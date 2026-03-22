require('@tensorflow/tfjs-node');
const express = require("express");
const multer = require("multer");
const cors = require("cors");
const faceapi = require("face-api.js");
const canvas = require("canvas");
const admin = require("firebase-admin");

const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

const app = express();
app.use(cors());
// Parse JSON and urlencoded bodies (for base64 image payloads)
app.use(express.json({ limit: '20mb' }));
app.use(express.urlencoded({ extended: true, limit: '20mb' }));

// Multer setup (in-memory)
const upload = multer({ storage: multer.memoryStorage() });

// Firebase setup
let serviceAccount;
if (process.env.FIREBASE_SERVICE_ACCOUNT) {
  serviceAccount = JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT);
} else {
  try {
    serviceAccount = require("./serviceAccountKey.json");
  } catch (err) {
    console.error("No serviceAccountKey.json found and FIREBASE_SERVICE_ACCOUNT not set.");
  }
}

if (serviceAccount) {
  admin.initializeApp({
    credential: admin.credential.cert(serviceAccount)
  });
} else {
  console.warn("Firebase not initialized: Missing credentials.");
}

const db = admin.firestore();
const PORT = process.env.PORT || 5000;

// Global in-memory cache for speed
let faceDescriptors = new Map();

// Load models
async function loadModels() {
  await faceapi.nets.tinyFaceDetector.loadFromDisk("./models");
  await faceapi.nets.faceLandmark68Net.loadFromDisk("./models");
  await faceapi.nets.faceRecognitionNet.loadFromDisk("./models");
  console.log("Ultra-fast TinyFaceDetector models loaded");
}

/**
 * Refresh the local in-memory cache from Firestore
 */
async function refreshCache() {
  try {
    console.log("Refreshing face descriptor cache...");
    const snapshot = await db.collection("users").get();
    const newCache = new Map();
    snapshot.forEach(doc => {
      const data = doc.data();
      if (data && data.descriptor) {
        newCache.set(doc.id, new Float32Array(data.descriptor));
      }
    });
    faceDescriptors = newCache;
    console.log(`Cache loaded with ${faceDescriptors.size} users`);
  } catch (err) {
    console.error("Cache refresh error:", err);
  }
}

function validateDocId(id) {
  if (!id || typeof id !== 'string') return false;
  const s = id.toString().trim();
  if (!s) return false;
  if (s.includes('/')) return false; 
  return s;
}

// Start models then server
async function startServer() {
  await loadModels();
  await refreshCache(); 

  // Register endpoint
  app.post("/register", upload.single("image"), async (req, res) => {
    try {
      const rawId = req.body.userId || req.body.uid || req.body.username || req.body.employeeId;
      const docId = validateDocId(rawId);
      if (!docId) return res.status(400).json({ error: "No valid user id provided" });

      if (!req.file && !req.body.image) {
        return res.status(400).json({ error: "No image provided" });
      }

      const imageBuffer = req.file
        ? req.file.buffer
        : Buffer.from((req.body.image || '').replace(/^data:image\/\w+;base64,/, ""), "base64");

      const img = await canvas.loadImage(imageBuffer);

      // Use Tiny Face Detector for speed
      const detection = await faceapi
        .detectSingleFace(img, new faceapi.TinyFaceDetectorOptions())
        .withFaceLandmarks()
        .withFaceDescriptor();

      if (!detection) {
        return res.status(400).json({ error: "No face detected" });
      }

      const descriptor = Array.from(detection.descriptor);

      await db.collection("users").doc(docId).set({ descriptor });
      faceDescriptors.set(docId, detection.descriptor);

      res.json({ success: true, message: "Face registered", userId: docId });

    } catch (err) {
      console.error('Register error:', err);
      res.status(500).json({ error: err.message });
    }
  });

  // Login / Verify Face (HIGH PERFORMANCE)
  app.post("/login", upload.single("image"), async (req, res) => {
    const startTime = Date.now();
    console.log("--- Starting Recognition Scan ---");
    try {
      if (!req.file && !req.body.image) {
        return res.status(400).json({ error: "No image provided" });
      }

      console.time("Step 1: Image Loading");
      const imageBuffer = req.file
        ? req.file.buffer
        : Buffer.from((req.body.image || '').replace(/^data:image\/\w+;base64,/, ""), "base64");

      const img = await canvas.loadImage(imageBuffer);
      console.timeEnd("Step 1: Image Loading");

      console.time("Step 2: Face Detection (Tiny)");
      // detect with Tiny Face Detector (FAST)
      const detection = await faceapi
        .detectSingleFace(img, new faceapi.TinyFaceDetectorOptions())
        .withFaceLandmarks()
        .withFaceDescriptor();
      console.timeEnd("Step 2: Face Detection (Tiny)");

      if (!detection) {
        console.log("No face detected in scan.");
        return res.status(400).json({ error: "No face detected" });
      }

      console.time("Step 3: Matching Loop");
      const inputDescriptor = detection.descriptor;

      let bestMatch = null;
      let minDistance = 0.6;

      for (let [userId, descriptor] of faceDescriptors) {
        const distance = faceapi.euclideanDistance(inputDescriptor, descriptor);
        if (distance < minDistance) {
          minDistance = distance;
          bestMatch = userId;
        }
      }
      console.timeEnd("Step 3: Matching Loop");

      const duration = Date.now() - startTime;
      console.log(`--- Scan Complete: ${duration}ms ---`);

      if (bestMatch) {
        res.json({ success: true, userId: bestMatch, time: duration });
      } else {
        res.json({ success: false, time: duration });
      }

    } catch (err) {
      console.error('Login error:', err);
      res.status(500).json({ error: err.message });
    }
  });

  app.listen(PORT, () => {
    console.log(`ULTRA-FAST Server running on port ${PORT}`);
  });
}

startServer();
