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
    const startTime = Date.now();
    console.log(`\n========== [REGISTER] Request received at ${new Date().toISOString()} ==========`);
    try {
      const rawId = req.body.userId || req.body.uid || req.body.username || req.body.employeeId;
      const docId = validateDocId(rawId);
      console.log(`[REGISTER] User ID: ${rawId} -> Validated: ${docId}`);
      if (!docId) {
        console.log(`[REGISTER] ❌ FAILED - Invalid user ID provided: "${rawId}"`);
        return res.status(400).json({ error: "No valid user id provided" });
      }

      if (!req.file && !req.body.image) {
        console.log(`[REGISTER] ❌ FAILED - No image provided for user: ${docId}`);
        return res.status(400).json({ error: "No image provided" });
      }

      const imageSource = req.file ? 'multipart file' : 'base64 body';
      const imageBuffer = req.file
        ? req.file.buffer
        : Buffer.from((req.body.image || '').replace(/^data:image\/\w+;base64,/, ""), "base64");
      console.log(`[REGISTER] Image received via ${imageSource}, size: ${(imageBuffer.length / 1024).toFixed(1)} KB`);

      const img = await canvas.loadImage(imageBuffer);
      console.log(`[REGISTER] Image loaded successfully, detecting face...`);

      // Use Tiny Face Detector for speed
      const detection = await faceapi
        .detectSingleFace(img, new faceapi.TinyFaceDetectorOptions())
        .withFaceLandmarks()
        .withFaceDescriptor();

      if (!detection) {
        const duration = Date.now() - startTime;
        console.log(`[REGISTER] ❌ FAILED - No face detected in image for user: ${docId} (took ${duration}ms)`);
        return res.status(400).json({ error: "No face detected" });
      }

      console.log(`[REGISTER] Face detected with confidence: ${(detection.detection.score * 100).toFixed(1)}%`);

      const descriptor = Array.from(detection.descriptor);

      await db.collection("users").doc(docId).set({ descriptor });
      faceDescriptors.set(docId, detection.descriptor);

      const duration = Date.now() - startTime;
      console.log(`[REGISTER] ✅ SUCCESS - Face registered for user: ${docId} (took ${duration}ms)`);
      console.log(`[REGISTER] Cache now has ${faceDescriptors.size} users`);
      res.json({ success: true, message: "Face registered", userId: docId });

    } catch (err) {
      const duration = Date.now() - startTime;
      console.error(`[REGISTER] ❌ ERROR after ${duration}ms:`, err.message);
      console.error(`[REGISTER] Stack trace:`, err.stack);
      res.status(500).json({ error: err.message });
    }
  });

  // Login / Verify Face (HIGH PERFORMANCE)
  app.post("/login", upload.single("image"), async (req, res) => {
    const startTime = Date.now();
    console.log(`\n========== [LOGIN] Request received at ${new Date().toISOString()} ==========`);
    console.log(`[LOGIN] Registered users in cache: ${faceDescriptors.size}`);
    try {
      if (!req.file && !req.body.image) {
        console.log(`[LOGIN] ❌ FAILED - No image provided in request`);
        return res.status(400).json({ error: "No image provided" });
      }

      const imageSource = req.file ? 'multipart file' : 'base64 body';
      const imageBuffer = req.file
        ? req.file.buffer
        : Buffer.from((req.body.image || '').replace(/^data:image\/\w+;base64,/, ""), "base64");
      console.log(`[LOGIN] Image received via ${imageSource}, size: ${(imageBuffer.length / 1024).toFixed(1)} KB`);

      const imgLoadStart = Date.now();
      const img = await canvas.loadImage(imageBuffer);
      console.log(`[LOGIN] Image loaded in ${Date.now() - imgLoadStart}ms`);

      // detect with Tiny Face Detector (FAST)
      const detectStart = Date.now();
      const detection = await faceapi
        .detectSingleFace(img, new faceapi.TinyFaceDetectorOptions())
        .withFaceLandmarks()
        .withFaceDescriptor();
      console.log(`[LOGIN] Face detection took ${Date.now() - detectStart}ms`);

      if (!detection) {
        const duration = Date.now() - startTime;
        console.log(`[LOGIN] ❌ FAILED - No face detected in image (took ${duration}ms)`);
        return res.status(400).json({ error: "No face detected" });
      }

      console.log(`[LOGIN] Face detected with confidence: ${(detection.detection.score * 100).toFixed(1)}%`);

      const inputDescriptor = detection.descriptor;

      let bestMatch = null;
      let minDistance = 0.6;
      let closestDistance = Infinity;

      const matchStart = Date.now();
      for (let [userId, descriptor] of faceDescriptors) {
        const distance = faceapi.euclideanDistance(inputDescriptor, descriptor);
        if (distance < closestDistance) {
          closestDistance = distance;
        }
        if (distance < minDistance) {
          minDistance = distance;
          bestMatch = userId;
        }
      }
      console.log(`[LOGIN] Matching against ${faceDescriptors.size} users took ${Date.now() - matchStart}ms`);

      const duration = Date.now() - startTime;

      if (bestMatch) {
        console.log(`[LOGIN] ✅ SUCCESS - Matched user: ${bestMatch} (distance: ${minDistance.toFixed(4)}, took ${duration}ms)`);
        res.json({ success: true, userId: bestMatch, time: duration });
      } else {
        console.log(`[LOGIN] ⚠️ NOT RECOGNIZED - No match found (closest distance: ${closestDistance.toFixed(4)}, threshold: 0.6, took ${duration}ms)`);
        res.json({ success: false, time: duration });
      }

    } catch (err) {
      const duration = Date.now() - startTime;
      console.error(`[LOGIN] ❌ ERROR after ${duration}ms:`, err.message);
      console.error(`[LOGIN] Stack trace:`, err.stack);
      res.status(500).json({ error: err.message });
    }
  });

  app.listen(PORT, () => {
    console.log(`ULTRA-FAST Server running on port ${PORT}`);
  });
}

startServer();
