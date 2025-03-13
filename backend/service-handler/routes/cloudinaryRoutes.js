// File: routes/uploadRoutes.js
const express = require("express");
const upload = require("../middleware/uploadMiddleware");
const {
  uploadImage,
  deleteImage,
  getImageById,
} = require("../controllers/cloudinaryController");

const router = express.Router();

// POST /api/upload
router.post("/upload", upload.single("image"), uploadImage);

// GET /api/images/:publicId
router.get("/images/:publicId", getImageById);

// DELETE /api/images/:publicId
router.delete("/images/:publicId", deleteImage);

module.exports = router;
