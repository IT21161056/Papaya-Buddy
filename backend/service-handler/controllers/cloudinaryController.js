const cloudinary = require("../config/cloudinary");
const fs = require("fs");
const asyncHandler = require("express-async-handler");

// Upload image to Cloudinary
const uploadImage = asyncHandler(async (req, res) => {
  if (!req.file) {
    res.status(400);
    throw new Error("No file uploaded");
  }

  try {
    const result = await cloudinary.uploader.upload(req.file.path, {
      folder: "uploads",
      resource_type: "auto",
    });

    // Remove file from server after upload
    fs.unlinkSync(req.file.path);

    res.status(200).json({
      success: true,
      message: "File uploaded successfully",
      data: {
        url: result.secure_url,
        public_id: result.public_id,
        format: result.format,
        width: result.width,
        height: result.height,
        bytes: result.bytes,
      },
    });
  } catch (error) {
    // Clean up the file if it exists
    if (req.file && req.file.path && fs.existsSync(req.file.path)) {
      fs.unlinkSync(req.file.path);
    }
    throw error;
  }
});

// Get image details by public ID
const getImageById = asyncHandler(async (req, res) => {
  console.log("hello");
  const { publicId } = req.params;
  console.log("publicId", publicId);

  const result = await cloudinary.api.resource(publicId);

  res.status(200).json({
    success: true,
    data: result,
  });
});

// Delete image by public ID
const deleteImage = asyncHandler(async (req, res) => {
  const { publicId } = req.params;

  const result = await cloudinary.uploader.destroy(publicId);

  res.status(200).json({
    success: true,
    message: "Image deleted successfully",
    data: result,
  });
});

module.exports = {
  uploadImage,
  getImageById,
  deleteImage,
};
