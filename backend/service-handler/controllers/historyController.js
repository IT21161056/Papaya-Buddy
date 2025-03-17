const History = require("../models/History");
const asyncHandler = require("express-async-handler");
const cloudinary = require("../config/cloudinary");
const fs = require("fs");

const createNewPredictionHistory = async (req, res) => {
  try {
    const { userId, diseaseId } = req.body;
    const uploaded_img = req.file;

    if (!diseaseId || !userId) {
      return res
        .status(400)
        .json({ message: "diseaseId and userId are required" });
    }

    if (!uploaded_img) {
      return res
        .status(400)
        .json({ message: "No image provided in the request body" });
    }

    let uploaded_img_url;
    try {
      const result = await cloudinary.uploader.upload(uploaded_img.path, {
        folder: "prediction_history",
        resource_type: "auto",
      });

      fs.unlinkSync(uploaded_img.path);
      uploaded_img_url = result.secure_url;
    } catch (error) {
      if (
        uploaded_img &&
        uploaded_img.path &&
        fs.existsSync(uploaded_img.path)
      ) {
        fs.unlinkSync(uploaded_img.path);
      }
      return res.status(500).json({
        message: "Error uploading image to Cloudinary",
        error: error.message,
      });
    }

    const historyObject = {
      uploaded_img_url,
      userId,
      diseaseId,
    };
    const createdHistory = await History.create(historyObject);
    if (createdHistory) {
      res.status(201).json({
        success: true,
        message: `New history created`,
        history: createdHistory,
      });
    } else {
      res
        .status(400)
        .json({ success: false, message: "Invalid history data received" });
    }
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

const getHistoryByUserId = async (req, res) => {
  try {
    const { userId } = req.params;
    if (!userId) {
      return res.status(400).json({ message: "userid is required" });
    }
    const historyList = await History.find({ userId }).populate({
      path: "diseaseId",
      model: "Disease",
      select: "name description", // Select which disease fields to include
    });

    if (historyList.length > 0) {
      res.status(200).json(historyList);
    } else {
      res.status(404).json({ message: "No history found for this user" });
    }
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

const getHistories = asyncHandler(async (req, res) => {
  const histories = await History.find().populate({
    path: "diseaseId",
    model: "Disease",
    select: "name affected_area disease_type description", // Select which disease fields to include
  });
  res.json(histories);
});

module.exports = {
  createNewPredictionHistory,
  getHistoryByUserId,
  getHistories,
};
