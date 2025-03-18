const asyncHandler = require("express-async-handler");
const Maturity = require("../models/maturity"); // Adjust path as needed

/**
 * @desc    Create a new papaya stage
 * @route   POST /api/papaya-stages
 * @access  Private
 */
const createPapayaStage = asyncHandler(async (req, res) => {
  const {
    stage,
    description,
    timeToReach,
    timeGapToNextStage,
    bestTimeToHarvest,
    image_urls,
  } = req.body;

  if (
    !stage ||
    !description ||
    !timeToReach ||
    !timeGapToNextStage ||
    !bestTimeToHarvest
  ) {
    res.status(400);
    throw new Error("All fields are required");
  }

  // Validate allowed enum values
  const allowedStages = ["Not Mature", "Partially Mature", "Mature", "Rotten"];
  if (!allowedStages.includes(stage)) {
    res.status(400);
    throw new Error(
      `Invalid stage. Allowed values are: ${allowedStages.join(", ")}`
    );
  }

  const papayaStageObject = {
    stage,
    description,
    timeToReach,
    timeGapToNextStage,
    bestTimeToHarvest,
    image_urls: image_urls || [],
  };

  const papayaStage = await Maturity.create(papayaStageObject);

  if (papayaStage) {
    res.status(201).json({
      success: true,
      message: `New papaya stage "${papayaStage.stage}" created successfully`,
      data: {
        id: papayaStage._id,
        stage: papayaStage.stage,
        description: papayaStage.description,
        timeToReach: papayaStage.timeToReach,
        timeGapToNextStage: papayaStage.timeGapToNextStage,
        bestTimeToHarvest: papayaStage.bestTimeToHarvest,
        image_urls: papayaStage.image_urls,
      },
    });
  } else {
    res.status(400);
    throw new Error("Invalid papaya stage data received");
  }
});

/**
 * @desc    Get all papaya stages
 * @route   GET /api/papaya-stages
 * @access  Public
 */
const getAllPapayaStages = asyncHandler(async (req, res) => {
  const papayaStages = await Maturity.find().sort({ _id: 1 });

  res.status(200).json({
    success: true,
    count: papayaStages.length,
    data: papayaStages,
  });
});

/**
 * @desc    Get a single papaya stage by ID
 * @route   GET /api/papaya-stages/:id
 * @access  Public
 */
const getPapayaStageById = asyncHandler(async (req, res) => {
  const papayaStage = await Maturity.findById(req.params.id);

  if (!papayaStage) {
    res.status(404);
    throw new Error("Papaya stage not found");
  }

  res.status(200).json({
    success: true,
    data: papayaStage,
  });
});

/**
 * @desc    Update a papaya stage
 * @route   PUT /api/papaya-stages/:id
 * @access  Private
 */
const updatePapayaStage = asyncHandler(async (req, res) => {
  const { id } = req.params;
  const {
    stage,
    description,
    timeToReach,
    timeGapToNextStage,
    bestTimeToHarvest,
    image_urls,
  } = req.body;

  const papayaStage = await Maturity.findById(id);

  if (!papayaStage) {
    res.status(404);
    throw new Error("Papaya stage not found");
  }

  if (stage) {
    const allowedStages = [
      "Not Mature",
      "Partially Mature",
      "Mature",
      "Rotten",
    ];
    if (!allowedStages.includes(stage)) {
      res.status(400);
      throw new Error(
        `Invalid stage. Allowed values are: ${allowedStages.join(", ")}`
      );
    }
  }

  const updateFields = {};
  if (stage) updateFields.stage = stage;
  if (description) updateFields.description = description;
  if (timeToReach) updateFields.timeToReach = timeToReach;
  if (timeGapToNextStage) updateFields.timeGapToNextStage = timeGapToNextStage;
  if (bestTimeToHarvest) updateFields.bestTimeToHarvest = bestTimeToHarvest;
  if (image_urls) updateFields.image_urls = image_urls;

  const updatedPapayaStage = await Maturity.findByIdAndUpdate(
    id,
    updateFields,
    { new: true, runValidators: true }
  );

  if (!updatedPapayaStage) {
    res.status(400);
    throw new Error("Failed to update papaya stage");
  }

  res.status(200).json({
    success: true,
    message: `Papaya stage "${updatedPapayaStage.stage}" updated successfully`,
    data: updatedPapayaStage,
  });
});

/**
 * @desc    Delete a papaya stage
 * @route   DELETE /api/papaya-stages/:id
 * @access  Private
 */
const deletePapayaStage = asyncHandler(async (req, res) => {
  const papayaStage = await Maturity.findById(req.params.id);

  if (!papayaStage) {
    res.status(404);
    throw new Error("Papaya stage not found");
  }

  await Maturity.findByIdAndDelete(req.params.id);

  res.status(200).json({
    success: true,
    message: `Papaya stage "${papayaStage.stage}" deleted successfully`,
  });
});

module.exports = {
  createPapayaStage,
  getAllPapayaStages,
  getPapayaStageById,
  updatePapayaStage,
  deletePapayaStage,
};
