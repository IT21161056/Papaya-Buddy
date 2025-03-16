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

  const papayaStageObject = {
    stage,
    description,
    timeToReach,
    timeGapToNextStage,
    bestTimeToHarvest,
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
  const papayaStage = await Maturity.findById(req.params.id);

  if (!papayaStage) {
    res.status(404);
    throw new Error("Papaya stage not found");
  }

  const updatedPapayaStage = await Maturity.findByIdAndUpdate(
    req.params.id,
    req.body,
    { new: true, runValidators: true }
  );

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
