const asyncHandler = require("express-async-handler");
const Treatment = require("../models/Treatment");

// @desc Get all treatments
// @route GET/treatment

const getAllTreatments = asyncHandler(async (req, res) => {
  const treatments = await Treatment.find().lean();

  if (!treatments?.length) {
    res.status(400);
    throw new Error("No treatments found");
  }

  res.status(200).json({
    success: true,
    count: treatments.length,
    data: treatments,
  });
});

// @desc Create new treatment
// @route POST/treatment

const createNewTreatment = asyncHandler(async (req, res) => {
  const {
    method,
    description,
    diseaseId,
    treatment_type,
    effectiveness,
    side_effects,
    precautions,
  } = req.body;

  if (!method || !description || !diseaseId || !treatment_type) {
    res.status(400);
    throw new Error(
      "Missing required fields: method, description, diseaseId, and treatment_type are required"
    );
  }

  // Validate treatment_type is in the enum
  const validTreatmentTypes = [
    "chemical",
    "biological",
    "cultural",
    "mechanical",
    "organic",
  ];
  if (!validTreatmentTypes.includes(treatment_type)) {
    res.status(400);
    throw new Error(
      `Invalid treatment_type. Must be one of: ${validTreatmentTypes.join(
        ", "
      )}`
    );
  }

  // Validate effectiveness if provided
  if (
    effectiveness &&
    !["high", "moderate", "low", "unknown"].includes(effectiveness)
  ) {
    res.status(400);
    throw new Error(
      "Invalid effectiveness value. Must be high, moderate, low, or unknown"
    );
  }

  const treatmentObject = {
    method,
    description,
    diseaseId,
    treatment_type,
    effectiveness,
    side_effects,
    precautions,
  };

  const createdTreatment = await Treatment.create(treatmentObject);

  if (createdTreatment) {
    res.status(201).json({
      success: true,
      message: `New treatment method ${method} created successfully`,
      data: createdTreatment,
    });
  } else {
    res.status(400);
    throw new Error("Invalid treatment data received");
  }
});

// @desc Get treatment by disease
// @route GET/treatment/:diseaseId

const getTreatmentsByDisease = asyncHandler(async (req, res) => {
  const { diseaseId } = req.params;

  if (!diseaseId) {
    res.status(400);
    throw new Error("Disease ID is required");
  }

  const treatments = await Treatment.find({ diseaseId });

  res.status(200).json({
    success: true,
    count: treatments.length,
    data: treatments,
  });
});

// @desc Get treatment by ID
// @route GET/treatment/:id

const getTreatmentById = asyncHandler(async (req, res) => {
  const treatment = await Treatment.findById(req.params.id).populate({
    path: "diseaseId",
    model: "Disease",
    select: "name affected_area disease_type description", // Select which disease fields to include
  });

  if (!treatment) {
    res.status(404);
    throw new Error("Treatment not found");
  }

  res.status(200).json(treatment);
});

module.exports = {
  getAllTreatments,
  createNewTreatment,
  getTreatmentsByDisease,
  getTreatmentById,
};
