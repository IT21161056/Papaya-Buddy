const asyncHandler = require("express-async-handler");
const Disease = require("../models/disease");

const createNewDisease = asyncHandler(async (req, res) => {
  const {
    name,
    affected_area,
    symptoms,
    disease_type,
    description,
    preventive_measures,
    suggested_image_urls,
  } = req.body;

  if (
    !name ||
    !affected_area ||
    !symptoms ||
    !disease_type ||
    !description ||
    !preventive_measures ||
    (!Array.isArray(suggested_image_urls) && suggested_image_urls.length === 0)
  ) {
    res.status(400);
    throw new Error("All fields are required");
  }

  const diseaseObject = {
    name,
    affected_area,
    symptoms,
    disease_type,
    description,
    preventive_measures,
    suggested_image_urls,
  };

  const disease = await Disease.create(diseaseObject);

  if (disease) {
    res.status(201).json({
      success: true,
      message: `New disease ${disease.name} created successfully`,
      data: {
        id: disease._id,
        name: disease.name,
        affected_area: disease.affected_area,
        disease_type: disease.disease_type,
      },
    });
  } else {
    res.status(400);
    throw new Error("Invalid disease data received");
  }
});

const getAllDiseases = asyncHandler(async (req, res) => {
  const { name } = req.query;
  let query = {};

  if (name) {
    query.name = { $regex: new RegExp(`^${name}$`, "i") };
  }

  const diseases = await Disease.find(query).sort({ name: 1 });

  if (name && diseases.length === 0) {
    res.status(404);
    throw new Error(`Disease with name "${name}" not found`);
  }

  res.status(200).json({
    success: true,
    count: diseases.length,
    data: diseases,
  });
});

const getDiseaseById = asyncHandler(async (req, res) => {
  const disease = await Disease.findById(req.params.id);

  if (!disease) {
    res.status(404);
    throw new Error("Disease not found");
  }

  res.status(200).json({
    success: true,
    data: disease,
  });
});

const updateDisease = asyncHandler(async (req, res) => {
  const { id } = req.params;

  const disease = await Disease.findById(id);

  if (!disease) {
    res.status(404);
    throw new Error("Disease not found");
  }

  const {
    name,
    affected_area,
    symptoms,
    disease_type,
    description,
    preventive_measures,
    suggested_image_urls,
  } = req.body;

  if (
    !name &&
    !affected_area &&
    !symptoms &&
    !disease_type &&
    !description &&
    !preventive_measures &&
    !suggested_image_urls
  ) {
    res.status(400);
    throw new Error("Please provide at least one field to update");
  }

  const updateData = {};
  if (name) updateData.name = name;
  if (affected_area) updateData.affected_area = affected_area;
  if (symptoms) updateData.symptoms = symptoms;
  if (disease_type) updateData.disease_type = disease_type;
  if (description) updateData.description = description;
  if (preventive_measures) updateData.preventive_measures = preventive_measures;
  if (suggested_image_urls.length)
    updateData.suggested_image_urls = suggested_image_urls;

  const updatedDisease = await Disease.findByIdAndUpdate(id, updateData, {
    new: true,
    runValidators: true,
  });

  res.status(200).json({
    success: true,
    message: `Disease "${updatedDisease.name}" updated successfully`,
    data: updatedDisease,
  });
});

const deleteDisease = asyncHandler(async (req, res) => {
  const { id } = req.params;

  if (!id) {
    res.status(400);
    throw new Error("Disease id is required.");
  }
  const result = await Disease.findOneAndDelete(id);

  res.status(200).json({
    success: true,
    message: `${result.name} disease deleted successfully`,
  });
});

module.exports = {
  createNewDisease,
  getAllDiseases,
  getDiseaseById,
  updateDisease,
  deleteDisease,
};
