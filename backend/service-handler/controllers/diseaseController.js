const asyncHandler = require("express-async-handler");
const Disease = require("../models/disease");

const createNewDisease = asyncHandler(async (req, res) => {
  const {
    name,
    affected_area,
    symptoms,
    disease_type,
    virus_family,
    virus_genus,
    description,
    transmission_method,
    causes,
    preventive_measures,
    suggested_image_urls,
    severity,
    common_in_season,
    common_regions,
    related_diseases,
  } = req.body;

  // Check required fields according to the schema
  if (
    !name ||
    !affected_area ||
    !symptoms ||
    !disease_type ||
    !description ||
    !preventive_measures
  ) {
    res.status(400);
    throw new Error("Required fields are missing");
  }

  // Create disease object with all possible fields
  const diseaseObject = {
    name,
    affected_area,
    symptoms,
    disease_type,
    description,
    preventive_measures,
  };

  // Add optional fields if they exist
  if (virus_family) diseaseObject.virus_family = virus_family;
  if (virus_genus) diseaseObject.virus_genus = virus_genus;
  if (transmission_method)
    diseaseObject.transmission_method = transmission_method;
  if (causes) diseaseObject.causes = causes;
  if (suggested_image_urls)
    diseaseObject.suggested_image_urls = suggested_image_urls;
  if (severity) diseaseObject.severity = severity;
  if (common_in_season) diseaseObject.common_in_season = common_in_season;
  if (common_regions) diseaseObject.common_regions = common_regions;
  if (related_diseases) diseaseObject.related_diseases = related_diseases;

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
  const { name, disease_type, affected_area, severity } = req.query;
  let query = {};

  // Enhanced filtering options
  if (name) {
    query.name = { $regex: new RegExp(name, "i") };
  }
  if (disease_type) {
    query.disease_type = disease_type;
  }
  if (affected_area) {
    query.affected_area = affected_area;
  }
  if (severity) {
    query.severity = severity;
  }

  const diseases = await Disease.find(query).sort({ name: 1 });

  if (Object.keys(query).length > 0 && diseases.length === 0) {
    res.status(404);
    throw new Error("No diseases found matching the criteria");
  }

  res.status(200).json({
    success: true,
    count: diseases.length,
    data: diseases,
  });
});

const getDiseaseById = asyncHandler(async (req, res) => {
  const disease = await Disease.findById(req.params.id).populate(
    "related_diseases",
    "name affected_area disease_type"
  );

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

  // Accept all fields from the model
  const {
    name,
    affected_area,
    symptoms,
    disease_type,
    virus_family,
    virus_genus,
    description,
    transmission_method,
    causes,
    preventive_measures,
    suggested_image_urls,
    severity,
    common_in_season,
    common_regions,
    related_diseases,
  } = req.body;

  // Ensure at least one field is provided for update
  if (Object.keys(req.body).length === 0) {
    res.status(400);
    throw new Error("Please provide at least one field to update");
  }

  // Create update object with all provided fields
  const updateData = {};
  if (name !== undefined) updateData.name = name;
  if (affected_area !== undefined) updateData.affected_area = affected_area;
  if (symptoms !== undefined) updateData.symptoms = symptoms;
  if (disease_type !== undefined) updateData.disease_type = disease_type;
  if (virus_family !== undefined) updateData.virus_family = virus_family;
  if (virus_genus !== undefined) updateData.virus_genus = virus_genus;
  if (description !== undefined) updateData.description = description;
  if (transmission_method !== undefined)
    updateData.transmission_method = transmission_method;
  if (causes !== undefined) updateData.causes = causes;
  if (preventive_measures !== undefined)
    updateData.preventive_measures = preventive_measures;
  if (suggested_image_urls !== undefined)
    updateData.suggested_image_urls = suggested_image_urls;
  if (severity !== undefined) updateData.severity = severity;
  if (common_in_season !== undefined)
    updateData.common_in_season = common_in_season;
  if (common_regions !== undefined) updateData.common_regions = common_regions;
  if (related_diseases !== undefined)
    updateData.related_diseases = related_diseases;

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

// Added new controller to handle related diseases
const getRelatedDiseases = asyncHandler(async (req, res) => {
  const { id } = req.params;

  const disease = await Disease.findById(id).populate(
    "related_diseases",
    "name affected_area disease_type description suggested_image_urls"
  );

  if (!disease) {
    res.status(404);
    throw new Error("Disease not found");
  }

  res.status(200).json({
    success: true,
    count: disease.related_diseases.length,
    data: disease.related_diseases,
  });
});

// Added search by symptoms
const searchDiseasesBySymptom = asyncHandler(async (req, res) => {
  const { symptom } = req.query;

  if (!symptom) {
    res.status(400);
    throw new Error("Symptom query parameter is required");
  }

  const diseases = await Disease.find({
    symptoms: { $regex: new RegExp(symptom, "i") },
  }).sort({ name: 1 });

  res.status(200).json({
    success: true,
    count: diseases.length,
    data: diseases,
  });
});

module.exports = {
  createNewDisease,
  getAllDiseases,
  getDiseaseById,
  updateDisease,
  getRelatedDiseases,
  searchDiseasesBySymptom,
};
