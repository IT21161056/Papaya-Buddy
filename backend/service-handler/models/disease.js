const mongoose = require("mongoose");
const Schema = mongoose.Schema;

const diseaseSchema = new Schema({
  name: { type: String, required: true },

  affected_area: {
    type: String,
    required: true,
    enum: ["leaves", "stem", "fruit", "root", "whole plant"],
  },

  symptoms: { type: [String], required: true },

  disease_type: {
    type: String,
    required: true,
    enum: [
      "viral",
      "bacterial",
      "fungal",
      "parasitic",
      "pest",
      "nutrient deficiency",
    ],
  },

  virus_family: { type: String },
  virus_genus: { type: String },

  description: { type: String, required: true, trim: true },

  transmission_method: {
    type: [String],
    enum: [
      "airborne",
      "soilborne",
      "waterborne",
      "vector-borne",
      "contact",
      "seed-borne",
    ],
  },

  causes: {
    type: [
      {
        name: String,
        type: String,
        description: String,
        image_url: String,
      },
    ],
  },

  preventive_measures: { type: String, required: true, trim: true },

  suggested_image_urls: [{ type: String }],

  severity: {
    type: String,
    enum: ["low", "moderate", "high"],
  },

  common_in_season: {
    type: [String],
    enum: ["spring", "summer", "autumn", "winter"],
  },

  common_regions: { type: [String] },

  related_diseases: [
    {
      type: mongoose.Types.ObjectId,
      ref: "Disease",
    },
  ],

  created_at: { type: Date, default: Date.now },
});

module.exports = mongoose.model("Disease", diseaseSchema);
