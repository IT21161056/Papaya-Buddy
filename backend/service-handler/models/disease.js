const mongoose = require("mongoose");
const Schema = mongoose.Schema;

const diseaseSchema = new Schema({
  name: {
    type: String,
    required: true,
  },
  affected_area: {
    type: String,
    required: true,
    enum: ["leaves", "stem", "fruit", "root"],
  },
  symptoms: {
    type: [String],
    required: true,
  },
  disease_type: {
    type: String,
    required: true,
    enum: ["viral", "bacterial", "fungal", "parasitic", "pest"],
  },
  description: {
    type: String,
    required: true,
    trim: true,
  },
  // transmission_method: {
  //   type: String,
  //   enum: ["airborne", "soilborne", "waterborne", "vector-borne"],
  // },
  preventive_measures: {
    type: String,
    required: true,
    trim: true,
  },
  suggested_image_urls: [
    {
      type: String,
      required: true,
    },
  ],
  created_at: {
    type: Date,
    default: Date.now,
  },
});

module.exports = mongoose.model("Disease", diseaseSchema);
