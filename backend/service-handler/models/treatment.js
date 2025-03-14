const mongoose = require("mongoose");
const Schema = mongoose.Schema;

const treatmentSchema = new Schema({
  method: {
    type: String,
    required: true,
  },
  description: {
    type: String,
    required: true,
    trim: true, // Optional: Ensures there's no extra space at the beginning/end
  },
  diseaseId: {
    type: mongoose.Types.ObjectId,
    ref: "Disease",
    required: true,
  },
  treatment_type: {
    type: String,
    enum: ["chemical", "biological", "cultural", "mechanical", "organic"],
    required: true,
  },
  effectiveness: {
    type: String,
    enum: ["high", "moderate", "low", "unknown"],
    required: false,
  },
  side_effects: {
    type: String,
    trim: false,
  },
  precautions: {
    type: String,
    trim: false,
  },
  created_at: {
    type: Date,
    default: Date.now,
  },
});

module.exports = mongoose.model("Treatment", treatmentSchema);
