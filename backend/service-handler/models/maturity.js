const mongoose = require("mongoose");
const Schema = mongoose.Schema;

const maturityStageSchema = new Schema({
  stage: {
    type: String,
    enum: ["Not Mature", "Partially Mature", "Mature", "Rotten"],
    required: true,
  },
  description: {
    type: String,
    required: true,
  },
  timeToReach: {
    type: String, // Example: "0–2 months after fruit set"
    required: true,
  },
  timeGapToNextStage: {
    type: String, // Example: "1–2 weeks"
    required: true,
  },
  bestTimeToHarvest: {
    type: String,
    required: true,
  },
});

module.exports = mongoose.model("Maturity", maturityStageSchema);
