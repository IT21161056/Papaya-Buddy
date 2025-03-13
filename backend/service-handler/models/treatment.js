const mongoose = require("mongoose");
const Schema = mongoose.Schema;

const treatmentSchema = new Schema({
    method: {
        type: String,
        required: true
    },
    description: {
        type: String,
        required: true
    },
    diseaseId: {
        type: mongoose.Types.ObjectId,
        ref: 'Disease',
        required: true
    },
    historyId: {
        type: mongoose.Types.ObjectId,
        ref: 'History',
        required: true
    },
    created_at: {
        type: Date,
        default: Date.now
    }
});

module.exports = mongoose.model("Treatment",treatmentSchema);