import mongoose from "mongoose";
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
    Disease: {
        type: mongoose.Types.ObjectId,
        ref: 'Disease',
        required: true
    },
    History: {
        type: mongoose.Types.ObjectId,
        ref: 'History',
        required: true
    },
    created_at: {
        type: Date,
        default: Date.now
    }
});

export default mongoose.model("Treatment",treatmentSchema);