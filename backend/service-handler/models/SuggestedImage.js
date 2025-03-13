import mongoose from "mongoose";
const Schema = mongoose.Schema;

const suggestedImagesSchema = new Schema({
    Disease: {
        type: mongoose.Types.ObjectId,
        ref: 'Disease',
        required: true
    },
    urls: [{
        type: String,
        required: true
    }],
});

export default mongoose.model("Suggested_Image", suggestedImagesSchema);