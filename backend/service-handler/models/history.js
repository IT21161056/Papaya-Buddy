import mongoose from "mongoose";
const Schema = mongoose.Schema;

const historySchema = new Schema({
    uploaded_img_url: {
        type: String,
        required: true
    },
    userid: {
        type: String,
        required: true
    },
    treatment: [{
        type: mongoose.Types.ObjectId,
        ref: 'Treatment'
    }],
    disease: {
        type: mongoose.Types.ObjectId,
        ref: 'Disease'
    },
    suggested_images: [{
        type: mongoose.Types.ObjectId,
        ref: 'Suggested_Image'
    }],
    created_at: {
        type: Date,
        default: Date.now
    }
});


export default mongoose.model("History",historySchema);