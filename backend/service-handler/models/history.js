const mongoose = require("mongoose");
const Schema = mongoose.Schema;

const historySchema = new Schema({
    uploaded_img: {
        type: String,
    },
    uploaded_img_url: {
        type: String,
        required: true
    },
    userId: {
        type: String,
        required: true
    },
    diseaseId: {
        type: mongoose.Types.ObjectId,
        ref: 'Disease'
    },
    created_at: {
        type: Date,
        default: Date.now
    }
    // treatmentId: [{
    //     type: mongoose.Types.ObjectId,
    //     ref: 'Treatment'
    // }],
    // suggested_image_list_id: {
    //     type: mongoose.Types.ObjectId,
    //     ref: 'Suggested_Image'
    // },
});


module.exports = mongoose.model("History",historySchema);