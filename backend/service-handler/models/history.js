const mongoose = require("mongoose");
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
    // treatmentId: [{
    //     type: mongoose.Types.ObjectId,
    //     ref: 'Treatment'
    // }],
    diseaseId: {
        type: mongoose.Types.ObjectId,
        ref: 'Disease'
    },
    // suggested_image_list_id: {
    //     type: mongoose.Types.ObjectId,
    //     ref: 'Suggested_Image'
    // },
    created_at: {
        type: Date,
        default: Date.now
    }
});


module.exports = mongoose.model("History",historySchema);