const mongoose = require("mongoose");
const Schema = mongoose.Schema;

const suggestedImagesSchema = new Schema({
    diseaseId: {
        type: mongoose.Types.ObjectId,
        ref: 'Disease',
        required: true
    },
    urls: [{
        type: String,
        required: true
    }],
});

module.exports = mongoose.model("Suggested_Image",suggestedImagesSchema);