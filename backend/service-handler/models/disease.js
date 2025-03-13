const mongoose = require('mongoose');
const Schema = mongoose.Schema;

const diseaseSchema = new Schema({
    name:{
        type:String,
        required:true
    },
    affected_area:{
        type:String,
        required:true
    },
    symptoms:{
        type:String,
        required:true
    },
    disease_type:{
        type:String,
        required:true
    },
    description:{
        type:String,
        required:true
    }
})

module.exports = mongoose.model("Disease",diseaseSchema);