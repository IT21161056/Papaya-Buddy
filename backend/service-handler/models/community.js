const mongoose = require("mongoose");
const Schema = mongoose.Schema;

const commentSchema = new Schema({
  user: {
    type: String,
    required: true,
  },
  text: {
    type: String,
    required: true,
  },
  createdAt: {
    type: Date,
    default: Date.now,
  },
});

const communitySchema = new Schema({
  userId: {
    type: String,
    required: true
  },
  name:{
    type: String,
    default: "Anonymous"
  },
  uploaded_img: {
    type: String,
  },
  question: {
    type: String,
    required: true,
  },
  description: {
    type: String,
    required: true,
  },
  comments:{
    type: [commentSchema],
    default:[]
  },
  createdAt: {
    type: Date,
    default: Date.now,
  },
});

module.exports = mongoose.model("Community", communitySchema);
