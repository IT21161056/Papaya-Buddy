const express = require("express");
const {createNewPost, addCommentToPost, getPostById, getAllPosts} = require("../controllers/communityController");
const upload = require("../middleware/uploadMiddleware");
const router = express.Router();

router.route("/").post(upload.single("uploaded_img"),createNewPost);
router.route("/comment/:postId").post(addCommentToPost)
router.route("/:postId").get(getPostById)
router.route("/").get(getAllPosts)

module.exports = router;