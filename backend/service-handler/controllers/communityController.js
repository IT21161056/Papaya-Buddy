const Community = require("../models/community");
const cloudinary = require("../config/cloudinary");
const fs = require("fs");

const createNewPost = async (req, res) => {
    try {
        const { userId, question, description } = req.body;
        const uploaded_img = req.file;

        if (!userId || !question || !description) {
            return res.status(400).json({ message: "userId, question, and description are required" });
        }

        let uploaded_img_url = null;

        if (uploaded_img) {
            try {
                const result = await cloudinary.uploader.upload(uploaded_img.path, {
                    folder: "community_images",
                    resource_type: "auto",
                });

                fs.unlinkSync(uploaded_img.path);
                uploaded_img_url = result.secure_url;
            } catch (error) {
                if (uploaded_img && uploaded_img.path && fs.existsSync(uploaded_img.path)) {
                    fs.unlinkSync(uploaded_img.path);
                }
                return res.status(500).json({
                    message: "Error uploading image to Cloudinary",
                    error: error.message,
                });
            }
        }
        const postObject = {
            userId,
            question,
            description,
            uploaded_img: uploaded_img_url,
        };

        const createdPost = await Community.create(postObject);

        if (createdPost) {
            res.status(201).json({
                success: true,
                message: "New community post created",
                post: createdPost,
            });
        } else {
            res.status(400).json({
                success: false,
                message: "Invalid post data received",
            });
        }
    } catch (err) {
        res.status(500).json({ message: err.message });
    }
};

const addCommentToPost = async (req, res) => {
    const { postId } = req.params;
    const { user, text } = req.body;

    if (!user || !text) {
        return res
            .status(400)
            .json({ message: "Both user and text are required for a comment" });
    }
    try {
        const post = await Community.findById(postId);

        if (!post) {
            return res.status(404).json({ message: "Post not found" });
        }
        const comment = {
            user,
            text,
            createdAt: new Date(),
        };
        post.comments.push(comment);

        const updatedPost = await post.save();

        res.status(200).json({
            success: true,
            message: "Comment added successfully",
            post: updatedPost,
        });
    } catch (err) {
        res.status(500).json({ message: err.message });
    }
};

const getPostById = async (req, res) => {
    const { postId } = req.params;

    try {
        const post = await Community.findById(postId);

        if (!post) {
            return res.status(404).json({ message: "Post not found" });
        }

        res.status(200).json(post);
    } catch (err) {
        res.status(500).json({ message: err.message });
    }
};

const getAllPosts = async (req, res) => {
    try {
        const posts = await Community.find().sort({ createdAt: -1 });

        res.status(200).json(posts);
    } catch (err) {
        res.status(500).json({ message: err.message });
    }
};


module.exports = {
    createNewPost,
    addCommentToPost,
    getPostById,
    getAllPosts
};
