const express = require("express");
const {
  createNewPredictionHistory,
  getHistoryByUserId,
  getHistories,
} = require("../controllers/historyController");
const upload = require("../middleware/uploadMiddleware");
const router = express.Router();

/**
 * @swagger
 * components:
 *   schemas:
 *     History:
 *       type: object
 *       required:
 *         - userId
 *         - predictionResult
 *       properties:
 *         _id:
 *           type: string
 *           description: Auto-generated MongoDB ID
 *         userId:
 *           type: string
 *           description: ID of the user associated with the prediction history
 *         predictionResult:
 *           type: string
 *           description: Result of the prediction (e.g., "Healthy Leaf")
 *         uploaded_img:
 *           type: string
 *           description: URL of the uploaded image
 *         createdAt:
 *           type: string
 *           format: date-time
 *           description: Date and time when the prediction history was created
 *         updatedAt:
 *           type: string
 *           format: date-time
 *           description: Date and time when the prediction history was last updated
 */

/**
 * @swagger
 * /api/v1/history/create_history:
 *   post:
 *     summary: Create a new prediction history
 *     tags: [History]
 *     requestBody:
 *       required: true
 *       content:
 *         multipart/form-data:
 *           schema:
 *             type: object
 *             properties:
 *               uploaded_img:
 *                 type: string
 *                 format: binary
 *                 description: Image file for the prediction
 *               userId:
 *                 type: string
 *                 description: ID of the user associated with the prediction
 *                 example: "64d5f7a9b1f8f8b8f8f8f8f8"
 *               predictionResult:
 *                 type: string
 *                 description: Result of the prediction
 *                 example: "Healthy Leaf"
 *     responses:
 *       201:
 *         description: Prediction history created successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 status:
 *                   type: string
 *                   example: success
 *                 data:
 *                   $ref: '#/components/schemas/History'
 *       400:
 *         description: Bad request - Invalid data
 *       500:
 *         description: Server error
 */
router
  .route("/create_history")
  .post(upload.single("uploaded_img"), createNewPredictionHistory);

/**
 * @swagger
 * /api/v1/history/get_user_history/{userId}:
 *   get:
 *     summary: Get prediction history by user ID
 *     tags: [History]
 *     parameters:
 *       - in: path
 *         name: userId
 *         schema:
 *           type: string
 *         required: true
 *         description: ID of the user to fetch prediction history for
 *         example: "64d5f7a9b1f8f8b8f8f8f8f8"
 *     responses:
 *       200:
 *         description: List of prediction histories for the user
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 status:
 *                   type: string
 *                   example: success
 *                 data:
 *                   type: array
 *                   items:
 *                     $ref: '#/components/schemas/History'
 *       404:
 *         description: User not found
 *       500:
 *         description: Server error
 */
router.route("/get_user_history/:userId").get(getHistoryByUserId);

/**
 * @swagger
 * /api/v1/history:
 *   get:
 *     summary: Get all prediction histories
 *     tags: [History]
 *     responses:
 *       200:
 *         description: List of all prediction histories
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 status:
 *                   type: string
 *                   example: success
 *                 data:
 *                   type: array
 *                   items:
 *                     $ref: '#/components/schemas/History'
 *       500:
 *         description: Server error
 */
router.get("/", getHistories);

module.exports = router;