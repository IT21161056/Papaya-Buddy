const express = require("express");
const suggestedImageController = require("../controllers/suggestedImageController");
const router = express.Router();

/**
 * @swagger
 * components:
 *   schemas:
 *     SuggestedImage:
 *       type: object
 *       required:
 *         - imageUrl
 *         - diseaseId
 *       properties:
 *         _id:
 *           type: string
 *           description: Auto-generated MongoDB ID
 *         imageUrl:
 *           type: string
 *           description: URL of the suggested image
 *           example: "https://res.cloudinary.com/dylqha5li/image/upload/v1742023649/mite_2_ahadez.jpg"
 *         diseaseId:
 *           type: string
 *           description: ID of the disease associated with the image
 *           example: "64d5f7a9b1f8f8b8f8f8f8f8"
 *         createdAt:
 *           type: string
 *           format: date-time
 *           description: Date and time when the image was created
 *         updatedAt:
 *           type: string
 *           format: date-time
 *           description: Date and time when the image was last updated
 */

/**
 * @swagger
 * /api/v1/suggested-images/suggested_image:
 *   post:
 *     summary: Create a new suggested image
 *     tags: [Suggested Images]
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - imageUrl
 *               - diseaseId
 *             properties:
 *               imageUrl:
 *                 type: string
 *                 description: URL of the suggested image
 *                 example: "https://res.cloudinary.com/dylqha5li/image/upload/v1742023649/mite_2_ahadez.jpg"
 *               diseaseId:
 *                 type: string
 *                 description: ID of the disease associated with the image
 *                 example: "64d5f7a9b1f8f8b8f8f8f8f8"
 *     responses:
 *       201:
 *         description: Suggested image created successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 status:
 *                   type: string
 *                   example: success
 *                 data:
 *                   $ref: '#/components/schemas/SuggestedImage'
 *       400:
 *         description: Bad request - Invalid data
 *       500:
 *         description: Server error
 */
router.route("/suggested_image").post(suggestedImageController.createNewSuggestedImage);

module.exports = router;