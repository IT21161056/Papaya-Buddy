const express = require("express");
const {
  createPapayaStage,
  deletePapayaStage,
  getAllPapayaStages,
  getPapayaStageById,
  updatePapayaStage,
} = require("../controllers/maturityController");
const router = express.Router();

/**
 * @swagger
 * components:
 *   schemas:
 *     PapayaStage:
 *       type: object
 *       required:
 *         - name
 *         - description
 *       properties:
 *         _id:
 *           type: string
 *           description: Auto-generated MongoDB ID
 *         name:
 *           type: string
 *           description: Name of the papaya growth stage
 *         description:
 *           type: string
 *           description: Description of the papaya growth stage
 *         images:
 *           type: array
 *           items:
 *             type: string
 *           description: URLs of images for the papaya growth stage
 *         createdAt:
 *           type: string
 *           format: date-time
 *           description: Date and time when the stage was created
 *         updatedAt:
 *           type: string
 *           format: date-time
 *           description: Date and time when the stage was last updated
 */

/**
 * @swagger
 * /api/v1/maturity:
 *   post:
 *     summary: Create a new papaya growth stage
 *     tags: [Maturity]
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - name
 *               - description
 *             properties:
 *               name:
 *                 type: string
 *                 description: Name of the papaya growth stage
 *                 example: "Seedling"
 *               description:
 *                 type: string
 *                 description: Description of the papaya growth stage
 *                 example: "The initial stage of papaya growth."
 *               images:
 *                 type: array
 *                 items:
 *                   type: string
 *                 description: URLs of images for the papaya growth stage
 *                 example: ["https://example.com/image1.jpg", "https://example.com/image2.jpg"]
 *     responses:
 *       201:
 *         description: Papaya growth stage created successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 status:
 *                   type: string
 *                   example: success
 *                 data:
 *                   $ref: '#/components/schemas/PapayaStage'
 *       400:
 *         description: Bad request - Invalid data
 *       500:
 *         description: Server error
 */
router.post("/", createPapayaStage);

/**
 * @swagger
 * /api/v1/maturity:
 *   get:
 *     summary: Get all papaya growth stages
 *     tags: [Maturity]
 *     responses:
 *       200:
 *         description: List of all papaya growth stages
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
 *                     $ref: '#/components/schemas/PapayaStage'
 *       500:
 *         description: Server error
 */
router.get("/", getAllPapayaStages);

/**
 * @swagger
 * /api/v1/maturity/{id}:
 *   get:
 *     summary: Get papaya growth stage by ID
 *     tags: [Maturity]
 *     parameters:
 *       - in: path
 *         name: id
 *         schema:
 *           type: string
 *         required: true
 *         description: ID of the papaya growth stage
 *         example: "64d5f7a9b1f8f8b8f8f8f8f8"
 *     responses:
 *       200:
 *         description: Papaya growth stage data
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 status:
 *                   type: string
 *                   example: success
 *                 data:
 *                   $ref: '#/components/schemas/PapayaStage'
 *       404:
 *         description: Papaya growth stage not found
 *       500:
 *         description: Server error
 */
router.get("/:id", getPapayaStageById);

/**
 * @swagger
 * /api/v1/maturity/{id}:
 *   patch:
 *     summary: Update papaya growth stage by ID
 *     tags: [Maturity]
 *     parameters:
 *       - in: path
 *         name: id
 *         schema:
 *           type: string
 *         required: true
 *         description: ID of the papaya growth stage
 *         example: "64d5f7a9b1f8f8b8f8f8f8f8"
 *     requestBody:
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               name:
 *                 type: string
 *                 description: Updated name of the papaya growth stage
 *                 example: "Flowering"
 *               description:
 *                 type: string
 *                 description: Updated description of the papaya growth stage
 *                 example: "The stage when papaya plants start flowering."
 *               images:
 *                 type: array
 *                 items:
 *                   type: string
 *                 description: Updated URLs of images for the papaya growth stage
 *                 example: ["https://example.com/image3.jpg", "https://example.com/image4.jpg"]
 *     responses:
 *       200:
 *         description: Papaya growth stage updated successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 status:
 *                   type: string
 *                   example: success
 *                 data:
 *                   $ref: '#/components/schemas/PapayaStage'
 *       404:
 *         description: Papaya growth stage not found
 *       500:
 *         description: Server error
 */
router.patch("/:id", updatePapayaStage);

/**
 * @swagger
 * /api/v1/maturity/{id}:
 *   delete:
 *     summary: Delete papaya growth stage by ID
 *     tags: [Maturity]
 *     parameters:
 *       - in: path
 *         name: id
 *         schema:
 *           type: string
 *         required: true
 *         description: ID of the papaya growth stage
 *         example: "64d5f7a9b1f8f8b8f8f8f8f8"
 *     responses:
 *       200:
 *         description: Papaya growth stage deleted successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 status:
 *                   type: string
 *                   example: success
 *                 message:
 *                   type: string
 *                   example: Papaya growth stage deleted successfully
 *       404:
 *         description: Papaya growth stage not found
 *       500:
 *         description: Server error
 */
router.delete("/:id", deletePapayaStage);

module.exports = router;