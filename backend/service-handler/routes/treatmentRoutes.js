const express = require("express");
const {
  getAllTreatments,
  createNewTreatment,
  getTreatmentById,
  getTreatmentsByDisease,
} = require("../controllers/treatmentController");
const router = express.Router();

/**
 * @swagger
 * components:
 *   schemas:
 *     Treatment:
 *       type: object
 *       required:
 *         - name
 *         - description
 *         - diseaseId
 *       properties:
 *         _id:
 *           type: string
 *           description: Auto-generated MongoDB ID
 *         name:
 *           type: string
 *           description: Name of the treatment
 *           example: "Neem Oil Spray"
 *         description:
 *           type: string
 *           description: Description of the treatment
 *           example: "A natural pesticide to control mites and other pests."
 *         diseaseId:
 *           type: string
 *           description: ID of the disease associated with the treatment
 *           example: "64d5f7a9b1f8f8b8f8f8f8f8"
 *         createdAt:
 *           type: string
 *           format: date-time
 *           description: Date and time when the treatment was created
 *         updatedAt:
 *           type: string
 *           format: date-time
 *           description: Date and time when the treatment was last updated
 */

/**
 * @swagger
 * /api/v1/treatment:
 *   get:
 *     summary: Get all treatments
 *     tags: [Treatments]
 *     responses:
 *       200:
 *         description: List of all treatments
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
 *                     $ref: '#/components/schemas/Treatment'
 *       500:
 *         description: Server error
 */
router.get("/", getAllTreatments);

/**
 * @swagger
 * /api/v1/treatment:
 *   post:
 *     summary: Create a new treatment
 *     tags: [Treatments]
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - name
 *               - description
 *               - diseaseId
 *             properties:
 *               name:
 *                 type: string
 *                 description: Name of the treatment
 *                 example: "Neem Oil Spray"
 *               description:
 *                 type: string
 *                 description: Description of the treatment
 *                 example: "A natural pesticide to control mites and other pests."
 *               diseaseId:
 *                 type: string
 *                 description: ID of the disease associated with the treatment
 *                 example: "64d5f7a9b1f8f8b8f8f8f8f8"
 *     responses:
 *       201:
 *         description: Treatment created successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 status:
 *                   type: string
 *                   example: success
 *                 data:
 *                   $ref: '#/components/schemas/Treatment'
 *       400:
 *         description: Bad request - Invalid data
 *       500:
 *         description: Server error
 */
router.post("/", createNewTreatment);

/**
 * @swagger
 * /api/v1/treatment/{id}:
 *   get:
 *     summary: Get treatment by ID
 *     tags: [Treatments]
 *     parameters:
 *       - in: path
 *         name: id
 *         schema:
 *           type: string
 *         required: true
 *         description: ID of the treatment
 *         example: "64d5f7a9b1f8f8b8f8f8f8f8"
 *     responses:
 *       200:
 *         description: Treatment data
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 status:
 *                   type: string
 *                   example: success
 *                 data:
 *                   $ref: '#/components/schemas/Treatment'
 *       404:
 *         description: Treatment not found
 *       500:
 *         description: Server error
 */
router.get("/:id", getTreatmentById);

/**
 * @swagger
 * /api/v1/treatment/by-disease/{id}:
 *   get:
 *     summary: Get treatments by disease ID
 *     tags: [Treatments]
 *     parameters:
 *       - in: path
 *         name: id
 *         schema:
 *           type: string
 *         required: true
 *         description: ID of the disease
 *         example: "64d5f7a9b1f8f8b8f8f8f8f8"
 *     responses:
 *       200:
 *         description: List of treatments for the specified disease
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
 *                     $ref: '#/components/schemas/Treatment'
 *       404:
 *         description: Disease not found
 *       500:
 *         description: Server error
 */
router.get("/by-disease/:id", getTreatmentsByDisease);

module.exports = router;