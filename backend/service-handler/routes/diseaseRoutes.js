const express = require("express");
const {
  createNewDisease,
  getAllDiseases,
  getDiseaseById,
  updateDisease,
  deleteDisease,
} = require("../controllers/diseaseController");
const router = express.Router();

/**
 * @swagger
 * components:
 *   schemas:
 *     Disease:
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
 *           description: Name of the disease
 *         description:
 *           type: string
 *           description: Description of the disease
 *         symptoms:
 *           type: array
 *           items:
 *             type: string
 *           description: List of symptoms associated with the disease
 *         images:
 *           type: array
 *           items:
 *             type: string
 *           description: URLs of disease images
 *         createdAt:
 *           type: string
 *           format: date-time
 *           description: Date and time when the disease was created
 *         updatedAt:
 *           type: string
 *           format: date-time
 *           description: Date and time when the disease was last updated
 */

/**
 * @swagger
 * /disease:
 *   post:
 *     summary: Create a new disease
 *     tags: [Diseases]
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
 *               description:
 *                 type: string
 *               symptoms:
 *                 type: array
 *                 items:
 *                   type: string
 *               images:
 *                 type: array
 *                 items:
 *                   type: string
 *     responses:
 *       201:
 *         description: Disease created successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 status:
 *                   type: string
 *                   example: success
 *                 data:
 *                   $ref: '#/components/schemas/Disease'
 *       400:
 *         description: Bad request - Invalid data
 *       500:
 *         description: Server error
 */
router.post("/", createNewDisease);

/**
 * @swagger
 * /disease:
 *   get:
 *     summary: Get all diseases
 *     tags: [Diseases]
 *     responses:
 *       200:
 *         description: List of all diseases
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
 *                     $ref: '#/components/schemas/Disease'
 *       500:
 *         description: Server error
 */
router.get("/", getAllDiseases);

/**
 * @swagger
 * /disease/{id}:
 *   get:
 *     summary: Get disease by ID
 *     tags: [Diseases]
 *     parameters:
 *       - in: path
 *         name: id
 *         schema:
 *           type: string
 *         required: true
 *         description: Disease ID
 *     responses:
 *       200:
 *         description: Disease data
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 status:
 *                   type: string
 *                   example: success
 *                 data:
 *                   $ref: '#/components/schemas/Disease'
 *       404:
 *         description: Disease not found
 *       500:
 *         description: Server error
 */
router.get("/:id", getDiseaseById);

/**
 * @swagger
 * /disease/{id}:
 *   patch:
 *     summary: Update disease by ID
 *     tags: [Diseases]
 *     parameters:
 *       - in: path
 *         name: id
 *         schema:
 *           type: string
 *         required: true
 *         description: Disease ID
 *     requestBody:
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               name:
 *                 type: string
 *               description:
 *                 type: string
 *               symptoms:
 *                 type: array
 *                 items:
 *                   type: string
 *               images:
 *                 type: array
 *                 items:
 *                   type: string
 *     responses:
 *       200:
 *         description: Disease updated successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 status:
 *                   type: string
 *                   example: success
 *                 data:
 *                   $ref: '#/components/schemas/Disease'
 *       404:
 *         description: Disease not found
 *       500:
 *         description: Server error
 */
router.patch("/:id", updateDisease);

/**
 * @swagger
 * /disease/{id}:
 *   delete:
 *     summary: Delete disease by ID
 *     tags: [Diseases]
 *     parameters:
 *       - in: path
 *         name: id
 *         schema:
 *           type: string
 *         required: true
 *         description: Disease ID
 *     responses:
 *       200:
 *         description: Disease deleted successfully
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
 *                   example: Disease deleted successfully
 *       404:
 *         description: Disease not found
 *       500:
 *         description: Server error
 */
router.delete("/:id", deleteDisease);

module.exports = router;
