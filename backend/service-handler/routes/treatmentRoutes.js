const express = require("express");
const {
  getAllTreatments,
  createNewTreatment,
  getTreatmentById,
  getTreatmentsByDisease,
} = require("../controllers/treatmentController");
const router = express.Router();

router.get("/", getAllTreatments);
router.post("/", createNewTreatment);
router.get("/:id", getTreatmentById);
router.get("/by-disease/:id", getTreatmentsByDisease);

module.exports = router;
