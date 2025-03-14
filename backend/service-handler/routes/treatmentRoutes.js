const express = require("express");
const {
  createNewTreatment,
  getTreatmentById,
  getTreatmentsByDisease,
} = require("../controllers/treatmentController");
const router = express.Router();

router.post("/", createNewTreatment);
router.get("/:id", getTreatmentById);
router.get("/by-disease/:id", getTreatmentsByDisease);

module.exports = router;
