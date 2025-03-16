const express = require("express");
const {
  createNewDisease,
  getAllDiseases,
  getDiseaseById,
  updateDisease,
  getRelatedDiseases,
  searchDiseasesBySymptom,
} = require("../controllers/diseaseController");
const router = express.Router();

router.post("/", createNewDisease);
router.get("/", getAllDiseases);
router.get("/:id", getDiseaseById);
router.patch("/:id", updateDisease);

router.get("/search/symptoms", searchDiseasesBySymptom);
router.get("/:id/related", getRelatedDiseases);

module.exports = router;
