const express = require("express");
const {
  createNewDisease,
  getAllDiseases,
  getDiseaseById,
  updateDisease,
  deleteDisease,
} = require("../controllers/diseaseController");
const router = express.Router();

router.post("/", createNewDisease);
router.get("/", getAllDiseases);
router.get("/:id", getDiseaseById);
router.patch("/:id", updateDisease);
router.delete("/:id", deleteDisease);

module.exports = router;
