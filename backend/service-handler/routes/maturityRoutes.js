const express = require("express");
const {
  createPapayaStage,
  deletePapayaStage,
  getAllPapayaStages,
  getPapayaStageById,
  updatePapayaStage,
} = require("../controllers/maturityController");
const router = express.Router();

router.post("/", createPapayaStage);
router.get("/", getAllPapayaStages);
router.get("/:id", getPapayaStageById);
router.patch("/:id", updatePapayaStage);
router.delete("/:id", deletePapayaStage);

module.exports = router;
