const express = require("express");
const treatmentController = require("../controllers/treatmentController");
const router = express.Router();

router.route("/create_treatment").post(treatmentController.createNewTreatment);


module.exports = router;
