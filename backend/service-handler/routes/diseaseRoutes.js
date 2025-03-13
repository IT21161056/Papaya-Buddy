const express = require("express");
const diseaseController = require("../controllers/diseaseController");
const router = express.Router();

router.route("/create_disease").post(diseaseController.createNewDisease);


module.exports = router;
