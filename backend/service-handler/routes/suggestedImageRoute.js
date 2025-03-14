const express = require("express");
const suggestedImageController = require("../controllers/suggestedImageController");
const router = express.Router();

router.route("/suggested_image").post(suggestedImageController.createNewSuggestedImage);


module.exports = router;
