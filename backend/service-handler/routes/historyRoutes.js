const express = require("express");
const historyController = require("../controllers/historyController");
const upload = require("../middleware/uploadMiddleware");
const router = express.Router();


router.route("/create_history").post(upload.single("uploaded_img"),historyController.createNewPredictionHistory);
router.route("/get_user_history/:userid").get(historyController.getHistoryByUserId);

module.exports = router;
