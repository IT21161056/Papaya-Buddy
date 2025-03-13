const express = require("express");
const historyController = require("../controllers/historyController");
const router = express.Router();

router.route("/create_history").post(historyController.createNewPredictionHistory);
router.route("/get_user_history/:userid").get(historyController.getHistoryByUserId);

module.exports = router;
