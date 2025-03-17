const express = require("express");
const {
  createNewPredictionHistory,
  getHistoryByUserId,
  getHistories,
} = require("../controllers/historyController");
const upload = require("../middleware/uploadMiddleware");
const router = express.Router();

router
  .route("/create_history")
  .post(upload.single("uploaded_img"), createNewPredictionHistory);
router.route("/get_user_history/:userId").get(getHistoryByUserId);
router.get("/", getHistories);

module.exports = router;
