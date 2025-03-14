const History = require("../models/History");

const createNewPredictionHistory = async (req, res) => {
    try {
        const { userid, treatmentId, diseaseId, suggested_image_list_id } = req.body;
        if (!diseaseId || !userid) {
            return res.status(400).json({ message: "diseaseId and userid are required" });
        }
        const historyObject = {
            uploaded_img_url,
            userid,
            treatmentId: treatmentId || [],
            diseaseId: diseaseId || null,
            suggested_image_list_id: suggested_image_list_id || {}
        };
        const createdHistory = await History.create(historyObject);

        if (createdHistory) {
            res.status(201).json({ message: `New history created`, history: createdHistory });
        } else {
            res.status(400).json({ message: "Invalid history data received" });
        }
    } catch (error) {
        res.status(500).json({ message: error.message });
    }
}

const getHistoryByUserId = async (req, res) => {
    try {
        const { userid } = req.params;
        if (!userid) {
            return res.status(400).json({ message: "userid is required" });
        }
        const historyList = await History.find({ userid });

        if (historyList.length > 0) {
            res.status(200).json({ message: "History list retrieved successfully", historyList });
        } else {
            res.status(404).json({ message: "No history found for this user" });
        }
    } catch (error) {
        res.status(500).json({ message: error.message });
    }
};

module.exports = {
    createNewPredictionHistory,
    getHistoryByUserId
}