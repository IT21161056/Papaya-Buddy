const SuggestedImage = require("../models/SuggestedImage");


const createNewSuggestedImage = async (req, res) => {
    try {
        const { diseaseId, urls } = req.body;
        
        if (!diseaseId || !urls || !Array.isArray(urls) || urls.length === 0) {
            return res.status(400).json({ message: "Disease ID and at least one image URL are required" });
        }
        const suggestedImageObject = { diseaseId, urls };

        const createdSuggestedImage = await SuggestedImage.create(suggestedImageObject);

        if (createdSuggestedImage) {
            res.status(201).json({ message: `New suggested image created`, suggestedImage: createdSuggestedImage });
        } else {
            res.status(400).json({ message: "Invalid suggested image data received" });
        }
    } catch (error) {
        res.status(500).json({ message: error.message });
    }
};

module.exports = {
    createNewSuggestedImage,
}