const Treatment = require("../models/Treatment");

const createNewTreatment = async (req, res) => {
    try {
        const { method, description, diseaseId, historyId } = req.body;

        if (!method || !description || !diseaseId || !historyId) {
            return res.status(400).json({ message: "All fields are required" });
        }

        const treatmentObject = { method, description, diseaseId, historyId };

        const createdTreatment = await Treatment.create(treatmentObject);

        if (createdTreatment) {
            res.status(201).json({ message: `New treatment created`, treatment: createdTreatment });
        } else {
            res.status(400).json({ message: "Invalid treatment data received" });
        }
    } catch (error) {
        res.status(500).json({ message: error.message });
    }
};


module.exports = {
    createNewTreatment,

}