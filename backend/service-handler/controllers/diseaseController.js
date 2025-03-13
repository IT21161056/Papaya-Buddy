const  Disease = require("../models/Disease");


const createNewDisease = async (req, res) => {
    try {
      const { name, affected_area, symptoms, disease_type, description } = req.body;
      if (!name || !affected_area || !symptoms || !disease_type || !description) {
        return res.status(400).json({ message: "All fields are required" });
      }
  
      const diseaseObject = { name, affected_area, symptoms, disease_type, description };
      const disease = await Disease.create(diseaseObject);
  
      if (disease) {
        res.status(201).json({ message: `New disease created` });
      } else {
        res.status(400).json({ message: "Invalid disease data received" });
      }
    } catch (error) {
      res.status(500).json({ message: error.message });
    }
  };
  

  module.exports ={
    createNewDisease,

  }