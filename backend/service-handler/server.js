// File: server.js
const express = require("express");
const dotenv = require("dotenv");
const cors = require("cors");
const fs = require("fs");
const path = require("path");
const cloudinaryRoutes = require("./routes/cloudinaryRoutes");
const errorMiddleware = require("./middleware/errorMiddleware");
const diseaseRoutes = require("./routes/diseaseRoutes");
const connectMongoDb = require("./config/dbConnection");
const suggestedImageRoutes = require("./routes/suggestedImageRoute");
const treatmentRoutes = require("./routes/treatmentRoutes");
const historyRoutes = require("./routes/historyRoutes");

connectMongoDb()

// Load environment variables
dotenv.config();

// Initialize Express app
const app = express();

// Middleware
app.use(cors());
app.use(express.json());

// Create uploads directory if it doesn't exist
const uploadsDir = path.join(__dirname, "uploads");
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir);
}

// external services routes
app.use("/api", cloudinaryRoutes);



//disease routes
app.use("/disease", diseaseRoutes);

//suggested image routes
app.use("/suggested_image",suggestedImageRoutes);

//treatment routes
app.use("/treatment", treatmentRoutes);

//history routes
app.use("/history", historyRoutes);



// Error handling middleware
app.use(errorMiddleware);

// Server setup
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
