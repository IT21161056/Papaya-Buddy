// File: server.js
const fs = require("fs");
const cors = require("cors");
const path = require("path");
const dotenv = require("dotenv");
const express = require("express");
const diseaseRoutes = require("./routes/diseaseRoutes");
const connectMongoDb = require("./config/dbConnection");
const historyRoutes = require("./routes/historyRoutes");
const treatmentRoutes = require("./routes/treatmentRoutes");
const cloudinaryRoutes = require("./routes/cloudinaryRoutes");
const errorMiddleware = require("./middleware/errorMiddleware");
const suggestedImageRoutes = require("./routes/suggestedImageRoute");

connectMongoDb();

dotenv.config();

const app = express();

const BASE_URL = process.env.API_BASE_URL || "/api/v1";

// Middleware
app.use(cors());
app.use(express.json());

const uploadsDir = path.join(__dirname, "uploads");
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir);
}

// external services routes
app.use(`${BASE_URL}/service`, cloudinaryRoutes);

// disease routes
app.use(`${BASE_URL}/disease`, diseaseRoutes);

app.use(`${BASE_URL}/suggested_image`, suggestedImageRoutes);

app.use(`${BASE_URL}/treatment`, treatmentRoutes);

app.use(`${BASE_URL}/history`, historyRoutes);

app.get(`${BASE_URL}`, (req, res) => {
  res.json({
    status: "success",
    message: "API is running",
  });
});

// Error handling middleware
app.use(errorMiddleware);

// Server setup
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log(`API base URL: ${BASE_URL}`);
});
