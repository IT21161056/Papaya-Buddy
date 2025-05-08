const fs = require("fs");
const cors = require("cors");
const path = require("path");
const dotenv = require("dotenv");
const express = require("express");
const swaggerJsDoc = require("swagger-jsdoc");
const swaggerUi = require("swagger-ui-express");
const diseaseRoutes = require("./routes/diseaseRoutes");
const connectMongoDb = require("./config/dbConnection");
const historyRoutes = require("./routes/historyRoutes");
const treatmentRoutes = require("./routes/treatmentRoutes");
const maturityRoutes = require("./routes/maturityRoutes");
const cloudinaryRoutes = require("./routes/cloudinaryRoutes");
const errorMiddleware = require("./middleware/errorMiddleware");
const suggestedImageRoutes = require("./routes/suggestedImageRoute");
const communityRoutes = require("./routes/communityRoutes");

connectMongoDb();

dotenv.config();

const app = express();

const PORT = process.env.PORT || 5016;
const BASE_URL = process.env.API_BASE_URL || "/api/v1";

const swaggerOptions = {
  definition: {
    openapi: "3.0.0",
    info: {
      title: "API Documentation",
      version: "1.0.0",
      description:
        "This collection provides a structured set of API requests to interact with the Papaya Buddy mobile application's backend, built with Node.js and Express. The API facilitates disease identification, prediction history tracking, plant maturity stage management, and treatment recommendations.",
      contact: {
        name: "API Support",
        email: "seprojectgroup123@gmail.com",
      },
    },
    servers: [
      {
        url: `http://localhost:${PORT || 5080}${BASE_URL}`,
        description: "Development server",
      },
    ],
  },
  apis: ["./routes/*.js", "./models/*.js", "./server.js"],
};

const swaggerDocs = swaggerJsDoc(swaggerOptions);
app.use(`${BASE_URL}/docs`, swaggerUi.serve, swaggerUi.setup(swaggerDocs));

// Middleware
app.use(cors());
app.use(express.json());

app.use((req, res, next) => {
  const timestamp = new Date().toISOString();
  console.log(`[${timestamp}] ${req.method} ${req.url}`);
  next();
});

const uploadsDir = path.join(__dirname, "uploads");
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir);
}

/**
 * @swagger
 * components:
 *   schemas:
 *     ApiResponse:
 *       type: object
 *       properties:
 *         status:
 *           type: string
 *           example: success
 *         message:
 *           type: string
 *           example: API is running
 */

/**
 * @swagger
 * /:
 *   get:
 *     summary: Check if API is running
 *     tags: [API Status]
 *     responses:
 *       200:
 *         description: API status
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ApiResponse'
 */
app.get(`${BASE_URL}`, (req, res) => {
  res.json({
    status: "success",
    message: "API is running",
  });
});

//cloudinary routes
app.use(`${BASE_URL}/upload`, cloudinaryRoutes);

// disease routes
app.use(`${BASE_URL}/disease`, diseaseRoutes);

// treatment routes
app.use(`${BASE_URL}/treatment`, treatmentRoutes);

// history routes
app.use(`${BASE_URL}/history`, historyRoutes);

// maturity routes
app.use(`${BASE_URL}/maturity`, maturityRoutes);

// suggested image routes (commented out in your original code)
app.use(`${BASE_URL}/suggested-images`, suggestedImageRoutes);

app.use(`${BASE_URL}/community`,communityRoutes);

// Error handling middleware
app.use(errorMiddleware);

// Server setup

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log(`API base URL: ${BASE_URL}`);
  console.log(`Swagger docs available at: ${BASE_URL}/docs`);
});
