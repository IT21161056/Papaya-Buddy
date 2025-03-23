from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from io import BytesIO
from PIL import Image

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load the CNN model
cnn_model = load_model("./models/custom_cnn_papaya_model.h5")  # Replace with your actual CNN model file

# Define class labels for CNN with proper formatting
CNN_CLASSES = {
    "healthy": "Healthy Fruit",
    "powdery_mildew": "Powdery Mildew",
    "ringspot": "Ring Spot"
}

def preprocess_image(img: Image.Image, target_size: tuple):
    """Preprocess image for model input."""
    img = img.resize(target_size)  # Resize to match model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
    return img_array

@app.post("/predict-fruit-disease")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
        cnn_input = preprocess_image(img, (256, 256))  # Use the input size for your CNN model
        cnn_pred = cnn_model.predict(cnn_input)
        predicted_class = list(CNN_CLASSES.keys())[np.argmax(cnn_pred)]
        formatted_class = CNN_CLASSES[predicted_class]  # Get formatted name

        return {
            "filename": file.filename,
            "disease_prediction": formatted_class
        }

    except Exception as e:
        return {"error": str(e)}

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Custom CNN Model API!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5011)
