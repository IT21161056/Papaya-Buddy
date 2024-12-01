from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess_input
from io import BytesIO
from PIL import Image
import tensorflow as tf
import uvicorn

# Initialize the FastAPI app
app = FastAPI()

# Enable CORS
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
dense_model = tf.keras.models.load_model("D:/papaya_diseases/training/saved_modal/papaya_disease_model.h5")
eff_model = tf.keras.models.load_model("D:/papaya_diseases/training/efficientNet_model/efficient_classify_model.h5")

# Define class labels
eff_class_labels = {0: "fruit", 1: "leaf", 2: "other"}  # EfficientNet classes
dense_class_labels = {0: "Healthy Fruit", 1: "Healthy Leaf", 2: "Mealy Bug", 3: "Mite Bug"}  # DenseNet classes

eff_threshold = 0.9  # Threshold for EfficientNet

# Preprocess the image
def preprocess_image(img_data: bytes, target_size, model_type='efficientnet'):
    img = Image.open(BytesIO(img_data))
    img = img.resize(target_size)
    img_array = np.array(img)

    if model_type == 'efficientnet':
        img_array = eff_preprocess_input(img_array)  # EfficientNet preprocessing
    elif model_type == 'densenet':
        img_array = img_array / 255.0  # Normalize for DenseNet

    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# EfficientNet prediction
def predict_eff(img_data: bytes):
    processed_img = preprocess_image(img_data, target_size=(224, 224), model_type='efficientnet')
    predictions = eff_model.predict(processed_img)
    eff_index = int(np.argmax(predictions[0]))
    eff_confidence = float(predictions[0][eff_index]) * 100

    if eff_confidence < eff_threshold * 100:
        return "Unrecognized", eff_confidence, None

    eff_label = eff_class_labels[eff_index]
    return eff_label, eff_confidence, processed_img

# DenseNet prediction
def predict_dense(img_array):
    predictions = dense_model.predict(img_array)
    dense_index = int(np.argmax(predictions[0]))
    dense_confidence = float(predictions[0][dense_index]) * 100
    dense_label = dense_class_labels[dense_index]
    return dense_label, dense_confidence

@app.get("/ping")
async def ping():
    return {"message": "Server is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_data = await file.read()

    # Predict using EfficientNet
    eff_label, eff_confidence, processed_img = predict_eff(img_data)

    response = {
        "EfficientNet": {
            "label": eff_label,
            "confidence": eff_confidence,
        }
    }

    # If classified as "fruit" or "leaf," pass to DenseNet
    if eff_label in ["fruit", "leaf"] and processed_img is not None:
        dense_img = preprocess_image(img_data, target_size=(224, 224), model_type='densenet')
        dense_label, dense_confidence = predict_dense(dense_img)
        response["DenseNet"] = {
            "label": dense_label,
            "confidence": dense_confidence,
        }
    else:
        response["DenseNet"] = {
            "label": "Not Applicable",
            "confidence": None,
        }

    return response

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
