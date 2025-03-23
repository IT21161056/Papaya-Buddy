from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from io import BytesIO
import numpy as np
print(np.__version__)

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
efficientnet_model = torch.load("./models/efficientnet_model_2.pth", map_location=device)
efficientnet_model.to(device)
efficientnet_model.eval()

vit_model = torch.load("./models/vit_entire_model.pth", map_location=device)
vit_model.to(device)
vit_model.eval()

# Define transforms
transform_health = transforms.Compose([
    transforms.Resize((224, 224)),   
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_disease = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),          
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

disease_labels = {0: "Cercospora", 1: "Mosaic"}

def predict_health(image: Image.Image):
    image = transform_health(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = efficientnet_model(image)
    probs = F.softmax(output, dim=1)
    confidence, predicted_class = torch.max(probs, dim=1)
    return predicted_class.item(), confidence.item()

def predict_disease(image: Image.Image):
    image = transform_disease(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = vit_model(image)
    probs = F.softmax(output.logits, dim=1)
    confidence, predicted_class = torch.max(probs, dim=1)
    disease_name = disease_labels.get(predicted_class.item(), "Unknown")
    return disease_name, confidence.item()

@app.post("/predict-leaf-disease")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(BytesIO(await file.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    health_status, health_confidence = predict_health(image)

    if health_status == 0:
        return JSONResponse(content={
            "health_status": "Healthy",
            "confidence": health_confidence
        })
    else:
        disease_name, disease_confidence = predict_disease(image)
        return JSONResponse(content={
            "health_status": "Unhealthy",
            "disease": disease_name,
            "confidence": disease_confidence
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5010)
