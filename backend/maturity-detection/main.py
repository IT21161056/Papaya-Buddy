from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import tensorflow as tf

app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter Web

# Load your trained model
model = tf.keras.models.load_model("./models/papaya_maturity_model.h5")  # Update with your model path

# Define class labels
class_labels = ['mature', 'not_mature', 'partially_mature', 'rotten']

def predict_maturity(image):
    image = image.resize((224, 224))  # Resize to MobileNet input size
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Expand dimensions for model

    prediction = model.predict(image_array)
    predicted_class = class_labels[np.argmax(prediction)]

    return predicted_class

@app.route('/predict-maturity', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    image = Image.open(io.BytesIO(file.read())).convert("RGB")  # Read image

    predicted_class = predict_maturity(image)

    return jsonify({'predicted_class': predicted_class})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5015, debug=True)

