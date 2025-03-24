Papaya Maturity Classification API (MobileNetV2 + Flask)

This project is a papaya maturity classification system using a MobileNetV2 model and a Flask API. Users can upload an image of a papaya fruit, and the model will classify it into one of the following categories:

Not Mature (All Green)

Partially Mature (Mostly Green with Some Yellow)

Mature (Mostly Yellow with Some Green)

Rotten (All Yellow with Damage)

📌 Features

MobileNetV2-based deep learning model for image classification

Flask API to handle image uploads and classification

Easy integration with mobile apps (Flutter, React Native, etc.)

🚀 Setup Instructions

1️⃣ Clone the Repository

git clone https://github.com/your-repo/papaya-classification.git
cd papaya-classification

2️⃣ Install Dependencies

pip install flask torch torchvision pillow

3️⃣ Load or Train the Model

If you already have a trained MobileNetV2 model, place it in the project folder (e.g., mobilenet_papaya.pth). If you need to train one, ensure you have a dataset and use PyTorch to train it.

4️⃣ Run the Flask API

python app.py

The server will start on http://127.0.0.1:5000/.
