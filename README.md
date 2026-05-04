<img width="192" height="150" alt="papaya_buddy_architecture" src="https://github.com/user-attachments/assets/4dbc607c-8ec5-444c-8660-7e076196a5a0" /># 🌿 Papaya Buddy

> **IT4010 Research Project** — An AI-powered mobile application for comprehensive papaya plant health monitoring, disease detection, and maturity assessment.

---

## 📖 Overview

Papaya Buddy is a cross-platform mobile application built with **Flutter** that leverages multiple deep learning models via a **microservices backend** to help farmers and agricultural professionals monitor the health of their papaya plants. The app can detect leaf diseases, fruit diseases, pest infestations, and assess fruit maturity — all from a simple photo taken with a smartphone.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🍃 **Leaf Disease Detection** | Identifies leaf conditions (Healthy, Cercospora, Mosaic) using EfficientNet & Vision Transformer models |
| 🍈 **Fruit Disease Detection** | Detects fruit diseases (Healthy, Powdery Mildew, Ring Spot) using a custom CNN model |
| 🐛 **Pest Detection** | Classifies pest types (Mealy Bug, Mite) using DenseNet & EfficientNet |
| 🌡️ **Maturity Detection** | Grades papaya maturity (Mature, Partially Mature, Not Mature, Rotten) using MobileNet |
| 💬 **Community Forum** | Allows users to ask questions and share knowledge with the farming community |
| 📋 **Diagnosis History** | View and track past diagnoses and treatment recommendations |
| 🔐 **Authentication** | Secure login with Email/Password and Google Sign-In via Firebase |
| 💳 **Subscription Plans** | Premium subscription options for advanced features |

---

## 🏗️ Architecture

Papaya Buddy follows a **microservices architecture** with a Flutter frontend communicating through a Kong API Gateway.

<img width="2125" height="1656" alt="papaya_buddy_architecture" src="https://github.com/user-attachments/assets/ebb01214-1b11-41e1-a8ef-6be8aa8ef344" />

---

## 🧠 ML Models

| Service | Framework | Model(s) | Classes |
|---|---|---|---|
| Leaf Detection | PyTorch | EfficientNet + Vision Transformer (ViT) | Healthy, Cercospora, Mosaic |
| Fruit Detection | TensorFlow/Keras | Custom CNN | Healthy Fruit, Powdery Mildew, Ring Spot |
| Pest Detection | TensorFlow/Keras | DenseNet + EfficientNet | Healthy, Mealy Bug, Mite |
| Maturity Detection | TensorFlow/Keras | MobileNet | Mature, Partially Mature, Not Mature, Rotten |

---

## 🛠️ Tech Stack

### Mobile App
- **Framework:** Flutter (Dart) `SDK ^3.6.1`
- **State Management:** Flutter built-in
- **Authentication:** Firebase Auth, Google Sign-In
- **Database:** Cloud Firestore
- **Key Packages:** `http`, `geolocator`, `geocoding`, `carousel_slider`, `lottie`, `flutter_animate`, `image_picker`

### Backend Services
- **API Gateway:** Kong (DB-less mode, declarative config)
- **Users Service:** Node.js + Express + MongoDB (Dockerized)
- **Service Handler:** Node.js + Express + Mongoose + Cloudinary + Swagger
- **ML Services:** FastAPI / Flask + TensorFlow + PyTorch
- **Containerization:** Docker + Docker Compose

---

## 📁 Project Structure

```
Papaya-Buddy/
├── mobile_app/                  # Flutter application
│   ├── lib/
│   │   ├── views/
│   │   │   ├── auth/            # Login, Signup, Profile
│   │   │   ├── home/            # Home screen
│   │   │   ├── diagonosisView/  # Diagnosis list & disease details
│   │   │   ├── diseaseView/     # Disease information view
│   │   │   ├── healthyView/     # Healthy plant view
│   │   │   ├── maturityView/    # Maturity detection & info
│   │   │   ├── predictionView/  # Prediction result view
│   │   │   ├── community/       # Community forum
│   │   │   └── subscription/    # Subscription plans
│   │   ├── models/              # Data models
│   │   ├── services/            # API service calls
│   │   ├── routes/              # App routing
│   │   ├── theme/               # App theming
│   │   └── widgets/             # Reusable widgets
│   ├── assets/                  # Images, icons, animations
│   └── pubspec.yaml
│
└── backend/
    ├── docker-compose.yml       # Orchestration for all services
    ├── gateway/                 # Kong API Gateway config
    │   └── kong.yml
    ├── users-service/           # User management (Node.js)
    ├── service-handler/         # Business logic handler (Node.js)
    ├── leaf-detection/          # Leaf disease ML service (FastAPI + PyTorch)
    ├── fruit-detection/         # Fruit disease ML service (FastAPI + TF)
    ├── pest-detection/          # Pest detection ML service (FastAPI + TF)
    └── maturity-detection/      # Maturity detection ML service (Flask + TF)
```

---

## 🚀 Getting Started

### Prerequisites

- Flutter SDK `^3.6.1`
- Dart SDK
- Docker & Docker Compose
- Python `3.11+` (for ML services)
- Node.js `18+`
- Firebase project configured

---

### 📱 Mobile App Setup

**1. Clone the repository**

```bash
git clone https://github.com/IT21161056/Papaya-Buddy.git
cd Papaya-Buddy/mobile_app
```

**2. Install Flutter dependencies**

```bash
flutter pub get
```

**3. Configure Firebase**

Follow the instructions in `mobile_app/FIREBASE_SETUP_README.md` to connect your Firebase project.

**4. Run the app**

```bash
flutter run
```

---

### 🐳 Backend Setup (Docker — Recommended)

The easiest way to run all services at once:

```bash
cd backend
docker-compose up --build
```

To stop all services:

```bash
docker-compose down
```

**Service Ports after startup:**

| Service | Port |
|---|---|
| Kong Proxy (API Gateway) | `5000` |
| Kong HTTPS | `5443` |
| Kong Admin API | `5003` |
| Users Service | `5001` |

---

### 🐍 ML Services Setup (Manual)

Each ML service can be run independently. Follow the steps below for each service inside `backend/<service-name>/`.

**1. Check available Python versions (macOS)**

```bash
ls /usr/local/Cellar/python@*
```

**2. Create a virtual environment (Python 3.11 recommended)**

```bash
rm -rf .venv
python3.11 -m venv .venv
source .venv/bin/activate       # macOS/Linux
.venv\Scripts\activate          # Windows
```

**3. Upgrade pip and install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**4. Run the service**

```bash
python main.py
```

> **Note:** Each ML service requires trained model files (`.h5` or `.pth`) placed in the service's `models/` directory. These are not included in the repository and must be trained or obtained separately.

#### Required Model Files

| Service | Required Files |
|---|---|
| `leaf-detection` | `models/efficientnet_model_2.pth`, `models/vit_entire_model.pth` |
| `fruit-detection` | `models/custom_cnn_papaya_model.h5` |
| `pest-detection` | `models/papaya_disease_model.h5`, `models/efficient_classify_model.h5` |
| `maturity-detection` | `models/papaya_maturity_model.h5` |

#### Torch Installation Issues

If `torch` fails to install:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### API Documentation (FastAPI services)

Once running, visit:
- Swagger UI: `http://127.0.0.1:<PORT>/docs`
- ReDoc: `http://127.0.0.1:<PORT>/redoc`

---

## 🔌 API Endpoints

| Method | Endpoint | Service | Description |
|---|---|---|---|
| `POST` | `/predict-leaf-disease` | Leaf Detection | Detect disease in a papaya leaf image |
| `POST` | `/predict-fruit-disease` | Fruit Detection | Detect disease in a papaya fruit image |
| `POST` | `/predict-pest` | Pest Detection | Classify pests from an uploaded image |
| `POST` | `/predict-maturity` | Maturity Detection | Assess papaya fruit maturity stage |

All ML endpoints accept a `multipart/form-data` request with a `file` field containing the image.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 👨‍💻 Authors

Developed as part of the **IT4010 Research Project**.

---

> 🌱 *Empowering papaya farmers with AI-driven plant health insights.*
