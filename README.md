# 📱 PapayaBuddy: AI-Powered Papaya Disease, Pest & Maturity Detection

A deep learning–based mobile solution designed to assist farmers in **papaya disease detection, pest identification, and fruit maturity classification** using smartphone images.

---

## 🚀 Overview

Papaya cultivation is significantly affected by **diseases, pests, and improper maturity assessment**, leading to reduced yield and economic loss. This project introduces an AI-powered mobile application that enables **real-time diagnosis and decision support** for farmers.

The system leverages advanced deep learning models to:

* Detect **papaya diseases**
* Identify **pests (Mite & Mealy Bug)**
* Classify **fruit maturity levels**
* Provide **actionable remedy suggestions**

---

## 🧠 Key Features

* 🌿 **Disease Detection**

  * Cercospora Leaf Spot
  * Papaya Mosaic Virus
  * Papaya Ring Spot Virus (PRSV)
  * Powdery Mildew

* 🐛 **Pest Identification**

  * Mite detection
  * Mealy bug detection
  * Severity classification (mild, moderate, severe)

* 🍈 **Maturity Classification**

  * Unripe
  * Partially Ripe
  * Ripe
  * Rotten

* 💡 **Smart Remedy Suggestions**

  * Tailored treatments based on detected condition
  * Environment-aware recommendations

* 📷 **Real-time Mobile Image Processing**

---

## 🏗️ System Architecture

The system is built using a **microservices-inspired architecture** with integrated AI components:

* **Mobile App (Flutter)**
* **Backend Services (Cloud-based)**
* **Database (MongoDB)**
* **Cloud Platform (Microsoft Azure)**
* **Weather API Integration (OpenWeatherAPI)**

---

## 🤖 AI Models Used

| Component                 | Model                                        |
| ------------------------- | -------------------------------------------- |
| Disease Detection (Leaf)  | EfficientNetV2B0 + Vision Transformers (ViT) |
| Disease Detection (Fruit) | YOLOv5                                       |
| Pest Detection            | DenseNet121                                  |
| Maturity Classification   | MobileNetV2                                  |
| Supporting Models         | Custom CNN                                   |

These models were trained and evaluated using metrics such as:

* Accuracy
* Precision
* Recall
* F1-score 

---

## 📊 Dataset

* Images collected using **mobile devices from real farms**
* Verified by agricultural experts
* Includes:

  * Healthy & diseased leaves
  * Pest-infected samples
  * Different maturity stages
* Data augmentation applied for robustness 

---

## 📱 How It Works

1. User captures or uploads a papaya image
2. Image is preprocessed and analyzed by AI models
3. System identifies:

   * Disease / pest / maturity level
4. Results are displayed instantly
5. Recommended remedies and actions are provided

---

## 🧪 Results

* High accuracy across all components
* EfficientNet & ViT showed best performance for disease detection
* MobileNetV2 optimized for real-time mobile inference 

---

## 🌍 Impact

* Enables **early detection** of diseases and pests
* Reduces **crop losses and pesticide overuse**
* Supports **data-driven farming decisions**
* Improves **yield, quality, and profitability**

---

## 🛠️ Tech Stack

* **Frontend:** Flutter
* **Backend:** REST APIs
* **AI/ML:** TensorFlow / PyTorch
* **Database:** MongoDB
* **Cloud:** Microsoft Azure
* **APIs:** OpenWeatherAPI

---

## 📌 Future Improvements

* Expand dataset with more real-world samples
* Integrate hyperspectral imaging
* Improve model generalization
* Large-scale deployment for farmers

---

## 👨‍🎓 Academic Context

* BSc (Hons) in Information Technology (Software Engineering)
* Sri Lanka Institute of Information Technology
* Research Year: 2025 

---

## 📜 License

This project is for academic and research purposes. Please check repository license for usage permissions.

---

## 🤝 Contributions

Contributions, suggestions, and improvements are welcome!
