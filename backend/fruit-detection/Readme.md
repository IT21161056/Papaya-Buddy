# FastAPI Fruit Disease Detection - Windows Setup Guide

This guide explains how to set up and run the **FastAPI** application for leaf disease detection on Windows.

## Prerequisites

- Windows 10/11
- Python **3.12** or higher
- Virtual environment for dependency management
- Required Python packages listed in `requirements.txt`

## Installation Steps

### **1. Install Python 3.12**

Download and install Python 3.12 from the official Python website: [https://www.python.org/downloads/](https://www.python.org/downloads/)

Ensure **Add Python to PATH** is checked during installation.

### **2. Set Up a Virtual Environment**

Remove any existing virtual environment and create a new one:

```powershell
if (Test-Path .venv) { Remove-Item -Recurse -Force .venv }
python -m venv .venv
```

### **3. Activate the Virtual Environment**

```powershell
.\.venv\Scripts\activate
```

### **4. Upgrade Pip**

Ensure you have the latest version of **pip**:

```powershell
pip install --upgrade pip
```

### **5. Install Dependencies**

Install all required dependencies from `requirements.txt`:

```powershell
pip install -r requirements.txt
```

## Running the FastAPI Server

Once the setup is complete, you can run the FastAPI application:

```powershell
python main.py
```

Alternatively, you can run it with Uvicorn manually:

```powershell
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### **Access API Documentation**

- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Redoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

## Troubleshooting

### **TensorFlow Installation Issues**

If you encounter issues installing TensorFlow, install it with:

```powershell
pip install tensorflow
```

_(For GPU support, install the correct CUDA and cuDNN versions as per TensorFlow’s official documentation.)_

### **Missing Model Files**

Ensure that model files exist in the `models/` directory:

- `models/custom_cnn_papaya.h5`

If missing, re-train or download them before running the API.

## License

This project is licensed under the MIT License.
