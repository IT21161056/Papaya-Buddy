# FastAPI Leaf Disease Detection - Setup Guide

This guide explains how to set up and run the **FastAPI** application for leaf disease detection.

## Prerequisites

- macOS with **Homebrew** installed
- Python **3.11** or higher
- Virtual environment for dependency management
- Required Python packages listed in `requirements.txt`

## Installation Steps

### **1. Install Python 3.11**

If you don't have Python 3.11 installed, use Homebrew:

```bash
brew install python@3.11
```

### **2. Set Up a Virtual Environment**

Remove any existing virtual environment and create a new one:

```bash
rm -rf .venv
python3.11 -m venv .venv
```

### **3. Activate the Virtual Environment**

```bash
source .venv/bin/activate
```

### **4. Upgrade Pip**

Ensure you have the latest version of **pip**:

```bash
pip install --upgrade pip
```

### **5. Install Dependencies**

Install all required dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Running the FastAPI Server

Once the setup is complete, you can run the FastAPI application:

```bash
python main.py
```

Alternatively, you can run it with Uvicorn manually:

```bash
uvicorn main:app --host 0.0.0.0 --port 5005 --reload
```

### **Access API Documentation**

- Swagger UI: [http://127.0.0.1:5005/docs](http://127.0.0.1:5005/docs)
- Redoc: [http://127.0.0.1:5005/redoc](http://127.0.0.1:5005/redoc)

## Troubleshooting

### **Torch Installation Issues**

If `torch` fails to install, try:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

_(For GPU support, install the correct CUDA version from PyTorch's official website.)_

### **Missing Model Files**

Ensure that model files exist in the `models/` directory:

- `models/efficientnet_model_2.pth`
- `models/vit_entire_model.pth`

If missing, re-train or download them before running the API.

## License

This project is licensed under the MIT License.
