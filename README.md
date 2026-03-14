# 🩺 Pneumonia Detection System with Explainable AI

**Final Year Project: Explainable Deep Learning Framework for Pneumonia Detection using Chest X-Ray Images**

A comprehensive deep learning system that detects pneumonia from chest X-ray images using transfer learning and provides clinically interpretable explanations through multiple XAI techniques (Grad-CAM, SHAP, LIME).

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [API Endpoints](#api-endpoints)
- [Frontend](#frontend)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- **Deep Learning Model**: Transfer learning with ResNet50/MobileNetV2 for accurate pneumonia detection
- **Multiple XAI Methods**:
  - **Grad-CAM**: Primary explanation method with heatmap visualization
  - **SHAP**: Feature importance analysis
  - **LIME**: Local interpretable explanations
- **Full-Stack Application**: React frontend + Flask backend
- **Clinical Interpretability**: Visual explanations for healthcare professionals
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

## 🏗 Architecture

```
Input (Chest X-ray Image)
    ↓
Preprocessing (Resize, Normalize)
    ↓
CNN Model (ResNet50/MobileNetV2)
    ↓
Prediction (Pneumonia / Normal)
    ↓
XAI Module (Grad-CAM / SHAP / LIME)
    ↓
Visualization (Heatmaps + Explanations)
    ↓
Display (Web Interface)
```

## 🛠 Tech Stack

### Backend (AI Engine)
- **Python 3.8+**
- **TensorFlow/Keras**: Deep learning framework
- **Flask**: REST API backend
- **OpenCV**: Image processing
- **SHAP/LIME**: XAI libraries

### Frontend
- **React.js**: Modern UI framework
- **Axios**: HTTP client
- **React Dropzone**: File upload component

### Data Processing
- **NumPy/Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualization
- **scikit-learn**: Evaluation metrics

## 📂 Project Structure

```
pneumonia-xai/
│
├── dataset/
│   ├── train/          # Training images
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── val/            # Validation images
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── test/           # Test images
│       ├── NORMAL/
│       └── PNEUMONIA/
│
├── models/
│   ├── train_model.py          # Model training script
│   ├── pneumonia_model.h5      # Trained model
│   └── pneumonia_model_finetuned.h5
│
├── xai/
│   ├── gradcam.py              # Grad-CAM implementation
│   ├── shap_explain.py         # SHAP implementation
│   └── lime_explain.py         # LIME implementation
│
├── app/
│   └── app.py                  # Flask backend API
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ImageUploader.js
│   │   │   ├── PredictionResult.js
│   │   │   └── XAIExplanations.js
│   │   ├── App.js
│   │   └── index.js
│   └── package.json
│
├── utils/
│   └── data_utils.py           # Data preprocessing utilities
│
├── evaluation/
│   ├── evaluate_model.py       # Model evaluation script
│   ├── results.json            # Evaluation results
│   └── confusion_matrix.png    # Confusion matrix visualization
│
├── static/
│   ├── uploads/                 # Uploaded images
│   └── results/                # Generated visualizations
│
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Node.js 16+ and npm
- CUDA-capable GPU (optional, for faster training)

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd pneumonia-xai
```

### Step 2: Install Python Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

## 📊 Dataset Setup

### Download Dataset

1. Download the Chest X-Ray Images (Pneumonia) dataset from Kaggle:
   - [Kaggle Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

2. Extract and organize the dataset:
   ```
   dataset/
   ├── train/
   │   ├── NORMAL/
   │   └── PNEUMONIA/
   ├── val/
   │   ├── NORMAL/
   │   └── PNEUMONIA/
   └── test/
       ├── NORMAL/
       └── PNEUMONIA/
   ```

### Dataset Statistics

- **Training**: ~5,000 images
- **Validation**: ~1,000 images
- **Test**: ~600 images
- **Classes**: Normal, Pneumonia

## 💻 Usage

### 1. Train the Model

```bash
python models/train_model.py
```

This will:
- Load and preprocess the dataset
- Train a ResNet50-based model with transfer learning
- Fine-tune the model
- Save the best model to `models/pneumonia_model.h5`

### 2. Evaluate the Model

```bash
python evaluation/evaluate_model.py
```

This generates:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Classification Report
- Saved results in `evaluation/` directory

### 3. Start Backend Server

```bash
python app/app.py
```

The Flask API will start on `http://localhost:5000`

### 4. Start Frontend

```bash
cd frontend
npm start
```

The React app will open at `http://localhost:3000`

## 🎯 Model Training

### Training Configuration

Edit `models/train_model.py` to customize:

- **Base Model**: ResNet50 or MobileNetV2
- **Image Size**: Default (224, 224)
- **Batch Size**: Default 32
- **Epochs**: Default 20 (initial) + 10 (fine-tuning)
- **Data Augmentation**: Rotation, flipping, zoom, etc.

### Training Process

1. **Transfer Learning**: Freeze base model, train classifier
2. **Fine-tuning**: Unfreeze top layers, train with lower learning rate
3. **Early Stopping**: Prevents overfitting
4. **Model Checkpointing**: Saves best model based on validation accuracy

## 📈 Evaluation

### Metrics Computed

- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown

### Expected Performance

- **Accuracy**: > 90%
- **Precision**: > 90%
- **Recall**: > 90%
- **F1-Score**: > 90%

## 🔌 API Endpoints

### Health Check
```
GET /api/health
```

### Prediction
```
POST /api/predict
Content-Type: multipart/form-data
Body: file (image)
```

### Grad-CAM Explanation
```
POST /api/explain/gradcam
Content-Type: multipart/form-data
Body: file (image)
```

### SHAP Explanation
```
POST /api/explain/shap
Content-Type: multipart/form-data
Body: file (image)
```

### LIME Explanation
```
POST /api/explain/lime
Content-Type: multipart/form-data
Body: file (image)
```

### All Explanations
```
POST /api/explain/all
Content-Type: multipart/form-data
Body: file (image)
```

## 🖥 Frontend

### Features

- **Drag & Drop Upload**: Easy image upload interface
- **Real-time Prediction**: Instant results
- **Interactive XAI Visualizations**: 
  - Grad-CAM heatmaps
  - SHAP feature importance
  - LIME explanations
- **Responsive Design**: Works on desktop and mobile
- **Clinical Interpretability**: Clear explanations for healthcare use

### Components

- **ImageUploader**: Handles file uploads
- **PredictionResult**: Displays prediction and confidence
- **XAIExplanations**: Shows XAI visualizations

## 📊 Results

### Model Performance

After training, you'll get:
- Training history plots
- Confusion matrix visualization
- Detailed evaluation metrics
- Saved model files

### XAI Visualizations

- **Grad-CAM**: Heatmaps showing important regions
- **SHAP**: Pixel-level feature importance
- **LIME**: Superpixel-based explanations

## 🎓 Project Report Sections

For your final year project report, include:

1. **Introduction**: Problem statement, objectives
2. **Literature Review**: Related work on pneumonia detection, XAI
3. **Methodology**: 
   - Dataset description
   - Model architecture (ResNet50/MobileNetV2)
   - XAI methods (Grad-CAM, SHAP, LIME)
4. **Implementation**: 
   - System architecture
   - Technology stack
   - Training process
5. **Results**: 
   - Model performance metrics
   - XAI visualizations
   - Clinical interpretation
6. **Discussion**: 
   - Comparison with baseline
   - XAI effectiveness
   - Limitations
7. **Conclusion**: Summary and future work

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size in `train_model.py`
2. **Model Not Found**: Train the model first using `train_model.py`
3. **SHAP Slow**: Reduce background samples in `shap_explain.py`
4. **Frontend Not Connecting**: Check backend is running on port 5000

## 📝 Notes

- The model requires GPU for efficient training (CPU works but slower)
- SHAP explanations can be slow; consider using fewer background samples
- Ensure dataset is properly organized before training
- Model files are large (~100MB+); add to `.gitignore` if using Git

## 🤝 Contributing

This is a final year project. Feel free to:
- Report issues
- Suggest improvements
- Fork and extend

## 📄 License

This project is for educational purposes.

