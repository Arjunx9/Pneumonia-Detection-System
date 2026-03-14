# 🚀 Quick Start Guide

Get your Pneumonia Detection System up and running in minutes!

## Prerequisites Checklist

- [ ] Python 3.8+ installed
- [ ] Node.js 16+ installed
- [ ] Dataset downloaded from Kaggle
- [ ] GPU (optional but recommended)

## Step-by-Step Setup

### 1. Install Python Dependencies (5 minutes)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### 2. Prepare Dataset (10 minutes)

1. Download from: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. Extract to `dataset/` folder
3. Structure should be:
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

### 3. Train Model (30-60 minutes)

```bash
python models/train_model.py
```

**Note**: First run will take longer. Subsequent runs use cached weights.

### 4. Evaluate Model (2 minutes)

```bash
python evaluation/evaluate_model.py
```

Check `evaluation/` folder for results.

### 5. Start Backend (1 minute)

```bash
python app/app.py
```

Backend runs on `http://localhost:5000`

### 6. Start Frontend (2 minutes)

```bash
cd frontend
npm install  # First time only
npm start
```

Frontend opens at `http://localhost:3000`

## 🎯 Test the System

1. Open `http://localhost:3000`
2. Upload a chest X-ray image
3. View prediction
4. Click "Generate Grad-CAM" to see explanations

## ⚡ Quick Test Without Training

If you want to test the UI without training:

1. Download a pre-trained model (if available)
2. Place it in `models/pneumonia_model.h5`
3. Start backend and frontend
4. Upload test images

## 🐛 Common Issues

**Issue**: `ModuleNotFoundError`
- **Solution**: Activate virtual environment and install requirements

**Issue**: `Model not found`
- **Solution**: Train model first using `train_model.py`

**Issue**: `CUDA out of memory`
- **Solution**: Reduce batch_size in `train_model.py` (e.g., 16 or 8)

**Issue**: Frontend can't connect to backend
- **Solution**: Ensure backend is running on port 5000

## 📚 Next Steps

- Read full [README.md](README.md) for detailed documentation
- Explore XAI methods in `xai/` folder
- Customize model architecture in `models/train_model.py`
- Modify UI components in `frontend/src/components/`

## 💡 Tips

- Use GPU for faster training (10x speedup)
- Start with smaller dataset for testing
- SHAP can be slow; use Grad-CAM for quick results
- Save model checkpoints during training

---

**Happy Coding! 🎉**
