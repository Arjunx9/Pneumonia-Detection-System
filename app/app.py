"""
Flask Backend API for Pneumonia Detection System
Handles image uploads, predictions, and XAI explanations
"""

import os
import sys
import matplotlib
matplotlib.use('Agg')  # Must be before any pyplot import — prevents Tkinter crash in threads

# Reduce TensorFlow memory use before any TF import (helps avoid OOM on CPU)
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # 0=all, 1=no INFO, 2=no WARNING, 3=no ERROR
# Optional: disable oneDNN custom ops if you see memory issues
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tensorflow as tf
# Limit CPU threading to reduce peak memory (helps with OOM and paging file)
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(1)
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from PIL import Image
import io
import base64
import json
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xai.gradcam import GradCAM
from xai.lime_explain import LIMEExplainer

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static/results', exist_ok=True)

# Global variables for models
model = None
gradcam = None
shap_explainer = None
lime_explainer = None
background_data = None


def sanitize_for_json(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_models():
    """Load all models and explainers (SHAP disabled if not installed)"""
    global model, gradcam, shap_explainer, lime_explainer, background_data
    
    # Skip if models already loaded (prevents reloading in Flask debug mode)
    if model is not None:
        return
    
    try:
        # Load main model – prefer fine-tuned (higher accuracy)
        model_path = 'models/pneumonia_model_finetuned.h5'
        if not os.path.exists(model_path):
            model_path = 'models/pneumonia_model.h5'
        
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}...")
            model = keras.models.load_model(model_path)
            
            # Initialize Grad-CAM
            try:
                gradcam = GradCAM(model)
                print("Grad-CAM initialized")
            except Exception as e:
                print(f"Warning: Grad-CAM initialization failed: {e}")
                gradcam = None
            
            # Initialize LIME
            try:
                lime_explainer = LIMEExplainer(model)
                print("LIME explainer initialized")
            except Exception as e:
                print(f"Warning: LIME initialization failed: {e}")
                lime_explainer = None

            # SHAP is optional and may not be installed in this environment.
            # We use a gradient-based fallback if SHAP is not available.
            try:
                import shap
                from xai.shap_explain import SHAPExplainer
                shap_explainer = SHAPExplainer(model)
                print("SHAP explainer initialized")
            except ImportError:
                print("SHAP library not installed — using gradient-based feature importance fallback")
                shap_explainer = 'gradient_fallback'
            except Exception as e:
                print(f"Warning: SHAP initialization failed: {e}")
                shap_explainer = 'gradient_fallback'
            
            print("All models loaded successfully!")
        else:
            print(f"Model file not found at {model_path}")
            print("Please train the model first using models/train_model.py")
    except Exception as e:
        print(f"Error loading models: {e}")
        import traceback
        traceback.print_exc()


def preprocess_image(image_file):
    """Preprocess uploaded image for ResNet50-based model"""
    img = Image.open(image_file).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img).astype('float32')  # [0, 255]
    img_array = resnet_preprocess(img_array)       # ResNet50 preprocessing
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img


def image_to_base64(image_array):
    """Convert image array to base64 string. Accepts [0,1] or [0,255] range."""
    arr = np.array(image_array)
    if arr.max() <= 1.0:
        arr = (arr * 255).astype('uint8')
    else:
        arr = arr.astype('uint8')
    img = Image.fromarray(arr)
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'gradcam_ready': gradcam is not None,
        'lime_ready': lime_explainer is not None,
        'shap_ready': shap_explainer is not None
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Preprocess image
        img_array, original_img = preprocess_image(file)
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        pred_class = int(np.argmax(predictions[0]))
        pred_prob = float(predictions[0][pred_class])
        
        result = {
            'prediction': 'Pneumonia' if pred_class == 1 else 'Normal',
            'probability': pred_prob,
            'confidence': pred_prob,
            'all_probabilities': {
                'Normal': float(predictions[0][0]),
                'Pneumonia': float(predictions[0][1])
            }
        }
        
        return jsonify(sanitize_for_json(result))
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/explain/gradcam', methods=['POST'])
def explain_gradcam():
    """Grad-CAM explanation endpoint"""
    if gradcam is None:
        return jsonify({'error': 'Grad-CAM not initialized'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Check if Grad-CAM is available
        if gradcam is None:
            return jsonify({'error': 'Grad-CAM not initialized. Model may not have been loaded correctly.'}), 500
        
        # Generate Grad-CAM explanation
        result = gradcam.generate_heatmap(filepath, img_size=(224, 224))
        
        # Convert images to base64
        response = {
            'prediction': sanitize_for_json(result['prediction']),
            'original_image': image_to_base64(result['original_image']),
            'heatmap': image_to_base64(result['heatmap']),
            'overlay': image_to_base64(result['overlay'])
        }
        
        # Clean up
        os.remove(filepath)
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/explain/shap', methods=['POST'])
def explain_shap():
    """SHAP explanation endpoint — uses gradient-based feature importance as fallback"""
    if shap_explainer is None:
        return jsonify({'error': 'SHAP explainer not available'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load and preprocess image
        img = Image.open(filepath).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img).astype('float32')
        img_array_preprocessed = resnet_preprocess(img_array.copy())
        img_array_preprocessed = np.expand_dims(img_array_preprocessed, axis=0)
        
        # Prepare input for gradient calculation
        input_tensor = tf.convert_to_tensor(img_array_preprocessed, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            # Use model call directly for gradient tracking
            predictions = model(input_tensor, training=False)
            
            # Predict the class if not provided (for saliency we want the winning class)
            pred_class = int(tf.argmax(predictions[0]).numpy())
            class_output = predictions[:, pred_class]
        
        # Compute gradients of the predicted class w.r.t. the input image
        grads = tape.gradient(class_output, input_tensor)
        
        if grads is None:
            return jsonify({'error': 'Could not compute gradients for SHAP. Model might not be differentiable at this point.'}), 500
            
        # Compute saliency map (absolute gradient values averaged across channels)
        saliency = np.abs(grads.numpy()[0])
        saliency = np.mean(saliency, axis=2)  # Average across RGB channels
        
        # Smooth the saliency map with Gaussian blur (using cv2 instead of scipy)
        import cv2
        saliency = cv2.GaussianBlur(saliency, (15, 15), 3)
        
        # Normalize to [0, 1]
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        
        # Create a colored heatmap visualization using matplotlib
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        # Apply colormap
        colored_saliency = cm.hot(saliency)
        colored_saliency = (colored_saliency[:, :, :3] * 255).astype('uint8')
        
        pred_prob = float(predictions[0][pred_class].numpy())
        
        response = {
            'prediction': sanitize_for_json({
                'class': 'Pneumonia' if pred_class == 1 else 'Normal',
                'probability': pred_prob,
            }),
            'original_image': image_to_base64(np.uint8(img_array)),
            'shap_heatmap': image_to_base64(colored_saliency)
        }
        
        # Clean up
        os.remove(filepath)
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/explain/lime', methods=['POST'])
def explain_lime():
    """LIME explanation endpoint"""
    if lime_explainer is None:
        return jsonify({'error': 'LIME explainer not initialized'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Check if LIME is available
        if lime_explainer is None:
            return jsonify({'error': 'LIME explainer not initialized. Model may not have been loaded correctly.'}), 500
        
        # Generate LIME explanation
        result = lime_explainer.visualize(
            filepath,
            'static/results/lime_temp.png',
            num_features=5,
            num_samples=250
        )
        
        # Read generated visualization
        with open('static/results/lime_temp.png', 'rb') as f:
            img_data = base64.b64encode(f.read()).decode()
        
        response = {
            'prediction': sanitize_for_json(result['prediction']),
            'original_image': image_to_base64(result['original_image']),
            'lime_explanation': f"data:image/png;base64,{img_data}"
        }
        
        # Clean up
        os.remove(filepath)
        if os.path.exists('static/results/lime_temp.png'):
            os.remove('static/results/lime_temp.png')
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/explain/all', methods=['POST'])
def explain_all():
    """Get all explanations (Grad-CAM, SHAP, LIME)"""
    if model is None or gradcam is None:
        return jsonify({'error': 'Models not loaded'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get all explanations
        results = {}
        
        # Grad-CAM
        if gradcam is None:
            return jsonify({'error': 'Grad-CAM not initialized. Model may not have been loaded correctly.'}), 500
        
        gradcam_result = gradcam.generate_heatmap(filepath)
        results['gradcam'] = {
            'prediction': sanitize_for_json(gradcam_result['prediction']),
            'original_image': image_to_base64(gradcam_result['original_image']),
            'heatmap': image_to_base64(gradcam_result['heatmap']),
            'overlay': image_to_base64(gradcam_result['overlay'])
        }
        
        # LIME
        if lime_explainer:
            lime_result = lime_explainer.visualize(
                filepath,
                'static/results/lime_temp.png',
                num_features=5
            )
            with open('static/results/lime_temp.png', 'rb') as f:
                img_data = base64.b64encode(f.read()).decode()
            results['lime'] = {
                'prediction': sanitize_for_json(lime_result['prediction']),
                'explanation_image': f"data:image/png;base64,{img_data}"
            }
        
        # Clean up
        os.remove(filepath)
        if os.path.exists('static/results/lime_temp.png'):
            os.remove('static/results/lime_temp.png')
        
        return jsonify(sanitize_for_json(results))
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("Loading models...")
    load_models()
    print("Starting Flask server...")
    # use_reloader=False avoids loading the model twice (saves memory; prevents "paging file too small")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
