"""
SHAP (SHapley Additive exPlanations) Implementation
Advanced XAI method for model interpretability
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import shap
import matplotlib.pyplot as plt
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


class SHAPExplainer:
    """
    SHAP explainer for CNN models
    Provides feature importance explanations
    """
    
    def __init__(self, model, background_images=None, background_size=50):
        """
        Initialize SHAP explainer
        
        Args:
            model: Trained Keras model
            background_images: Background dataset for SHAP (if None, will use random samples)
            background_size: Number of background samples to use
        """
        self.model = model
        self.background_images = background_images
        self.background_size = background_size
        
        # Create SHAP explainer
        self.explainer = None
        self._setup_explainer()
    
    def _setup_explainer(self):
        """Setup SHAP explainer with background data"""
        if self.background_images is None:
            print("Warning: No background images provided. SHAP may be slow.")
            return
        
        # Prepare background data
        if len(self.background_images) > self.background_size:
            indices = np.random.choice(len(self.background_images), self.background_size, replace=False)
            background = self.background_images[indices]
        else:
            background = self.background_images
        
        # Create DeepExplainer
        try:
            self.explainer = shap.DeepExplainer(self.model, background)
            print(f"SHAP explainer initialized with {len(background)} background samples")
        except Exception as e:
            print(f"Error initializing SHAP explainer: {e}")
            print("Falling back to GradientExplainer")
            self.explainer = shap.GradientExplainer(self.model, background)
    
    def explain_image(self, image_array, class_index=None):
        """
        Generate SHAP explanations for an image
        
        Args:
            image_array: Preprocessed image array (batch_size, height, width, channels)
            class_index: Class index to explain (None = use predicted class)
        
        Returns:
            SHAP values and visualization
        """
        if self.explainer is None:
            raise ValueError("SHAP explainer not initialized. Provide background images.")
        
        # Get prediction
        predictions = self.model.predict(image_array, verbose=0)
        if class_index is None:
            class_index = np.argmax(predictions[0])
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(image_array)
        
        # Handle multi-class output
        if isinstance(shap_values, list):
            shap_values = shap_values[class_index]
        
        return {
            'shap_values': shap_values,
            'prediction': {
                'class_index': class_index,
                'probability': float(predictions[0][class_index]),
                'all_probabilities': predictions[0].tolist()
            },
            'original_image': image_array[0]
        }
    
    def visualize(self, image_path, output_path, img_size=(224, 224)):
        """
        Generate and save SHAP visualization
        
        Args:
            image_path: Path to input image
            output_path: Path to save visualization
            img_size: Target image size
        """
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img = img.resize(img_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Get explanation
        result = self.explain_image(img_array)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        
        # Original image
        axes[0].imshow(result['original_image'])
        axes[0].set_title('Original X-ray')
        axes[0].axis('off')
        
        # SHAP values
        shap_image = result['shap_values'][0]
        # Take mean across channels if needed
        if len(shap_image.shape) == 3:
            shap_image = np.mean(np.abs(shap_image), axis=2)
        
        im = axes[1].imshow(shap_image, cmap='hot', interpolation='nearest')
        axes[1].set_title('SHAP Values (Feature Importance)')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        pred_text = f"Prediction: {'Pneumonia' if result['prediction']['class_index'] == 1 else 'Normal'}\n"
        pred_text += f"Probability: {result['prediction']['probability']:.2%}"
        plt.suptitle(pred_text, fontsize=12, y=0.98)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return result


def prepare_background_data(data_dir, num_samples=50, img_size=(224, 224)):
    """
    Prepare background dataset for SHAP
    
    Args:
        data_dir: Directory containing images
        num_samples: Number of samples to use
        img_size: Target image size
    
    Returns:
        Array of preprocessed images
    """
    import os
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    
    images = []
    image_files = []
    
    # Collect image files
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    
    # Sample images
    if len(image_files) > num_samples:
        indices = np.random.choice(len(image_files), num_samples, replace=False)
        image_files = [image_files[i] for i in indices]
    
    # Load and preprocess images
    for img_path in image_files:
        img = load_img(img_path, target_size=img_size)
        img_array = img_to_array(img) / 255.0
        images.append(img_array)
    
    return np.array(images)


if __name__ == '__main__':
    # Example usage
    model_path = 'models/pneumonia_model.h5'
    image_path = 'dataset/test/pneumonia/person1_virus_6.jpeg'
    background_dir = 'dataset/train'
    
    # Load model
    model = keras.models.load_model(model_path)
    
    # Prepare background data
    print("Preparing background data for SHAP...")
    background = prepare_background_data(background_dir, num_samples=50)
    
    # Create SHAP explainer
    shap_explainer = SHAPExplainer(model, background_images=background)
    
    # Generate explanation
    result = shap_explainer.visualize(image_path, 'evaluation/shap_example.png')
    
    print(f"Prediction: {'Pneumonia' if result['prediction']['class_index'] == 1 else 'Normal'}")
    print(f"Probability: {result['prediction']['probability']:.2%}")
