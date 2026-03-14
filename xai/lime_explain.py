"""
LIME (Local Interpretable Model-agnostic Explanations) Implementation
Model-agnostic XAI method for explanations
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — prevents Tkinter thread crash
import matplotlib.pyplot as plt
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


class LIMEExplainer:
    """
    LIME explainer for CNN models
    Provides local, interpretable explanations
    """
    
    def __init__(self, model, class_names=['Normal', 'Pneumonia']):
        """
        Initialize LIME explainer
        
        Args:
            model: Trained Keras model
            class_names: List of class names
        """
        self.model = model
        self.class_names = class_names
        
        # Create LIME explainer
        self.explainer = lime_image.LimeImageExplainer()
    
    def _predict_fn(self, images):
        """
        Prediction function for LIME
        
        Args:
            images: Array of images (n_samples, height, width, channels)
        
        Returns:
            Predictions array (n_samples, n_classes)
        """
        # LIME may pass images in different formats, ensure they're correct
        if isinstance(images, list):
            images = np.array(images)
        
        # Apply ResNet50 preprocessing (expects [0-255] range)
        # LIME passes images in the same range as the input image
        processed = resnet_preprocess(images.copy().astype('float32'))
        
        # Make predictions
        predictions = self.model.predict(processed, verbose=0)
        return predictions
    
    def explain_image(self, image_array, top_labels=2, num_features=5, num_samples=250):
        """
        Generate LIME explanation for an image
        
        Args:
            image_array: Preprocessed image array (height, width, channels)
            top_labels: Number of top labels to explain
            num_features: Number of features to highlight
            num_samples: Number of samples for LIME (reduced for speed)
        
        Returns:
            LIME explanation object
        """
        # Ensure image is in [0-255] range for LIME
        if image_array.max() <= 1.0:
            image_array = (image_array * 255.0).astype('float32')
        
        # Generate explanation
        explanation = self.explainer.explain_instance(
            image_array.astype('double'),
            self._predict_fn,
            top_labels=top_labels,
            hide_color=0,
            num_samples=num_samples
        )
        
        return explanation
    
    def visualize(self, image_path, output_path, img_size=(224, 224), 
                  num_features=5, num_samples=250):
        """
        Generate and save LIME visualization
        
        Args:
            image_path: Path to input image
            output_path: Path to save visualization
            img_size: Target image size
            num_features: Number of features to highlight
            num_samples: Number of samples for LIME
        """
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img = img.resize(img_size)
        img_array = np.array(img).astype('float32')  # keep [0, 255]
        
        # Make prediction
        img_batch = resnet_preprocess(np.expand_dims(img_array.copy(), axis=0))
        predictions = self.model.predict(img_batch, verbose=0)
        pred_class = int(np.argmax(predictions[0]))
        pred_prob = float(predictions[0][pred_class])
        
        # Get explanation
        explanation = self.explain_image(img_array, num_samples=num_samples)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(img_array)
        axes[0].set_title('Original X-ray')
        axes[0].axis('off')
        
        # LIME explanation for predicted class
        temp, mask = explanation.get_image_and_mask(
            pred_class,
            positive_only=True,
            num_features=num_features,
            hide_rest=False
        )
        axes[1].imshow(mark_boundaries(temp, mask))
        axes[1].set_title(f'LIME Explanation\n({self.class_names[pred_class]})')
        axes[1].axis('off')
        
        # LIME explanation with positive and negative
        temp2, mask2 = explanation.get_image_and_mask(
            pred_class,
            positive_only=False,
            num_features=num_features,
            hide_rest=False
        )
        axes[2].imshow(mark_boundaries(temp2, mask2))
        axes[2].set_title('LIME (Positive + Negative)')
        axes[2].axis('off')
        
        pred_text = f"Prediction: {self.class_names[pred_class]}\n"
        pred_text += f"Probability: {pred_prob:.2%}"
        plt.suptitle(pred_text, fontsize=12, y=0.98)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return {
            'explanation': explanation,
            'prediction': {
                'class': self.class_names[pred_class],
                'class_index': int(pred_class),
                'probability': float(pred_prob),
                'all_probabilities': predictions[0].tolist()
            },
            'original_image': img_array
        }


if __name__ == '__main__':
    # Example usage
    model_path = 'models/pneumonia_model.h5'
    image_path = 'dataset/test/pneumonia/person1_virus_6.jpeg'
    
    # Load model
    model = keras.models.load_model(model_path)
    
    # Create LIME explainer
    lime_explainer = LIMEExplainer(model)
    
    # Generate explanation
    result = lime_explainer.visualize(
        image_path,
        'evaluation/lime_example.png',
        num_features=5,
        num_samples=1000
    )
    
    print(f"Prediction: {result['prediction']['class']}")
    print(f"Probability: {result['prediction']['probability']:.2%}")
