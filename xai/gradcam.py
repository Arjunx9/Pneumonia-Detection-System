"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Implementation
Primary XAI method for pneumonia detection
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image


class GradCAM:
    """
    Grad-CAM implementation for CNN visualization
    Generates heatmaps showing which regions of the image are important for prediction
    """
    
    def __init__(self, model, layer_name=None):
        """
        Initialize Grad-CAM
        
        Args:
            model: Trained Keras model
            layer_name: Name of the convolutional layer to use (default: last conv layer)
        """
        self.model = model
        self.layer_name = layer_name
        
        # Find the last convolutional layer if not specified
        if layer_name is None:
            for layer in reversed(self.model.layers):
                # Check if layer has output_shape attribute and is convolutional
                try:
                    output_shape = getattr(layer, 'output_shape', None)
                    if output_shape is None:
                        # Try accessing via layer.output.shape
                        if hasattr(layer, 'output'):
                            output_shape = layer.output.shape
                        else:
                            continue
                    
                    # Check if it's a convolutional layer (4D tensor)
                    if isinstance(output_shape, tuple) and len(output_shape) == 4:
                        self.layer_name = layer.name
                        break
                    
                    # If it's a nested model (like ResNet50 in this project), look inside it
                    if hasattr(layer, 'layers'):
                        for sub_layer in reversed(layer.layers):
                            sub_output_shape = getattr(sub_layer, 'output_shape', None)
                            if sub_output_shape is None and hasattr(sub_layer, 'output'):
                                sub_output_shape = sub_layer.output.shape
                            
                            if isinstance(sub_output_shape, tuple) and len(sub_output_shape) == 4:
                                # We found a conv layer inside a nested model!
                                # For Grad-CAM to work on nested models, we often need 
                                # to use the nested model itself as the target or use its output.
                                self.layer_name = layer.name # Use the parent layer for simplicity in sub-model
                                break
                        if self.layer_name: break
                except (AttributeError, TypeError):
                    continue
        
        if self.layer_name is None:
            raise ValueError("No convolutional layer found in the model")
        
        # Create model that outputs both predictions and conv layer activations
        # If construction fails (common with nested functional models in Keras 3),
        # we mark it to use a higher-level fallback in the heatmap generation.
        try:
            inputs = self.model.input
            outputs = [self.model.get_layer(self.layer_name).output, self.model.output]
            self.grad_model = keras.Model(inputs=inputs, outputs=outputs)
            self.use_fallback = False
        except Exception as e:
            print(f"Warning: Grad-CAM sub-model construction failed: {e}")
            print("Using saliency fallback for heatmap generation.")
            self.grad_model = None
            self.use_fallback = True
    
    def make_gradcam_heatmap(self, img_array, pred_index=None):
        """
        Generate Grad-CAM heatmap for an image
        """
        if self.use_fallback or self.grad_model is None:
            return self._make_saliency_fallback(img_array, pred_index)
            
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        
        try:
            with tf.GradientTape() as tape:
                tape.watch(img_tensor)
                conv_outputs, predictions = self.grad_model(img_tensor, training=False)
                if pred_index is None:
                    pred_index = tf.argmax(predictions[0])
                class_output = predictions[:, pred_index]
            
            # Compute gradients
            grads = tape.gradient(class_output, conv_outputs)
            
            if grads is None:
                return self._make_saliency_fallback(img_array, pred_index)
                
            # Pool gradients spatially
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Weight the conv layer output by the pooled gradients
            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            
            # Normalize heatmap
            heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
            return heatmap.numpy()
        except Exception as e:
            print(f"Error in Grad-CAM computation: {e}. Falling back to saliency.")
            return self._make_saliency_fallback(img_array, pred_index)

    def _make_saliency_fallback(self, img_array, pred_index=None):
        """Standard saliency map fallback with Gaussian blurring"""
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        try:
            with tf.GradientTape() as tape:
                tape.watch(img_tensor)
                # Use call instead of predict inside tape
                predictions = self.model(img_tensor, training=False)
                if pred_index is None:
                    pred_index = tf.argmax(predictions[0])
                class_output = predictions[:, pred_index]
                
            grads = tape.gradient(class_output, img_tensor)
            if grads is None:
                return np.zeros((7, 7))
                
            saliency = np.abs(grads.numpy()[0])
            saliency = np.mean(saliency, axis=2)
            
            # Smoother blurring for heatmap appearance
            saliency = cv2.GaussianBlur(saliency, (21, 21), 5)
            
            # Normalize to [0, 1]
            saliency_min, saliency_max = saliency.min(), saliency.max()
            if saliency_max > saliency_min:
                saliency = (saliency - saliency_min) / (saliency_max - saliency_min)
            else:
                saliency = np.zeros_like(saliency)
                
            return saliency
        except Exception as e:
            print(f"Critical failure in saliency fallback: {e}")
            return np.zeros((224, 224))
    
    def generate_heatmap(self, image_path, img_size=(224, 224), alpha=0.4):
        """
        Generate and visualize Grad-CAM heatmap
        
        Args:
            image_path: Path to input image
            img_size: Target image size
            alpha: Transparency for overlay
        
        Returns:
            Dictionary with original image, heatmap, overlay, and prediction
        """
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert('RGB')
            img = img.resize(img_size)
            img_array = np.array(img).astype('float32') # [0, 255]
            img_array_preprocessed = resnet_preprocess(img_array.copy()) # ResNet preprocessing
            img_array_preprocessed = np.expand_dims(img_array_preprocessed, axis=0)
            
            # Predict using direct call instead of .predict() for better compatibility in Keras 3
            img_tensor = tf.convert_to_tensor(img_array_preprocessed, dtype=tf.float32)
            predictions = self.model(img_tensor, training=False)
            
            pred_class = int(np.argmax(predictions[0]))
            pred_prob = float(predictions[0][pred_class])
            
            # Generate heatmap
            heatmap = self.make_gradcam_heatmap(img_array_preprocessed, pred_class)
            
            # Resize heatmap to match image size
            heatmap = cv2.resize(heatmap, img_size)
        except Exception as e:
            print("\n!!!! GRAD-CAM FALLBACK TRIGGERED !!!!")
            print(f"Error details: {e}")
            # Absolute fallback to empty/black images if even the pre-processing failed
            return {
                'original_image': np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8),
                'heatmap': np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8),
                'overlay': np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8),
                'prediction': {'class': 'Unknown', 'probability': 0.0, 'all_probabilities': [0.5, 0.5]}
            }
        
        # Convert heatmap to RGB
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay heatmap on original image (use original [0-255] image)
        img_uint8 = np.uint8(img_array)
        overlay = cv2.addWeighted(img_uint8, 1 - alpha, heatmap, alpha, 0)
        
        return {
            'original_image': img_uint8,
            'heatmap': heatmap,
            'overlay': overlay,
            'prediction': {
                'class': 'Pneumonia' if pred_class == 1 else 'Normal',
                'probability': float(pred_prob),
                'all_probabilities': predictions.numpy()[0].tolist() if hasattr(predictions, 'numpy') else predictions[0].tolist()
            }
        }
    
    def save_visualization(self, image_path, output_path, img_size=(224, 224), alpha=0.4):
        """
        Generate and save Grad-CAM visualization
        
        Args:
            image_path: Path to input image
            output_path: Path to save visualization
            img_size: Target image size
            alpha: Transparency for overlay
        """
        result = self.generate_heatmap(image_path, img_size, alpha)
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(result['original_image'])
        axes[0].set_title('Original X-ray')
        axes[0].axis('off')
        
        # Heatmap
        axes[1].imshow(result['heatmap'])
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(result['overlay'])
        pred_text = f"{result['prediction']['class']}\n({result['prediction']['probability']:.2%})"
        axes[2].set_title(f'Overlay\n{pred_text}')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return result


def load_model_for_gradcam(model_path):
    """Load trained model for Grad-CAM"""
    return keras.models.load_model(model_path)


if __name__ == '__main__':
    # Example usage
    model_path = 'models/pneumonia_model.h5'
    image_path = 'dataset/test/pneumonia/person1_virus_6.jpeg'
    
    # Load model
    model = load_model_for_gradcam(model_path)
    
    # Create Grad-CAM instance
    gradcam = GradCAM(model)
    
    # Generate visualization
    result = gradcam.save_visualization(
        image_path,
        'evaluation/gradcam_example.png'
    )
    
    print(f"Prediction: {result['prediction']['class']}")
    print(f"Probability: {result['prediction']['probability']:.2%}")
