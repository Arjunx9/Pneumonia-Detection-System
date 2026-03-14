"""
Utility functions for data preprocessing and handling
"""

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import shutil


def split_dataset(source_dir, train_dir, val_dir, test_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split dataset into train, validation, and test sets
    
    Args:
        source_dir: Source directory with class subdirectories
        train_dir: Destination train directory
        val_dir: Destination validation directory
        test_dir: Destination test directory
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    # Get all class directories
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    for class_name in classes:
        class_path = os.path.join(source_dir, class_name)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Shuffle
        np.random.shuffle(images)
        
        # Split indices
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        # Create directories
        for split_dir in [train_dir, val_dir, test_dir]:
            os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)
        
        # Copy files
        for img in train_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))
        
        for img in val_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(val_dir, class_name, img))
        
        for img in test_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(test_dir, class_name, img))
        
        print(f"Class {class_name}: Train={len(train_images)}, Val={len(val_images)}, Test={len(test_images)}")


def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess a single image
    
    Args:
        image_path: Path to image
        target_size: Target size (height, width)
    
    Returns:
        Preprocessed image array
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    return img_array


def validate_image(image_path):
    """
    Validate if image is valid
    
    Args:
        image_path: Path to image
    
    Returns:
        True if valid, False otherwise
    """
    try:
        img = Image.open(image_path)
        img.verify()
        return True
    except Exception:
        return False
