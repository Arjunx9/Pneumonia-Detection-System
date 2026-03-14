"""
Model Evaluation Script
Computes accuracy, precision, recall, F1-score, and confusion matrix
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import json


def evaluate_model(model_path, test_dir, img_size=(224, 224), batch_size=32):
    """
    Evaluate trained model
    
    Args:
        model_path: Path to saved model
        test_dir: Test dataset directory
        img_size: Image size
        batch_size: Batch size
    """
    # Load model
    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    
    # Prepare test data generator
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    # Get class names
    class_names = list(test_generator.class_indices.keys())
    
    # Evaluate
    print("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
    
    # Predictions
    predictions = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None)
    recall_per_class = recall_score(y_true, y_pred, average=None)
    f1_per_class = f1_score(y_true, y_pred, average=None)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    print(f"\nPer-Class Metrics:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}:")
        print(f"    Precision: {precision_per_class[i]:.4f}")
        print(f"    Recall:    {recall_per_class[i]:.4f}")
        print(f"    F1-Score:  {f1_per_class[i]:.4f}")
    
    print(f"\nClassification Report:")
    print(report)
    
    # Save results
    results = {
        'overall': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        },
        'per_class': {
            class_names[i]: {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1_score': float(f1_per_class[i])
            }
            for i in range(len(class_names))
        },
        'confusion_matrix': cm.tolist(),
        'class_names': class_names
    }
    
    # Save to JSON
    with open('evaluation/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('evaluation/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nResults saved to:")
    print("  - evaluation/results.json")
    print("  - evaluation/confusion_matrix.png")
    
    return results


if __name__ == '__main__':
    model_path = 'models/pneumonia_model.h5'
    test_dir = 'dataset/test'
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train the model first using models/train_model.py")
    else:
        evaluate_model(model_path, test_dir)
