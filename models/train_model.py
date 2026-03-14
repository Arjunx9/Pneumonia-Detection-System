"""
Pneumonia Detection Model Training Script
Uses Transfer Learning with ResNet50 for high accuracy
Improved: better head architecture, proper fine-tuning, class weighting,
stronger augmentation, and correct preprocessing.
"""

import os
import gc
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns

# Configure GPU memory growth to prevent allocation issues
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)


class PneumoniaModelTrainer:
    def __init__(self, img_size=(224, 224), batch_size=16):
        """
        Initialize the model trainer

        Args:
            img_size: Target image size (height, width)
            batch_size: Batch size for training
        """
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.base_model = None
        self.history = None

    def create_model(self, num_classes=2):
        """Create the transfer learning model with improved head"""
        # Load pre-trained ResNet50
        self.base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )

        # Freeze all base model layers initially
        self.base_model.trainable = False

        # Build the model — preprocessing is handled by data generators
        inputs = keras.Input(shape=(*self.img_size, 3))

        x = self.base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)

        # Improved classifier head — deeper with batch norm for better gradient flow
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        outputs = layers.Dense(num_classes, activation='softmax')(x)

        model = keras.Model(inputs, outputs)

        # Compile with a moderate learning rate for training the head
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        print(f"Model created. Top-level layers: {len(model.layers)}")
        print(f"Base model layers: {len(self.base_model.layers)}")
        print(f"Trainable parameters: {sum(p.numpy().size for p in model.trainable_weights):,}")
        return model

    def prepare_data_generators(self, train_dir, val_dir, test_dir):
        """Prepare data generators with strong augmentation and ResNet preprocessing"""

        # Strong augmentation for training (important for small datasets)
        # preprocessing_function applies ResNet50 preprocess_input which
        # converts pixels from [0,255] to the range expected by ResNet.
        train_datagen = ImageDataGenerator(
            preprocessing_function=resnet_preprocess,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            zoom_range=0.25,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )

        # Validation and test: only preprocessing (no augmentation)
        val_test_datagen = ImageDataGenerator(preprocessing_function=resnet_preprocess)

        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )

        val_generator = val_test_datagen.flow_from_directory(
            val_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )

        test_generator = val_test_datagen.flow_from_directory(
            test_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )

        return train_generator, val_generator, test_generator

    @staticmethod
    def compute_class_weights(train_generator):
        """Compute class weights to handle any class imbalance"""
        labels = train_generator.classes
        class_weights = compute_class_weight(
            'balanced', classes=np.unique(labels), y=labels
        )
        return dict(enumerate(class_weights))

    def train(self, train_generator, val_generator, epochs=30, class_weight=None):
        """Phase 1: Train the classifier head (base model frozen)"""
        callbacks = [
            ModelCheckpoint(
                'models/pneumonia_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=8,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
        ]

        print("\n=== Phase 1: Training classifier head (base frozen) ===")
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )

        gc.collect()
        return self.history

    def fine_tune(self, train_generator, val_generator, epochs=20,
                  class_weight=None, unfreeze_from=140):
        """
        Phase 2: Fine-tune top layers of ResNet50

        For ResNet50 (175 layers), unfreezing from layer 140 means the last
        ~35 layers are trainable — roughly the last residual block.
        """
        print(f"\n=== Phase 2: Fine-tuning (unfreezing base model from layer {unfreeze_from}/{len(self.base_model.layers)}) ===")

        # Unfreeze the base model, then re-freeze layers before unfreeze_from
        self.base_model.trainable = True
        for layer in self.base_model.layers[:unfreeze_from]:
            layer.trainable = False
        # Layers from unfreeze_from onwards are now trainable

        # Recompile with a much lower learning rate to avoid destroying pretrained weights
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        trainable_count = sum(p.numpy().size for p in self.model.trainable_weights)
        print(f"Trainable parameters after unfreezing: {trainable_count:,}")

        callbacks = [
            ModelCheckpoint(
                'models/pneumonia_model_finetuned.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=6,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-7,
                verbose=1
            )
        ]

        history_fine = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )

        gc.collect()
        return history_fine

    def evaluate(self, test_generator):
        """Evaluate the model"""
        predictions = self.model.predict(test_generator)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_generator.classes

        test_loss, test_accuracy = self.model.evaluate(test_generator, verbose=0)

        class_names = list(test_generator.class_indices.keys())
        report = classification_report(y_true, y_pred, target_names=class_names)
        cm = confusion_matrix(y_true, y_pred)

        from sklearn.metrics import precision_score, recall_score, f1_score as f1_func
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_func(y_true, y_pred, average='weighted')

        return {
            'accuracy': test_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': report,
            'confusion_matrix': cm,
            'class_names': class_names
        }

    def plot_training_history(self, history, save_path='evaluation/training_history.png',
                              title_prefix=''):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        axes[0].plot(history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_title(f'{title_prefix}Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(history.history['loss'], label='Train Loss')
        axes[1].plot(history.history['val_loss'], label='Val Loss')
        axes[1].set_title(f'{title_prefix}Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Training history saved to {save_path}")

    def plot_confusion_matrix(self, cm, class_names,
                               save_path='evaluation/confusion_matrix.png'):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Confusion matrix saved to {save_path}")


def main():
    """Main training function"""
    # Paths
    train_dir = 'dataset/train'
    val_dir = 'dataset/val'
    test_dir = 'dataset/test'

    # Create trainer
    trainer = PneumoniaModelTrainer(img_size=(224, 224), batch_size=16)

    # Create model
    print("Creating model...")
    model = trainer.create_model(num_classes=2)
    model.summary()

    # Prepare data generators
    print("Preparing data generators...")
    train_gen, val_gen, test_gen = trainer.prepare_data_generators(
        train_dir, val_dir, test_dir
    )

    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Test samples: {test_gen.samples}")
    print(f"Class indices: {train_gen.class_indices}")

    # Compute class weights
    class_weight = trainer.compute_class_weights(train_gen)
    print(f"Class weights: {class_weight}")

    # ---- Phase 1: Train classifier head ----
    print("\nStarting Phase 1: Training classifier head...")
    history1 = trainer.train(
        train_generator=train_gen,
        val_generator=val_gen,
        epochs=30,
        class_weight=class_weight
    )
    trainer.plot_training_history(
        history1,
        save_path='evaluation/training_history_phase1.png',
        title_prefix='Phase 1 - '
    )

    # ---- Phase 2: Fine-tune top ResNet layers ----
    print("\nStarting Phase 2: Fine-tuning...")
    history2 = trainer.fine_tune(
        train_generator=train_gen,
        val_generator=val_gen,
        epochs=20,
        class_weight=class_weight,
        unfreeze_from=140
    )
    trainer.plot_training_history(
        history2,
        save_path='evaluation/training_history_phase2.png',
        title_prefix='Phase 2 (Fine-tune) - '
    )

    # ---- Evaluate ----
    print("\nEvaluating model...")
    results = trainer.evaluate(test_gen)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1_score']:.4f}")
    print("\nClassification Report:")
    print(results['classification_report'])

    # Plot confusion matrix
    trainer.plot_confusion_matrix(results['confusion_matrix'], results['class_names'])

    # Combined training history plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    total_epochs_1 = len(history1.history['accuracy'])
    total_epochs_2 = len(history2.history['accuracy'])
    all_acc = history1.history['accuracy'] + history2.history['accuracy']
    all_val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    all_loss = history1.history['loss'] + history2.history['loss']
    all_val_loss = history1.history['val_loss'] + history2.history['val_loss']

    epochs_range = range(1, total_epochs_1 + total_epochs_2 + 1)
    axes[0].plot(epochs_range, all_acc, label='Train Accuracy')
    axes[0].plot(epochs_range, all_val_acc, label='Val Accuracy')
    axes[0].axvline(x=total_epochs_1, color='r', linestyle='--', label='Fine-tune start')
    axes[0].set_title('Full Training - Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs_range, all_loss, label='Train Loss')
    axes[1].plot(epochs_range, all_val_loss, label='Val Loss')
    axes[1].axvline(x=total_epochs_1, color='r', linestyle='--', label='Fine-tune start')
    axes[1].set_title('Full Training - Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('evaluation/training_history.png', dpi=150)
    plt.close()

    print("\nTraining completed!")


if __name__ == '__main__':
    main()
