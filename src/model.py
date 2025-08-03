"""
Model creation and training module for CIFAR-10 image classification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
import numpy as np
import pickle
import os
import json
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CIFAR10Model:
    """
    CIFAR-10 CNN Model class with training and evaluation capabilities
    """
    
    def __init__(self, input_shape=(32, 32, 3), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
    
    def create_model(self, model_type='cnn'):
        """
        Create CNN model with optimization techniques
        """
        if model_type == 'cnn':
            self.model = self._create_cnn_model()
        elif model_type == 'resnet':
            self.model = self._create_resnet_model()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        logger.info(f"Model created: {model_type}")
        return self.model
    
    def _create_cnn_model(self):
        """
        Create optimized CNN model
        """
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            
            # Dense Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def _create_resnet_model(self):
        """
        Create ResNet-like model for better performance
        """
        inputs = layers.Input(shape=self.input_shape)
        
        # Initial convolution
        x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Residual blocks
        x = self._residual_block(x, 32)
        x = self._residual_block(x, 64, stride=2)
        x = self._residual_block(x, 128, stride=2)
        
        # Global average pooling and classification
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs)
        return model
    
    def _residual_block(self, x, filters, stride=1):
        """
        Create a residual block
        """
        shortcut = x
        
        # First convolution
        x = layers.Conv2D(filters, (3, 3), strides=stride, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Second convolution
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Adjust shortcut if needed
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, (1, 1), strides=stride, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        
        # Add shortcut and apply activation
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        
        return x
    
    def compile_model(self, learning_rate=0.001, optimizer_type='adam'):
        """
        Compile the model with specified optimizer and metrics
        """
        if not self.model:
            raise ValueError("Model must be created before compilation")
        
        # Choose optimizer
        if optimizer_type == 'adam':
            optimizer = optimizers.Adam(learning_rate=learning_rate, decay=1e-6)
        elif optimizer_type == 'sgd':
            optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=0.9, decay=1e-6)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        logger.info(f"Model compiled with {optimizer_type} optimizer")
    
    def get_callbacks(self, model_save_path='../models/best_model.h5'):
        """
        Get training callbacks for optimization
        """
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.0001,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        ]
        
        return callbacks_list
    
    def train_model(self, x_train, y_train, x_val, y_val, 
                   epochs=50, batch_size=32, use_augmentation=True):
        """
        Train the model with given data
        """
        if not self.model:
            raise ValueError("Model must be created and compiled before training")
        
        # Get callbacks
        callback_list = self.get_callbacks()
        
        if use_augmentation:
            # Create data augmentation
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            datagen = ImageDataGenerator(
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                zoom_range=0.1,
                fill_mode='nearest'
            )
            datagen.fit(x_train)
            
            # Train with augmentation
            self.history = self.model.fit(
                datagen.flow(x_train, y_train, batch_size=batch_size),
                steps_per_epoch=len(x_train) // batch_size,
                epochs=epochs,
                validation_data=(x_val, y_val),
                callbacks=callback_list,
                verbose=1
            )
        else:
            # Train without augmentation
            self.history = self.model.fit(
                x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_val, y_val),
                callbacks=callback_list,
                verbose=1
            )
        
        logger.info("Model training completed")
        return self.history
    
    def evaluate_model(self, x_test, y_test):
        """
        Evaluate model performance
        """
        if not self.model:
            raise ValueError("Model must be trained before evaluation")
        
        # Get predictions
        test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(
            x_test, y_test, verbose=0
        )
        
        # Calculate additional metrics
        y_pred = self.model.predict(x_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        from sklearn.metrics import f1_score, classification_report, confusion_matrix
        
        f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
        
        metrics = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1_score': f1,
            'confusion_matrix': confusion_matrix(y_true_classes, y_pred_classes).tolist(),
            'classification_report': classification_report(
                y_true_classes, y_pred_classes, 
                target_names=self.class_names, output_dict=True
            )
        }
        
        logger.info(f"Model evaluation completed - Accuracy: {test_accuracy:.4f}")
        return metrics
    
    def predict_single(self, image):
        """
        Predict a single image
        """
        if not self.model:
            raise ValueError("Model must be loaded before prediction")
        
        # Ensure correct shape
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Make prediction
        predictions = self.model.predict(image, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        # Get top 3 predictions
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [
            (self.class_names[idx], float(predictions[0][idx])) 
            for idx in top_3_idx
        ]
        
        return {
            'predicted_class': self.class_names[predicted_class_idx],
            'confidence': float(confidence),
            'top_3_predictions': top_3_predictions,
            'all_probabilities': {
                self.class_names[i]: float(predictions[0][i]) 
                for i in range(len(self.class_names))
            }
        }
    
    def predict_batch(self, images):
        """
        Predict a batch of images
        """
        if not self.model:
            raise ValueError("Model must be loaded before prediction")
        
        predictions = self.model.predict(images, verbose=0)
        results = []
        
        for i, pred in enumerate(predictions):
            predicted_class_idx = np.argmax(pred)
            confidence = pred[predicted_class_idx]
            
            results.append({
                'predicted_class': self.class_names[predicted_class_idx],
                'confidence': float(confidence),
                'all_probabilities': {
                    self.class_names[j]: float(pred[j]) 
                    for j in range(len(self.class_names))
                }
            })
        
        return results
    
    def save_model(self, model_path='../models/cifar10_model.h5', 
                   save_artifacts=True):
        """
        Save the trained model and artifacts
        """
        if not self.model:
            raise ValueError("Model must be created before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        self.model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        if save_artifacts:
            base_path = os.path.dirname(model_path)
            
            # Save model architecture
            model_json = self.model.to_json()
            with open(os.path.join(base_path, 'model_architecture.json'), 'w') as f:
                f.write(model_json)
            
            # Save training history
            if self.history:
                with open(os.path.join(base_path, 'training_history.pkl'), 'wb') as f:
                    pickle.dump(self.history.history, f)
            
            # Save class names
            with open(os.path.join(base_path, 'class_names.pkl'), 'wb') as f:
                pickle.dump(self.class_names, f)
            
            # Save model metadata
            metadata = {
                'input_shape': self.input_shape,
                'num_classes': self.num_classes,
                'class_names': self.class_names,
                'created_at': datetime.now().isoformat(),
                'model_path': model_path
            }
            
            with open(os.path.join(base_path, 'model_metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info("Model artifacts saved successfully")
    
    def load_model(self, model_path='../models/cifar10_model.h5'):
        """
        Load a saved model
        """
        try:
            self.model = keras.models.load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
            
            # Try to load metadata
            base_path = os.path.dirname(model_path)
            metadata_path = os.path.join(base_path, 'model_metadata.json')
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.input_shape = tuple(metadata['input_shape'])
                    self.num_classes = metadata['num_classes']
                    self.class_names = metadata['class_names']
                logger.info("Model metadata loaded")
            
            return self.model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def retrain_model(self, new_x_train, new_y_train, x_val, y_val, 
                     epochs=10, batch_size=32, learning_rate=0.0001):
        """
        Retrain the model with new data
        """
        if not self.model:
            raise ValueError("Model must be loaded before retraining")
        
        logger.info("Starting model retraining...")
        
        # Reduce learning rate for fine-tuning
        self.model.optimizer.learning_rate = learning_rate
        
        # Get callbacks with different patience for retraining
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        # Retrain the model
        retrain_history = self.model.fit(
            new_x_train, new_y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=callback_list,
            verbose=1
        )
        
        logger.info("Model retraining completed")
        return retrain_history
    
    def get_model_summary(self):
        """
        Get model summary information
        """
        if not self.model:
            return "Model not created"
        
        summary_list = []
        self.model.summary(print_fn=lambda x: summary_list.append(x))
        return '\n'.join(summary_list)

# Utility functions
def create_and_train_model(x_train, y_train, x_val, y_val, 
                          model_type='cnn', epochs=50, batch_size=32):
    """
    Convenience function to create and train a model
    """
    # Create model instance
    cifar_model = CIFAR10Model()
    
    # Create and compile model
    cifar_model.create_model(model_type=model_type)
    cifar_model.compile_model()
    
    # Train model
    history = cifar_model.train_model(
        x_train, y_train, x_val, y_val,
        epochs=epochs, batch_size=batch_size
    )
    
    return cifar_model, history

def load_pretrained_model(model_path):
    """
    Load a pretrained model
    """
    cifar_model = CIFAR10Model()
    cifar_model.load_model(model_path)
    return cifar_model

if __name__ == "__main__":
    # Example usage
    from preprocessing import DataPreprocessor
    
    # Load and preprocess data
    preprocessor = DataPreprocessor()
    (x_train, y_train), (x_test, y_test) = preprocessor.load_cifar10_data()
    x_train_proc, y_train_proc, x_test_proc, y_test_proc = preprocessor.preprocess_training_data(
        x_train, y_train, x_test, y_test
    )
    
    # Create train/validation split
    x_train_split, y_train_split, x_val_split, y_val_split = preprocessor.create_train_validation_split(
        x_train_proc, y_train_proc
    )
    
    # Create and train model
    cifar_model, history = create_and_train_model(
        x_train_split, y_train_split, x_val_split, y_val_split,
        epochs=2  # Reduced for testing
    )
    
    # Evaluate model
    metrics = cifar_model.evaluate_model(x_test_proc, y_test_proc)
    print("Evaluation metrics:", metrics)
    
    # Save model
    cifar_model.save_model()
    
    print("Model module test completed successfully!")
