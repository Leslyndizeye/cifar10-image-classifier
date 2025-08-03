"""
Prediction module for CIFAR-10 image classification
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from PIL import Image
import os
import json
import pickle
import logging
from datetime import datetime
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CIFAR10Predictor:
    """
    Prediction class for CIFAR-10 image classification
    """
    
    def __init__(self, model_path='../models/cifar10_model.h5'):
        self.model = None
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        self.model_path = model_path
        self.model_loaded = False
        self.prediction_history = []
        
    def load_model(self, model_path=None):
        """
        Load the trained model
        """
        if model_path:
            self.model_path = model_path
            
        try:
            self.model = keras.models.load_model(self.model_path)
            self.model_loaded = True
            logger.info(f"Model loaded successfully from {self.model_path}")
            
            # Try to load class names from saved artifacts
            base_path = os.path.dirname(self.model_path)
            class_names_path = os.path.join(base_path, 'class_names.pkl')
            
            if os.path.exists(class_names_path):
                with open(class_names_path, 'rb') as f:
                    self.class_names = pickle.load(f)
                logger.info("Class names loaded from saved artifacts")
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def preprocess_image(self, image_input):
        """
        Preprocess image for prediction
        """
        try:
            # Handle different input types
            if isinstance(image_input, str):
                # Load from file path
                if not os.path.exists(image_input):
                    raise FileNotFoundError(f"Image file not found: {image_input}")
                image = cv2.imread(image_input)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif isinstance(image_input, np.ndarray):
                # Use numpy array directly
                image = image_input.copy()
            elif hasattr(image_input, 'read'):
                # Handle file-like object (e.g., uploaded file)
                image_bytes = image_input.read()
                image_array = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # Handle PIL Image
                image = np.array(image_input)
            
            # Resize to CIFAR-10 dimensions (32x32)
            if image.shape[:2] != (32, 32):
                image = cv2.resize(image, (32, 32))
            
            # Ensure 3 channels
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
            # Normalize pixel values to [0, 1]
            image = image.astype('float32') / 255.0
            
            # Add batch dimension
            if len(image.shape) == 3:
                image = np.expand_dims(image, axis=0)
            
            return image
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise
    
    def predict_single(self, image_input, return_probabilities=True):
        """
        Predict a single image
        """
        if not self.model_loaded:
            raise ValueError("Model must be loaded before prediction")
        
        start_time = time.time()
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_input)
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            predicted_class = self.class_names[predicted_class_idx]
            
            # Calculate prediction time
            prediction_time = time.time() - start_time
            
            # Prepare result
            result = {
                'predicted_class': predicted_class,
                'predicted_class_index': int(predicted_class_idx),
                'confidence': confidence,
                'prediction_time': prediction_time,
                'timestamp': datetime.now().isoformat()
            }
            
            if return_probabilities:
                # Get top 3 predictions
                top_3_idx = np.argsort(predictions[0])[-3:][::-1]
                top_3_predictions = [
                    {
                        'class': self.class_names[idx],
                        'probability': float(predictions[0][idx])
                    }
                    for idx in top_3_idx
                ]
                
                # All class probabilities
                all_probabilities = {
                    self.class_names[i]: float(predictions[0][i])
                    for i in range(len(self.class_names))
                }
                
                result.update({
                    'top_3_predictions': top_3_predictions,
                    'all_probabilities': all_probabilities
                })
            
            # Store in prediction history
            self.prediction_history.append(result)
            
            logger.info(f"Prediction completed: {predicted_class} ({confidence:.4f}) in {prediction_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def predict_batch(self, image_list, return_probabilities=True):
        """
        Predict a batch of images
        """
        if not self.model_loaded:
            raise ValueError("Model must be loaded before prediction")
        
        start_time = time.time()
        results = []
        
        try:
            # Preprocess all images
            processed_images = []
            for image_input in image_list:
                processed_image = self.preprocess_image(image_input)
                processed_images.append(processed_image[0])  # Remove batch dimension
            
            # Stack into batch
            batch_images = np.array(processed_images)
            
            # Make batch prediction
            predictions = self.model.predict(batch_images, verbose=0)
            
            # Process results
            for i, pred in enumerate(predictions):
                predicted_class_idx = np.argmax(pred)
                confidence = float(pred[predicted_class_idx])
                predicted_class = self.class_names[predicted_class_idx]
                
                result = {
                    'image_index': i,
                    'predicted_class': predicted_class,
                    'predicted_class_index': int(predicted_class_idx),
                    'confidence': confidence
                }
                
                if return_probabilities:
                    all_probabilities = {
                        self.class_names[j]: float(pred[j])
                        for j in range(len(self.class_names))
                    }
                    result['all_probabilities'] = all_probabilities
                
                results.append(result)
            
            batch_time = time.time() - start_time
            logger.info(f"Batch prediction completed: {len(image_list)} images in {batch_time:.3f}s")
            
            return {
                'predictions': results,
                'batch_size': len(image_list),
                'total_time': batch_time,
                'avg_time_per_image': batch_time / len(image_list),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error during batch prediction: {str(e)}")
            raise
    
    def predict_from_folder(self, folder_path, output_file=None):
        """
        Predict all images in a folder
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        # Get all image files
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        image_files = [
            os.path.join(folder_path, f) 
            for f in os.listdir(folder_path) 
            if f.lower().endswith(image_extensions)
        ]
        
        if not image_files:
            raise ValueError(f"No image files found in {folder_path}")
        
        logger.info(f"Found {len(image_files)} images in {folder_path}")
        
        # Predict all images
        results = []
        for image_file in image_files:
            try:
                result = self.predict_single(image_file)
                result['image_path'] = image_file
                result['image_name'] = os.path.basename(image_file)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to predict {image_file}: {str(e)}")
        
        # Save results if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_file}")
        
        return results
    
    def get_model_info(self):
        """
        Get information about the loaded model
        """
        if not self.model_loaded:
            return {"error": "Model not loaded"}
        
        try:
            # Get model summary
            summary_list = []
            self.model.summary(print_fn=lambda x: summary_list.append(x))
            
            # Get model metadata if available
            base_path = os.path.dirname(self.model_path)
            metadata_path = os.path.join(base_path, 'model_metadata.json')
            
            metadata = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            return {
                'model_path': self.model_path,
                'input_shape': self.model.input_shape,
                'output_shape': self.model.output_shape,
                'total_params': self.model.count_params(),
                'class_names': self.class_names,
                'num_classes': len(self.class_names),
                'model_summary': '\n'.join(summary_list),
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {"error": str(e)}
    
    def get_prediction_stats(self):
        """
        Get statistics about predictions made
        """
        if not self.prediction_history:
            return {"message": "No predictions made yet"}
        
        # Calculate statistics
        total_predictions = len(self.prediction_history)
        avg_confidence = np.mean([p['confidence'] for p in self.prediction_history])
        avg_prediction_time = np.mean([p['prediction_time'] for p in self.prediction_history])
        
        # Class distribution
        class_counts = {}
        for pred in self.prediction_history:
            class_name = pred['predicted_class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        return {
            'total_predictions': total_predictions,
            'average_confidence': float(avg_confidence),
            'average_prediction_time': float(avg_prediction_time),
            'class_distribution': class_counts,
            'most_predicted_class': max(class_counts, key=class_counts.get) if class_counts else None
        }
    
    def clear_prediction_history(self):
        """
        Clear prediction history
        """
        self.prediction_history = []
        logger.info("Prediction history cleared")
    
    def validate_image(self, image_input):
        """
        Validate if image can be processed
        """
        try:
            processed_image = self.preprocess_image(image_input)
            return True, "Image is valid"
        except Exception as e:
            return False, str(e)

# Utility functions
def create_predictor(model_path='../models/cifar10_model.h5'):
    """
    Create and initialize a predictor
    """
    predictor = CIFAR10Predictor(model_path)
    predictor.load_model()
    return predictor

def predict_image_file(image_path, model_path='../models/cifar10_model.h5'):
    """
    Quick function to predict a single image file
    """
    predictor = create_predictor(model_path)
    return predictor.predict_single(image_path)

def benchmark_prediction_speed(model_path='../models/cifar10_model.h5', num_predictions=100):
    """
    Benchmark prediction speed
    """
    predictor = create_predictor(model_path)
    
    # Create random test images
    test_images = [np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8) for _ in range(num_predictions)]
    
    # Time predictions
    start_time = time.time()
    for image in test_images:
        predictor.predict_single(image, return_probabilities=False)
    total_time = time.time() - start_time
    
    return {
        'total_predictions': num_predictions,
        'total_time': total_time,
        'avg_time_per_prediction': total_time / num_predictions,
        'predictions_per_second': num_predictions / total_time
    }

if __name__ == "__main__":
    # Example usage
    try:
        # Create predictor
        predictor = CIFAR10Predictor()
        predictor.load_model()
        
        # Test with random image
        test_image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        result = predictor.predict_single(test_image)
        print("Prediction result:", result)
        
        # Get model info
        model_info = predictor.get_model_info()
        print("Model info:", model_info)
        
        # Get prediction stats
        stats = predictor.get_prediction_stats()
        print("Prediction stats:", stats)
        
        print("Prediction module test completed successfully!")
        
    except Exception as e:
        print(f"Error in prediction module test: {str(e)}")
