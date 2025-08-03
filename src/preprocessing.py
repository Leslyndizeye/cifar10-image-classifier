"""
Data preprocessing module for CIFAR-10 image classification
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from PIL import Image
import os
import logging

# Set up logging to DEBUG level for detailed output
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Data preprocessing class for CIFAR-10 images
    """
    
    def __init__(self, target_size=(32, 32), num_classes=10):
        self.target_size = target_size
        self.num_classes = num_classes
        self.data_augmentation = None
        
    def load_cifar10_data(self):
        """
        Load and return CIFAR-10 dataset
        """
        try:
            (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
            logger.info(f"CIFAR-10 data loaded successfully")
            logger.debug(f"Training data shape (raw): {x_train.shape}")
            logger.debug(f"Test data shape (raw): {x_test.shape}")
            return (x_train, y_train), (x_test, y_test)
        except Exception as e:
            logger.error(f"Error loading CIFAR-10 data: {str(e)}")
            raise
    
    def normalize_images(self, images):
        """
        Normalize image pixel values to [0, 1]
        """
        return images.astype('float32') / 255.0
    
    def encode_labels(self, labels):
        """
        Convert labels to categorical encoding
        """
        return keras.utils.to_categorical(labels, self.num_classes)
    
    def preprocess_training_data(self, x_train, y_train, x_test, y_test):
        """
        Complete preprocessing pipeline for training data
        """
        logger.info("Starting data preprocessing for training/test split...")
        logger.debug(f"preprocess_training_data - Input x_train shape: {x_train.shape}")
        logger.debug(f"preprocess_training_data - Input x_test shape: {x_test.shape}")
        
        # Normalize images
        x_train_processed = self.normalize_images(x_train)
        x_test_processed = self.normalize_images(x_test)
        
        # Encode labels
        y_train_processed = self.encode_labels(y_train)
        y_test_processed = self.encode_labels(y_test)
        
        logger.info("Data preprocessing for training/test split completed.")
        logger.debug(f"preprocess_training_data - Output x_train_processed shape: {x_train_processed.shape}")
        logger.debug(f"preprocess_training_data - Output x_test_processed shape: {x_test_processed.shape}")
        return x_train_processed, y_train_processed, x_test_processed, y_test_processed
    
    def create_data_augmentation(self):
        """
        Create data augmentation generator
        """
        self.data_augmentation = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            fill_mode='nearest'
        )
        logger.info("Data augmentation generator created")
        return self.data_augmentation
    
    def preprocess_single_image(self, image_input):
        """
        Preprocess a single image for prediction or batch processing.
        Ensures output is (1, 32, 32, 3)
        """
        try:
            image = None
            if isinstance(image_input, str):
                # Load from file path
                image = cv2.imread(image_input)
                if image is None:
                    raise ValueError(f"Could not read image from path: {image_input}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif isinstance(image_input, np.ndarray):
                # Use numpy array directly
                image = image_input.copy()
            elif hasattr(image_input, 'read'):
                # Handle file-like object (e.g., uploaded file)
                image_bytes = image_input.read()
                image_array = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("Could not decode image from bytes.")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif isinstance(image_input, Image.Image): # Handle PIL Image
                image = np.array(image_input)
            else:
                raise TypeError(f"Unsupported image input type: {type(image_input)}")
            
            logger.debug(f"Image shape after initial load/conversion: {image.shape}")
            
            # Resize to CIFAR-10 dimensions (32x32)
            if image.shape[:2] != self.target_size:
                image = cv2.resize(image, self.target_size)
                logger.debug(f"Image shape after resize: {image.shape}")
            
            # Ensure 3 channels (e.g., if grayscale or RGBA)
            if len(image.shape) == 2: # Grayscale to RGB
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                logger.debug(f"Image shape after grayscale to RGB: {image.shape}")
            elif image.shape[2] == 4: # RGBA to RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                logger.debug(f"Image shape after RGBA to RGB: {image.shape}")
            
            # Final check for (H, W, 3) format before normalization
            if len(image.shape) != 3 or image.shape[2] != 3:
                raise ValueError(f"Image not in (H, W, 3) format after preprocessing steps: {image.shape}")
            
            # Normalize pixel values to [0, 1]
            image = self.normalize_images(image)
            logger.debug(f"Image shape after normalization: {image.shape}")
            
            # Add batch dimension (output shape will be (1, H, W, C))
            image = np.expand_dims(image, axis=0)
            
            logger.debug(f"Final preprocessed single image shape: {image.shape}")
            return image
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise
    
    def preprocess_batch_images(self, image_paths):
        """
        Preprocess a batch of images.
        Returns a NumPy array of shape (num_images, 32, 32, 3).
        """
        processed_images_list = []
        
        for image_path in image_paths:
            try:
                # preprocess_single_image returns (1, 32, 32, 3)
                processed_image_batch_dim = self.preprocess_single_image(image_path)
                logger.debug(f"Shape of single processed_image (with batch dim) before [0]: {processed_image_batch_dim.shape}")
                
                # Remove the batch dimension to get (32, 32, 3)
                processed_image_no_batch_dim = processed_image_batch_dim[0]
                processed_images_list.append(processed_image_no_batch_dim)
                logger.debug(f"Shape of item appended to list: {processed_images_list[-1].shape}")
            except Exception as e:
                logger.warning(f"Failed to process {image_path}: {str(e)}")
                continue
        
        if processed_images_list:
            final_batch_array = np.array(processed_images_list)
            logger.info(f"Shape of final processed batch array (for retraining): {final_batch_array.shape}")
            return final_batch_array
        else:
            raise ValueError("No images were successfully processed for batch.")
    
    def save_processed_data(self, data, filepath):
        """
        Save processed data to disk
        """
        try:
            np.save(filepath, data)
            logger.info(f"Processed data saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            raise
    
    def load_processed_data(self, filepath):
        """
        Load processed data from disk
        """
        try:
            data = np.load(filepath)
            logger.info(f"Processed data loaded from {filepath}")
            return data
        except Exception as e:
            logger.error(f"Error loading processed data: {str(e)}")
            raise
    
    def validate_image_format(self, image):
        """
        Validate image format and dimensions
        """
        if not isinstance(image, np.ndarray):
            return False, "Image must be a numpy array"
        
        if len(image.shape) not in [3, 4]:
            return False, "Image must have 3 or 4 dimensions"
        
        if len(image.shape) == 3 and image.shape[2] != 3:
            return False, "Image must have 3 color channels"
        
        if len(image.shape) == 4 and image.shape[3] != 3:
            return False, "Batch images must have 3 color channels"
        
        return True, "Valid image format"
    
    def get_class_names(self):
        """
        Return CIFAR-10 class names
        """
        return [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
    
    def create_train_validation_split(self, x_train, y_train, validation_split=0.2):
        """
        Create train/validation split
        """
        split_idx = int(len(x_train) * (1 - validation_split))
        
        # Shuffle data
        indices = np.random.permutation(len(x_train))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        x_train_split = x_train[train_indices]
        y_train_split = y_train[train_indices]
        x_val_split = x_train[val_indices]
        y_val_split = y_train[val_indices]
        
        logger.info(f"Train/validation split created: {len(x_train_split)}/{len(x_val_split)}")
        
        return x_train_split, y_train_split, x_val_split, y_val_split

# Utility functions
def preprocess_uploaded_images(upload_folder, target_size=(32, 32)):
    """
    Preprocess all images in upload folder
    """
    preprocessor = DataPreprocessor(target_size=target_size)
    processed_images = []
    image_paths = []
    
    for filename in os.listdir(upload_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(upload_folder, filename)
            try:
                processed_image = preprocessor.preprocess_single_image(image_path)
                processed_images.append(processed_image[0])
                image_paths.append(image_path)
            except Exception as e:
                logger.warning(f"Failed to process {filename}: {str(e)}")
    
    if processed_images:
        return np.array(processed_images), image_paths
    else:
        return None, []

def calculate_dataset_statistics(images):
    """
    Calculate dataset statistics
    """
    stats = {
        'mean': np.mean(images, axis=(0, 1, 2)),
        'std': np.std(images, axis=(0, 1, 2)),
        'min': np.min(images),
        'max': np.max(images),
        'shape': images.shape
    }
    return stats

if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = preprocessor.load_cifar10_data()
    
    # Preprocess data
    x_train_proc, y_train_proc, x_test_proc, y_test_proc = preprocessor.preprocess_training_data(
        x_train, y_train, x_test, y_test
    )
    
    # Create data augmentation
    datagen = preprocessor.create_data_augmentation()
    
    # Calculate statistics
    stats = calculate_dataset_statistics(x_train_proc)
    print("Dataset statistics:", stats)
    
    print("Preprocessing module test completed successfully!")
