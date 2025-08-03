"""
FastAPI server for CIFAR-10 image classification - COMPLETE FIXED VERSION
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import os
import shutil
import json
from datetime import datetime
import logging
from typing import List, Optional
import asyncio
import time
import tensorflow as tf
from PIL import Image
import re
import traceback

# Import our modules
import sys
sys.path.append('../src')
from prediction import CIFAR10Predictor
from model import CIFAR10Model
from preprocessing import DataPreprocessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="CIFAR-10 Image Classification API",
    description="ML Pipeline for CIFAR-10 image classification with retraining capabilities",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
predictor = None
model_trainer = None
preprocessor = None
retraining_status = {"status": "idle", "progress": 0, "message": ""}
upload_folder = "../data/uploads"
model_path = "../models/cifar10_cnn_model.h5"

# Create necessary directories
os.makedirs(upload_folder, exist_ok=True)
os.makedirs("../models", exist_ok=True)

def preprocess_image_for_training(image_path, target_size=(32, 32)):
    """
    Properly preprocess images for CIFAR-10 training
    """
    try:
        # Load and resize image
        image = Image.open(image_path).convert('RGB')
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # Ensure shape is (32, 32, 3)
        if image_array.shape != (32, 32, 3):
            raise ValueError(f"Image shape {image_array.shape} doesn't match expected (32, 32, 3)")
        
        return image_array
    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {e}")
        return None

def extract_label_from_filename(filename):
    """
    Enhanced label extraction with better error handling
    """
    # CIFAR-10 class mapping
    class_mapping = {
        'airplane': 0, 'plane': 0, 'aircraft': 0,
        'automobile': 1, 'car': 1, 'auto': 1, 'vehicle': 1,
        'bird': 2, 'birds': 2,
        'cat': 3, 'cats': 3, 'kitten': 3,
        'deer': 4, 'deers': 4,
        'dog': 5, 'dogs': 5, 'puppy': 5,
        'frog': 6, 'frogs': 6, 'toad': 6,
        'horse': 7, 'horses': 7, 'pony': 7,
        'ship': 8, 'ships': 8, 'boat': 8, 'vessel': 8,
        'truck': 9, 'trucks': 9, 'lorry': 9
    }
    
    filename_lower = filename.lower()
    
    # Remove file extension
    filename_base = os.path.splitext(filename_lower)[0]
    
    # Check for class names in filename
    for class_name, class_id in class_mapping.items():
        if class_name in filename_base:
            logger.debug(f"Found class '{class_name}' (id={class_id}) in filename '{filename}'")
            return class_id
    
    # If underscore or dash separated, try first part
    for separator in ['_', '-', ' ']:
        if separator in filename_base:
            first_part = filename_base.split(separator)[0]
            if first_part in class_mapping:
                class_id = class_mapping[first_part]
                logger.debug(f"Found class '{first_part}' (id={class_id}) in filename '{filename}'")
                return class_id
    
    # Try to extract number from filename
    numbers = re.findall(r'\d+', filename_base)
    if numbers:
        try:
            num = int(numbers[0])
            if 0 <= num <= 9:
                logger.debug(f"Using number {num} as class id from filename '{filename}'")
                return num
        except ValueError:
            pass
    
    # Default to class 0 (airplane) for unlabeled data
    logger.warning(f"Could not extract label from filename '{filename}', defaulting to class 0")
    return 0

def validate_tensor_shapes(images, labels):
    """
    Enhanced tensor shape validation
    """
    logger.info(f"Images shape: {images.shape}, dtype: {images.dtype}")
    logger.info(f"Labels shape: {labels.shape}, dtype: {labels.dtype}")
    logger.info(f"Label values range: {labels.min()} to {labels.max()}")
    
    if len(images.shape) != 4:
        raise ValueError(f"Images must have 4 dimensions (batch, height, width, channels), got {len(images.shape)}")
    
    if images.shape[1:] != (32, 32, 3):
        raise ValueError(f"Expected image shape (32, 32, 3), got {images.shape[1:]}")
    
    if len(labels.shape) != 1:
        raise ValueError(f"Labels must have 1 dimension for sparse categorical, got {len(labels.shape)}")
    
    if images.shape[0] != labels.shape[0]:
        raise ValueError(f"Number of images ({images.shape[0]}) doesn't match number of labels ({labels.shape[0]})")
    
    # Validate label range for CIFAR-10
    if labels.min() < 0 or labels.max() > 9:
        raise ValueError(f"CIFAR-10 labels must be in range [0, 9], got range [{labels.min()}, {labels.max()}]")
    
    # Check data types
    if images.dtype != np.float32:
        logger.warning(f"Images dtype is {images.dtype}, should be float32")
    
    if labels.dtype not in [np.int32, np.int64]:
        logger.warning(f"Labels dtype is {labels.dtype}, should be int32 or int64")

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global predictor, model_trainer, preprocessor
    
    try:
        # Initialize predictor
        predictor = CIFAR10Predictor(model_path)
        if os.path.exists(model_path):
            predictor.load_model()
            logger.info("Model loaded successfully")
        else:
            logger.warning("Model file not found. Please train a model first.")
        
        # Initialize model trainer and preprocessor
        model_trainer = CIFAR10Model()
        preprocessor = DataPreprocessor()
        
        logger.info("API server initialized successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "CIFAR-10 Image Classification API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "upload_data": "/upload/data",
            "retrain": "/retrain",
            "model_info": "/model/info",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_loaded = predictor is not None and predictor.model_loaded
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_loaded,
        "retraining_status": retraining_status["status"],
        "uptime": "running"
    }

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """Predict a single image"""
    if not predictor or not predictor.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image file
        image_bytes = await file.read()
        
        # Create temporary file
        temp_path = os.path.join(upload_folder, f"temp_{int(time.time())}_{file.filename}")
        with open(temp_path, "wb") as temp_file:
            temp_file.write(image_bytes)
        
        # Make prediction
        result = predictor.predict_single(temp_path)
        
        # Clean up temporary file
        os.remove(temp_path)
        
        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "prediction": result
        })
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """Predict exactly two images"""
    if not predictor or not predictor.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Enforce exactly 2 images
    if len(files) != 2:
        raise HTTPException(status_code=400, detail="Exactly 2 images are required for batch prediction")
    
    try:
        temp_files = []
        results = []
        
        # Save all uploaded files temporarily
        for file in files:
            if not file.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail="File must be an image")
                
            image_bytes = await file.read()
            temp_path = os.path.join(upload_folder, f"batch_{int(time.time())}_{file.filename}")
            
            with open(temp_path, "wb") as temp_file:
                temp_file.write(image_bytes)
            
            temp_files.append((temp_path, file.filename))
        
        # Make batch prediction
        image_paths = [path for path, _ in temp_files]
        batch_result = predictor.predict_batch(image_paths)
        
        # Combine results with filenames
        for i, (_, filename) in enumerate(temp_files):
            if i < len(batch_result['predictions']):
                result = batch_result['predictions'][i]
                result['filename'] = filename
                results.append(result)
        
        # Clean up temporary files
        for temp_path, _ in temp_files:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        return JSONResponse(content={
            "success": True,
            "batch_size": len(results),
            "predictions": results,
            "total_time": batch_result['total_time'],
            "avg_time_per_image": batch_result['avg_time_per_image']
        })
        
    except Exception as e:
        logger.error(f"Error during batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/data")
async def upload_training_data(files: List[UploadFile] = File(...)):
    """Upload training data for retraining"""
    if len(files) > 1000:  # Limit number of files
        raise HTTPException(status_code=400, detail="Maximum 1000 images per upload")
    
    try:
        uploaded_files = []
        upload_timestamp = int(time.time())
        
        # Create upload directory for this batch
        batch_folder = os.path.join(upload_folder, f"batch_{upload_timestamp}")
        os.makedirs(batch_folder, exist_ok=True)
        
        for file in files:
            if not file.content_type.startswith('image/'):
                continue
            
            # Save file
            file_path = os.path.join(batch_folder, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            uploaded_files.append({
                "filename": file.filename,
                "path": file_path,
                "size": os.path.getsize(file_path)
            })
        
        # Save upload metadata
        metadata = {
            "upload_timestamp": upload_timestamp,
            "batch_folder": batch_folder,
            "files": uploaded_files,
            "total_files": len(uploaded_files)
        }
        
        metadata_path = os.path.join(batch_folder, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        return JSONResponse(content={
            "success": True,
            "message": f"Uploaded {len(uploaded_files)} files",
            "batch_id": upload_timestamp,
            "files": uploaded_files
        })
        
    except Exception as e:
        logger.error(f"Error uploading training data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def retrain_model_background(batch_id: int):
    """Background task for model retraining - COMPLETE FIXED VERSION"""
    global retraining_status, predictor
    
    try:
        retraining_status = {"status": "running", "progress": 0, "message": "Starting retraining..."}
        
        # Load uploaded data
        batch_folder = os.path.join(upload_folder, f"batch_{batch_id}")
        if not os.path.exists(batch_folder):
            raise FileNotFoundError(f"Batch folder not found: {batch_folder}")
        
        retraining_status["progress"] = 10
        retraining_status["message"] = "Loading training data..."
        
        # Get image files
        image_files = []
        for filename in os.listdir(batch_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(batch_folder, filename))
        
        if len(image_files) < 2:
            raise ValueError("Need at least 2 images for retraining")
        
        retraining_status["progress"] = 20
        retraining_status["message"] = "Preprocessing images..."
        
        # Properly preprocess images with correct shapes
        processed_images = []
        labels = []
        
        for image_path in image_files:
            # Preprocess image
            processed_img = preprocess_image_for_training(image_path)
            if processed_img is not None:
                processed_images.append(processed_img)
                # Extract label from filename
                filename = os.path.basename(image_path)
                label = extract_label_from_filename(filename)
                labels.append(label)
        
        if len(processed_images) == 0:
            raise ValueError("No valid images found after preprocessing")
        
        # Convert to numpy arrays with proper shapes and data types
        X_new = np.array(processed_images, dtype=np.float32)  # Shape: (num_images, 32, 32, 3)
        y_new = np.array(labels, dtype=np.int32)  # Shape: (num_images,) - INTEGER LABELS
        
        # Validate tensor shapes
        validate_tensor_shapes(X_new, y_new)
        
        retraining_status["progress"] = 40
        retraining_status["message"] = f"Processed {len(processed_images)} images successfully..."
        
        # Load existing model for retraining
        if not model_trainer.model:
            if os.path.exists(model_path):
                # Load the existing model
                try:
                    model_trainer.model = tf.keras.models.load_model(model_path)
                    logger.info("Loaded existing model for retraining")
                except Exception as e:
                    logger.error(f"Error loading existing model: {e}")
                    # Create new model if loading fails
                    model_trainer.create_model()
                    model_trainer.compile_model()
                    logger.info("Created new model due to loading error")
            else:
                # Create new model if none exists
                model_trainer.create_model()
                model_trainer.compile_model()
                logger.info("Created new model for training")
        
        retraining_status["progress"] = 50
        retraining_status["message"] = "Preparing validation data..."
        
        # Load some validation data (using a subset of CIFAR-10 test set)
        try:
            (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
            
            # Take a small subset for validation
            val_indices = np.random.choice(len(x_test), min(1000, len(x_test)), replace=False)
            x_val = x_test[val_indices].astype('float32') / 255.0
            y_val = y_test[val_indices].flatten().astype(np.int32)  # ENSURE INTEGER LABELS
            
            logger.info(f"Validation data shape: {x_val.shape}, {y_val.shape}")
            
        except Exception as e:
            logger.warning(f"Could not load validation data: {e}. Using training data split.")
            # Split the new data for validation
            split_idx = max(1, len(X_new) // 5)  # Use 20% for validation
            x_val = X_new[:split_idx]
            y_val = y_new[:split_idx]
            X_new = X_new[split_idx:]
            y_new = y_new[split_idx:]
        
        retraining_status["progress"] = 60
        retraining_status["message"] = "Starting model retraining..."
        
        # FIXED: Check model output shape and recompile if necessary
        try:
            # Get model output shape to determine the right loss function
            model_output_shape = model_trainer.model.output_shape
            logger.info(f"Model output shape: {model_output_shape}")
            
            # Determine the correct loss function based on model output
            if model_output_shape[-1] == 10:  # Multi-class classification
                # Check if labels need to be one-hot encoded
                if len(y_new.shape) == 1:  # Sparse labels
                    loss_function = 'sparse_categorical_crossentropy'
                    logger.info("Using sparse_categorical_crossentropy for integer labels")
                else:  # One-hot encoded labels
                    loss_function = 'categorical_crossentropy'
                    logger.info("Using categorical_crossentropy for one-hot labels")
            else:
                raise ValueError(f"Unexpected model output shape: {model_output_shape}")
            
            # Recompile the model with correct loss function
            model_trainer.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Lower LR for fine-tuning
                loss=loss_function,
                metrics=['accuracy']
            )
            
            logger.info(f"Model recompiled with {loss_function}")
            
            # Verify data shapes before training
            logger.info(f"Training data shapes: X={X_new.shape}, y={y_new.shape}")
            logger.info(f"Validation data shapes: X={x_val.shape}, y={y_val.shape}")
            logger.info(f"Label range: min={y_new.min()}, max={y_new.max()}")
            
            # Create TensorFlow dataset for better memory management
            train_dataset = tf.data.Dataset.from_tensor_slices((X_new, y_new))
            train_dataset = train_dataset.batch(16).prefetch(tf.data.AUTOTUNE)
            
            val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
            val_dataset = val_dataset.batch(16).prefetch(tf.data.AUTOTUNE)
            
            # Fine-tune the model with proper callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=3, 
                    restore_best_weights=True,
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5, 
                    patience=2,
                    min_lr=1e-7,
                    verbose=1
                )
            ]
            
            logger.info(f"Starting training with {len(X_new)} training samples")
            
            # Train the model
            history = model_trainer.model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=10,  # Reduced epochs for fine-tuning
                callbacks=callbacks,
                verbose=1
            )
            
            # Log training results
            final_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            final_acc = history.history['accuracy'][-1]
            final_val_acc = history.history['val_accuracy'][-1]
            
            logger.info(f"Training completed - Loss: {final_loss:.4f}, Val Loss: {final_val_loss:.4f}")
            logger.info(f"Training completed - Acc: {final_acc:.4f}, Val Acc: {final_val_acc:.4f}")
            
            retraining_status["progress"] = 80
            retraining_status["message"] = "Saving retrained model..."
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise e
        
        # Save retrained model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_model_path = f"../models/cifar10_model_retrained_{timestamp}.h5"
        
        # Ensure models directory exists
        os.makedirs(os.path.dirname(new_model_path), exist_ok=True)
        
        # Save the model
        model_trainer.model.save(new_model_path)
        logger.info(f"Model saved to {new_model_path}")
        
        # Update predictor with new model
        if hasattr(predictor, 'model'):
            predictor.model = model_trainer.model
            predictor.model_loaded = True
            predictor.model_path = new_model_path
        
        retraining_status["progress"] = 100
        retraining_status["message"] = "Retraining completed successfully!"
        retraining_status["status"] = "completed"
        
        logger.info(f"Model retraining completed. New model saved to {new_model_path}")
        
        # Clean up training data after successful retraining
        try:
            shutil.rmtree(batch_folder)
            logger.info(f"Cleaned up training data folder: {batch_folder}")
        except Exception as e:
            logger.warning(f"Could not clean up training data: {e}")
        
    except Exception as e:
        logger.error(f"Error during retraining: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        retraining_status = {
            "status": "failed",
            "progress": 0,
            "message": f"Retraining failed: {str(e)}"
        }

@app.post("/retrain")
async def trigger_retraining(background_tasks: BackgroundTasks, batch_id: int):
    """Trigger model retraining with uploaded data"""
    global retraining_status  # Fixed: Moved global declaration to the top
    
    if retraining_status["status"] == "running":
        raise HTTPException(status_code=409, detail="Retraining already in progress")
    
    # Check if batch exists
    batch_folder = os.path.join(upload_folder, f"batch_{batch_id}")
    if not os.path.exists(batch_folder):
        raise HTTPException(status_code=404, detail="Batch not found")
    
    # Reset retraining status
    retraining_status = {"status": "idle", "progress": 0, "message": ""}
    
    # Start retraining in background
    background_tasks.add_task(retrain_model_background, batch_id)
    
    return JSONResponse(content={
        "success": True,
        "message": "Retraining started",
        "batch_id": batch_id
    })

@app.get("/retrain/status")
async def get_retraining_status():
    """Get current retraining status"""
    return JSONResponse(content=retraining_status)

@app.get("/model/info")
async def get_model_info():
    """Get model information"""
    if not predictor:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    try:
        model_info = predictor.get_model_info()
        prediction_stats = predictor.get_prediction_stats()
        
        return JSONResponse(content={
            "model_info": model_info,
            "prediction_stats": prediction_stats,
            "retraining_status": retraining_status
        })
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return JSONResponse(content={
            "model_info": {"error": str(e)},
            "prediction_stats": {"error": str(e)},
            "retraining_status": retraining_status
        })

@app.get("/model/stats")
async def get_model_stats():
    """Get model performance statistics"""
    if not predictor:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    try:
        stats = predictor.get_prediction_stats()
        
        # Add system stats
        stats.update({
            "model_loaded": predictor.model_loaded if hasattr(predictor, 'model_loaded') else False,
            "model_path": predictor.model_path if hasattr(predictor, 'model_path') else None,
            "server_uptime": "running",
            "timestamp": datetime.now().isoformat()
        })
        
        return JSONResponse(content=stats)
    except Exception as e:
        logger.error(f"Error getting model stats: {str(e)}")
        return JSONResponse(content={
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })

@app.delete("/data/clear")
async def clear_uploaded_data():
    """Clear all uploaded data"""
    try:
        if os.path.exists(upload_folder):
            shutil.rmtree(upload_folder)
            os.makedirs(upload_folder, exist_ok=True)
        
        return JSONResponse(content={
            "success": True,
            "message": "All uploaded data cleared"
        })
        
    except Exception as e:
        logger.error(f"Error clearing data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/batches")
async def list_data_batches():
    """List all uploaded data batches"""
    try:
        batches = []
        
        if os.path.exists(upload_folder):
            for item in os.listdir(upload_folder):
                batch_path = os.path.join(upload_folder, item)
                if os.path.isdir(batch_path) and item.startswith("batch_"):
                    metadata_path = os.path.join(batch_path, "metadata.json")
                    
                    if os.path.exists(metadata_path):
                        try:
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                            batches.append(metadata)
                        except Exception as e:
                            logger.warning(f"Error reading metadata for {item}: {e}")
        
        return JSONResponse(content={
            "success": True,
            "batches": batches,
            "total_batches": len(batches)
        })
        
    except Exception as e:
        logger.error(f"Error listing batches: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Additional debugging endpoint
@app.get("/debug/model")
async def debug_model():
    """Debug endpoint to check model status"""
    debug_info = {
        "predictor_exists": predictor is not None,
        "model_trainer_exists": model_trainer is not None,
        "model_path_exists": os.path.exists(model_path) if model_path else False,
        "model_path": model_path,
        "retraining_status": retraining_status
    }
    
    if predictor:
        debug_info.update({
            "predictor_model_loaded": getattr(predictor, 'model_loaded', False),
            "predictor_model_path": getattr(predictor, 'model_path', None)
        })
    
    if model_trainer:
        debug_info.update({
            "model_trainer_has_model": hasattr(model_trainer, 'model') and model_trainer.model is not None
        })
    
    return JSONResponse(content=debug_info)

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Global exception: {str(exc)}")
    logger.error(f"Exception type: {type(exc).__name__}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "type": type(exc).__name__
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )