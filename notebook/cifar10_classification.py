"""
CIFAR-10 Image Classification Pipeline
Complete ML Pipeline with Model Training, Evaluation, and Deployment
"""

# Import required libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pickle
import os
from datetime import datetime

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

print("=== CIFAR-10 Image Classification Pipeline ===")
print("Complete ML Pipeline with Model Training, Evaluation, and Deployment")

# 1. Data Acquisition and Preprocessing
print("\n1. Loading CIFAR-10 Dataset...")

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print(f"Training data shape: {x_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test data shape: {x_test.shape}")
print(f"Test labels shape: {y_test.shape}")
print(f"Number of classes: {len(class_names)}")

# Data preprocessing
def preprocess_data(x_train, x_test, y_train, y_test):
    """Preprocess the data"""
    print("Preprocessing data...")
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Convert labels to categorical
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    return x_train, x_test, y_train, y_test

x_train_processed, x_test_processed, y_train_processed, y_test_processed = preprocess_data(
    x_train, x_test, y_train, y_test
)

print("Data preprocessing completed!")
print(f"Processed training data shape: {x_train_processed.shape}")
print(f"Processed training labels shape: {y_train_processed.shape}")

# 2. Data Visualization and Analysis
print("\n2. Data Visualization and Analysis...")

# Visualize sample images
def visualize_sample_images():
    """Visualize sample images from the dataset"""
    plt.figure(figsize=(15, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(x_train[i])
        plt.title(f'{class_names[y_train[i][0]]}')
        plt.axis('off')
    plt.suptitle('Sample Images from CIFAR-10 Dataset', fontsize=16)
    plt.tight_layout()
    plt.savefig('../models/sample_images.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Sample images visualization saved to ../models/sample_images.png")

# Class distribution analysis
def analyze_class_distribution():
    """Analyze class distribution in the dataset"""
    unique, counts = np.unique(y_train, return_counts=True)
    plt.figure(figsize=(12, 6))
    plt.bar([class_names[i] for i in unique], counts)
    plt.title('Class Distribution in Training Data')
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('../models/class_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Class distribution:")
    for i, count in enumerate(counts):
        print(f"{class_names[i]}: {count} samples")

# Pixel intensity analysis
def analyze_pixel_intensity():
    """Analyze pixel intensity distribution"""
    plt.figure(figsize=(15, 5))
    
    # RGB channel analysis
    for i, color in enumerate(['Red', 'Green', 'Blue']):
        plt.subplot(1, 3, i + 1)
        channel_data = x_train[:, :, :, i].flatten()
        plt.hist(channel_data, bins=50, alpha=0.7, color=['red', 'green', 'blue'][i])
        plt.title(f'{color} Channel Intensity Distribution')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('../models/pixel_intensity_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Pixel intensity analysis saved to ../models/pixel_intensity_analysis.png")

# Run visualizations
try:
    visualize_sample_images()
    analyze_class_distribution()
    analyze_pixel_intensity()
except Exception as e:
    print(f"Visualization error (continuing anyway): {e}")

# 3. Model Creation and Architecture
print("\n3. Creating CNN Model...")

def create_cnn_model(input_shape=(32, 32, 3), num_classes=10):
    """
    Create a CNN model with optimization techniques
    """
    model = keras.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
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
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Create the model
model = create_cnn_model()
print("Model architecture:")
model.summary()

# Compile the model with optimization techniques
print("\n4. Compiling Model...")
optimizer = keras.optimizers.Adam(learning_rate=0.001, decay=1e-6)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)

# Define callbacks for optimization
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=0.0001
    ),
    keras.callbacks.ModelCheckpoint(
        '../models/best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
]

print("Model compiled with optimization techniques!")

# 5. Model Training
print("\n5. Starting Model Training...")

# Create models directory if it doesn't exist
os.makedirs('../models', exist_ok=True)

# Data augmentation for better generalization
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

datagen.fit(x_train_processed)

# Train the model
batch_size = 32
epochs = 50

print(f"Training with {epochs} epochs and batch size {batch_size}")
print("This may take 30-60 minutes depending on your hardware...")

history = model.fit(
    datagen.flow(x_train_processed, y_train_processed, batch_size=batch_size),
    steps_per_epoch=len(x_train_processed) // batch_size,
    epochs=epochs,
    validation_data=(x_test_processed, y_test_processed),
    callbacks=callbacks,
    verbose=1
)

print("Model training completed!")

# 6. Model Evaluation and Metrics
print("\n6. Evaluating Model Performance...")

# Plot training history
def plot_training_history(history):
    """Plot training history"""
    plt.figure(figsize=(15, 5))
    
    # Accuracy plot
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss plot
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Learning rate plot (if available)
    plt.subplot(1, 3, 3)
    if 'lr' in history.history:
        plt.plot(history.history['lr'], label='Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'Learning Rate\nNot Recorded', ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig('../models/training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Training history plot saved to ../models/training_history.png")

plot_training_history(history)

# Comprehensive model evaluation
test_loss, test_accuracy, test_precision, test_recall = model.evaluate(x_test_processed, y_test_processed, verbose=0)

# Get predictions
y_pred = model.predict(x_test_processed)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test_processed, axis=1)

# Calculate additional metrics
accuracy = accuracy_score(y_true_classes, y_pred_classes)
precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')

print("=== MODEL EVALUATION METRICS ===")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1-Score: {f1:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Classification report
print("\n=== DETAILED CLASSIFICATION REPORT ===")
print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

# Confusion Matrix
def plot_confusion_matrix():
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('../models/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Per-class accuracy
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    print("\n=== PER-CLASS ACCURACY ===")
    for i, acc in enumerate(class_accuracy):
        print(f"{class_names[i]}: {acc:.4f}")
    
    return cm, class_accuracy

cm, class_accuracy = plot_confusion_matrix()

# 7. Model Saving and Serialization
print("\n7. Saving Model and Artifacts...")

# Save the trained model
model.save('../models/cifar10_cnn_model.h5')
print("Model saved as cifar10_cnn_model.h5")

# Save model architecture as JSON
model_json = model.to_json()
with open('../models/model_architecture.json', 'w') as json_file:
    json_file.write(model_json)

# Save training history
with open('../models/training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

# Save class names
with open('../models/class_names.pkl', 'wb') as f:
    pickle.dump(class_names, f)

# Save evaluation metrics
metrics = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'test_loss': test_loss,
    'confusion_matrix': cm.tolist(),
    'class_accuracy': class_accuracy.tolist(),
    'timestamp': datetime.now().isoformat()
}

with open('../models/evaluation_metrics.pkl', 'wb') as f:
    pickle.dump(metrics, f)

print("All model artifacts saved successfully!")

# 8. Model Testing and Prediction Functions
print("\n8. Testing Prediction Functions...")

def predict_single_image(model, image, class_names):
    """
    Predict a single image and return probabilities
    """
    # Ensure image is in correct format
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
    
    # Normalize if needed
    if image.max() > 1.0:
        image = image.astype('float32') / 255.0
    
    # Make prediction
    predictions = model.predict(image, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    
    # Get top 3 predictions
    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
    top_3_predictions = [(class_names[idx], predictions[0][idx]) for idx in top_3_idx]
    
    return {
        'predicted_class': class_names[predicted_class_idx],
        'confidence': float(confidence),
        'top_3_predictions': top_3_predictions,
        'all_probabilities': {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}
    }

# Test the prediction function
test_idx = 0
test_image = x_test[test_idx]
true_label = class_names[y_test[test_idx][0]]

prediction_result = predict_single_image(model, test_image, class_names)

print(f"True label: {true_label}")
print(f"Predicted: {prediction_result['predicted_class']} (Confidence: {prediction_result['confidence']:.4f})")
print("\nTop 3 predictions:")
for class_name, prob in prediction_result['top_3_predictions']:
    print(f"  {class_name}: {prob:.4f}")

# Visualize the test image
def visualize_prediction_sample():
    """Visualize prediction sample"""
    plt.figure(figsize=(8, 6))
    plt.imshow(test_image)
    plt.title(f"True: {true_label} | Predicted: {prediction_result['predicted_class']} ({prediction_result['confidence']:.3f})")
    plt.axis('off')
    plt.savefig('../models/prediction_sample.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Prediction sample saved to ../models/prediction_sample.png")

visualize_prediction_sample()

# Test multiple predictions
def test_multiple_predictions():
    """Test multiple predictions and visualize"""
    plt.figure(figsize=(15, 10))
    for i in range(12):
        plt.subplot(3, 4, i + 1)
        test_image = x_test[i]
        true_label = class_names[y_test[i][0]]
        
        prediction_result = predict_single_image(model, test_image, class_names)
        predicted_label = prediction_result['predicted_class']
        confidence = prediction_result['confidence']
        
        plt.imshow(test_image)
        color = 'green' if predicted_label == true_label else 'red'
        plt.title(f"True: {true_label}\nPred: {predicted_label} ({confidence:.2f})", color=color, fontsize=10)
        plt.axis('off')
    
    plt.suptitle('Sample Predictions on Test Set', fontsize=16)
    plt.tight_layout()
    plt.savefig('../models/multiple_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Multiple predictions sample saved to ../models/multiple_predictions.png")

test_multiple_predictions()

# 9. Model Performance Analysis
print("\n9. Analyzing Model Performance by Class...")

from sklearn.metrics import precision_recall_fscore_support
import pandas as pd

precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
    y_true_classes, y_pred_classes, average=None
)

# Create performance DataFrame
performance_df = pd.DataFrame({
    'Class': class_names,
    'Precision': precision_per_class,
    'Recall': recall_per_class,
    'F1-Score': f1_per_class,
    'Support': support_per_class,
    'Accuracy': class_accuracy
})

print("=== PER-CLASS PERFORMANCE METRICS ===")
print(performance_df.round(4))

# Visualize per-class performance
def plot_per_class_performance():
    """Plot per-class performance metrics"""
    plt.figure(figsize=(15, 8))
    
    x = np.arange(len(class_names))
    width = 0.2
    
    plt.bar(x - width, precision_per_class, width, label='Precision', alpha=0.8)
    plt.bar(x, recall_per_class, width, label='Recall', alpha=0.8)
    plt.bar(x + width, f1_per_class, width, label='F1-Score', alpha=0.8)
    
    plt.xlabel('Classes')
    plt.ylabel('Score')
    plt.title('Per-Class Performance Metrics')
    plt.xticks(x, class_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../models/per_class_performance.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Per-class performance plot saved to ../models/per_class_performance.png")

plot_per_class_performance()

# 10. Summary and Final Results
print("\n" + "="*60)
print("CIFAR-10 CLASSIFICATION PIPELINE COMPLETED SUCCESSFULLY!")
print("="*60)

print("\n=== FINAL RESULTS SUMMARY ===")
print(f"✅ Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"✅ Model Precision: {precision:.4f}")
print(f"✅ Model Recall: {recall:.4f}")
print(f"✅ Model F1-Score: {f1:.4f}")
print(f"✅ Training Epochs: {len(history.history['accuracy'])}")
print(f"✅ Best Validation Accuracy: {max(history.history['val_accuracy']):.4f}")

print("\n=== FILES CREATED ===")
model_files = [
    'cifar10_cnn_model.h5',
    'best_model.h5',
    'model_architecture.json',
    'training_history.pkl',
    'class_names.pkl',
    'evaluation_metrics.pkl',
    'sample_images.png',
    'class_distribution.png',
    'pixel_intensity_analysis.png',
    'training_history.png',
    'confusion_matrix.png',
    'prediction_sample.png',
    'multiple_predictions.png',
    'per_class_performance.png'
]

for file in model_files:
    file_path = f'../models/{file}'
    if os.path.exists(file_path):
        print(f"✅ {file}")
    else:
        print(f"❌ {file} (not created)")

print("\n=== NEXT STEPS ===")
print("1. 🚀 Start the API server: python scripts/api_server.py")
print("2. 🌐 Launch the web interface: npm run dev")
print("3. 🧪 Run load tests: locust -f scripts/locust_load_test.py")
print("4. 🐳 Deploy with Docker: docker-compose up --build")
print("5. 📹 Create your video demonstration")

print("\n=== MODEL READY FOR DEPLOYMENT ===")
print("Your CIFAR-10 classification model is now ready for production!")
print("All artifacts have been saved to the ../models/ directory.")
print("You can now use the FastAPI server and web interface for predictions and retraining.")

print("\n" + "="*60)
print("🎉 PIPELINE EXECUTION COMPLETED! 🎉")
print("="*60)
