"""
Locust load testing script for CIFAR-10 API
"""

from locust import HttpUser, task, between
import random
import io
import numpy as np
from PIL import Image
import time
import json

class MyBaseUser(HttpUser):
    host = "http://localhost:8000"  # Base host for all users
    wait_time = between(1, 3)
    
    def on_start(self):
        """Called when a user starts"""
        self.test_images = self.generate_test_images()
        
    def generate_test_images(self, count=10):
        """Generate test images for load testing"""
        images = []
        
        for i in range(count):
            # Create random 32x32 RGB image
            image_array = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            image = Image.fromarray(image_array)
            
            # Convert to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            images.append(img_byte_arr.getvalue())
        
        return images
    
    @task(10)
    def predict_single_image(self):
        """Test single image prediction endpoint"""
        # Select random test image
        image_data = random.choice(self.test_images)
        
        files = {
            'file': ('test_image.png', io.BytesIO(image_data), 'image/png')
        }
        
        with self.client.post("/predict", files=files, catch_response=True) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if result.get('success'):
                        response.success()
                    else:
                        response.failure("Prediction failed")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(3)
    def predict_batch_images(self):
        """Test batch image prediction endpoint"""
        # Select 2-5 random test images
        batch_size = random.randint(2, 5)
        selected_images = random.sample(self.test_images, min(batch_size, len(self.test_images)))
        
        files = []
        for i, image_data in enumerate(selected_images):
            files.append(
                ('files', (f'test_image_{i}.png', io.BytesIO(image_data), 'image/png'))
            )
        
        with self.client.post("/predict/batch", files=files, catch_response=True) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if result.get('success'):
                        response.success()
                    else:
                        response.failure("Batch prediction failed")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(2)
    def check_health(self):
        """Test health check endpoint"""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if result.get('status') == 'healthy':
                        response.success()
                    else:
                        response.failure("Service not healthy")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(1)
    def get_model_info(self):
        """Test model info endpoint"""
        with self.client.get("/model/info", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(1)
    def get_model_stats(self):
        """Test model stats endpoint"""
        with self.client.get("/model/stats", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

class HighLoadUser(HttpUser):
    """
    High-load user for stress testing
    """
    
    wait_time = between(0.1, 0.5)  # Very short wait time
    
    def on_start(self):
        """Called when a user starts"""
        # Generate a single test image for reuse
        image_array = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        image = Image.fromarray(image_array)
        
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        self.test_image = img_byte_arr.getvalue()
    
    @task
    def rapid_predictions(self):
        """Rapid fire predictions for stress testing"""
        files = {
            'file': ('test_image.png', io.BytesIO(self.test_image), 'image/png')
        }
        
        start_time = time.time()
        
        with self.client.post("/predict", files=files, catch_response=True) as response:
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            
            if response.status_code == 200:
                # Log response time for analysis
                if response_time > 1000:  # More than 1 second
                    response.failure(f"Slow response: {response_time:.2f}ms")
                else:
                    response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

# Custom load test scenarios
class ScenarioUser(HttpUser):
    """
    User class for specific testing scenarios
    """
    
    wait_time = between(2, 5)
    
    def on_start(self):
        """Initialize test data"""
        self.test_images = []
        
        # Generate different types of test images
        for _ in range(5):
            # Random noise image
            noise_image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            
            # Solid color image
            color = np.random.randint(0, 255, 3)
            solid_image = np.full((32, 32, 3), color, dtype=np.uint8)
            
            # Gradient image
            gradient = np.linspace(0, 255, 32, dtype=np.uint8)
            gradient_image = np.stack([gradient] * 32, axis=0)
            gradient_image = np.stack([gradient_image] * 3, axis=2)
            
            for img_array in [noise_image, solid_image, gradient_image]:
                image = Image.fromarray(img_array)
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                self.test_images.append(img_byte_arr.getvalue())
    
    @task(5)
    def test_different_image_types(self):
        """Test with different types of images"""
        image_data = random.choice(self.test_images)
        
        files = {
            'file': ('test_image.png', io.BytesIO(image_data), 'image/png')
        }
        
        with self.client.post("/predict", files=files, catch_response=True) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    prediction = result.get('prediction', {})
                    confidence = prediction.get('confidence', 0)
                    
                    # Check if confidence is reasonable
                    if 0 <= confidence <= 1:
                        response.success()
                    else:
                        response.failure(f"Invalid confidence: {confidence}")
                        
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(2)
    def test_concurrent_uploads(self):
        """Test uploading training data"""
        # Generate small batch of images
        batch_images = random.sample(self.test_images, 3)
        
        files = []
        for i, image_data in enumerate(batch_images):
            files.append(
                ('files', (f'upload_{i}.png', io.BytesIO(image_data), 'image/png'))
            )
        
        with self.client.post("/upload/data", files=files, catch_response=True) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if result.get('success'):
                        response.success()
                    else:
                        response.failure("Upload failed")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}")

# Performance monitoring user
class MonitoringUser(HttpUser):
    """
    User for monitoring system performance during load tests
    """
    
    wait_time = between(5, 10)  # Check less frequently
    
    @task
    def monitor_system_health(self):
        """Monitor system health and performance"""
        endpoints = ["/health", "/model/stats", "/model/info"]
        
        for endpoint in endpoints:
            with self.client.get(endpoint, catch_response=True) as response:
                if response.status_code == 200:
                    response.success()
                else:
                    response.failure(f"Monitoring endpoint {endpoint} failed: HTTP {response.status_code}")

if __name__ == "__main__":
    # This script is meant to be run with locust command
    # Example usage:
    # locust -f locust_load_test.py --host=http://localhost:8000
    print("Load testing script for CIFAR-10 API")
    print("Usage: locust -f locust_load_test.py --host=http://localhost:8000")
    print("\nAvailable user classes:")
    print("- CIFAR10APIUser: Standard load testing")
    print("- HighLoadUser: High-frequency stress testing")
    print("- ScenarioUser: Specific scenario testing")
    print("- MonitoringUser: System monitoring during tests")
