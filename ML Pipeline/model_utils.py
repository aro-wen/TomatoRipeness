import joblib
import numpy as np
import cv2
import os
from pathlib import Path
from .feature_extraction import TomatoFeatureExtractor

class TomatoClassifier:
    def __init__(self, model_path=None):
        self.model_path = model_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'models',
            'tomato_svm_model.pkl'
        )
        self.model = None
        self.label_encoder = None
        self.feature_names = None
        self.class_names = None
        self.extractor = TomatoFeatureExtractor()
        
    def load_model(self):
        """Load the trained model"""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        model_data = joblib.load(self.model_path)
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.class_names = model_data.get('class_names', ['unripe', 'turning', 'ripe', 'rotten'])
        
    def is_tomato_image(self, image):
        """Basic check to determine if image contains tomatoes"""
        # This is a simplified check - in practice you might want more sophisticated detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Check for red/green colors typical of tomatoes
        # Red color range
        red_lower1 = np.array([0, 50, 50])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 50, 50])
        red_upper2 = np.array([180, 255, 255])
        
        # Green color range
        green_lower = np.array([35, 50, 50])
        green_upper = np.array([85, 255, 255])
        
        # Create masks
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        
        # Combine masks
        tomato_mask = red_mask1 + red_mask2 + green_mask
        
        # Check if sufficient tomato-colored pixels exist
        tomato_pixels = np.sum(tomato_mask > 0)
        total_pixels = image.shape[0] * image.shape[1]
        tomato_ratio = tomato_pixels / total_pixels
        
        return tomato_ratio > 0.1  # At least 10% tomato-colored pixels
    
    def predict_image(self, image_path):
        """Predict ripeness class for an uploaded image"""
        if self.model is None:
            self.load_model()
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {
                'success': False,
                'error': 'Could not load image'
            }
        
        # Check if image contains tomatoes
        if not self.is_tomato_image(image):
            return {
                'success': False,
                'error': 'This image does not appear to contain tomatoes. Please upload a tomato image.'
            }
        
        # Resize for consistent processing
        image = cv2.resize(image, (224, 224))
        
        # Extract all analysis features
        analysis_result = self.extractor.process_image_for_analysis(image_path)
        
        if analysis_result is None:
            return {
                'success': False,
                'error': 'Could not process image'
            }
        
        # Prepare features for model prediction
        features_dict = analysis_result['features']
        features_array = []
        
        for name in self.feature_names:
            if name in features_dict:
                features_array.append(features_dict[name])
            else:
                features_array.append(0.0)  # Default value for missing features
        
        features_array = np.array(features_array).reshape(1, -1)
        features_array = np.nan_to_num(features_array, nan=0.0)
        
        # Make prediction
        try:
            prediction = self.model.predict(features_array)[0]
            probabilities = self.model.predict_proba(features_array)[0]
            
            # Convert to class names and probabilities
            predicted_class = self.label_encoder.inverse_transform([prediction])[0]
            
            class_probabilities = {}
            for i, class_name in enumerate(self.label_encoder.classes_):
                class_probabilities[class_name] = float(probabilities[i])
            
            confidence = float(np.max(probabilities))
            
            # Enhanced spoilage analysis
            spoilage_analysis = self._detailed_spoilage_analysis(analysis_result['spoilage_features'])
            
            # Get transport survivability
            transport_analysis = self.extractor.get_transport_survivability(
                predicted_class, analysis_result['spoilage_score']
            )
            
            return {
                'success': True,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'class_probabilities': class_probabilities,
                'cielab_analysis': analysis_result['cielab_analysis'],
                'spoilage_score': analysis_result['spoilage_score'],
                'spoilage_analysis': spoilage_analysis,
                'transport_survivability': transport_analysis
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Prediction error: {str(e)}'
            }
    
    def _detailed_spoilage_analysis(self, spoilage_features):
        """Create detailed spoilage analysis from features"""
        analysis = {
            'mold_detection': spoilage_features['brown_pixel_ratio'] * 100,
            'black_spots': spoilage_features['dark_pixel_ratio'] * 100,
            'brown_decay': spoilage_features['brown_pixel_ratio'] * 100,
            'texture_loss': spoilage_features['low_saturation_ratio'] * 100
        }
        
        # Add safety warnings
        if analysis['mold_detection'] > 10:
            analysis['safety_warning'] = "SEVERE SPOILAGE DETECTED - DO NOT CONSUME"
        elif analysis['mold_detection'] > 5:
            analysis['safety_warning'] = "MODERATE SPOILAGE - CAUTION ADVISED"
        elif any(v > 8 for v in analysis.values()):
            analysis['safety_warning'] = "POSSIBLE EARLY SPOILAGE - INSPECT CAREFULLY"
        else:
            analysis['safety_warning'] = None
        
        return analysis
    
    def analyze_batch_images(self, image_directory):
        """Analyze multiple images in a directory"""
        image_dir = Path(image_directory)
        results = []
        
        if not image_dir.exists():
            return {'error': 'Directory not found'}
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(image_dir.glob(f'*{ext}'))
            image_files.extend(image_dir.glob(f'*{ext.upper()}'))
        
        for image_path in image_files:
            result = self.predict_image(str(image_path))
            result['filename'] = image_path.name
            results.append(result)
        
        return {
            'total_images': len(results),
            'results': results
        }
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if self.model is None:
            self.load_model()
        
        return {
            'model_type': 'Support Vector Machine (SVM)',
            'classes': self.class_names,
            'feature_count': len(self.feature_names),
            'model_loaded': True
        }

class TomatoImageValidator:
    """Utility class for validating tomato images"""
    
    @staticmethod
    def validate_image_file(file_path, max_size_mb=10):
        """Validate uploaded image file"""
        errors = []
        
        # Check if file exists
        if not Path(file_path).exists():
            errors.append("File not found")
            return False, errors
        
        # Check file size
        file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
        if file_size_mb > max_size_mb:
            errors.append(f"File too large ({file_size_mb:.1f}MB). Maximum size: {max_size_mb}MB")
        
        # Check if it's a valid image
        try:
            image = cv2.imread(file_path)
            if image is None:
                errors.append("Invalid image file")
            else:
                # Check image dimensions
                h, w = image.shape[:2]
                if w < 50 or h < 50:
                    errors.append("Image too small (minimum 50x50 pixels)")
                if w > 5000 or h > 5000:
                    errors.append("Image too large (maximum 5000x5000 pixels)")
        except Exception as e:
            errors.append(f"Error reading image: {str(e)}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def preprocess_image(image_path, output_size=(224, 224)):
        """Preprocess image for consistent analysis"""
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Resize while maintaining aspect ratio
        h, w = image.shape[:2]
        if h != output_size[0] or w != output_size[1]:
            # Calculate scaling factor
            scale = min(output_size[0] / h, output_size[1] / w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize
            resized = cv2.resize(image, (new_w, new_h))
            
            # Pad to desired size
            delta_w = output_size[1] - new_w
            delta_h = output_size[0] - new_h
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)
            
            padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            return padded
        
        return image

def create_sample_predictions():
    """Create sample predictions for testing the interface"""
    return {
        'success': True,
        'predicted_class': 'ripe',
        'confidence': 0.87,
        'class_probabilities': {
            'unripe': 0.05,
            'turning': 0.08,
            'ripe': 0.87,
            'rotten': 0.0
        },
        'cielab_analysis': {
            'L_lightness': 45.2,
            'a_green_red': 23.1,
            'b_blue_yellow': 15.8
        },
        'spoilage_score': 2.1,
        'spoilage_analysis': {
            'mold_detection': 1.2,
            'black_spots': 0.8,
            'brown_decay': 2.1,
            'texture_loss': 5.3,
            'safety_warning': None
        },
        'transport_survivability': {
            'short_distance': {'safe': True, 'message': 'GOOD - Use soon after transport'},
            'medium_distance': {'safe': True, 'message': 'CAUTION - Refrigerate and use quickly'},
            'long_distance': {'safe': False, 'message': 'RISKY - May spoil during long transport'}
        }
    }

if __name__ == "__main__":
    # Test the classifier
    classifier = TomatoClassifier()
    
    try:
        # Try to load model
        classifier.load_model()
        print("Model loaded successfully!")
        
        # Get model info
        info = classifier.get_model_info()
        print(f"Model info: {info}")
        
    except FileNotFoundError:
        print("Model file not found. Please train the model first by running train_model.py")
        
    # Create sample prediction for testing
    sample = create_sample_predictions()
    print(f"Sample prediction: {sample}")