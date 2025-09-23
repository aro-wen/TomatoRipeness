import cv2
import numpy as np
from skimage import color, feature, segmentation
from sklearn.preprocessing import StandardScaler
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

class TomatoFeatureExtractor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def rgb_to_cielab(self, image):
        """Convert RGB image to CIELAB color space"""
        # Ensure image is in RGB format (OpenCV uses BGR)
        if len(image.shape) == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
            
        # Convert to CIELAB
        lab_image = color.rgb2lab(rgb_image)
        return lab_image
    
    def extract_cielab_features(self, image):
        """Extract statistical features from CIELAB color space"""
        lab_image = self.rgb_to_cielab(image)
        
        # Separate L*, a*, b* channels
        L_channel = lab_image[:, :, 0]  # Lightness (0-100)
        a_channel = lab_image[:, :, 1]  # Green-Red (-128 to 127)
        b_channel = lab_image[:, :, 2]  # Blue-Yellow (-128 to 127)
        
        features = {}
        
        # L* (Lightness) features
        features['L_mean'] = np.mean(L_channel)
        features['L_std'] = np.std(L_channel)
        features['L_median'] = np.median(L_channel)
        features['L_min'] = np.min(L_channel)
        features['L_max'] = np.max(L_channel)
        
        # a* (Green-Red) features
        features['a_mean'] = np.mean(a_channel)
        features['a_std'] = np.std(a_channel)
        features['a_median'] = np.median(a_channel)
        features['a_min'] = np.min(a_channel)
        features['a_max'] = np.max(a_channel)
        
        # b* (Blue-Yellow) features
        features['b_mean'] = np.mean(b_channel)
        features['b_std'] = np.std(b_channel)
        features['b_median'] = np.median(b_channel)
        features['b_min'] = np.min(b_channel)
        features['b_max'] = np.max(b_channel)
        
        # Color ratios and derived features
        features['a_b_ratio'] = features['a_mean'] / (features['b_mean'] + 1e-8)
        features['chroma'] = np.sqrt(features['a_mean']**2 + features['b_mean']**2)
        
        # Hue angle in CIELAB
        features['hue_angle'] = np.arctan2(features['b_mean'], features['a_mean']) * 180 / np.pi
        
        return features
    
    def extract_color_histograms(self, image, bins=32):
        """Extract color histogram features"""
        # Convert to different color spaces
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab_image = self.rgb_to_cielab(image)
        
        features = {}
        
        # RGB histograms
        for i, color in enumerate(['R', 'G', 'B']):
            hist = cv2.calcHist([rgb_image], [i], None, [bins], [0, 256])
            hist = hist.flatten() / np.sum(hist)  # Normalize
            for j, val in enumerate(hist):
                features[f'{color}_hist_{j}'] = val
        
        # HSV histograms
        for i, color in enumerate(['H', 'S', 'V']):
            if i == 0:  # Hue
                hist = cv2.calcHist([hsv_image], [i], None, [bins], [0, 180])
            else:
                hist = cv2.calcHist([hsv_image], [i], None, [bins], [0, 256])
            hist = hist.flatten() / np.sum(hist)
            for j, val in enumerate(hist):
                features[f'{color}_hist_{j}'] = val
        
        return features
    
    def extract_texture_features(self, image):
        """Extract texture features using Local Binary Patterns"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Local Binary Pattern
        lbp = feature.local_binary_pattern(gray, P=24, R=3, method='uniform')
        
        # Calculate histogram of LBP
        hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-8)  # Normalize
        
        features = {}
        for i, val in enumerate(hist):
            features[f'lbp_{i}'] = val
            
        # Additional texture measures
        features['texture_contrast'] = np.std(gray)
        features['texture_energy'] = np.sum(gray**2) / (gray.shape[0] * gray.shape[1])
        
        return features
    
    def extract_shape_features(self, image):
        """Extract shape and morphological features"""
        # Convert to grayscale and threshold to get tomato shape
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use Otsu thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        features = {}
        
        if contours:
            # Get largest contour (assumed to be the tomato)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Basic shape features
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            features['area'] = area
            features['perimeter'] = perimeter
            features['circularity'] = 4 * np.pi * area / (perimeter**2 + 1e-8)
            
            # Aspect ratio
            x, y, w, h = cv2.boundingRect(largest_contour)
            features['aspect_ratio'] = w / (h + 1e-8)
            
            # Solidity (convex hull ratio)
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            features['solidity'] = area / (hull_area + 1e-8)
            
        else:
            # Default values if no contours found
            features.update({
                'area': 0, 'perimeter': 0, 'circularity': 0,
                'aspect_ratio': 1, 'solidity': 0
            })
        
        return features
    
    def extract_spoilage_features(self, image):
        """Extract features related to spoilage detection"""
        features = {}
        
        # Convert to different color spaces
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = self.rgb_to_cielab(image)
        
        # Dark spot detection
        dark_threshold = 50
        dark_pixels = np.sum(gray < dark_threshold)
        features['dark_pixel_ratio'] = dark_pixels / (gray.shape[0] * gray.shape[1])
        
        # Low saturation areas (grayish areas indicating decay)
        low_sat_threshold = 30
        low_sat_pixels = np.sum(hsv[:, :, 1] < low_sat_threshold)
        features['low_saturation_ratio'] = low_sat_pixels / (hsv.shape[0] * hsv.shape[1])
        
        # Brown color detection (decay)
        h_channel = hsv[:, :, 0]
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]
        
        # Brown hue range (adjusted for HSV in OpenCV: 0-179)
        brown_mask = ((h_channel >= 8) & (h_channel <= 25) & 
                     (s_channel > 50) & (v_channel < 150))
        features['brown_pixel_ratio'] = np.sum(brown_mask) / (h_channel.shape[0] * h_channel.shape[1])
        
        # Color uniformity (less uniform = more likely to have spots)
        features['color_variance'] = np.var(gray)
        
        # Edge density (spoiled fruit often has irregular edges)
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        return features
    
    def extract_ripeness_features(self, image):
        """Extract features specifically for ripeness classification"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        features = {}
        
        # Red color detection (ripe tomatoes)
        h_channel = hsv[:, :, 0]
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]
        
        # Red hue ranges (HSV in OpenCV: 0-179)
        red_mask1 = ((h_channel <= 10) | (h_channel >= 170)) & (s_channel > 50) & (v_channel > 50)
        red_mask2 = ((h_channel >= 0) & (h_channel <= 20)) & (s_channel > 30) & (v_channel > 30)
        red_mask = red_mask1 | red_mask2
        features['red_pixel_ratio'] = np.sum(red_mask) / (h_channel.shape[0] * h_channel.shape[1])
        
        # Green color detection (unripe tomatoes)
        green_mask = ((h_channel >= 35) & (h_channel <= 85)) & (s_channel > 40) & (v_channel > 40)
        features['green_pixel_ratio'] = np.sum(green_mask) / (h_channel.shape[0] * h_channel.shape[1])
        
        # Yellow/orange detection (turning tomatoes)
        yellow_mask = ((h_channel >= 15) & (h_channel <= 35)) & (s_channel > 40) & (v_channel > 50)
        features['yellow_pixel_ratio'] = np.sum(yellow_mask) / (h_channel.shape[0] * h_channel.shape[1])
        
        # Overall brightness (ripe tomatoes tend to be brighter)
        features['brightness'] = np.mean(v_channel)
        
        # Color intensity (saturation)
        features['color_intensity'] = np.mean(s_channel)
        
        # Red-green ratio
        r_channel = rgb[:, :, 0]
        g_channel = rgb[:, :, 1]
        features['red_green_ratio'] = np.mean(r_channel) / (np.mean(g_channel) + 1e-8)
        
        return features
    
    def extract_all_features(self, image):
        """Extract all feature types and combine them"""
        features = {}
        
        # CIELAB features (primary for SVM)
        features.update(self.extract_cielab_features(image))
        
        # Color histogram features
        features.update(self.extract_color_histograms(image, bins=16))  # Reduced bins for efficiency
        
        # Texture features
        features.update(self.extract_texture_features(image))
        
        # Shape features
        features.update(self.extract_shape_features(image))
        
        # Spoilage-specific features
        features.update(self.extract_spoilage_features(image))
        
        # Ripeness-specific features
        features.update(self.extract_ripeness_features(image))
        
        return features
    
    def get_cielab_analysis(self, image):
        """Get detailed CIELAB analysis for display"""
        lab_image = self.rgb_to_cielab(image)
        
        # Calculate mean values for each channel
        L_mean = np.mean(lab_image[:, :, 0])  # Lightness
        a_mean = np.mean(lab_image[:, :, 1])  # Green-Red
        b_mean = np.mean(lab_image[:, :, 2])  # Blue-Yellow
        
        return {
            'L_lightness': round(L_mean, 1),
            'a_green_red': round(a_mean, 1),
            'b_blue_yellow': round(b_mean, 1)
        }
    
    def get_transport_survivability(self, ripeness_class, spoilage_score):
        """Calculate transport survivability based on ripeness and spoilage"""
        survivability = {
            'short_distance': {'safe': True, 'message': 'SAFE - Good for transport'},
            'medium_distance': {'safe': True, 'message': 'SAFE - Good for transport'}, 
            'long_distance': {'safe': True, 'message': 'SAFE - Good for transport'}
        }
        
        # Adjust based on spoilage
        if spoilage_score > 15:  # High spoilage
            survivability = {
                'short_distance': {'safe': False, 'message': 'UNSAFE - Do not transport or consume'},
                'medium_distance': {'safe': False, 'message': 'UNSAFE - Do not transport or consume'},
                'long_distance': {'safe': False, 'message': 'UNSAFE - Do not transport or consume'}
            }
        elif spoilage_score > 8:  # Moderate spoilage
            survivability = {
                'short_distance': {'safe': False, 'message': 'UNSAFE - Do not transport or consume'},
                'medium_distance': {'safe': False, 'message': 'UNSAFE - Do not transport or consume'},
                'long_distance': {'safe': False, 'message': 'UNSAFE - Do not transport or consume'}
            }
        else:
            # Adjust based on ripeness
            if ripeness_class == 'unripe':
                survivability = {
                    'short_distance': {'safe': True, 'message': 'EXCELLENT - Perfect for transport'},
                    'medium_distance': {'safe': True, 'message': 'EXCELLENT - Perfect for transport'},
                    'long_distance': {'safe': True, 'message': 'EXCELLENT - Perfect for transport'}
                }
            elif ripeness_class == 'turning':
                survivability = {
                    'short_distance': {'safe': True, 'message': 'GOOD - Suitable for transport'},
                    'medium_distance': {'safe': True, 'message': 'GOOD - Suitable for transport'},
                    'long_distance': {'safe': True, 'message': 'CAUTION - Monitor closely'}
                }
            elif ripeness_class == 'ripe':
                survivability = {
                    'short_distance': {'safe': True, 'message': 'GOOD - Use soon after transport'},
                    'medium_distance': {'safe': True, 'message': 'CAUTION - Refrigerate and use quickly'},
                    'long_distance': {'safe': False, 'message': 'RISKY - May spoil during long transport'}
                }
            elif ripeness_class == 'rotten':
                survivability = {
                    'short_distance': {'safe': False, 'message': 'UNSAFE - Do not transport or consume'},
                    'medium_distance': {'safe': False, 'message': 'UNSAFE - Do not transport or consume'},
                    'long_distance': {'safe': False, 'message': 'UNSAFE - Do not transport or consume'}
                }
        
        return survivability
    
    def process_image_for_analysis(self, image_path):
        """Process single image and return all analysis results"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        # Resize for consistent processing
        image = cv2.resize(image, (224, 224))
        
        # Extract features
        features = self.extract_all_features(image)
        
        # Get CIELAB analysis
        cielab_analysis = self.get_cielab_analysis(image)
        
        # Calculate spoilage score
        spoilage_features = self.extract_spoilage_features(image)
        spoilage_score = (spoilage_features['dark_pixel_ratio'] * 30 + 
                         spoilage_features['brown_pixel_ratio'] * 40 + 
                         spoilage_features['low_saturation_ratio'] * 20) * 100
        
        return {
            'features': features,
            'cielab_analysis': cielab_analysis,
            'spoilage_score': spoilage_score,
            'spoilage_features': spoilage_features
        }

def extract_features_from_dataset(dataset_dir, output_file="features.csv"):
    """Extract features from entire dataset and save to CSV"""
    extractor = TomatoFeatureExtractor()
    dataset_dir = Path(dataset_dir)
    
    all_features = []
    all_labels = []
    
    # Process each class directory
    for class_dir in dataset_dir.iterdir():
        if not class_dir.is_dir():
            continue
            
        class_name = class_dir.name
        print(f"Processing {class_name} images...")
        
        image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        
        for image_file in image_files:
            try:
                image = cv2.imread(str(image_file))
                if image is None:
                    continue
                    
                # Resize for consistency
                image = cv2.resize(image, (224, 224))
                
                # Extract features
                features = extractor.extract_all_features(image)
                
                all_features.append(features)
                all_labels.append(class_name)
                
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
                continue
    
    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)
    features_df['label'] = all_labels
    
    # Save to CSV
    features_df.to_csv(output_file, index=False)
    print(f"Features saved to {output_file}")
    print(f"Dataset shape: {features_df.shape}")
    print(f"Classes: {features_df['label'].value_counts()}")
    
    return features_df

if __name__ == "__main__":
    # Example usage
    extractor = TomatoFeatureExtractor()
    
    # Process dataset if available
    dataset_dir = "./data/crops"
    if Path(dataset_dir).exists():
        features_df = extract_features_from_dataset(dataset_dir)
        print("Feature extraction complete!")
    else:
        print("Dataset directory not found. Please run dataset_loader.py first.")