import os
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import zipfile
import kaggle

class LaboroDatasetLoader:
    def __init__(self, data_dir="./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Mapping from original Laboro classes to our simplified classes
        self.class_mapping = {
            'b_green': 'unripe',           # Big green -> unripe
            'l_green': 'unripe',           # Little green -> unripe  
            'b_half_ripened': 'turning',   # Big half-ripe -> turning
            'l_half_ripened': 'turning',   # Little half-ripe -> turning
            'b_fully_ripened': 'ripe',     # Big fully ripe -> ripe
            'l_fully_ripened': 'ripe'      # Little fully ripe -> ripe
        }
        
        # Add rotten class for enhanced spoilage detection
        self.target_classes = ['unripe', 'turning', 'ripe', 'rotten']
        
    def download_dataset(self):
        """Download Laboro Tomato dataset from Kaggle"""
        try:
            print("Downloading Laboro Tomato dataset from Kaggle...")
            kaggle.api.dataset_download_files(
                'nexuswho/laboro-tomato', 
                path=str(self.data_dir),
                unzip=True
            )
            print("Dataset downloaded successfully!")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Please ensure you have Kaggle API key configured")
            print("Visit: https://www.kaggle.com/docs/api")
            
    def load_annotations(self, annotation_file):
        """Load COCO-style annotations from Laboro dataset"""
        if not os.path.exists(annotation_file):
            print(f"Annotation file not found: {annotation_file}")
            return None
            
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
            
        return annotations
    
    def extract_image_crops(self, images_dir, annotations, output_dir):
        """Extract tomato crops from images using bounding boxes"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Create class directories
        for class_name in self.target_classes:
            (output_dir / class_name).mkdir(exist_ok=True)
            
        image_data = []
        
        # Create mapping from image_id to filename
        image_id_to_filename = {img['id']: img['file_name'] for img in annotations['images']}
        
        # Group annotations by image_id
        annotations_by_image = {}
        for ann in annotations['annotations']:
            image_id = ann['image_id']
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(ann)
        
        # Category mapping
        category_id_to_name = {cat['id']: cat['name'] for cat in annotations['categories']}
        
        crop_count = 0
        
        for image_id, anns in tqdm(annotations_by_image.items(), desc="Extracting crops"):
            filename = image_id_to_filename[image_id]
            image_path = Path(images_dir) / filename
            
            if not image_path.exists():
                continue
                
            image = cv2.imread(str(image_path))
            if image is None:
                continue
                
            for ann in anns:
                # Extract bounding box
                bbox = ann['bbox']  # [x, y, width, height]
                x, y, w, h = [int(v) for v in bbox]
                
                # Ensure bbox is within image bounds
                x = max(0, x)
                y = max(0, y)
                w = min(w, image.shape[1] - x)
                h = min(h, image.shape[0] - y)
                
                if w <= 0 or h <= 0:
                    continue
                    
                # Extract crop
                crop = image[y:y+h, x:x+w]
                
                # Resize crop to standard size
                crop_resized = cv2.resize(crop, (224, 224))
                
                # Get class name and map to our simplified classes
                category_id = ann['category_id']
                original_class = category_id_to_name[category_id]
                
                if original_class in self.class_mapping:
                    target_class = self.class_mapping[original_class]
                    
                    # Save crop
                    crop_filename = f"crop_{crop_count:05d}.jpg"
                    crop_path = output_dir / target_class / crop_filename
                    cv2.imwrite(str(crop_path), crop_resized)
                    
                    image_data.append({
                        'filename': crop_filename,
                        'class': target_class,
                        'original_class': original_class,
                        'bbox': bbox,
                        'original_image': filename
                    })
                    
                    crop_count += 1
        
        # Save metadata
        metadata_df = pd.DataFrame(image_data)
        metadata_df.to_csv(output_dir / "metadata.csv", index=False)
        
        print(f"Extracted {crop_count} tomato crops")
        print(f"Class distribution:")
        for class_name in self.target_classes:
            class_count = len([x for x in image_data if x['class'] == class_name])
            print(f"  {class_name}: {class_count} samples")
            
        return image_data
    
    def create_synthetic_rotten_samples(self, clean_dir, rotten_dir, num_samples=200):
        """Create synthetic rotten samples by applying image augmentation"""
        rotten_dir = Path(rotten_dir)
        rotten_dir.mkdir(exist_ok=True)
        
        # Collect ripe samples to augment
        ripe_dir = Path(clean_dir) / "ripe"
        if not ripe_dir.exists():
            print("No ripe samples found for generating rotten samples")
            return
            
        ripe_images = list(ripe_dir.glob("*.jpg"))
        
        if len(ripe_images) == 0:
            print("No ripe images found")
            return
            
        print(f"Creating {num_samples} synthetic rotten samples...")
        
        for i in tqdm(range(num_samples), desc="Generating rotten samples"):
            # Select random ripe image
            source_img_path = np.random.choice(ripe_images)
            img = cv2.imread(str(source_img_path))
            
            if img is None:
                continue
                
            # Apply rotten transformations
            rotten_img = self._apply_spoilage_effects(img)
            
            # Save rotten sample
            rotten_filename = f"rotten_{i:05d}.jpg"
            rotten_path = rotten_dir / rotten_filename
            cv2.imwrite(str(rotten_path), rotten_img)
    
    def _apply_spoilage_effects(self, image):
        """Apply various spoilage effects to simulate rotten tomatoes"""
        result = image.copy()
        
        # Random combination of effects
        effects = np.random.choice([1, 2, 3, 4], size=np.random.randint(2, 4), replace=False)
        
        for effect in effects:
            if effect == 1:
                # Add dark spots (mold)
                result = self._add_dark_spots(result)
            elif effect == 2:
                # Darken overall image
                result = self._darken_image(result)
            elif effect == 3:
                # Add brown decay patches
                result = self._add_brown_patches(result)
            elif effect == 4:
                # Reduce saturation (decay effect)
                result = self._reduce_saturation(result)
                
        return result
    
    def _add_dark_spots(self, image, num_spots=None):
        """Add dark spots to simulate mold"""
        if num_spots is None:
            num_spots = np.random.randint(3, 8)
            
        result = image.copy()
        h, w = result.shape[:2]
        
        for _ in range(num_spots):
            # Random spot parameters
            center_x = np.random.randint(0, w)
            center_y = np.random.randint(0, h)
            radius = np.random.randint(5, 20)
            
            # Create circular mask
            y, x = np.ogrid[:h, :w]
            mask = ((x - center_x) ** 2 + (y - center_y) ** 2) <= radius ** 2
            
            # Apply dark color
            dark_color = np.random.randint(0, 50, 3)
            result[mask] = dark_color
            
        return result
    
    def _darken_image(self, image, factor=None):
        """Darken the image to simulate decay"""
        if factor is None:
            factor = np.random.uniform(0.3, 0.7)
            
        return (image * factor).astype(np.uint8)
    
    def _add_brown_patches(self, image, num_patches=None):
        """Add brown decay patches"""
        if num_patches is None:
            num_patches = np.random.randint(2, 5)
            
        result = image.copy()
        h, w = result.shape[:2]
        
        for _ in range(num_patches):
            # Random patch parameters
            x = np.random.randint(0, w//2)
            y = np.random.randint(0, h//2)
            patch_w = np.random.randint(10, w//3)
            patch_h = np.random.randint(10, h//3)
            
            # Ensure patch is within bounds
            x2 = min(x + patch_w, w)
            y2 = min(y + patch_h, h)
            
            # Apply brown color
            brown_color = [40, 90, 120]  # BGR brown
            alpha = np.random.uniform(0.4, 0.8)
            
            result[y:y2, x:x2] = (result[y:y2, x:x2] * (1 - alpha) + 
                                 np.array(brown_color) * alpha).astype(np.uint8)
            
        return result
    
    def _reduce_saturation(self, image, factor=None):
        """Reduce color saturation to simulate decay"""
        if factor is None:
            factor = np.random.uniform(0.3, 0.6)
            
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = (hsv[:, :, 1] * factor).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def prepare_dataset(self, force_download=False):
        """Complete dataset preparation pipeline"""
        dataset_ready = (self.data_dir / "crops").exists()
        
        if force_download or not dataset_ready:
            print("Preparing Laboro Tomato dataset...")
            
            # Download if needed
            if force_download or not any(self.data_dir.glob("*.json")):
                self.download_dataset()
            
            # Find annotation files
            annotation_files = list(self.data_dir.glob("**/*.json"))
            if not annotation_files:
                print("No annotation files found. Please check dataset structure.")
                return None
                
            # Use the first annotation file found
            annotation_file = annotation_files[0]
            print(f"Using annotations: {annotation_file}")
            
            # Load annotations
            annotations = self.load_annotations(annotation_file)
            if annotations is None:
                return None
                
            # Find images directory
            images_dir = self.data_dir / "images"
            if not images_dir.exists():
                # Try alternative paths
                for subdir in self.data_dir.rglob("*"):
                    if subdir.is_dir() and any(subdir.glob("*.jpg")):
                        images_dir = subdir
                        break
                        
            if not images_dir.exists():
                print("Images directory not found")
                return None
                
            # Extract crops
            crops_dir = self.data_dir / "crops"
            image_data = self.extract_image_crops(images_dir, annotations, crops_dir)
            
            # Create synthetic rotten samples
            self.create_synthetic_rotten_samples(crops_dir, crops_dir / "rotten")
            
            print("Dataset preparation complete!")
            
        return str(self.data_dir / "crops")

if __name__ == "__main__":
    loader = LaboroDatasetLoader()
    crops_dir = loader.prepare_dataset(force_download=False)
    print(f"Dataset ready at: {crops_dir}")