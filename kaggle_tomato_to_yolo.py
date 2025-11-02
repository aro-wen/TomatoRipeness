"""
Kaggle Tomato Dataset to YOLO Format Converter
Uses the TomatoD dataset and COCO annotations from Kaggle
"""

import json
import os
from pathlib import Path
import shutil
from tqdm import tqdm

class KaggleTomatoDatasetConverter:
    def __init__(self, dataset_path, output_path="tomato_yolo_dataset"):
        """
        Convert Kaggle TomatoD dataset to YOLO format
        
        Args:
            dataset_path: Path to downloaded Kaggle dataset
            output_path: Where to save YOLO format dataset
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        
        # Create output directories
        self.output_path.mkdir(parents=True, exist_ok=True)
        (self.output_path / "images" / "train").mkdir(parents=True, exist_ok=True)
        (self.output_path / "images" / "val").mkdir(parents=True, exist_ok=True)
        (self.output_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (self.output_path / "labels" / "val").mkdir(parents=True, exist_ok=True)
        
        print(f"Dataset path: {self.dataset_path}")
        print(f"Output path: {self.output_path}")
    
    def convert_bbox_coco2yolo(self, img_width, img_height, bbox):
        """
        Convert COCO bbox format to YOLO format
        
        COCO: [x_min, y_min, width, height]
        YOLO: [x_center, y_center, width, height] (normalized 0-1)
        """
        x_min, y_min, w, h = bbox
        
        x_center = (x_min + w / 2) / img_width
        y_center = (y_min + h / 2) / img_height
        width = w / img_width
        height = h / img_height
        
        return [x_center, y_center, width, height]
    
    def convert_annotations(self, coco_json_path, split="train"):
        """
        Convert COCO JSON annotations to YOLO format
        
        Args:
            coco_json_path: Path to COCO JSON file
            split: 'train' or 'val'
        """
        print(f"\nProcessing {split} split...")
        
        # Load COCO data
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)
        
        # Create category mapping
        categories = {cat['id']: idx for idx, cat in enumerate(coco_data['categories'])}
        category_names = {cat['id']: cat['name'] for cat in coco_data['categories']}
        
        print(f"Found {len(coco_data['categories'])} categories:")
        for cat_id, cat_name in category_names.items():
            print(f"  - Class {categories[cat_id]}: {cat_name}")
        
        # Create image mapping
        images = {img['id']: img for img in coco_data['images']}
        
        # Group annotations by image
        img_annotations = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in img_annotations:
                img_annotations[img_id] = []
            img_annotations[img_id].append(ann)
        
        # Convert each image's annotations
        converted_count = 0
        skipped_count = 0
        
        for img_id, annotations in tqdm(img_annotations.items(), desc=f"Converting {split}"):
            if img_id not in images:
                skipped_count += 1
                continue
            
            img_info = images[img_id]
            img_width = img_info['width']
            img_height = img_info['height']
            img_filename = img_info['file_name']
            
            # Source image path
            src_img_path = self.dataset_path / img_filename
            if not src_img_path.exists():
                # Try different possible paths
                src_img_path = self.dataset_path / "images" / img_filename
                if not src_img_path.exists():
                    src_img_path = self.dataset_path / split / img_filename
                    if not src_img_path.exists():
                        skipped_count += 1
                        continue
            
            # Destination paths
            dst_img_path = self.output_path / "images" / split / img_filename
            label_filename = Path(img_filename).stem + '.txt'
            dst_label_path = self.output_path / "labels" / split / label_filename
            
            # Copy image
            shutil.copy(src_img_path, dst_img_path)
            
            # Convert annotations
            yolo_annotations = []
            for ann in annotations:
                category_id = ann['category_id']
                bbox = ann['bbox']
                
                if category_id not in categories:
                    continue
                
                # Convert to YOLO format
                yolo_bbox = self.convert_bbox_coco2yolo(img_width, img_height, bbox)
                class_id = categories[category_id]
                
                # Format: <class_id> <x_center> <y_center> <width> <height>
                yolo_line = f"{class_id} {' '.join(map(str, yolo_bbox))}"
                yolo_annotations.append(yolo_line)
            
            # Write label file
            with open(dst_label_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))
            
            converted_count += 1
        
        print(f"✓ Converted {converted_count} images")
        print(f"⚠ Skipped {skipped_count} images")
        
        return category_names, categories
    
    def create_yaml_config(self, category_names):
        """
        Create YOLO dataset configuration file
        """
        yaml_content = f"""# Tomato Detection Dataset Configuration
# Generated for YOLO training

path: {self.output_path.absolute()}
train: images/train
val: images/val

# Classes
nc: {len(category_names)}  # number of classes
names: {list(category_names.values())}  # class names
"""
        
        yaml_path = self.output_path / "tomato_data.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"\n✓ Created dataset config: {yaml_path}")
        return yaml_path
    
    def create_classes_file(self, category_names):
        """
        Create classes.names file
        """
        classes_path = self.output_path / "classes.names"
        sorted_names = [category_names[cat_id] for cat_id in sorted(category_names.keys())]
        
        with open(classes_path, 'w') as f:
            f.write('\n'.join(sorted_names))
        
        print(f"✓ Created classes file: {classes_path}")


def main():
    """
    Main conversion function
    """
    print("=" * 70)
    print("Kaggle TomatoD Dataset → YOLO Format Converter")
    print("=" * 70)
    
    # Configuration
    # After downloading from Kaggle, update these paths:
    DATASET_PATH = "path/to/tomatod"  # Update this
    COCO_TRAIN_JSON = "path/to/train_annotations.json"  # Update this
    COCO_VAL_JSON = "path/to/val_annotations.json"  # Update this (if available)
    
    print(f"\nDataset location: {DATASET_PATH}")
    print(f"Training annotations: {COCO_TRAIN_JSON}")
    
    # Create converter
    converter = KaggleTomatoDatasetConverter(
        dataset_path=DATASET_PATH,
        output_path="tomato_yolo_dataset"
    )
    
    # Convert training data
    print("\n" + "=" * 70)
    print("CONVERTING TRAINING DATA")
    print("=" * 70)
    category_names, categories = converter.convert_annotations(
        COCO_TRAIN_JSON, 
        split="train"
    )
    
    # Convert validation data (if available)
    if os.path.exists(COCO_VAL_JSON):
        print("\n" + "=" * 70)
        print("CONVERTING VALIDATION DATA")
        print("=" * 70)
        converter.convert_annotations(COCO_VAL_JSON, split="val")
    else:
        print("\n⚠ No validation set found. You can use training data for both.")
    
    # Create config files
    print("\n" + "=" * 70)
    print("CREATING CONFIGURATION FILES")
    print("=" * 70)
    yaml_path = converter.create_yaml_config(category_names)
    converter.create_classes_file(category_names)
    
    # Summary
    print("\n" + "=" * 70)
    print("CONVERSION COMPLETE!")
    print("=" * 70)
    print(f"\n✓ YOLO dataset created at: {converter.output_path}")
    print(f"✓ Dataset config: {yaml_path}")
    print(f"\nDataset structure:")
    print(f"  tomato_yolo_dataset/")
    print(f"  ├── images/")
    print(f"  │   ├── train/")
    print(f"  │   └── val/")
    print(f"  ├── labels/")
    print(f"  │   ├── train/")
    print(f"  │   └── val/")
    print(f"  ├── tomato_data.yaml")
    print(f"  └── classes.names")
    
    print(f"\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("1. Train YOLOv4 or YOLOv5 with this dataset")
    print("2. Use the trained weights with your ESP32-CAM")
    print("3. Detect tomato ripeness in real-time!")
    print("\nTraining command example (YOLOv5):")
    print(f"  python train.py --data {yaml_path} --weights yolov5s.pt --epochs 100")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert Kaggle TomatoD to YOLO format')
    parser.add_argument('--dataset', required=True, help='Path to TomatoD dataset')
    parser.add_argument('--train-json', required=True, help='Path to training COCO JSON')
    parser.add_argument('--val-json', help='Path to validation COCO JSON (optional)')
    parser.add_argument('--output', default='tomato_yolo_dataset', help='Output directory')
    
    args = parser.parse_args()
    
    converter = KaggleTomatoDatasetConverter(args.dataset, args.output)
    
    # Convert training
    category_names, _ = converter.convert_annotations(args.train_json, "train")
    
    # Convert validation if provided
    if args.val_json:
        converter.convert_annotations(args.val_json, "val")
    
    # Create configs
    converter.create_yaml_config(category_names)
    converter.create_classes_file(category_names)
    
    print("\n✓ Conversion complete!")
