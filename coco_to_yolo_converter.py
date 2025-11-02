"""
COCO to YOLO Annotation Converter
Converts COCO JSON format annotations to YOLO darknet format
"""

import json
import os
from pathlib import Path
from tqdm import tqdm

class COCOtoYOLO:
    def __init__(self, coco_json_path, output_dir, img_dir=None):
        """
        Initialize the converter
        
        Args:
            coco_json_path: Path to COCO JSON annotation file
            output_dir: Directory to save YOLO format labels
            img_dir: Optional image directory path
        """
        self.coco_json_path = coco_json_path
        self.output_dir = Path(output_dir)
        self.img_dir = Path(img_dir) if img_dir else None
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load COCO data
        with open(coco_json_path, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create category mapping
        self.categories = {cat['id']: idx for idx, cat in enumerate(self.coco_data['categories'])}
        self.category_names = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        
        # Create image id to filename mapping
        self.images = {img['id']: img for img in self.coco_data['images']}
        
        print(f"Loaded {len(self.coco_data['images'])} images")
        print(f"Found {len(self.coco_data['categories'])} categories")
        print(f"Processing {len(self.coco_data['annotations'])} annotations")
    
    def convert_bbox_coco2yolo(self, img_width, img_height, bbox):
        """
        Convert COCO bbox format to YOLO format
        
        COCO format: [x_min, y_min, width, height]
        YOLO format: [x_center, y_center, width, height] (normalized 0-1)
        
        Args:
            img_width: Image width
            img_height: Image height
            bbox: COCO format bounding box [x, y, w, h]
        
        Returns:
            YOLO format bbox [x_center, y_center, width, height]
        """
        x_min, y_min, w, h = bbox
        
        # Calculate center point
        x_center = (x_min + w / 2) / img_width
        y_center = (y_min + h / 2) / img_height
        
        # Normalize width and height
        width = w / img_width
        height = h / img_height
        
        return [x_center, y_center, width, height]
    
    def convert(self):
        """
        Convert COCO annotations to YOLO format
        """
        # Group annotations by image
        img_annotations = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in img_annotations:
                img_annotations[img_id] = []
            img_annotations[img_id].append(ann)
        
        # Convert each image's annotations
        converted_count = 0
        for img_id, annotations in tqdm(img_annotations.items(), desc="Converting"):
            if img_id not in self.images:
                continue
            
            img_info = self.images[img_id]
            img_width = img_info['width']
            img_height = img_info['height']
            img_filename = img_info['file_name']
            
            # Create output file path (same name as image but .txt)
            output_filename = Path(img_filename).stem + '.txt'
            output_path = self.output_dir / output_filename
            
            # Convert annotations for this image
            yolo_annotations = []
            for ann in annotations:
                category_id = ann['category_id']
                bbox = ann['bbox']
                
                # Skip if category not found
                if category_id not in self.categories:
                    continue
                
                # Convert to YOLO format
                yolo_bbox = self.convert_bbox_coco2yolo(img_width, img_height, bbox)
                class_id = self.categories[category_id]
                
                # Format: <class_id> <x_center> <y_center> <width> <height>
                yolo_line = f"{class_id} {' '.join(map(str, yolo_bbox))}"
                yolo_annotations.append(yolo_line)
            
            # Write to file
            with open(output_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))
            
            converted_count += 1
        
        print(f"\nConversion complete!")
        print(f"Converted {converted_count} images")
        print(f"Labels saved to: {self.output_dir}")
        
        # Save class names file
        self.save_classes_file()
    
    def save_classes_file(self):
        """
        Save class names to a .names file for YOLO
        """
        classes_file = self.output_dir / 'classes.names'
        
        # Sort by class index
        sorted_classes = sorted(self.categories.items(), key=lambda x: x[1])
        class_names = [self.category_names[cat_id] for cat_id, _ in sorted_classes]
        
        with open(classes_file, 'w') as f:
            f.write('\n'.join(class_names))
        
        print(f"Class names saved to: {classes_file}")
        print(f"Number of classes: {len(class_names)}")


def main():
    """
    Example usage
    """
    # Example paths - modify these for your dataset
    coco_json_path = "path/to/your/annotations.json"  # Your COCO JSON file
    output_dir = "yolo_labels"  # Where to save YOLO format labels
    img_dir = "path/to/images"  # Optional: your images directory
    
    print("=" * 60)
    print("COCO to YOLO Annotation Converter")
    print("=" * 60)
    
    # Convert
    converter = COCOtoYOLO(coco_json_path, output_dir, img_dir)
    converter.convert()
    
    print("\n" + "=" * 60)
    print("Conversion Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Your YOLO labels are in:", output_dir)
    print("2. Use these labels to train your YOLO model")
    print("3. Deploy to ESP32-CAM for object detection")


if __name__ == "__main__":
    # If you want to run directly with arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert COCO to YOLO format')
    parser.add_argument('--json', required=True, help='Path to COCO JSON file')
    parser.add_argument('--output', default='yolo_labels', help='Output directory for YOLO labels')
    parser.add_argument('--images', help='Optional: Images directory')
    
    args = parser.parse_args()
    
    converter = COCOtoYOLO(args.json, args.output, args.images)
    converter.convert()
