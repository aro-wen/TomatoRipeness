"""
ESP32-CAM with YOLO Object Detection (YOLOv4-tiny)
Captures video stream from ESP32-CAM and performs real-time object detection
"""

import cv2
import numpy as np
import requests
from io import BytesIO
import time
import os

class ESP32CAM_YOLO:
    def __init__(self, esp32_ip, yolo_cfg="yolov4-tiny.cfg", yolo_weights="yolov4-tiny.weights", 
                 coco_names="coco.names", confidence_threshold=0.5, nms_threshold=0.4):
        """
        Initialize ESP32-CAM YOLO detector
        
        Args:
            esp32_ip: IP address of ESP32-CAM (e.g., "192.168.1.100")
            yolo_cfg: Path to YOLO config file
            yolo_weights: Path to YOLO weights file
            coco_names: Path to COCO class names file
            confidence_threshold: Minimum confidence for detection
            nms_threshold: Non-maximum suppression threshold
        """
        self.esp32_ip = esp32_ip
        # Try to detect the correct stream URL
        self.stream_url = self._find_stream_url()
        self.capture_url = f"http://{esp32_ip}/capture"
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        print("Loading YOLO model...")
        # Load YOLO
        self.net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Get output layer names
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # Load class names
        with open(coco_names, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Generate random colors for each class
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.classes), 3), dtype='uint8')
        
        print(f"✓ YOLO model loaded successfully!")
        print(f"✓ Loaded {len(self.classes)} object classes")
        print(f"✓ ESP32-CAM URL: {self.stream_url}")
    
    def detect_objects(self, image):
        """
        Perform object detection on an image
        
        Args:
            image: Input image (numpy array)
        
        Returns:
            image with bounding boxes, list of detections
        """
        height, width = image.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        
        # Set input to network
        self.net.setInput(blob)
        
        # Forward pass
        start = time.time()
        outputs = self.net.forward(self.output_layers)
        end = time.time()
        inference_time = end - start
        
        # Process detections
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.confidence_threshold:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                confidence = confidences[i]
                class_id = class_ids[i]
                label = self.classes[class_id]
                color = self.colors[class_id].tolist()
                
                # Draw bounding box
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                
                # Draw label
                text = f"{label}: {confidence:.2f}"
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(image, (x, y - text_height - 10), (x + text_width, y), color, -1)
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                detections.append({
                    'label': label,
                    'confidence': confidence,
                    'box': [x, y, w, h]
                })
        
        # Display FPS and inference time
        fps_text = f"Inference: {inference_time*1000:.0f}ms | FPS: {1/inference_time:.1f}"
        cv2.putText(image, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return image, detections
    
    def capture_single_image(self):
        """
        Capture a single image from ESP32-CAM and detect objects
        """
        try:
            print(f"\nConnecting to {self.capture_url}...")
            response = requests.get(self.capture_url, timeout=5)
            if response.status_code == 200:
                # Convert to image
                image_array = np.frombuffer(response.content, dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                
                if image is not None:
                    print("✓ Image captured, detecting objects...")
                    # Detect objects
                    result_image, detections = self.detect_objects(image)
                    
                    # Display results
                    cv2.imshow('ESP32-CAM YOLO Detection', result_image)
                    print("Press any key to close window...")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    
                    # Print detections
                    if detections:
                        print(f"\n✓ Detected {len(detections)} object(s):")
                        for det in detections:
                            print(f"  - {det['label']}: {det['confidence']:.2%}")
                    else:
                        print("\nNo objects detected. Try adjusting confidence threshold or camera position.")
                    
                    return result_image, detections
                else:
                    print("✗ Failed to decode image")
            else:
                print(f"✗ Failed to capture image. Status code: {response.status_code}")
                print("  Make sure ESP32-CAM is powered on and accessible.")
        except requests.exceptions.ConnectionError:
            print(f"✗ Cannot connect to ESP32-CAM at {self.esp32_ip}")
            print("  Check:")
            print("  1. ESP32-CAM is powered on")
            print("  2. ESP32-CAM is connected to WiFi")
            print("  3. IP address is correct")
            print("  4. Both devices are on same network")
        except Exception as e:
            print(f"✗ Error capturing image: {e}")
        
        return None, []
    
    def stream_detection(self, show_fps=True):
        """
        Stream video from ESP32-CAM and perform real-time object detection
        
        Args:
            show_fps: Display FPS counter
        """
        print("\n" + "=" * 60)
        print("Starting live stream detection...")
        print("=" * 60)
        print("Controls:")
        print("  - Press 'q' to quit")
        print("  - Press 's' to save screenshot")
        print("=" * 60 + "\n")
        
        try:
            print(f"Connecting to stream: {self.stream_url}")
            # Open stream with longer timeout
            stream = requests.get(self.stream_url, stream=True, timeout=10)
            
            if stream.status_code != 200:
                print(f"✗ Failed to connect. Status code: {stream.status_code}")
                return
                
            print("✓ Connected! Processing frames...")
            bytes_data = bytes()
            
            frame_count = 0
            start_time = time.time()
            frames_received = 0
            
            for chunk in stream.iter_content(chunk_size=4096):  # Increased chunk size
                bytes_data += chunk
                
                # Find JPEG boundaries
                a = bytes_data.find(b'\xff\xd8')  # JPEG start
                b = bytes_data.find(b'\xff\xd9')  # JPEG end
                
                if a != -1 and b != -1:
                    jpg = bytes_data[a:b+2]
                    bytes_data = bytes_data[b+2:]
                    
                    frames_received += 1
                    if frames_received == 1:
                        print(f"✓ First frame received! Opening window...")
                    
                    # Decode image
                    image = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    
                    if image is not None:
                        # Detect objects
                        result_image, detections = self.detect_objects(image)
                        
                        # Calculate average FPS
                        frame_count += 1
                        if frame_count % 30 == 0:
                            elapsed = time.time() - start_time
                            avg_fps = frame_count / elapsed
                            print(f"📊 Stats: Frame #{frame_count} | Avg FPS: {avg_fps:.2f} | Objects detected: {len(detections)}")
                        
                        # Display
                        cv2.imshow('ESP32-CAM YOLO Detection', result_image)
                        
                        # Handle key presses
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            print("\n✓ Quitting...")
                            break
                        elif key == ord('s'):
                            filename = f"detection_{int(time.time())}.jpg"
                            cv2.imwrite(filename, result_image)
                            print(f"✓ Screenshot saved: {filename}")
                    else:
                        print("⚠ Failed to decode frame")
                            
        except KeyboardInterrupt:
            print("\n✓ Stream interrupted by user")
        except requests.exceptions.Timeout:
            print(f"\n✗ Connection timeout. Make sure stream is accessible at {self.stream_url}")
        except requests.exceptions.ConnectionError:
            print(f"\n✗ Cannot connect to ESP32-CAM stream at {self.stream_url}")
            print("  Make sure ESP32-CAM is accessible via web browser first")
        except Exception as e:
            print(f"\n✗ Error during streaming: {e}")
        finally:
            cv2.destroyAllWindows()


def download_yolo_files():
    """
    Helper function to download YOLO files if not present
    """
    files = {
        'yolov4-tiny.weights': 'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights',
        'yolov4-tiny.cfg': 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg',
        'coco.names': 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'
    }
    
    print("Checking YOLO files...")
    for filename, url in files.items():
        if not os.path.exists(filename):
            print(f"Downloading {filename}... (this may take a moment)")
            try:
                response = requests.get(url, stream=True)
                total_size = int(response.headers.get('content-length', 0))
                with open(filename, 'wb') as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"  Progress: {percent:.1f}%", end='\r')
                print(f"\n✓ {filename} downloaded successfully")
            except Exception as e:
                print(f"\n✗ Failed to download {filename}: {e}")
                print("  Please download manually from:", url)
                return False
        else:
            print(f"✓ {filename} already exists")
    return True


def main():
    """
    Main function - Example usage
    """
    print("\n" + "=" * 60)
    print("       ESP32-CAM YOLO Object Detection (YOLOv4-tiny)")
    print("=" * 60 + "\n")
    
    # Download YOLO files if needed
    if not download_yolo_files():
        print("\n✗ Failed to download required files. Exiting.")
        return
    
    # Configuration
    ESP32_IP = "192.168.1.31"  # CHANGE THIS to your ESP32-CAM IP address
    
    print(f"\n📷 ESP32-CAM Configuration")
    print(f"   IP Address: {ESP32_IP}")
    print(f"   Make sure your ESP32-CAM is powered on and connected to WiFi!\n")
    
    # Initialize detector
    try:
        detector = ESP32CAM_YOLO(
            esp32_ip=ESP32_IP,
            yolo_cfg="yolov4-tiny.cfg",
            yolo_weights="yolov4-tiny.weights",
            coco_names="coco.names",
            confidence_threshold=0.5,
            nms_threshold=0.4
        )
    except Exception as e:
        print(f"\n✗ Error loading YOLO model: {e}")
        print("\nTry deleting the weights file and running again:")
        print("  Remove-Item yolov4-tiny.weights")
        return
    
    # Choose mode
    print("\n" + "=" * 60)
    print("Select detection mode:")
    print("  1. Single image capture")
    print("  2. Continuous stream detection (recommended)")
    print("=" * 60)
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        detector.capture_single_image()
    elif choice == "2":
        detector.stream_detection()
    else:
        print("✗ Invalid choice!")


if __name__ == "__main__":
    main()