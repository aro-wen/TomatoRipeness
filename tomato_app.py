from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO, emit
import cv2
import torch
import requests
import numpy as np
from io import BytesIO
from PIL import Image
import time
import torchvision.transforms as transforms
from datetime import datetime
import threading

app = Flask(__name__)
app.config['SECRET_KEY'] = 'tomato-classifier-2024'
socketio = SocketIO(app, cors_allowed_origins="*")

# Classification data storage
class ClassificationData:
    def __init__(self):
        self.current_class = 'Waiting...'
        self.confidence = 0.0
        self.counts = {'Ripe': 0, 'Unripe': 0, 'Old': 0, 'Damaged': 0}
        self.total = 0
        self.last_update = 'Never'
        self.is_running = False
        
classification_data = ClassificationData()

class ESP32TomatoClassifier:
    def __init__(self, esp32_ip, model_path):
        print("\n🍅 Loading AI Model...")
        self.esp32_ip = esp32_ip
        
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                         path=model_path, force_reload=False)
            self.model.eval()
            print("✓ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.classes = ['Damaged', 'Old', 'Ripe', 'Unripe']
        self.last_class = None
        
    def capture_image(self):
        """Capture image from ESP32-CAM"""
        try:
            url = f"http://{self.esp32_ip}/capture"
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                return img
            return None
        except Exception as e:
            print(f"⚠️ Capture error: {e}")
            return None
    
    def classify(self, image):
        """Classify tomato state"""
        try:
            img_tensor = self.transform(image).unsqueeze(0)
            
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                class_id = predicted.item()
                conf = confidence.item()
                class_name = self.classes[class_id]
                
                return class_name, conf
        except Exception as e:
            print(f"⚠️ Classification error: {e}")
            return None, 0.0
    
    def get_annotated_frame(self):
        """Get frame with classification overlay"""
        img_pil = self.capture_image()
        if img_pil is None:
            return None, None, 0.0
        
        # Classify
        class_name, confidence = self.classify(img_pil)
        if class_name is None:
            return None, None, 0.0
        
        # Update counts when class changes
        if self.last_class and class_name != self.last_class:
            classification_data.counts[class_name] += 1
            classification_data.total += 1
        self.last_class = class_name
        
        # Convert to OpenCV format
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        height, width = img_cv.shape[:2]
        
        # Color mapping
        color_map = {
            'Ripe': (82, 255, 168),      # Beautiful green
            'Unripe': (255, 215, 0),     # Golden yellow
            'Old': (255, 140, 0),        # Orange
            'Damaged': (220, 20, 60)     # Crimson red
        }
        color = color_map.get(class_name, (255, 255, 255))
        
        # Create semi-transparent overlay
        overlay = img_cv.copy()
        
        # Draw top banner
        banner_height = int(height * 0.12)
        cv2.rectangle(overlay, (0, 0), (width, banner_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img_cv, 0.4, 0, img_cv)
        
        # Draw classification text
        text = f"{class_name.upper()}"
        font = cv2.FONT_HERSHEY_BOLD
        font_scale = min(width, height) / 400
        thickness = max(2, int(font_scale * 2))
        
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = int(banner_height * 0.5)
        
        # Text with outline
        cv2.putText(img_cv, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 2)
        cv2.putText(img_cv, text, (text_x, text_y), font, font_scale, color, thickness)
        
        # Draw confidence
        conf_text = f"{confidence:.1%}"
        conf_size = cv2.getTextSize(conf_text, font, font_scale * 0.7, thickness)[0]
        conf_x = (width - conf_size[0]) // 2
        conf_y = int(banner_height * 0.8)
        
        cv2.putText(img_cv, conf_text, (conf_x, conf_y), font, font_scale * 0.7, (255, 255, 255), thickness)
        
        # Draw colored border
        border_thickness = max(5, int(min(width, height) * 0.01))
        cv2.rectangle(img_cv, (0, 0), (width-1, height-1), color, border_thickness)
        
        return img_cv, class_name, confidence

# Initialize classifier
print("\n" + "="*60)
print("🍅 ESP32-CAM TOMATO CLASSIFIER")
print("="*60)

try:
    classifier = ESP32TomatoClassifier(
        esp32_ip="192.168.1.31",  # ⚠️ UPDATE THIS WITH YOUR ESP32 IP!
        model_path="yolov5/tomato_training/run1/weights/best.pt"
    )
    print("✓ System ready!")
except Exception as e:
    print(f"❌ Initialization failed: {e}")
    print("Please check your model path and ESP32 connection")
    classifier = None

def generate_frames():
    """Video streaming generator"""
    while True:
        if not classification_data.is_running or classifier is None:
            time.sleep(0.5)
            continue
            
        try:
            frame, class_name, confidence = classifier.get_annotated_frame()
            
            if frame is None:
                time.sleep(0.5)
                continue
            
            # Update global data
            classification_data.current_class = class_name
            classification_data.confidence = confidence
            classification_data.last_update = datetime.now().strftime("%I:%M:%S %p")
            
            # Emit to websocket
            socketio.emit('update', {
                'class': class_name,
                'confidence': round(confidence * 100, 1),
                'counts': classification_data.counts,
                'total': classification_data.total,
                'timestamp': classification_data.last_update
            })
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.1)  # ~10 FPS
            
        except Exception as e:
            print(f"⚠️ Stream error: {e}")
            time.sleep(1)

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/start')
def start_classification():
    classification_data.is_running = True
    return jsonify({'success': True, 'message': 'Classification started'})

@app.route('/api/stop')
def stop_classification():
    classification_data.is_running = False
    return jsonify({'success': True, 'message': 'Classification stopped'})

@app.route('/api/reset')
def reset_counts():
    classification_data.counts = {'Ripe': 0, 'Unripe': 0, 'Old': 0, 'Damaged': 0}
    classification_data.total = 0
    return jsonify({'success': True, 'message': 'Counts reset'})

@app.route('/api/status')
def get_status():
    return jsonify({
        'class': classification_data.current_class,
        'confidence': round(classification_data.confidence * 100, 1),
        'counts': classification_data.counts,
        'total': classification_data.total,
        'last_update': classification_data.last_update,
        'is_running': classification_data.is_running
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌐 DASHBOARD SERVER STARTING")
    print("="*60)
    print("\n📱 Open your browser and navigate to:")
    print("   → http://localhost:5000")
    print("\n🔗 Or from your phone (same WiFi):")
    print("   → http://YOUR_COMPUTER_IP:5000")
    print("\n⚠️  Make sure your ESP32-CAM is powered on!")
    print("\nPress Ctrl+C to stop the server\n")
    print("="*60 + "\n")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)