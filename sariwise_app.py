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

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sariwise-tomato-magic-2024'
socketio = SocketIO(app, cors_allowed_origins="*")

class SariWiseData:
    """SariWise - Tomato Classifier"""
    def __init__(self):
        self.current_class = 'Waiting...'
        self.confidence = 0.0
        self.counts = {'Ripe': 0, 'Unripe': 0, 'Old': 0, 'Damaged': 0}
        self.total = 0
        self.last_update = 'Never'
        self.is_running = False
        self.history = []
        self.max_history = 100
        
    def add_to_history(self, class_name, confidence):
        """Add tomato to history"""
        entry = {
            'id': len(self.history) + 1,
            'class': class_name,
            'confidence': round(confidence * 100, 1),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'action': self.get_sariwise_action(class_name),
            'emoji': self.get_tomato_emoji(class_name)
        }
        self.history.insert(0, entry)
        
        if len(self.history) > self.max_history:
            self.history = self.history[:self.max_history]
    
    def get_tomato_emoji(self, class_name):
        """Get fun tomato emoji for each state"""
        emojis = {
            'Ripe': '🍅',
            'Unripe': '🟢',
            'Old': '🟠',
            'Damaged': '💔'
        }
        return emojis.get(class_name, '🍅')
    
    def get_sariwise_action(self, class_name):
        """SariWise recommendations for each tomato"""
        actions = {
            'Ripe': '🍅 Perfect! Package for sari-sari store',
            'Unripe': '⏰ Store for 3-5 days to ripen',
            'Old': '🥫 Make tomato sauce or paste',
            'Damaged': '🗑️ Discard or compost'
        }
        return actions.get(class_name, 'No action')
    
    def get_sariwise_wisdom(self):
        """SariWise's smart logistics advice"""
        total = sum(self.counts.values())
        if total == 0:
            return {
                'wisdom': 'Start scanning tomatoes to get SariWise insights',
                'actions': [],
                'quality_score': 0,
                'recommendations': [],
                'logistics_tip': 'Logistics Tip: Scan at least 10 tomatoes for accurate recommendations'
            }
        
        percentages = {k: (v/total)*100 for k, v in self.counts.items()}
        
        # Quality score with tomato theme
        quality_score = (
            percentages['Ripe'] * 1.0 +
            percentages['Unripe'] * 0.7 +
            percentages['Old'] * 0.3 +
            percentages['Damaged'] * 0.0
        )
        
        # Generate actions
        actions = []
        if self.counts['Damaged'] > 0:
            actions.append({
                'priority': 'URGENT',
                'action': f"Remove {self.counts['Damaged']} damaged tomato(es) - they'll spoil others!",
                'icon': '🚨',
                'emoji': '💔'
            })
        
        if self.counts['Ripe'] > 5:
            actions.append({
                'priority': 'TODAY',
                'action': f"Display {self.counts['Ripe']} ripe tomato(es) in your sari-sari store",
                'icon': '🏪',
                'emoji': '🍅'
            })
        
        if self.counts['Old'] > 0:
            actions.append({
                'priority': 'TODAY',
                'action': f"Cook {self.counts['Old']} old tomato(es) for sauce - don't waste!",
                'icon': '🥘',
                'emoji': '🟠'
            })
        
        if self.counts['Unripe'] > 0:
            actions.append({
                'priority': 'LATER',
                'action': f"Store {self.counts['Unripe']} green tomato(es) in cool place",
                'icon': '📦',
                'emoji': '🟢'
            })
        
        # SariWise recommendations
        recommendations = []
        
        if percentages['Ripe'] > 60:
            recommendations.append("Excellent timing for your sari-sari store")
        elif percentages['Ripe'] < 30:
            recommendations.append("Tip: Wait 2-3 days before harvesting more tomatoes")
        
        if percentages['Damaged'] > 20:
            recommendations.append("Too many damaged! Check your handling and storage")
        
        if percentages['Unripe'] > 50:
            recommendations.append("These need more sun time. Harvest when redder")
        
        if percentages['Old'] > 30:
            recommendations.append("Pick your tomatoes more often to stay fresh")
        
        if total < 10:
            recommendations.append("Scan more tomatoes for better SariWise insights")
        
        # Logistics tip
        logistics_tip = self.get_logistics_tip(percentages, quality_score)
        
        return {
            'wisdom': f"Quality Score: {round(quality_score, 1)}/100",
            'actions': actions,
            'quality_score': round(quality_score, 1),
            'recommendations': recommendations,
            'percentages': percentages,
            'logistics_tip': logistics_tip
        }
    
    def get_logistics_tip(self, percentages, score):
        """Get logistics tip based on score"""
        if score >= 90:
            return "Excellent batch! Price these premium and sell fast"
        elif score >= 70:
            return "Good quality! Perfect for your regular customers"
        elif score >= 50:
            return "Mixed batch - separate by quality for better sales"
        else:
            return "Low quality - consider making sauce or selling cheaper"

sariwise_data = SariWiseData()

class TomatoClassifier:
    def __init__(self, esp32_ip, model_path):
        print("\n🍅 SariWise is loading...")
        self.esp32_ip = esp32_ip
        
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                         path=model_path, force_reload=False)
            self.model.eval()
            print("✓ SariWise AI is ready!")
        except Exception as e:
            print(f"❌ Error: {e}")
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
        try:
            url = f"http://{self.esp32_ip}/capture"
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                return img
            return None
        except Exception as e:
            print(f"⚠️ Camera error: {e}")
            return None
    
    def classify(self, image):
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
            return None, 0.0
    
    def get_annotated_frame(self):
        img_pil = self.capture_image()
        if img_pil is None:
            return None, None, 0.0
        
        class_name, confidence = self.classify(img_pil)
        if class_name is None:
            return None, None, 0.0
        
        # Update counts
        if self.last_class and class_name != self.last_class:
            sariwise_data.counts[class_name] += 1
            sariwise_data.total += 1
            sariwise_data.add_to_history(class_name, confidence)
        self.last_class = class_name
        
        # Convert to OpenCV
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        height, width = img_cv.shape[:2]
        
        # Tomato colors!
        color_map = {
            'Ripe': (82, 255, 168),      # Bright tomato red
            'Unripe': (144, 238, 144),   # Light green
            'Old': (255, 180, 0),        # Deep orange
            'Damaged': (220, 20, 60)     # Dark red
        }
        color = color_map.get(class_name, (255, 255, 255))
        
        # Create overlay
        overlay = img_cv.copy()
        banner_height = int(height * 0.15)
        cv2.rectangle(overlay, (0, 0), (width, banner_height), (139, 0, 0), -1)  # Dark red
        cv2.addWeighted(overlay, 0.7, img_cv, 0.3, 0, img_cv)
        
        # Draw text
        font = cv2.FONT_HERSHEY_BOLD
        font_scale = min(width, height) / 350
        thickness = max(2, int(font_scale * 2))
        
        # Main text
        text = f"🍅 {class_name.upper()}"
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = int(banner_height * 0.45)
        
        cv2.putText(img_cv, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 2)
        cv2.putText(img_cv, text, (text_x, text_y), font, font_scale, color, thickness)
        
        # Confidence
        conf_text = f"SariWise: {confidence:.1%}"
        conf_size = cv2.getTextSize(conf_text, font, font_scale * 0.6, thickness)[0]
        conf_x = (width - conf_size[0]) // 2
        conf_y = int(banner_height * 0.75)
        cv2.putText(img_cv, conf_text, (conf_x, conf_y), font, font_scale * 0.6, (255, 255, 255), thickness)
        
        # Tomato border
        border = max(8, int(min(width, height) * 0.015))
        cv2.rectangle(img_cv, (0, 0), (width-1, height-1), color, border)
        
        return img_cv, class_name, confidence

# Initialize SariWise
print("\n" + "="*60)
print("SARIWISE - TOMATO CLASSIFIER")
print("="*60)

try:
    classifier = TomatoClassifier(
        esp32_ip="192.168.1.31",  # ⚠️ UPDATE YOUR ESP32 IP!
        model_path="yolov5/tomato_training/run1/weights/best.pt"
    )
    print("✓ SariWise ready to help your business!")
except Exception as e:
    print(f"❌ Failed: {e}")
    classifier = None

def generate_frames():
    while True:
        if not sariwise_data.is_running or classifier is None:
            time.sleep(0.5)
            continue
            
        try:
            frame, class_name, confidence = classifier.get_annotated_frame()
            
            if frame is None:
                time.sleep(0.5)
                continue
            
            # Update data
            sariwise_data.current_class = class_name
            sariwise_data.confidence = confidence
            sariwise_data.last_update = datetime.now().strftime("%I:%M:%S %p")
            
            # Get SariWise wisdom
            wisdom = sariwise_data.get_sariwise_wisdom()
            
            # Emit update
            socketio.emit('update', {
                'class': class_name,
                'confidence': round(confidence * 100, 1),
                'counts': sariwise_data.counts,
                'total': sariwise_data.total,
                'timestamp': sariwise_data.last_update,
                'wisdom': wisdom,
                'history': sariwise_data.history[:10]
            })
            
            # Encode
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.1)
            
        except Exception as e:
            print(f"⚠️ Error: {e}")
            time.sleep(1)

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/start')
def start():
    sariwise_data.is_running = True
    return jsonify({'success': True})

@app.route('/api/stop')
def stop():
    sariwise_data.is_running = False
    return jsonify({'success': True})

@app.route('/api/reset')
def reset():
    sariwise_data.counts = {'Ripe': 0, 'Unripe': 0, 'Old': 0, 'Damaged': 0}
    sariwise_data.total = 0
    sariwise_data.history = []
    return jsonify({'success': True})

@app.route('/api/status')
def status():
    wisdom = sariwise_data.get_sariwise_wisdom()
    return jsonify({
        'class': sariwise_data.current_class,
        'confidence': round(sariwise_data.confidence * 100, 1),
        'counts': sariwise_data.counts,
        'total': sariwise_data.total,
        'last_update': sariwise_data.last_update,
        'is_running': sariwise_data.is_running,
        'wisdom': wisdom,
        'history': sariwise_data.history[:20]
    })

@app.route('/api/export_csv')
def export():
    import csv
    from flask import make_response
    from io import StringIO
    
    si = StringIO()
    writer = csv.writer(si)
    writer.writerow(['ID', 'Timestamp', 'Tomato State', 'Confidence (%)', 'SariWise Action'])
    
    for entry in sariwise_data.history:
        writer.writerow([
            entry['id'],
            entry['timestamp'],
            f"{entry['emoji']} {entry['class']}",
            entry['confidence'],
            entry['action']
        ])
    
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=sariwise_tomato_report.csv"
    output.headers["Content-type"] = "text/csv"
    return output

if __name__ == '__main__':
    print("\n" + "="*60)
    print("SARIWISE SERVER STARTING")
    print("="*60)
    print("\n📱 Open: http://localhost:5000")
    print("⚠️  ESP32-CAM must be powered on!")
    print("\nPress Ctrl+C to stop\n")
    print("="*60 + "\n")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)