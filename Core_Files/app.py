from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from pathlib import Path
import tempfile
import shutil
from ML_Pipeline.model_utils import TomatoClassifier, TomatoImageValidator, create_sample_predictions

# Initialize Flask app
app = Flask(__name__,
           template_folder='../Interface/templates',
           static_folder='../Interface/static')
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize classifier
classifier = TomatoClassifier()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        # Try to load model
        if classifier.model is None:
            classifier.load_model()
        
        return jsonify({
            'status': 'healthy',
            'model_loaded': True,
            'message': 'TomatoVision Pro API is running'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'model_loaded': False,
            'message': f'Model loading error: {str(e)}'
        }), 500

@app.route('/predict', methods=['POST'])
def predict_tomato():
    """Main prediction endpoint"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file uploaded'
            }), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Check file type
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Please upload an image file.'
            }), 400
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1])
        file.save(temp_file.name)
        
        try:
            # Validate image
            is_valid, validation_errors = TomatoImageValidator.validate_image_file(temp_file.name)
            if not is_valid:
                return jsonify({
                    'success': False,
                    'error': f'Image validation failed: {"; ".join(validation_errors)}'
                }), 400
            
            # Check if model is loaded
            if classifier.model is None:
                try:
                    classifier.load_model()
                except FileNotFoundError:
                    # Return sample predictions if model is not available
                    return jsonify({
                        'success': True,
                        'using_sample_data': True,
                        'message': 'Model not found, using sample prediction',
                        **create_sample_predictions()
                    })
            
            # Make prediction
            prediction = classifier.predict_image(temp_file.name)
            
            # Add processing time info
            prediction['processing_time'] = 'Real-time analysis completed'
            prediction['model_version'] = '1.0'
            
            return jsonify(prediction)
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file.name)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/model-info')
def model_info():
    """Get model information"""
    try:
        if classifier.model is None:
            try:
                classifier.load_model()
            except FileNotFoundError:
                return jsonify({
                    'model_available': False,
                    'message': 'Model not trained yet. Using demo mode.',
                    'classes': ['unripe', 'turning', 'ripe', 'rotten'],
                    'demo_mode': True
                })
        
        info = classifier.get_model_info()
        info['model_available'] = True
        info['demo_mode'] = False
        return jsonify(info)
        
    except Exception as e:
        return jsonify({
            'error': f'Error getting model info: {str(e)}',
            'model_available': False
        }), 500

@app.route('/demo-prediction')
def demo_prediction():
    """Get a demo prediction for testing the interface"""
    # Create realistic sample data for demo
    demo_results = {
        'success': True,
        'demo_mode': True,
        'predicted_class': 'ripe',
        'confidence': 0.89,
        'class_probabilities': {
            'unripe': 0.03,
            'turning': 0.08,
            'ripe': 0.89,
            'rotten': 0.0
        },
        'cielab_analysis': {
            'L_lightness': 42.7,
            'a_green_red': 28.3,
            'b_blue_yellow': 18.9
        },
        'spoilage_score': 1.2,
        'spoilage_analysis': {
            'mold_detection': 0.8,
            'black_spots': 0.3,
            'brown_decay': 1.2,
            'texture_loss': 3.1,
            'safety_warning': None
        },
        'transport_survivability': {
            'short_distance': {'safe': True, 'message': 'GOOD - Use soon after transport'},
            'medium_distance': {'safe': True, 'message': 'CAUTION - Refrigerate and use quickly'},
            'long_distance': {'safe': False, 'message': 'RISKY - May spoil during long transport'}
        },
        'processing_time': 'Demo mode - instant results',
        'model_version': 'Demo 1.0'
    }
    
    return jsonify(demo_results)

@app.route('/train-status')
def train_status():
    """Check training status and provide guidance"""
    model_exists = Path('tomato_svm_model.pkl').exists()
    dataset_exists = Path('data/crops').exists()
    features_exist = Path('features.csv').exists()
    
    status = {
        'model_trained': model_exists,
        'dataset_available': dataset_exists,
        'features_extracted': features_exist,
        'ready_for_training': False,
        'next_steps': []
    }
    
    if not dataset_exists:
        status['next_steps'].append('1. Run: python dataset_loader.py (to download and prepare Laboro dataset)')
    
    if not features_exist:
        status['next_steps'].append('2. Run: python feature_extraction.py (to extract CIELAB features)')
    
    if dataset_exists and not model_exists:
        status['next_steps'].append('3. Run: python train_model.py (to train SVM classifier)')
        status['ready_for_training'] = True
    
    if model_exists:
        status['message'] = 'Model is trained and ready for predictions!'
    elif len(status['next_steps']) > 0:
        status['message'] = f"Training pipeline incomplete. Complete {len(status['next_steps'])} steps to train the model."
    
    return jsonify(status)

@app.route('/sample-images')
def sample_images():
    """Provide information about sample images"""
    sample_dir = Path('sample_images')
    samples = []
    
    if sample_dir.exists():
        for img_file in sample_dir.glob('*.jpg'):
            samples.append({
                'filename': img_file.name,
                'path': f'/sample_images/{img_file.name}',
                'description': f'Sample tomato image for testing'
            })
    
    return jsonify({
        'available_samples': len(samples),
        'samples': samples,
        'instructions': 'Upload your own tomato images for classification'
    })

@app.route('/sample_images/<filename>')
def serve_sample_image(filename):
    """Serve sample images"""
    return send_from_directory('sample_images', filename)

@app.errorhandler(413)
def file_too_large(error):
    """Handle file too large error"""
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 16MB.'
    }), 413

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error. Please try again.'
    }), 500

# Development route for testing
@app.route('/test')
def test_endpoint():
    """Test endpoint for development"""
    return jsonify({
        'message': 'TomatoVision Pro API is working!',
        'endpoints': {
            'predict': '/predict (POST with image file)',
            'health': '/health',
            'model_info': '/model-info',
            'demo': '/demo-prediction',
            'train_status': '/train-status'
        },
        'version': '1.0.0'
    })

# CLI commands for model management
def create_cli_commands():
    """Create CLI commands for model management"""
    import click
    
    @app.cli.command()
    def prepare_dataset():
        """Prepare the Laboro tomato dataset"""
        from ML_Pipeline.dataset_loader import LaboroDatasetLoader
        
        click.echo("Preparing Laboro Tomato dataset...")
        loader = LaboroDatasetLoader()
        dataset_dir = loader.prepare_dataset(force_download=False)
        
        if dataset_dir:
            click.echo(f"Dataset prepared at: {dataset_dir}")
        else:
            click.echo("Failed to prepare dataset")
    
    @app.cli.command()
    def extract_features():
        """Extract features from dataset"""
        from ML_Pipeline.feature_extraction import extract_features_from_dataset
        
        dataset_dir = "data/crops"
        if not Path(dataset_dir).exists():
            click.echo("Dataset not found. Run 'flask prepare-dataset' first.")
            return
        
        click.echo("Extracting CIELAB and other features...")
        features_df = extract_features_from_dataset(dataset_dir)
        click.echo(f"Features extracted: {features_df.shape}")
    
    @app.cli.command()
    def train_model():
        """Train the SVM model"""
        from ML_Pipeline.train_model import main as train_main
        
        click.echo("Training SVM model...")
        train_main()
        click.echo("Training completed!")
    
    @app.cli.command()
    def model_info():
        """Show model information"""
        try:
            classifier.load_model()
            info = classifier.get_model_info()
            click.echo(f"Model Type: {info['model_type']}")
            click.echo(f"Classes: {info['classes']}")
            click.echo(f"Features: {info['feature_count']}")
        except FileNotFoundError:
            click.echo("Model not found. Run 'flask train-model' first.")

# Create CLI commands
create_cli_commands()

if __name__ == '__main__':
    print("Starting TomatoVision Pro API Server...")
    print("Access the interface at: http://localhost:5000")
    print("\nAPI Endpoints:")
    print("  POST /predict - Upload image for classification")
    print("  GET /health - Health check")
    print("  GET /model-info - Model information")
    print("  GET /demo-prediction - Demo prediction")
    print("  GET /train-status - Training status")
    
    # Check if model exists
    if Path('tomato_svm_model.pkl').exists():
        print("\nModel Status: READY")
    else:
        print("\nModel Status: NOT TRAINED")
        print("To train the model:")
        print("1. flask prepare-dataset")
        print("2. flask extract-features")
        print("3. flask train-model")
    
    app.run(debug=True, host='0.0.0.0', port=5000)