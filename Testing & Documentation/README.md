A comprehensive machine learning system for tomato ripeness classification using SVM with CIELAB color features, built with the Laboro Tomato dataset.
Features

Real SVM Classification: Trained on Laboro Tomato dataset with 4 classes (unripe, turning, ripe, rotten)
CIELAB Color Analysis: Advanced color space analysis for accurate ripeness detection
Spoilage Detection: AI-powered identification of mold, black spots, and decay
Transport Survivability: Recommendations for different shipping distances
Beautiful Web Interface: User-friendly interface for image upload and analysis
Real-time Processing: Fast image analysis with detailed results

System Requirements

Python 3.8 or higher
4GB+ RAM recommended
Modern web browser
Internet connection (for dataset download)

Quick Start
1. Clone and Setup
bash# Clone the repository (or extract files to a folder)
cd tomato-vision-pro

# Run automated setup
python setup.py
2. Manual Setup (Alternative)
bash# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p data uploads models templates logs
3. Configure Kaggle API

Go to Kaggle Account Settings
Click "Create New API Token"
Download kaggle.json
Place it in ~/.kaggle/kaggle.json
Set permissions: chmod 600 ~/.kaggle/kaggle.json

4. Train the Model
bash# Option 1: Use Flask CLI commands
flask prepare-dataset    # Download and prepare Laboro dataset
flask extract-features   # Extract CIELAB and other features
flask train-model        # Train SVM classifier

# Option 2: Run scripts directly
python dataset_loader.py
python feature_extraction.py
python train_model.py
5. Start the Web Interface
bashpython app.py
Visit http://localhost:5000 to use the interface.
Project Structure
tomato-vision-pro/
├── app.py                 # Flask web application
├── setup.py              # Automated setup script
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── config.py             # Configuration settings
│
├── dataset_loader.py     # Laboro dataset preparation
├── feature_extraction.py # CIELAB feature extraction
├── train_model.py        # SVM model training
├── model_utils.py        # Model utilities and inference
│
├── templates/
│   └── index.html        # Web interface
│
├── data/                 # Dataset storage
│   └── crops/           # Extracted tomato crops
│
├── models/              # Trained models
├── uploads/             # Uploaded images
├── sample_images/       # Sample test images
└── logs/               # Application logs
Dataset Information
This system uses the Laboro Tomato Dataset from Kaggle:

Source: Laboro Tomato Dataset
Classes: 6 original classes mapped to 4 target classes

b_green + l_green → unripe
b_half_ripened + l_half_ripened → turning
b_fully_ripened + l_fully_ripened → ripe
Synthetic augmentation → rotten


Images: 800+ high-resolution tomato images
Features: Instance segmentation annotations

Feature Engineering
The system extracts multiple feature types for robust classification:
CIELAB Color Features (Primary)

L* (Lightness): 0-100 scale
a* (Green-Red): -128 to +127
b* (Blue-Yellow): -128 to +127
Statistical measures: mean, std, median, min, max
Derived features: chroma, hue angle, ratios

Additional Features

Color Histograms: RGB and HSV distributions
Texture Features: Local Binary Patterns (LBP)
Shape Features: Area, perimeter, circularity, aspect ratio
Spoilage Features: Dark pixels, mold detection, decay analysis
Ripeness Features: Red/green/yellow pixel ratios

Model Architecture

Algorithm: Support Vector Machine (SVM) with RBF kernel
Preprocessing: StandardScaler for feature normalization
Hyperparameters: Grid search optimized (C, gamma, kernel)
Target Accuracy: 75%+ on held-out test set
Cross-validation: 5-fold CV for robust evaluation

API Endpoints
Web Interface

GET / - Main web interface
POST /predict - Image upload and classification
GET /health - Health check
GET /model-info - Model information
GET /demo-prediction - Demo prediction for testing

Model Management

GET /train-status - Check training progress
Flask CLI commands for dataset preparation and training

Usage Examples
Web Interface

Open http://localhost:5000
Upload a tomato image
View classification results with:

Ripeness class and confidence
CIELAB color analysis
Spoilage detection results
Transport survivability recommendations



Programmatic Usage
pythonfrom model_utils import TomatoClassifier

# Load trained model
classifier = TomatoClassifier()
classifier.load_model()

# Classify an image
result = classifier.predict_image('path/to/tomato.jpg')

print(f"Class: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.3f}")
print(f"CIELAB: {result['cielab_analysis']}")
Batch Processing
python# Analyze multiple images
results = classifier.analyze_batch_images('images_directory/')

for result in results['results']:
    if result['success']:
        print(f"{result['filename']}: {result['predicted_class']}")
Model Performance
Expected performance metrics:

Accuracy: 75%+ on test set
Classes: 4-class classification (unripe/turning/ripe/rotten)
Processing Time: <2 seconds per image
Image Size: Optimized for 224x224 pixels

Customization
Adding New Features

Modify feature_extraction.py
Add new feature extraction methods
Update extract_all_features() method
Retrain model with new features

Changing Classification Classes

Update class_mapping in dataset_loader.py
Modify target_classes list
Update web interface labels
Retrain model

Tuning Hyperparameters

Modify parameter grid in train_model.py
Adjust cross-validation settings
Run hyperparameter optimization

Deployment
Development
bashpython app.py
# Runs on http://localhost:5000 with debug mode
Production
bash# Install production WSGI server
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app
Docker (Optional)
dockerfileFROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "app.py"]
Troubleshooting
Common Issues
"Model not found" error
bash# Train the model first
python train_model.py
"Dataset not found" error
bash# Check Kaggle API setup
cat ~/.kaggle/kaggle.json

# Download dataset manually
python dataset_loader.py
"Import errors" with packages
bash# Reinstall requirements
pip install -r requirements.txt --force-reinstall
Low accuracy results

Check dataset quality and balance
Verify feature extraction is working correctly
Try different SVM hyperparameters
Ensure sufficient training data

Debug Mode
bash# Enable detailed logging
export FLASK_DEBUG=True
export FLASK_ENV=development
python app.py
Contributing

Fork the repository
Create a feature branch
Make your changes
Add tests for new functionality
Submit a pull request

Development Setup
bash# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest tests/

# Format code
black *.py

# Check code style
flake8 *.py
Scientific Background
This system implements research-based approaches for fruit quality assessment:

Color Analysis: CIELAB color space provides perceptually uniform color representation
Machine Learning: SVM with RBF kernel for non-linear classification
Feature Engineering: Multi-modal features combining color, texture, and shape
Agricultural Applications: Real-world applicability for quality control and sorting

License
This project is open source. The Laboro Tomato dataset is under CC BY-NC-SA 4.0 license.
References

Laboro Tomato Dataset: https://github.com/laboroai/LaboroTomato
CIELAB Color Space: CIE 1976 Lab* color space
Support Vector Machines: Scikit-learn SVM implementation
Computer Vision: OpenCV for image processing

Support
For issues and questions:

Check this README and troubleshooting section
Review error messages and logs
Ensure all dependencies are installed correctly
Verify dataset and model files exist

Acknowledgments

Laboro.AI for the Laboro Tomato dataset
Scikit-learn team for machine learning tools
OpenCV community for computer vision libraries
Flask team for the web framework

