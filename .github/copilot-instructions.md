# Copilot Instructions for TomatoRipeness

## Project Overview
This is a machine learning system for tomato ripeness classification using SVM and CIELAB color features, built on the Laboro Tomato dataset. It includes a Flask web interface for image upload and analysis, and supports spoilage detection and transport recommendations.

## Architecture & Key Components
- **Core Files/**: Main application logic (`app.py` for Flask web app, `config.py` for settings, `setup.py` for setup automation, `Dockerfile` for containerization).
- **ML Pipeline/**: Core ML functionality for data loading (`dataset_loader.py`), feature extraction (`feature_extraction.py`), model training (`train_model.py`), and inference utilities (`model_utils.py`).
- **Interface/templates/**: Flask HTML templates (`index.html` with interactive image upload and analysis UI).
- **Testing & Documentation/**: Project documentation and testing resources.

### Data Flow
1. **Dataset Preparation**: Download and preprocess Laboro Tomato dataset (Kaggle API required).
2. **Data Augmentation**: Generate synthetic rotten samples from ripe images for balanced training.
3. **Feature Extraction**: Extract multi-modal features (CIELAB color space, texture, shape, spoilage indicators).
4. **Model Training**: Train SVM classifier with RBF kernel, using grid search for optimal hyperparameters.
5. **Inference**: Process uploaded images through validation, feature extraction, and classification pipeline.

## Developer Workflows
- **Setup**: Run `python setup.py` for automated setup, or manually install with `pip install -r requirements.txt`.
- **Dataset**: Place `kaggle.json` in `~/.kaggle/` for dataset download.
- **Training**: Use Flask CLI (`flask prepare-dataset`, `flask extract-features`, `flask train-model`) or run scripts directly.
- **Web Interface**: Start with `python app.py` and visit `http://localhost:5000`.
- **Testing**: Use `pytest` for tests (see `Testing & Documentation/`).
- **Formatting/Style**: Use `black` and `flake8` for code style.

## Project-Specific Patterns
- **Feature Engineering**: Extend `feature_extraction.py` and update `extract_all_features()` for new features.
- **Class Mapping**: Update `class_mapping` in `dataset_loader.py` to change classification targets.
- **Model Management**: Use `model_utils.py` for loading, saving, and batch inference.
- **API Endpoints**: See `app.py` for `/predict`, `/health`, `/model-info`, `/demo-prediction`, `/train-status`.

## Integration Points
- **External**: Kaggle API for dataset, scikit-learn for ML, OpenCV for image processing, Flask for web.
- **Directories**: `data/` for datasets, `models/` for trained models, `uploads/` for user images, `logs/` for logs.
- **Environment**: `.env` or `.env.example` for configuration (see `setup.py`).

## Conventions
- All scripts expect relative paths from project root.
- Model files: `model_utils.py` expects `tomato_svm_model.pkl` in root, but can be configured in `config.py`.
- Template files: All templates should be in `Interface/templates/`.
- Images for analysis: Upload to `uploads/`, must be ≤16MB and ≤5000x5000 pixels.
- Image preprocessing: All images are resized to 224x224 pixels for consistent analysis.

## Example: Batch Inference
```python
from ML Pipeline.model_utils import TomatoClassifier
classifier = TomatoClassifier()
classifier.load_model()
results = classifier.analyze_batch_images('uploads/')

# Example output format:
# {
#   'predicted_class': 'ripe',
#   'confidence': 0.87,
#   'class_probabilities': {'unripe': 0.05, 'turning': 0.08, 'ripe': 0.87, 'rotten': 0.0},
#   'cielab_analysis': {'L_lightness': 45.2, 'a_green_red': 23.1, 'b_blue_yellow': 15.8},
#   'spoilage_score': 2.1,
#   'transport_survivability': {'short_distance': {...}, ...}
# }
```

## Troubleshooting
- If model/dataset not found, run training scripts and check Kaggle setup.
- For package errors, reinstall with `pip install -r requirements.txt --force-reinstall`.
- Enable debug with `export FLASK_DEBUG=True` and `python app.py`.

---
For more details, see `Testing & Documentation/README.md`.
