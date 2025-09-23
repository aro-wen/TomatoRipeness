import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "uploads"
LOG_DIR = BASE_DIR / "logs"

# Ensure directories exist
for directory in [DATA_DIR, MODEL_DIR, UPLOAD_DIR, LOG_DIR]:
    directory.mkdir(exist_ok=True)

# Model Configuration
MODEL_CONFIG = {
    'model_path': MODEL_DIR / "tomato_svm_model.pkl",
    'backup_model_path': MODEL_DIR / "tomato_svm_model_backup.pkl",
    'model_version': "1.0.0",
    'target_accuracy': 0.75,
    'confidence_threshold': 0.7
}

# Dataset Configuration
DATASET_CONFIG = {
    'kaggle_dataset': 'nexuswho/laboro-tomato',
    'raw_data_dir': DATA_DIR / "raw",
    'processed_data_dir': DATA_DIR / "crops",
    'features_file': BASE_DIR / "features.csv",
    'metadata_file': DATA_DIR / "metadata.csv",
    'target_classes': ['unripe', 'turning', 'ripe', 'rotten'],
    'original_classes': {
        'b_green': 'unripe',
        'l_green': 'unripe',
        'b_half_ripened': 'turning',
        'l_half_ripened': 'turning',
        'b_fully_ripened': 'ripe',
        'l_fully_ripened': 'ripe'
    }
}

# Feature Extraction Configuration
FEATURE_CONFIG = {
    'image_size': (224, 224),
    'cielab_bins': 32,
    'histogram_bins': 16,
    'texture_radius': 3,
    'texture_points': 24,
    'color_spaces': ['RGB', 'HSV', 'LAB'],
    'extract_texture': True,
    'extract_shape': True,
    'extract_spoilage': True,
    'extract_ripeness': True
}

# Training Configuration
TRAINING_CONFIG = {
    'random_state': 42,
    'test_size': 0.2,
    'validation_size': 0.2,
    'cv_folds': 5,
    'tune_hyperparameters': True,
    'algorithms_to_compare': ['SVM', 'RandomForest', 'LogisticRegression'],
    'svm_param_grid': {
        'svm__C': [0.1, 1, 10, 100],
        'svm__kernel': ['rbf', 'poly', 'sigmoid'],
        'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
    },
    'scoring_metric': 'accuracy',
    'n_jobs': -1
}

# Web Application Configuration
FLASK_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': True,
    'upload_folder': str(UPLOAD_DIR),
    'max_content_length': 16 * 1024 * 1024,  # 16MB
    'allowed_extensions': {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'},
    'secret_key': os.environ.get('SECRET_KEY', 'tomato-vision-pro-secret-key'),
    'cors_origins': ['http://localhost:3000', 'http://127.0.0.1:3000']
}

# Image Processing Configuration
IMAGE_CONFIG = {
    'max_image_size': (1024, 1024),
    'thumbnail_size': (300, 300),
    'supported_formats': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
    'quality_settings': {
        'jpeg_quality': 85,
        'png_compress_level': 6
    },
    'preprocessing': {
        'normalize': True,
        'resize_method': 'maintain_aspect_ratio',
        'padding_color': [0, 0, 0]
    }
}

# Classification Thresholds
CLASSIFICATION_CONFIG = {
    'spoilage_thresholds': {
        'low': 5.0,
        'moderate': 10.0,
        'high': 20.0,
        'severe': 30.0
    },
    'ripeness_color_ranges': {
        'red_hue_range': [(0, 20), (160, 180)],
        'green_hue_range': (35, 85),
        'yellow_hue_range': (15, 35),
        'saturation_threshold': 30,
        'value_threshold': 40
    },
    'transport_survivability': {
        'distance_categories': {
            'short': 100,    # miles
            'medium': 500,   # miles
            'long': 1000     # miles
        },
        'temperature_factor': True,
        'humidity_factor': True,
        'handling_factor': True
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_handler': {
        'filename': LOG_DIR / 'tomato_vision.log',
        'max_bytes': 10 * 1024 * 1024,  # 10MB
        'backup_count': 5
    },
    'console_handler': {
        'enabled': True,
        'level': 'INFO'
    }
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    'enable_caching': True,
    'cache_duration': 3600,  # 1 hour
    'max_concurrent_predictions': 10,
    'prediction_timeout': 30,  # seconds
    'memory_limit': '2GB',
    'cpu_cores': None  # None = auto-detect
}

# Development/Testing Configuration
DEV_CONFIG = {
    'create_sample_data': True,
    'sample_images_count': 10,
    'enable_debug_routes': True,
    'mock_predictions': False,
    'detailed_errors': True,
    'profiling_enabled': False
}

# Production Configuration
PROD_CONFIG = {
    'debug': False,
    'testing': False,
    'log_level': 'WARNING',
    'enable_monitoring': True,
    'rate_limiting': {
        'enabled': True,
        'requests_per_minute': 60,
        'requests_per_hour': 1000
    },
    'security': {
        'csrf_protection': True,
        'secure_headers': True,
        'file_validation': 'strict'
    }
}

# API Configuration
API_CONFIG = {
    'version': 'v1',
    'base_url': '/api/v1',
    'rate_limits': {
        'predict': '30/minute',
        'batch_predict': '5/minute',
        'model_info': '60/minute'
    },
    'response_format': 'json',
    'enable_swagger': True,
    'cors_enabled': True
}

# Monitoring and Analytics
MONITORING_CONFIG = {
    'enable_metrics': True,
    'metrics_endpoint': '/metrics',
    'track_predictions': True,
    'track_accuracy': True,
    'track_performance': True,
    'analytics_retention_days': 30
}

# Database Configuration (Optional)
DATABASE_CONFIG = {
    'enabled': False,  # Set to True to enable database logging
    'url': os.environ.get('DATABASE_URL', 'sqlite:///tomato_vision.db'),
    'track_predictions': True,
    'track_user_sessions': False,  # Privacy consideration
    'backup_enabled': True,
    'backup_frequency': 'daily'
}

# Email/Notification Configuration (Optional)
NOTIFICATION_CONFIG = {
    'enabled': False,
    'email_backend': 'smtp',
    'smtp_settings': {
        'host': os.environ.get('SMTP_HOST'),
        'port': int(os.environ.get('SMTP_PORT', 587)),
        'username': os.environ.get('SMTP_USERNAME'),
        'password': os.environ.get('SMTP_PASSWORD'),
        'use_tls': True
    },
    'admin_email': os.environ.get('ADMIN_EMAIL'),
    'notification_types': ['error', 'model_retrain', 'system_alert']
}

# Environment-specific configuration
ENVIRONMENT = os.environ.get('ENVIRONMENT', 'development')

if ENVIRONMENT == 'production':
    # Override with production settings
    FLASK_CONFIG.update(PROD_CONFIG)
    FLASK_CONFIG['debug'] = False
    LOGGING_CONFIG['level'] = 'WARNING'
elif ENVIRONMENT == 'testing':
    # Override with testing settings
    FLASK_CONFIG['testing'] = True
    DATASET_CONFIG['target_classes'] = ['test_class_1', 'test_class_2']
    MODEL_CONFIG['model_path'] = MODEL_DIR / "test_model.pkl"

def get_config():
    """Get complete configuration dictionary"""
    return {
        'model': MODEL_CONFIG,
        'dataset': DATASET_CONFIG,
        'features': FEATURE_CONFIG,
        'training': TRAINING_CONFIG,
        'flask': FLASK_CONFIG,
        'image': IMAGE_CONFIG,
        'classification': CLASSIFICATION_CONFIG,
        'logging': LOGGING_CONFIG,
        'performance': PERFORMANCE_CONFIG,
        'development': DEV_CONFIG,
        'api': API_CONFIG,
        'monitoring': MONITORING_CONFIG,
        'database': DATABASE_CONFIG,
        'notification': NOTIFICATION_CONFIG,
        'environment': ENVIRONMENT
    }

def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check required directories
    required_dirs = [DATA_DIR, MODEL_DIR, UPLOAD_DIR]
    for directory in required_dirs:
        if not directory.exists():
            errors.append(f"Required directory missing: {directory}")
    
    # Check model file if not in development
    if ENVIRONMENT != 'development':
        if not MODEL_CONFIG['model_path'].exists():
            errors.append(f"Model file not found: {MODEL_CONFIG['model_path']}")
    
    # Validate file size limits
    if FLASK_CONFIG['max_content_length'] > 50 * 1024 * 1024:  # 50MB
        errors.append("File size limit too high (security risk)")
    
    # Validate image size limits
    if IMAGE_CONFIG['max_image_size'][0] > 2048 or IMAGE_CONFIG['max_image_size'][1] > 2048:
        errors.append("Image size limit too high (memory risk)")
    
    return errors

def print_config_summary():
    """Print configuration summary"""
    print("🍅 TomatoVision Pro Configuration Summary")
    print("=" * 50)
    print(f"Environment: {ENVIRONMENT}")
    print(f"Model Path: {MODEL_CONFIG['model_path']}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Target Classes: {DATASET_CONFIG['target_classes']}")
    print(f"Image Size: {FEATURE_CONFIG['image_size']}")
    print(f"Flask Host:Port: {FLASK_CONFIG['host']}:{FLASK_CONFIG['port']}")
    print(f"Debug Mode: {FLASK_CONFIG['debug']}")
    print(f"Max File Size: {FLASK_CONFIG['max_content_length'] // (1024*1024)}MB")
    print()
    
    # Validate configuration
    validation_errors = validate_config()
    if validation_errors:
        print("⚠️  Configuration Issues:")
        for error in validation_errors:
            print(f"   - {error}")
    else:
        print("✅ Configuration Valid")

if __name__ == "__main__":
    print_config_summary()