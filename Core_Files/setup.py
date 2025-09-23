import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description):
    """Run a command with error handling"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}: {e}")
        print(f"Command output: {e.output}")
        return False

def check_python_version():
    """Check if Python version is suitable"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        'data',
        'uploads',
        'models',
        'sample_images',
        'templates',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")

def install_requirements():
    """Install Python packages"""
    requirements_exist = Path('requirements.txt').exists()
    
    if not requirements_exist:
        print("❌ requirements.txt not found")
        return False
    
    # Upgrade pip first
    run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip")
    
    # Install requirements
    return run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing Python packages")

def setup_kaggle_api():
    """Guide user through Kaggle API setup"""
    print("\n🔧 Kaggle API Setup")
    print("=" * 50)
    
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    if kaggle_json.exists():
        print("✅ Kaggle API credentials found")
        return True
    
    print("📋 Kaggle API Setup Required:")
    print("1. Go to https://www.kaggle.com/account")
    print("2. Click 'Create New API Token'")
    print("3. Download kaggle.json file")
    print(f"4. Place it in: {kaggle_dir}")
    
    # Create .kaggle directory
    kaggle_dir.mkdir(exist_ok=True)
    
    # Set permissions on Unix systems
    if platform.system() != 'Windows':
        os.chmod(kaggle_dir, 0o700)
    
    print(f"\n⚠️  After placing kaggle.json, run: chmod 600 {kaggle_json}")
    print("Then re-run this setup script or continue manually.")
    
    return False

def create_sample_data():
    """Create sample configuration and test files"""
    
    # Create config file
    config_content = """# TomatoVision Pro Configuration
MODEL_PATH = "tomato_svm_model.pkl"
UPLOAD_FOLDER = "uploads"
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# Training parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Feature extraction
IMAGE_SIZE = (224, 224)
CIELAB_BINS = 32
TEXTURE_RADIUS = 3
TEXTURE_POINTS = 24

# Classification thresholds
SPOILAGE_THRESHOLD_LOW = 5.0
SPOILAGE_THRESHOLD_HIGH = 15.0
CONFIDENCE_THRESHOLD = 0.7
"""
    
    with open('config.py', 'w') as f:
        f.write(config_content)
    print("📄 Created config.py")
    
    # Create .env file template
    env_content = """# Environment variables for TomatoVision Pro
FLASK_ENV=development
FLASK_DEBUG=True
FLASK_APP=app.py

# Model settings
MODEL_PATH=tomato_svm_model.pkl
DATA_DIR=data/crops

# Security (change in production)
SECRET_KEY=your-secret-key-here

# Optional: Database URL if you add database functionality
# DATABASE_URL=sqlite:///tomato_classifications.db
"""
    
    with open('.env.example', 'w') as f:
        f.write(env_content)
    print("📄 Created .env.example")

def run_tests():
    """Run basic tests to verify setup"""
    print("\n🧪 Running Setup Tests")
    print("=" * 30)
    
    # Test imports
    test_imports = [
        'numpy',
        'pandas',
        'sklearn',
        'cv2',
        'flask',
        'joblib'
    ]
    
    for module in test_imports:
        try:
            __import__(module)
            print(f"✅ {module} import successful")
        except ImportError:
            print(f"❌ {module} import failed")
            return False
    
    return True

def print_next_steps():
    """Print next steps for the user"""
    print("\n🎉 Setup Complete!")
    print("=" * 50)
    print("Next steps:")
    print("1. Set up Kaggle API credentials (if not done)")
    print("2. Download and prepare dataset:")
    print("   python dataset_loader.py")
    print("3. Extract features:")
    print("   python feature_extraction.py")
    print("4. Train the model:")
    print("   python train_model.py")
    print("5. Start the web interface:")
    print("   python app.py")
    print("\nOr use Flask CLI commands:")
    print("   flask prepare-dataset")
    print("   flask extract-features")
    print("   flask train-model")
    print("\n🌐 Web interface will be available at: http://localhost:5000")
    print("\n📚 See README.md for detailed instructions")

def main():
    """Main setup function"""
    print("🍅 TomatoVision Pro Setup")
    print("=" * 30)
    print("Setting up your tomato classification system...\n")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Install requirements
    print("\n📦 Installing Python packages...")
    if not install_requirements():
        print("❌ Failed to install requirements. Please install manually:")
        print(f"   {sys.executable} -m pip install -r requirements.txt")
        sys.exit(1)
    
    # Setup Kaggle API
    kaggle_setup = setup_kaggle_api()
    
    # Create sample files
    print("\n📄 Creating configuration files...")
    create_sample_data()
    
    # Run tests
    if not run_tests():
        print("❌ Some tests failed. Please check your installation.")
        sys.exit(1)
    
    # Print next steps
    print_next_steps()
    
    if not kaggle_setup:
        print("\n⚠️  Warning: Kaggle API not configured.")
        print("You'll need to set this up to download the dataset.")
    
    print("\n✨ TomatoVision Pro is ready to use!")

if __name__ == "__main__":
    main()