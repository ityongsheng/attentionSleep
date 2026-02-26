"""
Setup and installation script
"""
import subprocess
import sys
import os
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("Error: Python 3.9 or higher is required")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True


def install_dependencies():
    """Install required packages"""
    print("\nInstalling dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("✗ Failed to install dependencies")
        return False


def create_directories():
    """Create necessary directories"""
    print("\nCreating directories...")
    directories = [
        'data/raw',
        'data/processed',
        'models/checkpoints',
        'logs/screenshots',
        'database',
        'config'
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {directory}")


def download_sample_data():
    """Download or prepare sample data"""
    print("\nPreparing sample data...")
    print("Note: You need to manually download the NTHU-DDD dataset")
    print("Dataset URL: https://github.com/NTHU-DDD/NTHU-DDD")
    print("Place the dataset in: data/raw/")


def test_imports():
    """Test if all required packages can be imported"""
    print("\nTesting imports...")
    required_packages = [
        'torch',
        'torchvision',
        'cv2',
        'mediapipe',
        'PyQt5',
        'pyttsx3',
        'albumentations',
        'yaml'
    ]

    all_ok = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - Failed to import")
            all_ok = False

    return all_ok


def check_cuda():
    """Check CUDA availability"""
    print("\nChecking CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
        else:
            print("⚠ CUDA not available - will use CPU")
    except ImportError:
        print("✗ PyTorch not installed")


def main():
    """Main setup function"""
    print("=" * 60)
    print("Fatigue Detection System - Setup")
    print("=" * 60)

    # Check Python version
    if not check_python_version():
        return

    # Create directories
    create_directories()

    # Install dependencies
    if not install_dependencies():
        print("\nSetup failed. Please install dependencies manually.")
        return

    # Test imports
    if not test_imports():
        print("\nSome packages failed to import. Please check installation.")
        return

    # Check CUDA
    check_cuda()

    # Download data instructions
    download_sample_data()

    print("\n" + "=" * 60)
    print("Setup completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Download the NTHU-DDD dataset to data/raw/")
    print("2. Run data preprocessing: python data/preprocess.py")
    print("3. Train the model: python train/trainer.py")
    print("4. Run the application: python main.py")
    print("\nFor more information, see README.md")


if __name__ == "__main__":
    main()
