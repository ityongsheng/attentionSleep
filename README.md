# Fatigue Driving Detection System

## Project Overview
A PC-based fatigue driving detection system using attention mechanisms (MobileNetV3 + ECA + CBAM + LSTM) for real-time driver monitoring.

## Features
- Real-time facial landmark detection
- Fatigue state classification (eye closure, yawning, head nodding)
- Hybrid attention mechanism for robust feature extraction
- LSTM temporal analysis for continuous monitoring
- Voice alert system
- Detection logging with screenshots
- Qt-based user interface

## System Requirements
- Python 3.9+
- PyTorch 2.1+
- CUDA 12.1+ (for GPU acceleration)
- Webcam or video input device

## Installation

### 1. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Pre-trained Models (if available)
```bash
# Place models in models/checkpoints/
```

## Project Structure
```
Fatigue_Detection_System/
├── data/               # Data processing scripts
├── models/             # Model definitions
├── train/              # Training scripts
├── deployment/         # Inference and optimization
├── gui/                # UI system
├── alerts/             # Alert system
├── database/           # Database management
├── config/             # Configuration files
├── logs/               # Training and detection logs
└── main.py             # System entry point
```

## Usage

### Training
```bash
python train/trainer.py --config config/train_config.yaml
```

### Inference
```bash
python main.py
```

## Performance Targets
- Detection accuracy: >90%
- Inference latency: <100ms
- False positive rate: <5%

## References
See task document for complete reference list.

## License
Academic use only.
