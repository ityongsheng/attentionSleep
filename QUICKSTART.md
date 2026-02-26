# Quick Start Guide

## Installation

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Run setup script
python setup.py
```

### 2. Manual Installation (if setup.py fails)

```bash
pip install -r requirements.txt
```

## Data Preparation

### Download Dataset

1. Download NTHU-DDD dataset from: https://github.com/NTHU-DDD/NTHU-DDD
2. Extract to `data/raw/`

### Preprocess Data

```bash
# Extract frames from videos
python data/preprocess.py

# Organize data structure:
# data/processed/
#   ├── normal/
#   ├── drowsy/
#   └── very_drowsy/
```

## Training

### Basic Training

```bash
python train/trainer.py --config config/train_config.yaml
```

### Monitor Training

```bash
# In another terminal
tensorboard --logdir logs
```

### Training Configuration

Edit `config/train_config.yaml` to adjust:
- Batch size
- Learning rate
- Number of epochs
- Model architecture options

## Model Export

```bash
# Export to ONNX format
python deployment/onnx_export.py \
    --model models/checkpoints/best_model.pth \
    --output models/checkpoints/model.onnx
```

## Running the Application

### GUI Application

```bash
python main.py
```

### Command Line Testing

```bash
# Test model components
python tests/test_models.py
```

## Usage

### GUI Controls

1. **Start Monitoring**: Begin real-time fatigue detection
2. **Stop Monitoring**: Stop detection and save session
3. **View Logs**: Check detection events in the log panel

### Detection States

- **Normal**: Alert and focused
- **Drowsy**: Signs of fatigue detected
- **Very Drowsy**: Severe fatigue, immediate rest recommended

### Voice Alerts

The system will automatically:
- Warn when drowsiness is detected
- Alert urgently for severe fatigue
- Log all events to database

## Troubleshooting

### Camera Not Working

```bash
# Test camera access
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Error')"
```

### Model Not Found

Ensure you have trained the model first:
```bash
python train/trainer.py
```

### CUDA Out of Memory

Reduce batch size in `config/train_config.yaml`:
```yaml
training:
  batch_size: 16  # Reduce from 32
```

### Import Errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## Performance Optimization

### For CPU-only Systems

1. Reduce model complexity in config:
```yaml
model:
  use_lstm: false  # Disable LSTM for faster inference
```

2. Use smaller image size:
```yaml
data:
  image_size: [160, 160]  # Reduce from [224, 224]
```

### For GPU Systems

1. Enable CUDA in config:
```yaml
training:
  device: "cuda"
```

2. Use TensorRT for faster inference:
```bash
# Convert to TensorRT (requires NVIDIA GPU)
python deployment/tensorrt_export.py
```

## Database

### View Detection Logs

```python
from database.db_manager import DatabaseManager

db = DatabaseManager()
events = db.get_recent_events(limit=100)
for event in events:
    print(event)
```

### Export Session Data

```python
session_summary = db.get_session_summary(session_id)
print(session_summary)
```

## Advanced Configuration

### Custom Thresholds

Edit `config/train_config.yaml`:

```yaml
thresholds:
  ear_threshold: 0.25      # Eye closure threshold
  mar_threshold: 0.6       # Yawn detection threshold
  ear_consecutive_frames: 15  # Frames before alert
```

### Voice Settings

Modify in `alerts/voice_engine.py`:
```python
engine = VoiceAlertEngine(
    rate=150,      # Speech rate
    volume=1.0     # Volume level
)
```

## Project Structure

```
attentionSleep/
├── data/              # Data processing
├── models/            # Model definitions
├── train/             # Training scripts
├── deployment/        # Inference and export
├── gui/               # User interface
├── alerts/            # Alert system
├── database/          # Database management
├── config/            # Configuration files
├── logs/              # Training and detection logs
├── tests/             # Unit tests
├── main.py            # Application entry point
└── requirements.txt   # Dependencies
```

## Next Steps

1. Collect or download training data
2. Train the model with your data
3. Test the system with webcam
4. Fine-tune thresholds for your use case
5. Deploy to production environment

## Support

For issues and questions:
- Check README.md for detailed documentation
- Review task document for project requirements
- Test individual components with test scripts

## Citation

If you use this system in your research, please cite:

```
@misc{fatigue_detection_2025,
  title={Fatigue Driving Detection System with Attention Mechanisms},
  author={Your Name},
  year={2025},
  institution={Hebei Oriental University}
}
```
