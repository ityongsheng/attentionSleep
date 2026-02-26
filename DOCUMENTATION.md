# Technical Documentation

## System Architecture

### Overview

The Fatigue Detection System is built on a hybrid deep learning architecture combining:

1. **Spatial Feature Extraction**: MobileNetV3 with ECA and CBAM attention mechanisms
2. **Temporal Analysis**: LSTM network for sequence modeling
3. **Real-time Processing**: Multi-threaded video capture and inference
4. **User Interface**: PyQt5-based GUI with live monitoring
5. **Alert System**: Voice alerts and database logging

### Architecture Diagram

```
Input Video Stream
       ↓
Face Detection (MediaPipe)
       ↓
Feature Extraction
       ├─→ Spatial Features (MobileNetV3 + ECA + CBAM)
       └─→ Temporal Features (EAR/MAR → LSTM)
       ↓
Feature Fusion
       ↓
Classification (Normal/Drowsy/Very Drowsy)
       ↓
       ├─→ GUI Display
       ├─→ Voice Alert
       └─→ Database Logging
```

## Model Components

### 1. MobileNetV3 Backbone

**Purpose**: Lightweight feature extraction for real-time performance

**Key Features**:
- Efficient inverted residual blocks
- Hard-swish activation
- Squeeze-and-excitation modules
- Optimized for mobile/edge devices

**Implementation**: `models/mobilenet_v3.py`

### 2. ECA (Efficient Channel Attention)

**Purpose**: Enhance important feature channels (eyes, mouth)

**Mechanism**:
- Global average pooling
- 1D convolution with adaptive kernel size
- Channel-wise feature recalibration

**Formula**:
```
k = |log₂(C) + b| / γ
where C = number of channels
```

**Implementation**: `models/attention.py:ECALayer`

### 3. CBAM (Convolutional Block Attention Module)

**Purpose**: Focus on important spatial regions and channels

**Components**:
- **Channel Attention**: What to focus on
- **Spatial Attention**: Where to focus

**Implementation**: `models/attention.py:CBAM`

### 4. LSTM Temporal Branch

**Purpose**: Analyze temporal patterns in EAR/MAR sequences

**Input**: Sequence of (EAR, MAR) values over 30 frames

**Architecture**:
- 2-layer LSTM with 128 hidden units
- Dropout for regularization
- Fully connected classifier

**Implementation**: `models/lstm_branch.py`

## Feature Engineering

### Eye Aspect Ratio (EAR)

**Formula**:
```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
```

Where p1-p6 are eye landmark points.

**Threshold**: EAR < 0.25 indicates closed eyes

### Mouth Aspect Ratio (MAR)

**Formula**:
```
MAR = (||p2 - p10|| + ||p4 - p8||) / (2 * ||p0 - p6||)
```

**Threshold**: MAR > 0.6 indicates yawning

## Training Pipeline

### Data Augmentation

Implemented in `data/preprocess.py`:

1. **Geometric Transformations**:
   - Random rotation (±15°)
   - Horizontal flip (50%)

2. **Color Augmentation**:
   - Random brightness (±20%)
   - Random contrast (±20%)

3. **Noise and Occlusion**:
   - Gaussian noise
   - Random coarse dropout (simulates occlusion)

### Training Strategy

**Optimizer**: AdamW
- Learning rate: 0.001
- Weight decay: 0.0001

**Scheduler**: CosineAnnealingLR
- Smooth learning rate decay

**Loss Function**: CrossEntropyLoss

**Batch Size**: 32 (adjustable)

**Epochs**: 100 (with early stopping)

### Training Script

```bash
python train/trainer.py --config config/train_config.yaml
```

## Inference Pipeline

### Real-time Detection Flow

1. **Video Capture**: OpenCV captures frames from webcam
2. **Face Detection**: MediaPipe Face Mesh detects 468 landmarks
3. **Feature Calculation**: Compute EAR and MAR
4. **Sequence Buffering**: Maintain 30-frame history
5. **Model Inference**:
   - Spatial branch processes current frame
   - Temporal branch processes sequence
   - Fusion layer combines predictions
6. **Post-processing**: Apply thresholds and temporal logic
7. **Alert Generation**: Trigger voice alerts if needed
8. **Logging**: Save events to database

### Performance Optimization

**Target Metrics**:
- Inference latency: < 100ms
- FPS: > 10 (real-time)
- Accuracy: > 90%

**Optimization Techniques**:
1. Model quantization (FP32 → FP16/INT8)
2. ONNX export for cross-platform deployment
3. TensorRT acceleration (NVIDIA GPUs)
4. Multi-threading (separate capture and inference)

## Database Schema

### Tables

#### fatigue_events
```sql
CREATE TABLE fatigue_events (
    id INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL,
    fatigue_class TEXT NOT NULL,
    confidence REAL NOT NULL,
    ear REAL,
    mar REAL,
    duration INTEGER,
    screenshot_path TEXT,
    notes TEXT
);
```

#### monitoring_sessions
```sql
CREATE TABLE monitoring_sessions (
    id INTEGER PRIMARY KEY,
    start_time TEXT NOT NULL,
    end_time TEXT,
    total_events INTEGER,
    normal_count INTEGER,
    drowsy_count INTEGER,
    very_drowsy_count INTEGER
);
```

## GUI Components

### Main Window

**Components**:
1. **Video Display**: Live camera feed with overlays
2. **Control Panel**: Start/stop buttons
3. **Info Panel**: Real-time EAR, MAR, classification
4. **Event Log**: Scrolling log of detections

**Threading**:
- Main thread: GUI event loop
- Video thread: Capture and inference
- Alert thread: Voice synthesis

### Implementation

File: `gui/main_window.py`

Key classes:
- `MainWindow`: Main application window
- `VideoThread`: Background video processing

## Alert System

### Voice Engine

**Library**: pyttsx3 (cross-platform TTS)

**Features**:
- Configurable speech rate and volume
- Alert cooldown to prevent spam
- Asynchronous speech synthesis

**Alert Messages**:
- Drowsy: "Warning! You appear drowsy. Please take a break."
- Very Drowsy: "Danger! Severe fatigue detected. Please stop and rest immediately."

### Implementation

File: `alerts/voice_engine.py`

## Configuration

### Main Config File

`config/train_config.yaml`

**Sections**:
1. **Training**: Hyperparameters
2. **Model**: Architecture options
3. **Data**: Dataset paths and splits
4. **Augmentation**: Data augmentation settings
5. **Thresholds**: Detection thresholds
6. **Logging**: Output directories

## Testing

### Unit Tests

File: `tests/test_models.py`

**Tests**:
1. ECA layer forward pass
2. CBAM forward pass
3. MobileNetV3 with attention
4. LSTM branch
5. Hybrid model
6. Forward and backward pass
7. Parameter counting

### Run Tests

```bash
python tests/test_models.py
```

## Deployment

### Export to ONNX

```bash
python deployment/onnx_export.py \
    --model models/checkpoints/best_model.pth \
    --output models/checkpoints/model.onnx
```

### ONNX Runtime Inference

Benefits:
- Cross-platform compatibility
- Optimized inference
- Smaller model size

### TensorRT (NVIDIA GPUs)

For maximum performance on NVIDIA hardware:
1. Export to ONNX
2. Convert ONNX to TensorRT engine
3. Use TensorRT runtime for inference

## Performance Benchmarks

### Expected Performance

**Hardware**: Intel i7 CPU
- FPS: 15-20
- Latency: 50-70ms

**Hardware**: NVIDIA RTX 3060
- FPS: 60+
- Latency: 15-20ms

### Model Size

- PyTorch (.pth): ~15 MB
- ONNX (.onnx): ~12 MB
- TensorRT (.engine): ~8 MB

## Troubleshooting

### Common Issues

1. **Low FPS**:
   - Reduce image size
   - Disable LSTM branch
   - Use GPU acceleration

2. **High False Positives**:
   - Adjust EAR/MAR thresholds
   - Increase consecutive frame requirement
   - Collect more training data

3. **Memory Issues**:
   - Reduce batch size
   - Use model quantization
   - Clear old logs

## Future Improvements

1. **Model Enhancements**:
   - Transformer-based temporal modeling
   - Multi-task learning (head pose, gaze)
   - Self-supervised pre-training

2. **Features**:
   - Multi-person detection
   - Cloud synchronization
   - Mobile app version

3. **Performance**:
   - Model pruning
   - Knowledge distillation
   - Edge TPU deployment

## References

See task document for complete reference list.

Key papers:
1. MobileNetV3 (Howard et al., 2019)
2. ECA-Net (Zhang et al., 2020)
3. CBAM (Woo et al., 2018)
4. NTHU-DDD Dataset (Chen et al., 2020)

## License

Academic use only. See LICENSE file for details.

## Contact

For technical questions or contributions, please refer to the project repository.
