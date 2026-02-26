"""
Real-time fatigue detector using trained model
"""
import torch
import cv2
import numpy as np
from collections import deque
import sys
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

sys.path.append(str(Path(__file__).parent.parent))

from models.hybrid_model import HybridFatigueDetector
from data.preprocess import calculate_ear, calculate_mar

# Path to the FaceLandmarker task model file
_FACE_LANDMARKER_MODEL = str(Path(__file__).parent.parent / 'models' / 'face_landmarker.task')


class FatigueDetector:
    """Real-time fatigue detection system"""

    def __init__(self, model_path, config, device='cpu'):
        self.device = torch.device(device)
        self.config = config

        # Load model
        self.model = self._load_model(model_path)

        # Initialize MediaPipe FaceLandmarker (new Tasks API)
        base_options = mp_python.BaseOptions(model_asset_path=_FACE_LANDMARKER_MODEL)
        face_landmarker_options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.face_landmarker = mp_vision.FaceLandmarker.create_from_options(face_landmarker_options)

        # Temporal buffers for LSTM
        self.sequence_length = config['model']['sequence_length']
        self.ear_buffer = deque(maxlen=self.sequence_length)
        self.mar_buffer = deque(maxlen=self.sequence_length)

        # Fatigue state tracking
        self.ear_consecutive_frames = 0
        self.mar_consecutive_frames = 0

        # Thresholds
        self.ear_threshold = config['thresholds']['ear_threshold']
        self.mar_threshold = config['thresholds']['mar_threshold']
        self.ear_frame_threshold = config['thresholds']['ear_consecutive_frames']
        self.mar_frame_threshold = config['thresholds']['mar_consecutive_frames']

        # Class names
        self.class_names = ['Normal', 'Drowsy', 'Very Drowsy']

    def _load_model(self, model_path):
        """Load trained model"""
        model = HybridFatigueDetector(
            num_classes=self.config['model']['num_classes'],
            pretrained=False,
            use_eca=self.config['model']['use_eca'],
            use_cbam=self.config['model']['use_cbam'],
            use_lstm=self.config['model']['use_lstm'],
            lstm_hidden_size=self.config['model']['lstm_hidden_size']
        ).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.float()  # Ensure model weights are float32 to match input tensors
        model.eval()

        return model


    def _preprocess_image(self, image):
        """Preprocess image for model input"""
        # Resize
        img = cv2.resize(image, (224, 224))

        # Normalize
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

        # Convert to tensor and match model's dtype (handles float32/float64 mismatch)
        model_dtype = next(self.model.parameters()).dtype
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(dtype=model_dtype)

        return img.to(self.device)

    def _extract_landmarks(self, image):
        """Extract facial landmarks using MediaPipe FaceLandmarker (new Tasks API)"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        results = self.face_landmarker.detect(mp_image)

        if not results.face_landmarks:
            return None

        landmarks = results.face_landmarks[0]

        # Convert to numpy array
        h, w = image.shape[:2]
        points = []
        for landmark in landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            points.append([x, y])

        return np.array(points)

    def _calculate_features(self, landmarks):
        """Calculate EAR and MAR from landmarks"""
        # Eye landmarks (MediaPipe indices)
        left_eye_indices = [33, 160, 158, 133, 153, 144]
        right_eye_indices = [362, 385, 387, 263, 373, 380]

        # Mouth landmarks
        mouth_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]

        # Extract eye landmarks
        left_eye = landmarks[left_eye_indices]
        right_eye = landmarks[right_eye_indices]

        # Calculate EAR
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Extract mouth landmarks
        mouth = landmarks[mouth_indices]

        # Calculate MAR
        mar = calculate_mar(mouth)

        return ear, mar

    def detect(self, frame):
        """
        Detect fatigue state from a single frame

        Args:
            frame: Input image (BGR format)

        Returns:
            dict with detection results
        """
        # Extract landmarks
        landmarks = self._extract_landmarks(frame)

        if landmarks is None:
            return {
                'status': 'no_face',
                'class': 'Unknown',
                'confidence': 0.0,
                'ear': 0.0,
                'mar': 0.0,
                'alert': False
            }

        # Calculate features
        ear, mar = self._calculate_features(landmarks)

        # Update buffers
        self.ear_buffer.append(ear)
        self.mar_buffer.append(mar)

        # Check thresholds
        if ear < self.ear_threshold:
            self.ear_consecutive_frames += 1
        else:
            self.ear_consecutive_frames = 0

        if mar > self.mar_threshold:
            self.mar_consecutive_frames += 1
        else:
            self.mar_consecutive_frames = 0

        # Determine alert status
        alert = (
            self.ear_consecutive_frames >= self.ear_frame_threshold or
            self.mar_consecutive_frames >= self.mar_frame_threshold
        )

        # Preprocess image for model
        img_tensor = self._preprocess_image(frame)

        # Prepare sequence for LSTM
        sequence = None
        if len(self.ear_buffer) == self.sequence_length:
            ear_seq = np.array(list(self.ear_buffer))
            mar_seq = np.array(list(self.mar_buffer))
            sequence = np.stack([ear_seq, mar_seq], axis=1)
            sequence = torch.from_numpy(sequence).unsqueeze(0).float().to(self.device)

        # Model inference
        with torch.no_grad():
            outputs = self.model(img_tensor, sequence)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        predicted_class = self.class_names[predicted.item()]
        confidence_value = confidence.item()

        return {
            'status': 'success',
            'class': predicted_class,
            'confidence': confidence_value,
            'ear': ear,
            'mar': mar,
            'alert': alert,
            'landmarks': landmarks
        }

    def release(self):
        """Release resources"""
        self.face_landmarker.close()
