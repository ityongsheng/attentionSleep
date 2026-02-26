"""
Data preprocessing and augmentation utilities
"""
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from pathlib import Path
from tqdm import tqdm


class DataAugmentation:
    """Data augmentation pipeline for fatigue detection"""

    def __init__(self, image_size=(224, 224), is_training=True):
        if is_training:
            self.transform = A.Compose([
                A.Resize(image_size[0], image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.Rotate(limit=15, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=20,
                    max_width=20,
                    p=0.3
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(image_size[0], image_size[1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])

    def __call__(self, image):
        return self.transform(image=image)['image']


def extract_frames_from_video(video_path, output_dir, frame_interval=5):
    """
    Extract frames from video file

    Args:
        video_path: Path to video file
        output_dir: Directory to save extracted frames
        frame_interval: Extract every N frames
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with tqdm(total=total_frames // frame_interval) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frame_path = os.path.join(
                    output_dir,
                    f"frame_{saved_count:06d}.jpg"
                )
                cv2.imwrite(frame_path, frame)
                saved_count += 1
                pbar.update(1)

            frame_count += 1

    cap.release()
    print(f"Extracted {saved_count} frames from {video_path}")


def calculate_ear(eye_landmarks):
    """
    Calculate Eye Aspect Ratio (EAR)

    Args:
        eye_landmarks: Array of 6 (x, y) coordinates for eye landmarks
    Returns:
        EAR value
    """
    # Vertical distances
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])

    # Horizontal distance
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])

    # EAR formula
    ear = (A + B) / (2.0 * C)
    return ear


def calculate_mar(mouth_landmarks):
    """
    Calculate Mouth Aspect Ratio (MAR)

    Args:
        mouth_landmarks: Array of mouth landmark coordinates
    Returns:
        MAR value
    """
    # Vertical distances
    A = np.linalg.norm(mouth_landmarks[2] - mouth_landmarks[10])
    B = np.linalg.norm(mouth_landmarks[4] - mouth_landmarks[8])

    # Horizontal distance
    C = np.linalg.norm(mouth_landmarks[0] - mouth_landmarks[6])

    # MAR formula
    mar = (A + B) / (2.0 * C)
    return mar


if __name__ == "__main__":
    # Example usage
    print("Data preprocessing utilities loaded")
