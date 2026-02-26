"""
Custom Dataset for Fatigue Detection
"""
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
from pathlib import Path
import json


class FatigueDataset(Dataset):
    """Dataset for fatigue detection training"""

    def __init__(self, data_dir, transform=None, sequence_length=30):
        """
        Args:
            data_dir: Root directory containing images and labels
            transform: Data augmentation transforms
            sequence_length: Length of temporal sequence for LSTM
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.sequence_length = sequence_length

        # Load annotations
        self.samples = self._load_annotations()

        # Class mapping
        self.class_to_idx = {
            'normal': 0,
            'drowsy': 1,
            'very_drowsy': 2
        }

    def _load_annotations(self):
        """Load image paths and labels"""
        samples = []

        # Assuming structure: data_dir/class_name/images
        for class_dir in self.data_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                for img_path in class_dir.glob('*.jpg'):
                    samples.append({
                        'image_path': str(img_path),
                        'class': class_name
                    })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        image = cv2.imread(sample['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Get label
        label = self.class_to_idx[sample['class']]

        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long)
        }


class FatigueSequenceDataset(Dataset):
    """Dataset with temporal sequences for LSTM training"""

    def __init__(self, data_dir, transform=None, sequence_length=30):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.sequence_length = sequence_length

        # Load sequence data (EAR, MAR values)
        self.sequences = self._load_sequences()

    def _load_sequences(self):
        """Load pre-computed EAR/MAR sequences"""
        sequences = []

        # Load from JSON or CSV files
        sequence_file = self.data_dir / 'sequences.json'
        if sequence_file.exists():
            with open(sequence_file, 'r') as f:
                sequences = json.load(f)

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_data = self.sequences[idx]

        # Extract EAR and MAR values
        ear_values = np.array(seq_data['ear'])
        mar_values = np.array(seq_data['mar'])

        # Combine into sequence
        sequence = np.stack([ear_values, mar_values], axis=1)

        # Pad or truncate to sequence_length
        if len(sequence) < self.sequence_length:
            padding = np.zeros((self.sequence_length - len(sequence), 2))
            sequence = np.vstack([sequence, padding])
        else:
            sequence = sequence[:self.sequence_length]

        label = seq_data['label']

        return {
            'sequence': torch.tensor(sequence, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long)
        }
