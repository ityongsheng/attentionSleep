"""
Training script for fatigue detection model
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import yaml
import os
import sys
from pathlib import Path
from tqdm import tqdm
import argparse

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.hybrid_model import HybridFatigueDetector
from data.dataset import FatigueDataset
from data.preprocess import DataAugmentation


class Trainer:
    """Training manager for fatigue detection model"""

    def __init__(self, config_path):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Setup device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

        # Create model
        self.model = HybridFatigueDetector(
            num_classes=self.config['model']['num_classes'],
            pretrained=self.config['model']['pretrained'],
            use_eca=self.config['model']['use_eca'],
            use_cbam=self.config['model']['use_cbam'],
            use_lstm=self.config['model']['use_lstm'],
            lstm_hidden_size=self.config['model']['lstm_hidden_size']
        ).to(self.device)

        # Setup data
        self._setup_data()

        # Setup training components
        self._setup_training()

        # Setup logging
        self.writer = SummaryWriter(self.config['logging']['log_dir'])

    def _setup_data(self):
        """Setup datasets and dataloaders"""
        # Transforms
        train_transform = DataAugmentation(
            image_size=tuple(self.config['data']['image_size']),
            is_training=True
        )
        val_transform = DataAugmentation(
            image_size=tuple(self.config['data']['image_size']),
            is_training=False
        )

        # Full dataset
        full_dataset = FatigueDataset(
            data_dir=self.config['data']['dataset_path'],
            transform=None
        )

        # Split dataset
        train_size = int(self.config['data']['train_split'] * len(full_dataset))
        val_size = int(self.config['data']['val_split'] * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size]
        )

        # Apply transforms
        train_dataset.dataset.transform = train_transform
        val_dataset.dataset.transform = val_transform
        test_dataset.dataset.transform = val_transform

        # Dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['training']['num_workers']
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['training']['num_workers']
        )

        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")

    def _setup_training(self):
        """Setup optimizer, scheduler, and loss function"""
        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        if self.config['training']['optimizer'] == 'AdamW':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['training']['learning_rate']
            )

        # Scheduler
        if self.config['training']['scheduler'] == 'CosineAnnealingLR':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs']
            )

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc': 100. * correct / total
            })

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total

        return val_loss, val_acc

    def train(self):
        """Main training loop"""
        best_acc = 0.0

        for epoch in range(1, self.config['training']['epochs'] + 1):
            print(f"\nEpoch {epoch}/{self.config['training']['epochs']}")

            # Train
            train_loss, train_acc = self.train_epoch(epoch)

            # Validate
            val_loss, val_acc = self.validate(epoch)

            # Log to tensorboard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)

            # Print results
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Save checkpoint
            if val_acc > best_acc:
                best_acc = val_acc
                checkpoint_path = os.path.join(
                    self.config['logging']['checkpoint_dir'],
                    'best_model.pth'
                )
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, checkpoint_path)
                print(f"Saved best model with accuracy: {best_acc:.2f}%")

            # Step scheduler
            if hasattr(self, 'scheduler'):
                self.scheduler.step()

        print(f"\nTraining completed! Best validation accuracy: {best_acc:.2f}%")
        self.writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default='config/train_config.yaml',
        help='Path to config file'
    )
    args = parser.parse_args()

    trainer = Trainer(args.config)
    trainer.train()
