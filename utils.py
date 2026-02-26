"""
Utility functions for the fatigue detection system
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


def visualize_attention_maps(model, image, layer_name):
    """
    Visualize attention maps from the model

    Args:
        model: Trained model
        image: Input image
        layer_name: Name of the layer to visualize

    Returns:
        Attention map visualization
    """
    # This is a placeholder for attention visualization
    # Implement Grad-CAM or similar technique
    pass


def plot_training_history(history, save_path=None):
    """
    Plot training history

    Args:
        history: Dictionary with training metrics
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot confusion matrix

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def calculate_metrics(y_true, y_pred, class_names):
    """
    Calculate and print classification metrics

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names

    Returns:
        Classification report as string
    """
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)
    return report


def draw_facial_landmarks(image, landmarks, color=(0, 255, 0)):
    """
    Draw facial landmarks on image

    Args:
        image: Input image
        landmarks: Array of landmark coordinates
        color: Color for drawing

    Returns:
        Image with landmarks drawn
    """
    img_copy = image.copy()

    for point in landmarks:
        cv2.circle(img_copy, tuple(point.astype(int)), 2, color, -1)

    return img_copy


def create_video_from_frames(frame_dir, output_path, fps=30):
    """
    Create video from image frames

    Args:
        frame_dir: Directory containing frames
        output_path: Output video path
        fps: Frames per second
    """
    from pathlib import Path
    import glob

    frame_files = sorted(glob.glob(str(Path(frame_dir) / "*.jpg")))

    if not frame_files:
        print("No frames found")
        return

    # Read first frame to get dimensions
    first_frame = cv2.imread(frame_files[0])
    h, w = first_frame.shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        out.write(frame)

    out.release()
    print(f"Video saved to {output_path}")


def benchmark_model(model, test_loader, device):
    """
    Benchmark model performance

    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run on

    Returns:
        Dictionary with performance metrics
    """
    import time
    import torch

    model.eval()
    total_time = 0
    num_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)

            start_time = time.time()
            outputs = model(images)
            end_time = time.time()

            total_time += (end_time - start_time)
            num_samples += images.size(0)

    avg_time = total_time / num_samples
    fps = 1.0 / avg_time

    return {
        'avg_inference_time': avg_time * 1000,  # ms
        'fps': fps,
        'total_samples': num_samples
    }


if __name__ == "__main__":
    print("Utility functions loaded")
