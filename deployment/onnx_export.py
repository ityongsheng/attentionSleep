"""
Export PyTorch model to ONNX format for deployment
"""
import torch
import sys
from pathlib import Path
import argparse

sys.path.append(str(Path(__file__).parent.parent))

from models.hybrid_model import HybridFatigueDetector


def export_to_onnx(model_path, output_path, input_size=(1, 3, 224, 224)):
    """
    Export PyTorch model to ONNX format

    Args:
        model_path: Path to trained PyTorch model
        output_path: Path to save ONNX model
        input_size: Input tensor size (batch, channels, height, width)
    """
    # Load model
    device = torch.device('cpu')
    model = HybridFatigueDetector(
        num_classes=3,
        pretrained=False,
        use_eca=True,
        use_cbam=True,
        use_lstm=False  # Export spatial model only for simplicity
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(input_size).to(device)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"Model exported to {output_path}")

    # Verify ONNX model
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verified successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to PyTorch model checkpoint'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/checkpoints/model.onnx',
        help='Output path for ONNX model'
    )
    args = parser.parse_args()

    export_to_onnx(args.model, args.output)
