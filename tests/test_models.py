"""
Test script for model components
"""
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from models.attention import ECALayer, CBAM
from models.mobilenet_v3 import MobileNetV3WithAttention
from models.lstm_branch import LSTMBranch
from models.hybrid_model import HybridFatigueDetector


def test_eca_layer():
    """Test ECA attention layer"""
    print("Testing ECA Layer...")
    eca = ECALayer(channels=64)
    x = torch.randn(2, 64, 56, 56)
    out = eca(x)
    assert out.shape == x.shape, "ECA output shape mismatch"
    print(f"✓ ECA Layer: Input {x.shape} -> Output {out.shape}")


def test_cbam():
    """Test CBAM attention module"""
    print("\nTesting CBAM...")
    cbam = CBAM(channels=64)
    x = torch.randn(2, 64, 56, 56)
    out = cbam(x)
    assert out.shape == x.shape, "CBAM output shape mismatch"
    print(f"✓ CBAM: Input {x.shape} -> Output {out.shape}")


def test_mobilenet_with_attention():
    """Test MobileNetV3 with attention"""
    print("\nTesting MobileNetV3 with Attention...")
    model = MobileNetV3WithAttention(num_classes=3, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 3), "MobileNetV3 output shape mismatch"
    print(f"✓ MobileNetV3: Input {x.shape} -> Output {out.shape}")


def test_lstm_branch():
    """Test LSTM branch"""
    print("\nTesting LSTM Branch...")
    lstm = LSTMBranch(input_size=2, hidden_size=128, num_layers=2, num_classes=3)
    x = torch.randn(2, 30, 2)  # batch, sequence, features
    out = lstm(x)
    assert out.shape == (2, 3), "LSTM output shape mismatch"
    print(f"✓ LSTM Branch: Input {x.shape} -> Output {out.shape}")


def test_hybrid_model():
    """Test hybrid model"""
    print("\nTesting Hybrid Model...")
    model = HybridFatigueDetector(
        num_classes=3,
        pretrained=False,
        use_eca=True,
        use_cbam=True,
        use_lstm=True
    )

    # Test with image only
    image = torch.randn(2, 3, 224, 224)
    out = model(image)
    assert out.shape == (2, 3), "Hybrid model output shape mismatch (image only)"
    print(f"✓ Hybrid Model (image only): Input {image.shape} -> Output {out.shape}")

    # Test with image and sequence
    sequence = torch.randn(2, 30, 2)
    out = model(image, sequence)
    assert out.shape == (2, 3), "Hybrid model output shape mismatch (with sequence)"
    print(f"✓ Hybrid Model (with sequence): Image {image.shape}, Seq {sequence.shape} -> Output {out.shape}")


def test_model_parameters():
    """Test model parameter count"""
    print("\nTesting Model Parameters...")
    model = HybridFatigueDetector(num_classes=3, pretrained=False)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")


def test_forward_backward():
    """Test forward and backward pass"""
    print("\nTesting Forward and Backward Pass...")
    model = HybridFatigueDetector(num_classes=3, pretrained=False)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Forward pass
    image = torch.randn(2, 3, 224, 224)
    labels = torch.tensor([0, 1])

    outputs = model(image)
    loss = criterion(outputs, labels)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"✓ Forward pass: Loss = {loss.item():.4f}")
    print(f"✓ Backward pass: Gradients computed successfully")


def main():
    """Run all tests"""
    print("=" * 60)
    print("Model Component Tests")
    print("=" * 60)

    try:
        test_eca_layer()
        test_cbam()
        test_mobilenet_with_attention()
        test_lstm_branch()
        test_hybrid_model()
        test_model_parameters()
        test_forward_backward()

        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
