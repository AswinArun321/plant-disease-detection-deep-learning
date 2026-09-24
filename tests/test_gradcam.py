"""
Tests for Grad-CAM Explainable AI generator.
"""
import os
import sys
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.train import build_baseline_cnn
from src.gradcam import find_last_conv_layer, compute_gradcam, overlay_heatmap

def test_find_last_conv_layer():
    model = build_baseline_cnn()
    last_conv = find_last_conv_layer(model)
    assert last_conv is not None
    assert "conv" in last_conv.lower()

def test_compute_gradcam_and_overlay():
    model = build_baseline_cnn()
    img_tensor = np.random.randn(1, 224, 224, 3).astype(np.float32)
    
    # Compute heatmap
    heatmap = compute_gradcam(model, img_tensor, class_idx=0)
    assert heatmap.ndim == 2
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.05

    # Test overlay onto image
    dummy_pil = Image.new("RGB", (224, 224), color=(50, 150, 50))
    superimposed, resized_hm = overlay_heatmap(dummy_pil, heatmap)
    
    assert superimposed.shape == (224, 224, 3)
    assert superimposed.dtype == np.uint8
    assert resized_hm.shape == (224, 224)
