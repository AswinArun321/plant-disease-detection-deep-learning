"""
Tests for deep learning model architectures.
"""
import os
import sys
import numpy as np
import pytest
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.train import build_baseline_cnn, build_mobilenet_v2

def test_baseline_cnn_architecture():
    model = build_baseline_cnn(input_shape=(224, 224, 3), num_classes=38)
    assert model.name == "baseline_cnn"
    assert model.output_shape == (None, 38)
    
    # Test forward pass with random batch
    dummy_input = np.random.randn(2, 224, 224, 3).astype(np.float32)
    preds = model(dummy_input)
    assert preds.shape == (2, 38)
    # Check probabilities sum to 1.0 (softmax)
    sums = np.sum(preds.numpy(), axis=1)
    np.testing.assert_allclose(sums, np.ones(2), atol=1e-5)

def test_mobilenet_v2_architecture():
    model = build_mobilenet_v2(input_shape=(224, 224, 3), num_classes=38, unfreeze_layers=0)
    assert model.name == "mobilenet_v2"
    assert model.output_shape == (None, 38)
    
    dummy_input = np.random.randn(1, 224, 224, 3).astype(np.float32)
    preds = model(dummy_input)
    assert preds.shape == (1, 38)
    np.testing.assert_allclose(np.sum(preds.numpy()), 1.0, atol=1e-5)
