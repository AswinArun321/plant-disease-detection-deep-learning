"""
Tests for image preprocessing pipeline.
"""
import os
import sys
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing import preprocess_image_array, load_and_preprocess_image

def test_preprocess_image_array_normalization():
    # Test [0, 255] RGB array
    raw = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
    
    # mobilenet_v2: [-1, 1]
    norm_mob = preprocess_image_array(raw, model_type="mobilenet_v2")
    assert norm_mob.shape == (224, 224, 3)
    assert norm_mob.min() >= -1.05 and norm_mob.max() <= 1.05

    # standard: [0, 1]
    norm_std = preprocess_image_array(raw, model_type="standard")
    assert norm_std.shape == (224, 224, 3)
    assert norm_std.min() >= 0.0 and norm_std.max() <= 1.0

def test_load_and_preprocess_pil_image():
    # Test loading from PIL Image
    pil_img = Image.new("RGB", (300, 400), color=(100, 150, 200))
    batch_tensor, resized_pil = load_and_preprocess_image(pil_img, target_size=(224, 224))
    
    assert batch_tensor.shape == (1, 224, 224, 3)
    assert resized_pil.size == (224, 224)
    assert resized_pil.mode == "RGB"

def test_load_and_preprocess_grayscale_conversion():
    # Test converting 1-channel grayscale to 3-channel RGB
    gray_img = Image.new("L", (150, 150), color=128)
    batch_tensor, resized_pil = load_and_preprocess_image(gray_img, target_size=(224, 224))
    
    assert batch_tensor.shape == (1, 224, 224, 3)
    assert resized_pil.mode == "RGB"

def test_invalid_source_raises_error():
    with pytest.raises(ValueError):
        load_and_preprocess_image(12345)
