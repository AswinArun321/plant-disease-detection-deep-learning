"""
Image preprocessing utilities for inference and training.
"""
import io
import numpy as np
from PIL import Image
import tensorflow as tf
from src import config

def preprocess_image_array(img_array, model_type="mobilenet_v2"):
    """
    Apply model-specific normalization:
    - mobilenet_v2: tf.keras.applications.mobilenet_v2.preprocess_input (scales to [-1, 1])
    - efficientnet: tf.keras.applications.efficientnet.preprocess_input
    - resnet50: tf.keras.applications.resnet50.preprocess_input
    - standard: scale [0, 255] to [0, 1]
    """
    x = img_array.astype(np.float32)
    if model_type == "mobilenet_v2":
        return tf.keras.applications.mobilenet_v2.preprocess_input(x)
    elif model_type == "efficientnet":
        return tf.keras.applications.efficientnet.preprocess_input(x)
    elif model_type == "resnet50":
        return tf.keras.applications.resnet50.preprocess_input(x)
    else:
        return x / 255.0

def load_and_preprocess_image(source, target_size=config.IMAGE_SIZE, model_type="mobilenet_v2"):
    """
    Accepts:
      - File path (str)
      - Raw bytes (bytes)
      - PIL Image object
    Returns:
      (batch_tensor, rgb_pil_image)
    """
    if isinstance(source, str):
        pil_img = Image.open(source)
    elif isinstance(source, bytes):
        pil_img = Image.open(io.BytesIO(source))
    elif isinstance(source, Image.Image):
        pil_img = source
    else:
        raise ValueError(f"Unsupported image source type: {type(source)}")

    # Ensure RGB
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    # Resize with high quality resampling
    resized_pil = pil_img.resize(target_size, Image.Resampling.BILINEAR)
    img_array = np.array(resized_pil)

    # Normalize
    processed_array = preprocess_image_array(img_array, model_type=model_type)
    batch_tensor = np.expand_dims(processed_array, axis=0)

    return batch_tensor, resized_pil
