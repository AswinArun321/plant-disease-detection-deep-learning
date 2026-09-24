"""
Explainable AI (XAI) Grad-CAM implementation for deep learning models.
"""
import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

def find_last_conv_layer(model):
    """
    Search recursively or directly for the last 4D convolutional layer.
    """
    # If model has nested base model (e.g. Sequential with MobileNetV2 base)
    for layer in reversed(model.layers):
        if hasattr(layer, "layers"):  # Nested model
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                    return sub_layer.name
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            return layer.name
            
    # Fallback to string matching
    for layer in reversed(model.layers):
        if "conv" in layer.name.lower():
            return layer.name
            
    raise ValueError("Could not automatically identify a convolutional layer in the model.")

def compute_gradcam(model, img_tensor, class_idx=None, conv_layer_name=None):
    """
    Compute Grad-CAM heatmap for a given input tensor and class index.
    img_tensor: numpy array or tf.Tensor of shape (1, H, W, 3)
    """
    if conv_layer_name is None:
        conv_layer_name = find_last_conv_layer(model)

    # Check if target layer is in outer model or inner base model
    target_layer = None
    try:
        target_layer = model.get_layer(conv_layer_name)
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[target_layer.output, model.output]
        )
    except Exception:
        # If target is nested inside a sub-model (like layer.layers)
        for layer in model.layers:
            if hasattr(layer, "layers"):
                try:
                    target_layer = layer.get_layer(conv_layer_name)
                    # Reconstruct functional pipeline
                    grad_model = tf.keras.Model(
                        inputs=model.inputs,
                        outputs=[target_layer.output, model.output]
                    )
                    break
                except Exception:
                    continue

    if target_layer is None:
        raise ValueError(f"Target layer {conv_layer_name} could not be linked for gradients.")

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        if class_idx is None:
            class_idx = tf.argmax(predictions[0])
        class_score = predictions[:, class_idx]

    # Gradients of class score with respect to convolutional feature map
    grads = tape.gradient(class_score, conv_outputs)
    
    # Global average pooling of gradients gives feature map importance weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight convolutional feature maps
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Apply ReLU to focus only on features positively contributing to the class
    heatmap = tf.maximum(heatmap, 0.0)
    
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
        
    return heatmap.numpy()

def overlay_heatmap(original_image, heatmap, alpha=0.45, colormap=cv2.COLORMAP_JET):
    """
    Overlay Grad-CAM heatmap onto the original image.
    original_image: PIL Image or numpy array (H, W, 3) with RGB values in [0, 255]
    """
    if isinstance(original_image, Image.Image):
        img = np.array(original_image)
    else:
        img = original_image.copy()

    h, w = img.shape[:2]
    # Resize heatmap to match original image dimensions
    heatmap_resized = cv2.resize(heatmap, (w, h))
    
    # Convert heatmap to uint8 [0, 255]
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    
    # Apply colormap (yields BGR)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
    
    # Blend overlay
    superimposed = np.uint8(alpha * colored_heatmap + (1.0 - alpha) * img)
    return superimposed, heatmap_resized

def generate_and_save_gradcam(model, img_tensor, original_image, save_path, class_idx=None, conv_layer_name=None):
    """
    Compute Grad-CAM and save the superimposed visualization to disk.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    heatmap = compute_gradcam(model, img_tensor, class_idx=class_idx, conv_layer_name=conv_layer_name)
    superimposed, _ = overlay_heatmap(original_image, heatmap)
    
    result_pil = Image.fromarray(superimposed)
    result_pil.save(save_path)
    return save_path
