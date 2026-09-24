"""
Data augmentation pipeline for plant leaf imagery.
"""
import tensorflow as tf

def get_augmentation_layer():
    """
    Returns a Keras Sequential layer applying realistic leaf variations:
    - Horizontal and vertical flips (leaves have natural orientation invariance)
    - Small rotation (+/- 15 degrees)
    - Slight zoom (+/- 10%)
    - Subtle contrast adjustment
    """
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(factor=0.08),  # approx +/- 15-20 degrees
        tf.keras.layers.RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1)),
        tf.keras.layers.RandomContrast(factor=0.1)
    ], name="leaf_augmentation")
    
    return data_augmentation

def apply_augmentation(image_tensor):
    """Functional wrapper for tf.data map operations."""
    aug_layer = get_augmentation_layer()
    return aug_layer(image_tensor, training=True)
