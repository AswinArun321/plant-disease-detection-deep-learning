"""
Model architecture builders and training pipelines for Baseline CNN and Transfer Learning models.
"""
import os
import time
import json
import numpy as np
import tensorflow as tf
from src import config
from src.data_loader import get_dataset_splits, create_tf_dataset
from src.augmentation import get_augmentation_layer
from src.utils import plot_training_history, save_json

def build_baseline_cnn(input_shape=config.INPUT_SHAPE, num_classes=config.NUM_CLASSES):
    """
    Standard Convolutional Neural Network trained from scratch as a baseline.
    """
    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    # Rescaling [0, 255] to [0, 1]
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)
    
    # Block 1
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    # Block 2
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    # Block 3
    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    # Classification Head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="baseline_cnn")
    return model

def build_mobilenet_v2(input_shape=config.INPUT_SHAPE, num_classes=config.NUM_CLASSES, unfreeze_layers=20):
    """
    Transfer learning with MobileNetV2.
    """
    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    # MobileNetV2 preprocessing: scales to [-1, 1]
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    
    if unfreeze_layers > 0:
        base_model.trainable = True
        for layer in base_model.layers[:-unfreeze_layers]:
            layer.trainable = False
    else:
        base_model.trainable = False

    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mobilenet_v2")
    return model

def build_resnet50(input_shape=config.INPUT_SHAPE, num_classes=config.NUM_CLASSES):
    """
    Transfer learning with ResNet50.
    """
    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    x = tf.keras.applications.resnet50.preprocess_input(inputs)
    
    base_model = tf.keras.applications.ResNet50(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False
    
    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="resnet50")
    return model

def build_efficientnet_b0(input_shape=config.INPUT_SHAPE, num_classes=config.NUM_CLASSES, unfreeze_layers=20):
    """
    Transfer learning with EfficientNetB0.
    """
    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    
    if unfreeze_layers > 0:
        base_model.trainable = True
        for layer in base_model.layers[:-unfreeze_layers]:
            layer.trainable = False
    else:
        base_model.trainable = False

    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="efficientnet_b0")
    return model

def train_model(model, train_ds, val_ds, epochs=10, lr=1e-3, checkpoint_path=None):
    """
    Compile and fit a Keras model with EarlyStopping, ReduceLROnPlateau, and Checkpoints.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    if checkpoint_path:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1
            )
        )

    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )
    training_time = time.time() - start_time
    
    return history, training_time
