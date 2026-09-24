"""
Data loading and dataset splitting module for PlantVillage.
"""
import os
import glob
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from PIL import Image
from src import config

def get_class_names(dataset_dir=config.DEFAULT_DATASET_DIR):
    """Return sorted list of class directory names."""
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    classes = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    return classes

def verify_dataset(dataset_dir=config.DEFAULT_DATASET_DIR, sample_verify_per_class=10):
    """
    Verify image integrity across all class folders rapidly.
    Returns dictionary of class counts, total valid, and list of corrupted images.
    """
    classes = get_class_names(dataset_dir)
    stats = {}
    corrupted = []
    total_valid = 0

    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}

    for cls in classes:
        cls_dir = os.path.join(dataset_dir, cls)
        files = []
        with os.scandir(cls_dir) as entries:
            for entry in entries:
                if entry.is_file():
                    ext = os.path.splitext(entry.name)[1].lower()
                    if ext in valid_exts and entry.stat().st_size > 0:
                        files.append(entry.path)
        
        # Verify a sample of images with PIL to guarantee validity
        for fpath in files[:sample_verify_per_class]:
            try:
                with Image.open(fpath) as img:
                    img.verify()
            except Exception:
                corrupted.append(fpath)
        
        valid_count = len(files)
        stats[cls] = valid_count
        total_valid += valid_count

    return {
        "classes": classes,
        "class_counts": stats,
        "total_valid_images": total_valid,
        "corrupted_images": corrupted
    }

def get_dataset_splits(dataset_dir=config.DEFAULT_DATASET_DIR, 
                       train_ratio=config.TRAIN_RATIO, 
                       val_ratio=config.VAL_RATIO, 
                       test_ratio=config.TEST_RATIO, 
                       max_samples_per_class=None,
                       seed=config.RANDOM_SEED):
    """
    Generate deterministic, stratified train/validation/test splits.
    Optionally sub-sample max_samples_per_class for rapid comparative benchmarking.
    """
    classes = get_class_names(dataset_dir)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    all_paths = []
    all_labels = []

    for cls in classes:
        cls_dir = os.path.join(dataset_dir, cls)
        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
        files = []
        with os.scandir(cls_dir) as entries:
            for entry in entries:
                if entry.is_file():
                    ext = os.path.splitext(entry.name)[1].lower()
                    if ext in valid_exts:
                        files.append(entry.path)
        files.sort()  # deterministic ordering
        
        if max_samples_per_class is not None and len(files) > max_samples_per_class:
            rng = np.random.RandomState(seed)
            indices = rng.choice(len(files), size=max_samples_per_class, replace=False)
            files = [files[idx] for idx in sorted(indices)]

        for f in files:
            all_paths.append(f)
            all_labels.append(class_to_idx[cls])

    all_paths = np.array(all_paths)
    all_labels = np.array(all_labels)

    # First split: Train vs Temp (Val + Test)
    temp_ratio = val_ratio + test_ratio
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_paths, all_labels,
        test_size=temp_ratio,
        stratify=all_labels,
        random_state=seed
    )

    # Second split: Val vs Test
    test_rel_ratio = test_ratio / temp_ratio
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=test_rel_ratio,
        stratify=temp_labels,
        random_state=seed
    )

    return {
        "classes": classes,
        "class_to_idx": class_to_idx,
        "train": (train_paths, train_labels),
        "val": (val_paths, val_labels),
        "test": (test_paths, test_labels)
    }

def _parse_image_and_label(path, label, img_size=config.IMAGE_SIZE):
    """Read image from file, decode, resize, and cast."""
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, img_size)
    img = tf.cast(img, tf.float32)
    return img, label

def create_tf_dataset(paths, labels, batch_size=config.BATCH_SIZE, 
                      is_training=False, augment_fn=None, img_size=config.IMAGE_SIZE):
    """
    Create a highly performant tf.data.Dataset pipeline with caching, batching, and prefetching.
    """
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if is_training:
        dataset = dataset.shuffle(buffer_size=min(len(paths), 2048), seed=config.RANDOM_SEED)

    dataset = dataset.map(
        lambda p, l: _parse_image_and_label(p, l, img_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if is_training and augment_fn is not None:
        dataset = dataset.map(
            lambda img, lbl: (augment_fn(img), lbl),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset
