"""
Utility functions for plotting, serialization, metrics, and visualization.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script execution
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from src import config

def save_json(data, filepath):
    """Save dictionary or list to JSON with formatting."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_json(filepath):
    """Load JSON file from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def plot_training_history(history, save_path, title_prefix="Model"):
    """Plot and save training and validation loss and accuracy curves."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history.history["loss"]) + 1)
    
    # Accuracy
    ax1.plot(epochs, history.history["accuracy"], "b-o", label="Training Accuracy", markersize=4)
    if "val_accuracy" in history.history:
        ax1.plot(epochs, history.history["val_accuracy"], "g--s", label="Validation Accuracy", markersize=4)
    ax1.set_title(f"{title_prefix} - Accuracy", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right")
    
    # Loss
    ax1_loss = ax2
    ax1_loss.plot(epochs, history.history["loss"], "r-o", label="Training Loss", markersize=4)
    if "val_loss" in history.history:
        ax1_loss.plot(epochs, history.history["val_loss"], "m--s", label="Validation Loss", markersize=4)
    ax1_loss.set_title(f"{title_prefix} - Loss", fontsize=13, fontweight="bold")
    ax1_loss.set_xlabel("Epoch", fontsize=11)
    ax1_loss.set_ylabel("Cross Entropy Loss", fontsize=11)
    ax1_loss.grid(True, linestyle="--", alpha=0.6)
    ax1_loss.legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_confusion_matrix(y_true, y_pred, classes, save_path, title="Confusion Matrix"):
    """Plot and save high-resolution normalized confusion matrix."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-7)
    
    fig, ax = plt.subplots(figsize=(18, 16))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.YlGnBu)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label"
    )
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_class_distribution(class_counts, save_path):
    """Plot and save class distribution bar chart."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    classes = list(class_counts.keys())
    counts = [class_counts[c] for c in classes]
    
    fig, ax = plt.subplots(figsize=(16, 8))
    bars = ax.barh(range(len(classes)), counts, color="#2e7d32", edgecolor="#1b5e20", alpha=0.85)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel("Number of Images", fontsize=12, fontweight="bold")
    ax.set_title("PlantVillage Dataset - Class Sample Distribution (38 Classes)", fontsize=14, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    
    # Annotate bar values
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 30, bar.get_y() + bar.get_height() / 2, f"{int(w)}", va="center", ha="left", fontsize=8)
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_model_comparison_chart(comparison_df, save_path):
    """Plot and save comparison of models across Accuracy, Precision, Recall, F1."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = comparison_df["model_name"]
    x = np.arange(len(models))
    width = 0.2
    
    ax.bar(x - 1.5 * width, comparison_df["test_accuracy"] * 100, width, label="Accuracy (%)", color="#1976d2")
    ax.bar(x - 0.5 * width, comparison_df["precision"] * 100, width, label="Precision (%)", color="#388e3c")
    ax.bar(x + 0.5 * width, comparison_df["recall"] * 100, width, label="Recall (%)", color="#f57c00")
    ax.bar(x + 1.5 * width, comparison_df["f1_score"] * 100, width, label="F1-Score (%)", color="#7b1fa2")
    
    ax.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
    ax.set_title("Architectural Model Comparison on PlantVillage Test Set", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
