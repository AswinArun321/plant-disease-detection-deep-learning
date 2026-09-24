"""
Comprehensive model evaluation, metrics calculation, and reporting.
"""
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
from src import config
from src.utils import plot_confusion_matrix

def evaluate_model(model, test_ds, class_names, model_name="model", save_figures=True):
    """
    Evaluate model on a test tf.data.Dataset and calculate full metrics suite.
    """
    y_true = []
    y_pred_probs = []

    print(f"Evaluating {model_name} on test dataset...")
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred_probs.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.argmax(np.array(y_pred_probs), axis=1)

    acc = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    report_text = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

    # Save classification report text
    report_path = os.path.join(config.METRICS_DIR, f"{model_name}_classification_report.txt")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # Confusion matrix
    if save_figures:
        cm_path = os.path.join(config.FIGURES_DIR, f"{model_name}_confusion_matrix.png")
        plot_confusion_matrix(y_true, y_pred, class_names, cm_path, title=f"{model_name} - Confusion Matrix")

    metrics = {
        "model_name": model_name,
        "test_accuracy": round(acc, 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "macro_f1": round(float(macro_f1), 4),
        "total_test_samples": len(y_true)
    }

    print(f"Results for {model_name}: Accuracy={acc*100:.2f}%, Precision={precision*100:.2f}%, Recall={recall*100:.2f}%, F1={f1*100:.2f}%")
    return metrics, report_dict
