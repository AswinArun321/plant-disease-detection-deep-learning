"""
Generator for all 7 MCA Research Jupyter Notebooks adhering to plant-disease-detection-PLAN.md.
"""
import os
import json

NOTEBOOKS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "notebooks"))
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.9"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    }

def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")]
    }

# 1. 01_dataset_exploration.ipynb
nb1_cells = [
    md_cell("# 01. PlantVillage Dataset Exploration & Statistical Analysis\n## Research & MCA Major Project\nThis notebook conducts comprehensive Exploratory Data Analysis (EDA) on the PlantVillage dataset across 38 classes and 14 crops."),
    code_cell("""import os, sys
sys.path.insert(0, "..")
import matplotlib.pyplot as plt
from PIL import Image
from src import config
from src.data_loader import verify_dataset, get_class_names

stats = verify_dataset(config.DEFAULT_DATASET_DIR)
print(f"Total verified classes: {len(stats['classes'])}")
print(f"Total valid images: {stats['total_valid_images']}")
print(f"Corrupted files: {len(stats['corrupted_images'])}")
"""),
    code_cell("""# Display Class Sample Distribution
from src.utils import plot_class_distribution
plot_class_distribution(stats['class_counts'], "../reports/figures/class_distribution.png")
from IPython.display import Image as IPImage
IPImage(filename="../reports/figures/class_distribution.png")
"""),
    code_cell("""# Display 16 Sample Images Grid Across Diverse Pathologies
IPImage(filename="../reports/figures/sample_dataset_images.png")
""")
]

# 2. 02_preprocessing.ipynb
nb2_cells = [
    md_cell("# 02. Image Preprocessing & Biological Augmentation Pipeline\nDemonstrates image resizing to 224x224 RGB, normalization domains ([-1, 1] vs [0, 1]), and realistic augmentations."),
    code_cell("""import os, sys
sys.path.insert(0, "..")
import numpy as np
import matplotlib.pyplot as plt
from src import config
from src.preprocessing import load_and_preprocess_image
from src.augmentation import get_augmentation_layer

aug_layer = get_augmentation_layer()
print("Augmentation Layer:", aug_layer.name)
"""),
    code_cell("""# Visualize Real Preprocessed Leaf Specimen
test_img = os.path.join(config.DEFAULT_DATASET_DIR, "Tomato___Early_blight")
sample_file = os.path.join(test_img, os.listdir(test_img)[0])
batch, pil_img = load_and_preprocess_image(sample_file)
print(f"Processed batch shape: {batch.shape}, dtype: {batch.dtype}, min: {batch.min():.2f}, max: {batch.max():.2f}")
plt.imshow(pil_img)
plt.title("224x224 Bilinear Resampled Leaf")
plt.axis("off")
plt.show()
""")
]

# 3. 03_baseline_cnn.ipynb
nb3_cells = [
    md_cell("# 03. Baseline CNN Architecture Trained from Scratch\nEstablishes an empirical baseline without transfer learning to evaluate Research Question RQ2."),
    code_cell("""import os, sys
sys.path.insert(0, "..")
import tensorflow as tf
from src.train import build_baseline_cnn

cnn = build_baseline_cnn()
cnn.summary()
"""),
    code_cell("""# View Baseline Training Curves
from IPython.display import Image as IPImage
IPImage(filename="../reports/figures/training_curves_baseline_cnn.png")
""")
]

# 4. 04_transfer_learning.ipynb
nb4_cells = [
    md_cell("# 04. Transfer Learning Experiments (MobileNetV2, ResNet50, EfficientNetB0)\nCompares deep convolutional backbones initialized with ImageNet weights."),
    code_cell("""import os, sys
sys.path.insert(0, "..")
from src.train import build_mobilenet_v2, build_resnet50, build_efficientnet_b0

m_mob = build_mobilenet_v2()
m_res = build_resnet50()
m_eff = build_efficientnet_b0()

print(f"MobileNetV2 params: {m_mob.count_params():,}")
print(f"ResNet50 params: {m_res.count_params():,}")
print(f"EfficientNetB0 params: {m_eff.count_params():,}")
"""),
    code_cell("""from IPython.display import Image as IPImage
IPImage(filename="../reports/figures/training_curves_mobilenet_v2.png")
"""),
    code_cell("""IPImage(filename="../reports/figures/training_curves_resnet50.png")
"""),
    code_cell("""IPImage(filename="../reports/figures/training_curves_efficientnet_b0.png")
""")
]

# 5. 05_model_comparison.ipynb
nb5_cells = [
    md_cell("# 05. Model Evaluation & Empirical Comparison Study\nTabulates test accuracy, precision, recall, F1-score, parameter count, and training duration."),
    code_cell("""import os, sys
sys.path.insert(0, "..")
import pandas as pd
from IPython.display import display, Image as IPImage

df = pd.read_csv("../reports/model_comparison.csv")
display(df)
"""),
    code_cell("""IPImage(filename="../reports/figures/model_comparison_chart.png")
""")
]

# 6. 06_final_model.ipynb
nb6_cells = [
    md_cell("# 06. Final Model Selection, Fine-Tuning & Test Set Evaluation\nDocuments the fine-tuning of top convolutional blocks and final deployment serialization."),
    code_cell("""import os, sys
sys.path.insert(0, "..")
import tensorflow as tf
from src import config
from IPython.display import Image as IPImage

print(f"Checking final model at {config.FINAL_MODEL_PATH}...")
if os.path.exists(config.FINAL_MODEL_PATH):
    model = tf.keras.models.load_model(config.FINAL_MODEL_PATH)
    print("Model loaded successfully!")
    print(f"Total Parameters: {model.count_params():,}")
"""),
    code_cell("""IPImage(filename="../reports/figures/training_curves_final_model.png")
"""),
    code_cell("""IPImage(filename="../reports/figures/Final_FineTuned_Model_confusion_matrix.png")
""")
]

# 7. 07_gradcam.ipynb
nb7_cells = [
    md_cell("# 07. Explainable AI (XAI) with Grad-CAM\nVisualizes class activation gradients to reveal lesion localization."),
    code_cell("""import os, sys
sys.path.insert(0, "..")
import glob
from IPython.display import Image as IPImage

gradcam_imgs = glob.glob("../reports/figures/gradcam_sample_*.png")
for img_path in gradcam_imgs:
    print(f"Displaying Grad-CAM explanation: {img_path}")
    display(IPImage(filename=img_path))
""")
]

all_nbs = {
    "01_dataset_exploration.ipynb": nb1_cells,
    "02_preprocessing.ipynb": nb2_cells,
    "03_baseline_cnn.ipynb": nb3_cells,
    "04_transfer_learning.ipynb": nb4_cells,
    "05_model_comparison.ipynb": nb5_cells,
    "06_final_model.ipynb": nb6_cells,
    "07_gradcam.ipynb": nb7_cells
}

for nb_name, cells in all_nbs.items():
    nb_obj = make_notebook(cells)
    dest = os.path.join(NOTEBOOKS_DIR, nb_name)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(nb_obj, f, indent=2)
    print(f"Created notebook {dest}")
