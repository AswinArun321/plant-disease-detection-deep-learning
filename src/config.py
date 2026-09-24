"""
Configuration settings for Plant Disease Detection and Crop Advisory System.
"""
import os

# Base Directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Default PlantVillage location in this workspace
DEFAULT_DATASET_DIR = os.path.join(BASE_DIR, "plantvillage dataset", "color")

# Models Directory
MODELS_DIR = os.path.join(BASE_DIR, "models")
BASELINE_MODEL_DIR = os.path.join(MODELS_DIR, "baseline")
EXPERIMENTS_MODEL_DIR = os.path.join(MODELS_DIR, "experiments")
FINAL_MODEL_DIR = os.path.join(MODELS_DIR, "final")
FINAL_MODEL_PATH = os.path.join(FINAL_MODEL_DIR, "plant_disease_model.keras")
CLASS_NAMES_PATH = os.path.join(FINAL_MODEL_DIR, "class_names.json")

# Advisory Directory
ADVISORY_DIR = os.path.join(BASE_DIR, "advisory")
DISEASE_INFO_PATH = os.path.join(ADVISORY_DIR, "disease_information.json")

# Reports Directory
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
METRICS_DIR = os.path.join(REPORTS_DIR, "metrics")
EXPERIMENT_RESULTS_CSV = os.path.join(REPORTS_DIR, "experiment_results.csv")
MODEL_COMPARISON_CSV = os.path.join(REPORTS_DIR, "model_comparison.csv")

# App Directory
APP_DIR = os.path.join(BASE_DIR, "app")
UPLOAD_FOLDER = os.path.join(APP_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

# Model Hyperparameters & Specs
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
NUM_CLASSES = 38
RANDOM_SEED = 42

# Training Splitting Ratio
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# Confidence Threshold for Advisory Warnings
CONFIDENCE_THRESHOLD = 0.60
