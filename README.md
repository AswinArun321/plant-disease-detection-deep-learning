# FloraScan AI

## AI-Based Plant Disease Detection and Crop Advisory System

**An intelligent computer vision system for plant disease detection using deep learning, transfer learning, and Explainable AI (Grad-CAM).**

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3-red?logo=keras)](https://keras.io/)
[![Flask](https://img.shields.io/badge/Flask-3.1-black?logo=flask)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-Academic-lightgrey)](#license)

---

## Overview

**FloraScan AI** is an AI-powered plant disease detection and crop advisory system designed to identify plant diseases from leaf images.

The system uses **deep learning and transfer learning** to classify plant diseases and combines the prediction with **Explainable AI (XAI)** using **Grad-CAM** to visually indicate the regions of the leaf that influenced the model's prediction.

After identifying the disease, the system provides structured information including:

* Crop and disease name
* Prediction confidence
* Grad-CAM visual explanation
*  Disease symptoms
* Treatment information
* Prevention recommendations

The project is designed as an **MCA major project and research-oriented deep learning system** combining computer vision, transfer learning, explainable AI, and agricultural decision support.

---

## Problem Statement

Plant diseases can significantly affect crop productivity and agricultural sustainability.

Traditional disease identification often depends on manual inspection by farmers or agricultural experts. This process can be time-consuming and difficult to scale.

Deep learning-based computer vision can automate disease classification from plant leaf images. However, conventional deep learning models can behave as **black boxes**, making it difficult for users to understand why a particular prediction was generated.

FloraScan AI addresses these challenges by combining:

1. Deep learning-based disease classification
2. Transfer learning
3. Multiple model experiments
4. Explainable AI using Grad-CAM
5. Disease information
6. Treatment guidance
7. Prevention recommendations
8. A Flask-based web application

---

## Key Features

### AI-Based Disease Detection

Classifies plant leaf images into supported healthy and disease categories using deep learning.

### Transfer Learning

Experiments with multiple pretrained architectures:

* MobileNetV2
* ResNet50
* EfficientNetB0

A baseline CNN trained from scratch is also implemented for comparison.

### Explainable AI

Grad-CAM generates visual heatmaps showing the image regions that contributed to the model's prediction.

### Confidence Score

The application displays the model's prediction confidence and provides cautious messaging for low-confidence predictions.

### Crop Advisory

Provides structured information related to:

* Disease description
* Symptoms
* Treatment
* Prevention

### Image Upload

Supports common image formats such as:

* JPG
* JPEG
* PNG

### Web Application

A responsive Flask application provides a simple interface for:

* Uploading leaf images
* Viewing predictions
* Viewing confidence
* Inspecting Grad-CAM explanations
* Reading crop advisory information

### REST API

The application also provides a prediction API for potential integration with other applications and systems.

---

## System Architecture

```text
                    ┌──────────────────────┐
                    │        User          │
                    │ Farmer / Researcher  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Leaf Image Upload  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Image Validation   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Image Preprocessing  │
                    │ Resize / Normalize   │
                    └──────────┬───────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │     Deep Learning Model        │
              │  CNN / Transfer Learning       │
              └───────────────┬────────────────┘
                              │
                 ┌────────────┼────────────┐
                 │            │            │
                 ▼            ▼            ▼
          ┌────────────┐ ┌──────────┐ ┌─────────────┐
          │ Prediction │ │Confidence│ │   Grad-CAM  │
          └─────┬──────┘ └────┬─────┘ └──────┬──────┘
                │             │              │
                └─────────────┼──────────────┘
                              │
                              ▼
                    ┌──────────────────────┐
                    │  Crop Advisory      │
                    │ Symptoms / Treatment│
                    │ Prevention           │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Diagnostic Result    │
                    │      Web Page        │
                    └──────────────────────┘
```

---

## Dataset

### PlantVillage Dataset

The project uses the **PlantVillage dataset** for plant disease classification.

The project plan specifies approximately:

* **54,000+ leaf images**
* **38 classes**
* **14 crop species**

The supported crop categories include:

* Apple
* Blueberry
* Cherry
* Corn / Maize
* Grape
* Orange
* Peach
* Bell Pepper
* Potato
* Raspberry
* Soybean
* Squash
* Strawberry
* Tomato

The dataset is used for:

* Dataset exploration
* Image preprocessing
* Data augmentation
* CNN training
* Transfer learning
* Model comparison
* Final model evaluation

### Dataset Structure

```text
data/
├── raw/
├── processed/
└── README.md
```

The complete PlantVillage dataset is **not included in this repository**.

Refer to:

```text
data/README.md
```

for dataset acquisition instructions.

---

## 🤖 Machine Learning Approach

The project follows a structured experimental workflow.

### 1. Baseline CNN

A convolutional neural network is trained from scratch to establish a baseline.

### 2. Transfer Learning

Three pretrained architectures are evaluated:

```text
MobileNetV2
ResNet50
EfficientNetB0
```

Each model is evaluated using a consistent dataset configuration and evaluation procedure.

### 3. Model Selection

The final architecture is selected based on actual experimental results rather than predefined assumptions.

Evaluation considers:

* Accuracy
* Precision
* Recall
* F1-score
* Training time
* Model size
* Computational requirements

### 4. Fine-Tuning

The selected transfer-learning model is fine-tuned using a staged training approach:

```text
Pretrained Model
       ↓
Freeze Backbone
       ↓
Train Classification Head
       ↓
Unfreeze Selected Layers
       ↓
Fine-Tune with Lower Learning Rate
       ↓
Final Model
```

---

## Explainable AI — Grad-CAM

FloraScan AI uses **Gradient-weighted Class Activation Mapping (Grad-CAM)** to provide visual explanations for model predictions.

### Grad-CAM Pipeline

```text
Input Leaf Image
       ↓
Deep Learning Model
       ↓
Predicted Disease
       ↓
Feature Activations
       ↓
Gradient Calculation
       ↓
Activation Weighting
       ↓
Grad-CAM Heatmap
       ↓
Overlay with Original Image
```

The heatmap helps visualize whether the model is focusing on relevant leaf regions such as:

* Lesions
* Discoloration
* Disease symptoms
* Other visually relevant regions

Grad-CAM is treated as an **interpretability tool**, not proof that a prediction is correct.

---

## Crop Advisory Module

The advisory module maps the predicted disease to structured agricultural information.

### Information Provided

```text
Disease
   ↓
Crop
   ↓
Description
   ↓
Symptoms
   ↓
Treatment
   ↓
Prevention
```

The advisory information is stored in:

```text
advisory/disease_information.json
```

The business logic is implemented in:

```text
advisory/advisory_service.py
```

Agricultural recommendations should be supported by reliable references and should not be treated as a replacement for professional agricultural guidance.

---

## Technology Stack

| Category             | Technology                       |
| -------------------- | -------------------------------- |
| Programming Language | Python 3.11                      |
| Deep Learning        | TensorFlow / Keras               |
| Computer Vision      | OpenCV, Pillow                   |
| Data Processing      | NumPy, Pandas                    |
| Machine Learning     | Scikit-learn                     |
| Visualization        | Matplotlib                       |
| Web Framework        | Flask                            |
| Frontend             | HTML5, CSS3, JavaScript          |
| Testing              | Pytest                           |
| Development          | VS Code / Jupyter / Google Colab |
| Version Control      | Git / GitHub                     |

---

## Project Structure

```text
plant-disease-detection-advisory/
│
├── README.md
├── requirements.txt
├── .gitignore
├── plant-disease-detection-PLAN.md
│
├── advisory/
│   ├── advisory_service.py
│   └── disease_information.json
│
├── app/
│   ├── app.py
│   ├── templates/
│   │   ├── index.html
│   │   ├── result.html
│   │   └── error.html
│   │
│   └── static/
│       ├── css/
│       ├── js/
│       └── uploads/
│
├── data/
│   ├── README.md
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── baseline/
│   ├── experiments/
│   └── final/
│       └── class_names.json
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_cnn.ipynb
│   ├── 04_transfer_learning.ipynb
│   ├── 05_model_comparison.ipynb
│   ├── 06_final_model.ipynb
│   └── 07_gradcam.ipynb
│
├── reports/
│   ├── figures/
│   ├── metrics/
│   ├── model_comparison.csv
│   └── experiment_results.csv
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── augmentation.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── gradcam.py
│   └── utils.py
│
└── tests/
    ├── test_model.py
    ├── test_preprocessing.py
    └── test_app.py
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AswinArun321/plant-disease-detection-advisory.git
cd plant-disease-detection-advisory
```

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the Dataset

Download the PlantVillage dataset according to the instructions in:

```text
data/README.md
```

Place the dataset in the appropriate local data directory.

### 5. Verify the Dataset

Run the dataset exploration notebook:

```text
notebooks/01_dataset_exploration.ipynb
```

---

## Model Training

The recommended development sequence is:

```text
Dataset Exploration
        ↓
Preprocessing
        ↓
Data Augmentation
        ↓
Baseline CNN
        ↓
MobileNetV2
        ↓
ResNet50
        ↓
EfficientNetB0
        ↓
Model Comparison
        ↓
Final Model Selection
        ↓
Fine-Tuning
        ↓
Final Evaluation
```

The notebooks are organized accordingly:

```text
01_dataset_exploration.ipynb
02_preprocessing.ipynb
03_baseline_cnn.ipynb
04_transfer_learning.ipynb
05_model_comparison.ipynb
06_final_model.ipynb
07_gradcam.ipynb
```

---

## Model Evaluation

The project evaluates models using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* Classification Report
* Training/Validation Curves
* Error Analysis

Experimental results are recorded in:

```text
reports/experiment_results.csv
reports/model_comparison.csv
```

### Important

All reported metrics must come from **actual experiments**.

No estimated or fabricated performance values should be included.

---

## Running the Web Application

After the final model and required resources are available:

```bash
python app/app.py
```

Then open:

```text
http://127.0.0.1:5000
```

### Application Workflow

```text
Upload Leaf Image
        ↓
Validate Image
        ↓
Preprocess Image
        ↓
Run Model Prediction
        ↓
Display Disease + Confidence
        ↓
Generate Grad-CAM
        ↓
Load Advisory Information
        ↓
Display Result
```

---

## REST API

FloraScan AI can expose a prediction endpoint for programmatic use.

### Endpoint

```text
POST /api/predict
```

### Example

```bash
curl -X POST \
  -F "file=@sample_leaf.jpg" \
  http://127.0.0.1:5000/api/predict
```

### Example Response

```json
{
  "status": "success",
  "crop": "Tomato",
  "disease_name": "Early Blight",
  "confidence": 0.98,
  "is_healthy": false,
  "symptoms": [],
  "treatment": [],
  "prevention": []
}
```

The exact response fields depend on the implemented application version and advisory database.

---

## Testing

The project includes automated tests for major components.

Run:

```bash
pytest tests/ -v
```

Tests cover areas such as:

### Model

* Model loading
* Input dimensions
* Prediction structure
* Class mapping

### Preprocessing

* Image resizing
* Normalization
* Invalid images
* Unsupported formats

### Flask Application

* Home page
* File upload
* Prediction flow
* Invalid file handling
* Error handling

### Grad-CAM

* Heatmap generation
* Output dimensions
* Overlay generation

---

## Security Considerations

The application includes safeguards for uploaded images.

These include:

* File type validation
* File size validation
* Secure filename handling
* Temporary upload handling
* Error handling
* Avoiding exposure of internal file paths

The application should not expose debug information or stack traces in production.

---

## Performance Evaluation

The project can measure:

* Model loading time
* Image preprocessing time
* Prediction time
* Grad-CAM generation time
* Total response time

These measurements can help evaluate the practical performance of the deployed system.

---

## Limitations

The system has several important limitations.

### Dataset Limitations

PlantVillage images may not fully represent real-world field conditions.

Real agricultural environments can contain:

* Different lighting conditions
* Complex backgrounds
* Different camera qualities
* Occluded leaves
* Multiple diseases
* Symptoms at different stages

### Model Limitations

The model can produce incorrect predictions.

A high confidence score does not guarantee correctness.

Diseases outside the supported classes cannot be reliably classified.

### Advisory Limitations

The system is an **AI-based decision-support tool**.

Its recommendations should not be considered a guaranteed diagnosis or a replacement for professional agricultural advice.

---

## Research Questions

The project investigates the following questions:

### RQ1

Can deep learning accurately classify plant diseases from leaf images?

### RQ2

Does transfer learning provide better performance than a CNN trained from scratch?

### RQ3

Which pretrained architecture provides a suitable balance between classification performance and computational efficiency?

### RQ4

Can Grad-CAM provide meaningful visual explanations for plant disease predictions?

### RQ5

Can the trained model be integrated into a practical web-based crop disease advisory system?

---

## Project Roadmap

* [x] Project planning
* [ ] Dataset acquisition
* [ ] Dataset exploration
* [ ] Image preprocessing
* [ ] Data augmentation
* [ ] Baseline CNN
* [ ] MobileNetV2 experiment
* [ ] ResNet50 experiment
* [ ] EfficientNetB0 experiment
* [ ] Model comparison
* [ ] Final model selection
* [ ] Fine-tuning
* [ ] Final evaluation
* [ ] Error analysis
* [ ] Grad-CAM implementation
* [ ] Advisory module
* [ ] Flask integration
* [ ] Automated testing
* [ ] End-to-end testing
* [ ] Deployment
* [ ] Final documentation
* [ ] MCA dissertation
* [ ] Final presentation

> Update the checklist as each stage of the actual project is completed.

---

## Academic Deliverables

The project is intended to support the following MCA deliverables:

* Project source code
* Dataset documentation
* Training notebooks
* Baseline CNN
* Transfer-learning experiments
* Final trained model
* Class mapping
* Grad-CAM implementation
* Crop advisory module
* Flask web application
* Automated tests
* Experiment report
* Dissertation
* Presentation
* Application screenshots
* Model evaluation results

---

## Documentation

Important project documentation includes:

```text
README.md
plant-disease-detection-PLAN.md
data/README.md
reports/experiment_report.md
```

The complete development roadmap is documented in:

```text
plant-disease-detection-PLAN.md
```

---

## Future Scope

Potential future improvements include:

* Real-world field image datasets
* Mobile application
* Multilingual crop advisory
* Voice-based agricultural assistance
* Offline inference
* Additional crop species
* Additional disease classes
* Weather-based disease risk analysis
* IoT-based crop monitoring
* Cloud-based model serving
* Disease severity estimation
* Continuous model improvement using new field images

---

### Domain

* Artificial Intelligence
* Machine Learning
* Deep Learning
* Computer Vision
* Transfer Learning
* Explainable AI
* Agriculture Technology

---

## Disclaimer

FloraScan AI is an **academic research and decision-support system**.

AI-generated predictions may be incorrect and should be verified by qualified agricultural professionals before taking significant crop-management or chemical-treatment decisions.

The system should not be presented as a guaranteed replacement for agricultural experts.

---

## License

This project is developed for **academic and educational purposes**.

Refer to the repository license file for the applicable usage terms.

---

## Acknowledgements

* PlantVillage dataset and its contributors
* TensorFlow / Keras
* OpenCV
* Scikit-learn
* Flask
* The open-source Python ecosystem

---

## Project Repository

**Repository:** `plant-disease-detection-advisory`

**Project:** **FloraScan AI — AI-Based Plant Disease Detection and Crop Advisory System**

---

### From Leaf Image to Intelligent Crop Advisory

```text
Leaf Image
    ↓
AI Disease Detection
    ↓
Confidence Analysis
    ↓
Grad-CAM Explanation
    ↓
Disease Information
    ↓
Treatment & Prevention
    ↓
Intelligent Crop Advisory
```
