# FloraScan AI: Plant Disease Detection and Intelligent Crop Advisory System
### Using Deep Transfer Learning and Explainable AI (Grad-CAM)
**MCA Major Project & Agricultural Deep Learning Research**

---

## 1. Project Overview
FloraScan AI is an end-to-end computer vision and agricultural decision-support platform designed to accurately classify plant foliar pathologies from leaf photographs and deliver agronomic advisory guidance.

Traditional manual identification of plant diseases by agronomists is time-consuming, labor-intensive, and hard to scale across extensive farming regions. Computer vision systems can automate early diagnosis; however, typical deep learning models operate as "black boxes" that lack transparency. 

FloraScan AI solves this by integrating:
1. **Transfer Learning**: Evaluating MobileNetV2, ResNet50, and EfficientNetB0 against a baseline convolutional neural network (CNN) trained from scratch.
2. **Explainable AI (XAI)**: Generating Gradient-weighted Class Activation Mapping (**Grad-CAM**) heatmaps that highlight which leaf regions (lesions, discolorations, pustules) influenced the prediction.
3. **Structured Agricultural Advisory**: Translating classifications into verified biological symptoms, cultural management techniques, organic remedies, and chemical controls across **38 classes** and **14 crop species**.
4. **Cautious AI Guardrails**: Displaying confidence scores and flagging diagnoses below 60% with advisory warnings to consult local agricultural extension officers.
5. **Modern Web Interface**: A responsive Flask web application featuring drag-and-drop image uploads, live visual inspections, and a RESTful API.

---

## 2. Research Questions (RQs)
- **RQ1**: Can deep convolutional neural networks accurately classify plant diseases across 38 diverse classes from leaf photographs?
- **RQ2**: Does transfer learning using pretrained ImageNet weights outperform a convolutional network trained from scratch?
- **RQ3**: Which pretrained architecture (MobileNetV2, ResNet50, or EfficientNetB0) provides the optimal balance between classification performance and inference speed?
- **RQ4**: Can Grad-CAM provide reliable, interpretable visual evidence that correlates with genuine foliar lesions rather than spurious background artifacts?
- **RQ5**: Can these deep learning and XAI components be successfully deployed into a practical, latency-conscious web advisory tool?

---

## 3. System Architecture

```
User (Farmer / Agronomist)
       │
       ▼ Upload Leaf Image (JPG/PNG)
┌────────────────────────────────────────────────────────┐
│ Flask Web Application & Upload Pipeline                │
│ • Secure filename sanitization                         │
│ • File size limit check (<16MB)                        │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│ Image Preprocessing Pipeline (src/preprocessing.py)    │
│ • Bilinear resize to 224×224 RGB                       │
│ • Color-space verification (3-channel)                 │
│ • Model-specific normalization                         │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│ Deep Learning Model (models/final/)                    │
│ • MobileNetV2 Transfer Learning Backbone               │
│ • Global Average Pooling + Batch Normalization         │
│ • Dense(256, ReLU) + Dropout(0.35)                     │
│ • Dense(38, Softmax) Output Classification             │
└───────────┬───────────────────────────────┬────────────┘
            │ Predicted Class & Probs       │ Feature Activations
            ▼                               ▼
┌──────────────────────────┐    ┌────────────────────────┐
│ Crop Advisory Service    │    │ Explainable AI Engine  │
│ (advisory/)              │    │ (src/gradcam.py)       │
│ • 38-Class Knowledge Base│    │ • Gradient Computation │
│ • Symptoms & Etiology    │    │ • Pooled Gradients     │
│ • Organic & Chemical     │    │ • ReLU Activation Map  │
│ • Preventive Agronomy    │    │ • Jet Heatmap Overlay  │
└───────────┬──────────────┘    └───────────┬────────────┘
            │                               │
            └───────────────┬───────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Responsive Diagnostic Report (app/templates/result.html│
│ • Health status badge & confidence gauge               │
│ • Side-by-side original vs Grad-CAM overlay            │
│ • Top-3 candidate differential diagnoses               │
│ • Tabbed symptom, treatment & prevention advice        │
└────────────────────────────────────────────────────────┘
```

---

## 4. Dataset Overview: PlantVillage
- **Total Images**: 54,305 curated leaf images
- **Classes**: 38 distinct classes (12 healthy classes, 26 disease classes)
- **Crops (14 Species)**: Apple, Blueberry, Cherry, Corn (Maize), Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato
- **Format**: 256×256 RGB JPEG
- **Splits**: 80% Training, 10% Validation, 10% Held-Out Testing (stratified with deterministic seed 42)

---

## 5. Technology Stack
- **Core Language**: Python 3.11
- **Deep Learning Framework**: TensorFlow 2.20 / Keras 3
- **Computer Vision**: OpenCV (`cv2`), Pillow (`PIL`)
- **Scientific Computing**: NumPy, Pandas, Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Web Framework**: Flask 3.1
- **Frontend**: HTML5, Vanilla CSS3 (Custom Glassmorphism Design System), JavaScript (ES6)
- **Unit Testing**: Pytest

---

## 6. Project Directory Layout

```
Plant-Disease-Detection-and-Intelligent-Crop-Advisory-System/
├── README.md                          # Project documentation
├── requirements.txt                   # Dependency specifications
├── .gitignore                         # Version control ignore rules
│
├── advisory/                          # Agronomic Advisory System
│   ├── advisory_service.py            # Diagnostic & advisory business logic
│   └── disease_information.json       # 38-class botanical knowledge base
│
├── app/                               # Flask Web Application
│   ├── app.py                         # Application routes & REST endpoints
│   ├── templates/
│   │   ├── index.html                 # Modern landing page & uploader
│   │   ├── result.html                # Diagnostic dashboard & Grad-CAM viewer
│   │   └── error.html                 # Friendly error recovery page
│   └── static/
│       ├── css/style.css              # Custom styling & responsive tokens
│       ├── js/main.js                 # Drag & drop upload handler
│       └── uploads/                   # Temporary specimen storage
│
├── data/
│   ├── README.md                      # Dataset acquisition guide
│   ├── raw/                           # Raw PlantVillage directory
│   └── processed/                     # Preprocessed splits & caches
│
├── models/
│   ├── baseline/                      # Baseline CNN checkpoints
│   ├── experiments/                   # Comparative architecture models
│   └── final/
│       ├── class_names.json           # Ordered 38 class mapping
│       └── plant_disease_model.keras  # Final fine-tuned production model
│
├── notebooks/                         # MCA Dissertation Research Notebooks
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_cnn.ipynb
│   ├── 04_transfer_learning.ipynb
│   ├── 05_model_comparison.ipynb
│   ├── 06_final_model.ipynb
│   └── 07_gradcam.ipynb
│
├── reports/
│   ├── dissertation.md                # Complete MCA Dissertation
│   ├── experiment_report.md           # Research experimental log
│   ├── presentation.md                # Project defense presentation guide
│   ├── model_comparison.csv           # Tabulated empirical metrics
│   ├── experiment_results.csv         # Full experiment tracking log
│   ├── figures/                       # Plots, charts, confusion matrices
│   └── metrics/                       # Text classification reports
│
├── src/                               # Modular Source Package
│   ├── __init__.py
│   ├── config.py                      # Central constants & path configuration
│   ├── data_loader.py                 # Dataset verification & tf.data pipeline
│   ├── preprocessing.py               # Image resizing & normalizations
│   ├── augmentation.py                # Biological data augmentations
│   ├── train.py                       # CNN and transfer learning builders
│   ├── evaluate.py                    # Metrics, confusion matrix, reporting
│   ├── predict.py                     # Integrated inference pipeline
│   ├── gradcam.py                     # Explainable AI Grad-CAM engine
│   ├── run_experiments.py             # Master experimental runner
│   └── utils.py                       # Plotting and serialization utilities
│
└── tests/                             # Automated Test Suite
    ├── test_app.py                    # Flask web route tests
    ├── test_gradcam.py                # Grad-CAM heatmap validation
    ├── test_model.py                  # Model architecture tests
    └── test_preprocessing.py          # Image transformation tests
```

---

## 7. Setup & Installation

### Step 1: Clone Repository & Create Virtual Environment
```bash
git clone <repository_url>
cd "Plant Disease Detection and Intelligent Crop Advisory System"

# Create Python virtual environment
python -m venv venv
# Activate on Windows:
venv\Scripts\activate
# Activate on Linux/macOS:
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run Master Experiments & Training Pipeline
To generate the dataset analysis, train baseline and transfer learning architectures, evaluate comparative metrics, and save the final fine-tuned model:
```bash
python src/run_experiments.py
```

### Step 4: Run Automated Verification Tests
```bash
pytest tests/ -v
```

### Step 5: Launch the Flask Web Application
```bash
python app/app.py
```
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 8. REST API Usage

FloraScan AI provides a headless JSON API for integration into mobile applications or agricultural IoT sensors:

### Endpoint: `POST /api/predict`
**Request**:
```bash
curl -X POST -F "file=@sample_leaf.jpg" http://127.0.0.1:5000/api/predict
```

**Response Format**:
```json
{
  "status": "success",
  "crop": "Tomato",
  "disease_name": "Early Blight",
  "pathogen": "Alternaria linariae / Alternaria solani (Fungus)",
  "confidence": 0.9842,
  "confidence_percent": "98.4%",
  "is_healthy": false,
  "original_image_url": "/static/uploads/leaf_abc123.jpg",
  "gradcam_image_url": "/static/uploads/gradcam_abc123.png",
  "top_candidates": [
    {
      "class_name": "Tomato___Early_blight",
      "crop": "Tomato",
      "disease": "Early blight",
      "confidence": 0.9842,
      "confidence_percent": "98.4%"
    }
  ],
  "symptoms": [
    "Circular brown spots with distinct concentric rings on lower leaves",
    "Yellow chlorotic halos surrounding lesions"
  ],
  "treatment": [
    "Apply copper protectants or chlorothalonil",
    "Prune bottom 12 inches of foliage to prevent soil splash"
  ],
  "prevention": [
    "Apply organic or black plastic mulch",
    "Stake plants and water only at the soil line"
  ]
}
```

---

## 9. Academic Dissertation & Defense Preparation
Full academic deliverables are provided within the `reports/` directory:
- [MCA Dissertation Manuscript](file:///d:/Plant%20Disease%20Detection%20and%20Intelligent%20Crop%20Advisory%20System/reports/dissertation.md)
- [Experiment Report](file:///d:/Plant%20Disease%20Detection%20and%20Intelligent%20Crop%20Advisory%20System/reports/experiment_report.md)
- [Final Presentation & Viva Guide](file:///d:/Plant%20Disease%20Detection%20and%20Intelligent%20Crop%20Advisory%20System/reports/presentation.md)

---

## 10. License & Disclaimers
This system is an academic research decision-support tool. Artificial intelligence predictions should be complemented with professional agricultural inspection before making chemical fungicide investments.
