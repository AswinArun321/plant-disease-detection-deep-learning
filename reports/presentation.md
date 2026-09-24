# FloraScan AI: Final MCA Project Presentation & Viva Defense Guide

---

## Slide 1: Title & Authorship
- **Project Title**: FloraScan AI: Plant Disease Detection and Intelligent Crop Advisory System Using Transfer Learning and Explainable AI
- **Domain**: Deep Learning, Computer Vision, Agriculture Technology, Explainable AI (XAI)
- **Degree**: Master of Computer Applications (MCA) Major Project

---

## Slide 2: Problem Statement & Real-World Motivation
- **The Challenge**: Plant diseases cause 20-40% annual crop yield losses globally. Traditional manual diagnosis by agricultural experts is slow, subjective, and inaccessible in rural agricultural communities.
- **The Need**: An automated, accurate, and trustworthy diagnosis system that farmers can access via web or mobile devices.
- **The Core Barrier**: Deep learning models are typically "black boxes" — farmers and agronomists cannot trust an unexplainable prediction when chemical fungicide decisions are at stake.

---

## Slide 3: Project Aim & Objectives
- **Aim**: Develop an explainable deep learning decision-support system capable of detecting crop pathologies across 38 conditions and providing structured agronomic guidance.
- **Objectives**:
  1. Acquire and verify the 54,305-image PlantVillage dataset.
  2. Implement baseline CNN vs. multiple transfer learning models (MobileNetV2, ResNet50, EfficientNetB0).
  3. Empirically select and fine-tune the superior architecture.
  4. Implement Grad-CAM to visualize model reasoning.
  5. Build a comprehensive 38-class agronomic advisory knowledge base.
  6. Deploy an interactive Flask web application and REST API.

---

## Slide 4: Dataset Overview (PlantVillage)
- **Dataset Size**: 54,305 RGB leaf photographs.
- **Class Count**: 38 classes (12 healthy classes, 26 disease classes).
- **Crop Species (14)**: Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato.
- **Splits**: 80% Train, 10% Validation, 10% Test (Stratified, deterministic seed 42).
- **Data Integrity**: 100% verified with 0 corrupted images.

---

## Slide 5: System Pipeline Architecture
```
Input Leaf Image
     ↓
Image Verification & Preprocessing (224×224 RGB, [-1, 1] normalized)
     ↓
Fine-Tuned MobileNetV2 Deep Feature Extractor
     ↓
┌──────────────────────┬──────────────────────┐
↓                      ↓                      ↓
Disease Prediction   Confidence Score     Grad-CAM XAI Heatmap
(38-Class Softmax)   (Safety Threshold)   (Last Conv Feature Maps)
└──────────────────────┬──────────────────────┘
                       ↓
         Intelligent Crop Advisory
  (Symptoms, Organic Treatment, Chemical Control, Prevention)
                       ↓
             Flask Web Dashboard
```

---

## Slide 6: Model Comparison & Experimental Results

| Model Architecture | Approach | Test Accuracy | Precision | Recall | F1-Score | Parameter Count |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline CNN** | Trained from scratch | 31.4% | 0.342 | 0.314 | 0.298 | ~350K |
| **ResNet50** | Transfer Learning (Frozen) | 82.6% | 0.841 | 0.826 | 0.821 | ~24.1M |
| **EfficientNetB0**| Transfer Learning (Frozen) | 86.8% | 0.875 | 0.868 | 0.864 | ~4.4M |
| **MobileNetV2** | Transfer Learning (Frozen) | 91.2% | 0.920 | 0.912 | 0.911 | ~2.6M |
| **Final Fine-Tuned** | MobileNetV2 (Top 30 Unfrozen) | **96.8%** | **0.971** | **0.968** | **0.969** | **~2.6M** |

---

## Slide 7: Why MobileNetV2 Was Selected as the Final Backbone
1. **Accuracy**: Achieved 91.2% out of the box and reached 96.8% upon fine-tuning.
2. **Computational Footprint**: Requires only 2.6 million parameters, compared to 24.1 million for ResNet50 (nearly 10x smaller).
3. **Inference Latency**: Fast forward pass (~25ms per image), ideal for edge deployment and web serving.
4. **Depthwise Separable Convolutions**: Drastically reduces multiply-accumulate operations without sacrificing representational capacity.

---

## Slide 8: Explainable AI: How Grad-CAM Works
1. Compute gradient of predicted class score with respect to feature maps of the final convolutional layer: $\frac{\partial y^c}{\partial A^k}$.
2. Global average pool gradients to produce importance weights $\alpha_k^c$.
3. Compute weighted sum of feature maps and apply ReLU: $L^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$.
4. Normalize to $[0, 1]$, resize to $224 \times 224$, and blend with original image using OpenCV JET colormap.
5. **Result**: Red/yellow highlight proves the network detected necrotic lesions rather than spurious background soil.

---

## Slide 9: Crop Advisory & Decision Support System
- Structured database covering all 38 classes:
  - **Symptom Profile**: Morphological cues (spots, wilting, curling).
  - **Organic Controls**: Neem oil, copper soap, Trichoderma biocontrols, baking soda sprays.
  - **Chemical Treatments**: Mancozeb, Chlorothalonil, Azoxystrobin, Myclobutanil.
  - **Preventive Cultural Practices**: Drip irrigation, wide plant spacing, sanitizing tools, crop rotation.
- **Cautious AI Notice**: Predictions under 60% confidence trigger a safety warning advising extension verification.

---

## Slide 10: Web Application Architecture
- **Backend**: Flask 3.1 with secure upload validation, model caching, and modular service separation.
- **Frontend**: Glassmorphism design system built with custom CSS, Google Fonts, responsive flex/grid layouts.
- **Features**:
  - Drag-and-drop uploader with live image preview.
  - Animated diagnostic progress stages.
  - Interactive side-by-side inspection (Original vs. Grad-CAM).
  - Top-3 candidate differential diagnosis breakdown.
  - Tabbed advisory panels and printable clinical reports.
  - Headless REST API (`/api/predict`).

---

## Slide 11: Live Demonstration Scenarios
1. **Demo 1 — Healthy Foliage**: Upload healthy apple leaf $\rightarrow$ Confirmed "Healthy", diffuse low-intensity Grad-CAM, maintenance guidance.
2. **Demo 2 — Fungal Pathology**: Upload tomato early blight leaf $\rightarrow$ Confirmed "Tomato Early Blight" with 98% confidence, Grad-CAM pinpointing concentric target spots, fungicide protocol.
3. **Demo 3 — Edge Case / Low Confidence**: Upload ambiguous or corrupted image $\rightarrow$ Handled gracefully with explanatory error or cautious advisory flag.

---

## Slide 12: Viva Voce — Expected Questions & Answers

### Q1: Why did you use transfer learning instead of training exclusively from scratch?
> **Answer**: Training deep networks from scratch requires massive domain-specific datasets and risks severe overfitting or slow convergence (our scratch baseline only achieved 31.4%). Transfer learning harnesses low-level edge and texture representations already learned on 1.4 million ImageNet samples, enabling faster convergence and higher accuracy (96.8%) on specialized agricultural leaf datasets.

### Q2: What layer is targeted for Grad-CAM in MobileNetV2?
> **Answer**: We target the final convolutional expansion/projection layer (`Conv_1` / `out_relu`), which is the last point in the network retaining spatial 2D coordinates before global average pooling collapses the tensor into a 1D vector.

### Q3: How do you prevent data leakage during training?
> **Answer**: We partition the dataset into train, validation, and test splits *before* any preprocessing or augmentation. Augmentation is applied exclusively to the training stream. The test set is held out completely and evaluated only once at the conclusion of training.

### Q4: What makes this project an MCA major project rather than a toy model?
> **Answer**: The project delivers a complete scientific and engineering pipeline: empirical multi-architecture comparison, genuine statistical evaluation, explainable AI with Grad-CAM, a biologically accurate 38-class advisory engine, cautious confidence safety thresholds, full unit test coverage, and a production-ready web application with a REST API.
