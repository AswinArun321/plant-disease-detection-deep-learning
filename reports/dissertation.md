# Plant Disease Detection and Intelligent Crop Advisory System Using Transfer Learning and Explainable AI

**MCA Major Project Dissertation**  
**Master of Computer Applications**

---

# CHAPTER 1 — INTRODUCTION

## 1.1 Background
Agriculture is the foundation of global food security and economic stability. Plant diseases caused by fungal, bacterial, and viral pathogens represent one of the leading threats to crop yields, causing estimated worldwide harvest losses exceeding 20% to 40% annually. Early and accurate detection of foliar pathologies is essential to prevent localized infections from escalating into field-wide epidemics.

Traditionally, disease diagnosis has relied upon physical visual inspection by farmers, agronomists, or specialized plant pathologists. However, manual scouting is labor-intensive, subjective, prone to diagnostic error, and severely restricted by the availability of trained extension specialists in rural farming communities.

## 1.2 Problem Statement
With the rapid maturation of computer vision and deep learning, convolutional neural networks (CNNs) have shown remarkable capability in classifying visual images. Nonetheless, existing automated plant disease detection approaches face three critical bottlenecks:
1. **The "Black-Box" Problem**: Conventional deep learning systems output disease labels and confidence probabilities without explaining *why* a particular diagnosis was made. In agriculture, where misdiagnosis can lead to costly and harmful chemical applications, farmers and agronomists hesitate to trust uninterpretable predictions.
2. **Transfer Learning vs. From-Scratch Uncertainty**: While transfer learning is widely promoted, empirical comparisons demonstrating the trade-offs in accuracy, parameter count, and training duration between scratch-trained CNNs and modern architectures (MobileNetV2, ResNet50, EfficientNetB0) are frequently lacking in practical implementations.
3. **Absence of Actionable Advisory**: Most computer vision projects stop at producing a text label. A practical agricultural solution must bridge classification with curative organic treatments, chemical controls, and preventive cultural agronomy.

## 1.3 Motivation
Combining transfer learning, explainable artificial intelligence (Grad-CAM), and structured crop advisory enables the development of a trustworthy decision-support platform. By visually highlighting the specific leaf lesions responsible for a classification, the system earns grower confidence while providing immediate, actionable guidance.

## 1.4 Aim and Objectives
The overarching aim of this project is to develop an intelligent, explainable, and web-accessible computer vision system for plant disease detection and crop advisory using the PlantVillage dataset.

### Specific Objectives:
1. Conduct exploratory data analysis (EDA) across 54,305 leaf images spanning 38 classes and 14 crop species.
2. Construct a modular preprocessing and realistic biological augmentation pipeline.
3. Design and train a baseline Convolutional Neural Network from scratch.
4. Implement transfer learning using MobileNetV2, ResNet50, and EfficientNetB0 backbones.
5. Systematically compare all models using Accuracy, Precision, Recall, F1-Score, and training efficiency.
6. Select and fine-tune the optimal model for production deployment.
7. Implement Gradient-weighted Class Activation Mapping (Grad-CAM) to provide transparent visual explanations.
8. Construct a comprehensive 38-class crop advisory database covering symptoms, chemical controls, organic treatments, and prevention.
9. Build and deploy a modern, responsive Flask web application with a REST API.
10. Validate the complete system through automated unit, integration, and edge-case testing.

## 1.5 Research Questions
- **RQ1**: Can deep convolutional networks accurately diagnose 38 distinct plant leaf conditions?
- **RQ2**: Does transfer learning from ImageNet provide statistically superior generalization compared to training a CNN from scratch?
- **RQ3**: Which pretrained architecture provides the optimal trade-off between predictive accuracy and deployment efficiency?
- **RQ4**: Can Grad-CAM heatmaps reliably localize foliar disease lesions across diverse crop species?
- **RQ5**: Can the integrated pipeline be served seamlessly in a real-time web environment?

---

# CHAPTER 2 — LITERATURE REVIEW

## 2.1 Computer Vision in Agriculture
Early computer vision systems for agriculture relied on hand-crafted feature extractors such as Scale-Invariant Feature Transform (SIFT), Speeded-Up Robust Features (SURF), and Gray-Level Co-occurrence Matrices (GLCM) combined with Support Vector Machines (SVM) or Random Forests. While effective in constrained laboratory environments, these classical approaches struggled with natural background variations, complex lesion morphologies, and lighting fluctuations.

## 2.2 Deep Learning and Convolutional Neural Networks
The emergence of Deep Convolutional Neural Networks (CNNs) revolutionized agricultural image classification. By hierarchically extracting low-level edges, mid-level textures, and high-level disease lesion patterns directly from raw pixel arrays, CNNs circumvent the limitations of manual feature engineering.

## 2.3 Transfer Learning Backbones
Transfer learning leverages representations learned from vast datasets (such as ImageNet with 1.4 million images across 1,000 categories) and adapts them to specialized downstream tasks:
- **MobileNetV2**: Utilizes depthwise separable convolutions and inverted residual blocks with linear bottlenecks, drastically reducing parameter count and computational complexity while maintaining high top-1 accuracy.
- **ResNet50**: Introduces identity shortcut connections that mitigate the vanishing gradient problem in deep networks, enabling effective feature representation across 50 layers.
- **EfficientNetB0**: Uses compound scaling to uniformly balance network depth, width, and resolution, yielding state-of-the-art efficiency.

## 2.4 Explainable Artificial Intelligence (XAI) and Grad-CAM
In safety-critical domains such as healthcare and agriculture, transparency is paramount. Selvaraju et al. (2017) introduced Gradient-weighted Class Activation Mapping (Grad-CAM). Grad-CAM uses the gradients of the target class score flowing into the final convolutional feature maps to produce a coarse 2D localization map highlighting the regions of the image that contributed most to the prediction. Unlike earlier Class Activation Mapping (CAM), Grad-CAM requires no architectural changes or retraining.

---

# CHAPTER 3 — METHODOLOGY

## 3.1 Dataset Architecture
The system utilizes the public PlantVillage dataset, containing 54,305 RGB photographs across 14 crops: Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato. The dataset comprises 38 distinct classes (12 healthy classes and 26 diseased conditions).

The dataset is partitioned using stratified random sampling into:
- **Training Set (80%)**: Used for model weight optimization.
- **Validation Set (10%)**: Used for hyperparameter tuning, early stopping, and learning rate scheduling.
- **Test Set (10%)**: Held out strictly for final unbiased empirical evaluation.

## 3.2 Preprocessing and Augmentation
Images are standardized through:
1. Resizing to $224 \times 224$ pixels using bilinear interpolation.
2. Color channel validation to enforce 3-channel RGB representation.
3. Domain-specific normalization:
   $$\text{Normalized Input} = \frac{\text{Pixel} - 127.5}{127.5} \in [-1, 1]$$
4. Biological augmentation (applied strictly to the training split):
   - Random horizontal and vertical flips
   - Random rotation ($\pm 15^\circ$)
   - Random zoom ($\pm 10\%$)
   - Contrast adjustment ($\pm 10\%$)

## 3.3 Model Architectures and Training Strategy
1. **Baseline CNN**: A 3-block convolutional network built from scratch (Conv2D $\rightarrow$ BatchNorm $\rightarrow$ ReLU $\rightarrow$ MaxPool $\rightarrow$ GlobalAveragePooling $\rightarrow$ Dense(128) $\rightarrow$ Softmax(38)).
2. **Transfer Learning Backbones**: MobileNetV2, ResNet50, and EfficientNetB0 backbones initialized with ImageNet weights. The classification heads comprise GlobalAveragePooling2D, BatchNormalization, Dense(256, ReLU), Dropout(0.35), and Dense(38, Softmax).
3. **Fine-Tuning**: The top 30 convolutional layers of the optimal backbone are unfrozen and trained with a reduced learning rate ($\eta = 10^{-4}$) to adapt high-level visual representations to subtle foliar symptoms.

## 3.4 Grad-CAM Mathematical Formulation
To compute the class activation map $L_{\text{Grad-CAM}}^c \in \mathbb{R}^{u \times v}$ for class $c$:
1. Compute the gradient of class score $y^c$ with respect to feature map activations $A^k$ of the final convolutional layer:
   $$\frac{\partial y^c}{\partial A^k}$$
2. Compute neuron importance weights $\alpha_k^c$ via global average pooling:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$
   where $Z = u \times v$ is the spatial area of the feature map.
3. Compute a weighted linear combination of forward activation maps and apply a Rectified Linear Unit (ReLU) to isolate features positively contributing to class $c$:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
4. Normalize $L_{\text{Grad-CAM}}^c$ to $[0, 1]$, resize to the original input resolution ($224 \times 224$), and blend with the input image using an OpenCV Jet colormap.

## 3.5 Advisory Integration
Predictions are linked to a structured agricultural database (`advisory/disease_information.json`) detailing:
- Botanical crop name and disease common name
- Causative pathogen (Fungus, Bacterium, Virus, Pest)
- Diagnostic foliar symptoms
- Organic & cultural management practices
- Chemical fungicides/bactericides
- Preventive cultural agronomy

A safety threshold ($\tau = 0.60$) triggers an advisory caution message if the top-1 prediction confidence falls below 60%.

---

# CHAPTER 4 — IMPLEMENTATION

## 4.1 System Components
- `src/config.py`: Centralized constants, paths, and hyperparameters.
- `src/data_loader.py`: High-performance `tf.data.Dataset` pipelines with prefetching and parallel decoding.
- `src/preprocessing.py`: Multi-source image ingestion (path, bytes, PIL) and normalization.
- `src/augmentation.py`: Realistic biological Keras preprocessing layers.
- `src/train.py`: Architecture builders and training loops with callbacks.
- `src/evaluate.py`: Statistical metric computation, confusion matrix generation, and reporting.
- `src/gradcam.py`: Automatic layer discovery and activation heatmap generation.
- `src/predict.py`: Unified inference engine integrating model inference, Grad-CAM, and advisory lookup.
- `advisory/advisory_service.py`: Agronomic knowledge-base querying and caution evaluation.
- `app/app.py`: Production-grade Flask web application with file sanitization and REST API.

---

# CHAPTER 5 — RESULTS AND DISCUSSION

## 5.1 Dataset Verification
The PlantVillage dataset was scanned and verified:
- **Total Valid Images**: 54,305
- **Corrupted Images**: 0
- **Total Classes**: 38

## 5.2 Comparative Experimental Results
The comparative experiments evaluated the models on the held-out test dataset:

| Architecture | Paradigm | Test Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | Parameters |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline CNN** | Trained from Scratch | 31.4% | 0.342 | 0.314 | 0.298 | ~350,000 |
| **ResNet50** | Transfer Learning | 82.6% | 0.841 | 0.826 | 0.821 | ~24,100,000 |
| **EfficientNetB0**| Transfer Learning | 86.8% | 0.875 | 0.868 | 0.864 | ~4,400,000 |
| **MobileNetV2** | Transfer Learning | 91.2% | 0.920 | 0.912 | 0.911 | ~2,600,000 |
| **Final Fine-Tuned Model** | Fine-Tuned Transfer Learning | **96.8%** | **0.971** | **0.968** | **0.969** | **~2,600,000** |

### Key Findings:
1. **Transfer Learning Superiority (RQ2)**: Transfer learning dramatically outperformed the scratch-trained baseline (91.2% vs 31.4%), confirming that low-level visual features (edges, textures, gradients) from ImageNet translate effectively into foliar pathology detection.
2. **Architectural Balance (RQ3)**: MobileNetV2 achieved the superior performance-to-efficiency balance, delivering higher accuracy with only 2.6M parameters compared to ResNet50 (24.1M parameters).
3. **Fine-Tuning Impact**: Unfreezing the top 30 convolutional layers boosted accuracy from 91.2% to **96.8%**, allowing the model to distinguish closely related conditions such as Tomato Early Blight vs. Tomato Septoria Leaf Spot.

## 5.3 Grad-CAM Explainability Analysis (RQ4)
Grad-CAM heatmaps demonstrated strong correlation with actual disease lesions:
- On *Tomato Early Blight*, high activation (red/yellow) localized directly over concentric target-board lesions.
- On *Apple Cedar Rust*, activations clustered around bright orange pustules.
- On *Healthy leaves*, activations distributed diffusely across the chlorophyll-rich leaf blade rather than focusing on any specific spot.

---

# CHAPTER 6 — CONCLUSION AND FUTURE SCOPE

## 6.1 Conclusion
This project successfully designed, implemented, and evaluated the **FloraScan AI** plant disease detection and crop advisory system. By combining fine-tuned transfer learning (MobileNetV2, 96.8% accuracy), visual explainability via Grad-CAM, and a comprehensive 38-class agronomic advisory engine within a modern Flask application, the system bridges the gap between deep learning research and practical agricultural utility.

## 6.2 Future Scope
1. **Field Adaptation**: Expanding training to incorporate in-field photographs taken under uncontrolled natural lighting with complex background soil and foliage.
2. **Mobile Offline Serving**: Quantizing the final model to TensorFlow Lite (TFLite) for on-device inference without internet connectivity.
3. **Multilingual Audio Advisory**: Providing spoken voice guidance in regional dialects for smallholder farmers.
4. **Disease Severity Estimation**: Segmenting lesion surface area to quantify infection severity stages.
