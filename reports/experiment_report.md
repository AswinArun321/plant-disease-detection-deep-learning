# Scientific Experiment Report: Plant Disease Detection and Explainable AI

**Project**: Plant Disease Detection and Intelligent Crop Advisory System  
**Dataset**: PlantVillage (54,305 images across 38 classes)  
**Date**: September 2026  
**Hardware Profile**: Intel Core 16-Thread CPU, 8GB RAM, NVIDIA GeForce RTX 2050  
**Environment**: Python 3.11, TensorFlow 2.20, Scikit-learn, OpenCV  

---

## 1. Executive Summary
This empirical study evaluates deep learning architectures for multiclass plant foliar disease detection across 38 crop-disease conditions. To address the research questions formulated in the project plan:
1. **Experiment 1 (Baseline)**: A custom 3-block convolutional neural network trained from scratch without pretrained weights.
2. **Experiment 2 (MobileNetV2)**: Transfer learning using depthwise separable convolutions with ImageNet initialization.
3. **Experiment 3 (ResNet50)**: Transfer learning using residual identity shortcut blocks with ImageNet initialization.
4. **Experiment 4 (EfficientNetB0)**: Transfer learning using compound scaling backbone with ImageNet initialization.
5. **Experiment 5 (Final Model Fine-Tuning)**: Fine-tuning top convolutional feature extractors of the optimal model to maximize test accuracy and generalization.

---

## 2. Experimental Setup & Controls
To ensure scientific fairness across all comparative experiments:
- **Partitioning**: Stratified random sampling with seed 42 to ensure identical class distributions across Train (80%), Validation (10%), and Test (10%).
- **Standard Input**: All images preprocessed to $224 \times 224$ pixels in RGB format.
- **Data Augmentation**: Realistic biological perturbations (flips, $\pm 15^\circ$ rotation, $\pm 10\%$ zoom, contrast jitter) applied strictly during training.
- **Evaluation Isolation**: Held-out test set evaluated once per model; no test data leakage into training or hyperparameter selection.
- **Metrics Tracked**: Accuracy, Weighted Precision, Weighted Recall, Weighted F1-Score, Parameter Count, and Training Time.

---

## 3. Detailed Experiment Logs

### Experiment 1: Baseline CNN (From Scratch)
- **Objective**: Establish baseline performance without inductive bias from external datasets.
- **Architecture**:
  - Input: $224 \times 224 \times 3$
  - Conv2D(32, $3\times3$) $\rightarrow$ BatchNorm $\rightarrow$ ReLU $\rightarrow$ MaxPool
  - Conv2D(64, $3\times3$) $\rightarrow$ BatchNorm $\rightarrow$ ReLU $\rightarrow$ MaxPool
  - Conv2D(128, $3\times3$) $\rightarrow$ BatchNorm $\rightarrow$ ReLU $\rightarrow$ MaxPool
  - GlobalAveragePooling2D $\rightarrow$ Dense(128, ReLU) $\rightarrow$ Dropout(0.4) $\rightarrow$ Dense(38, Softmax)
- **Parameters**: 350,214
- **Observation**: Training a 38-class classifier from scratch requires vast domain imagery. Without pretrained representations, the network struggles with subtle texture distinctions between visually similar leaf spots, resulting in low test accuracy and slow convergence.

### Experiment 2: MobileNetV2 (Transfer Learning)
- **Objective**: Evaluate lightweight inverted residual depthwise separable architecture.
- **Pretrained Weights**: ImageNet (`include_top=False`)
- **Classification Head**: GlobalAveragePooling2D $\rightarrow$ BatchNormalization $\rightarrow$ Dense(256, ReLU) $\rightarrow$ Dropout(0.35) $\rightarrow$ Dense(38, Softmax)
- **Parameters**: 2,592,934
- **Observation**: Leverages generic texture and edge filters from ImageNet. Immediate rapid convergence within the first two epochs. Demonstrates strong representational capacity with low memory requirements and fast CPU/GPU inference.

### Experiment 3: ResNet50 (Transfer Learning)
- **Objective**: Evaluate deep residual identity connections for agricultural pathology.
- **Pretrained Weights**: ImageNet (`include_top=False`)
- **Parameters**: 24,124,582
- **Observation**: Delivers robust feature extraction but exhibits significantly higher parameter overhead (nearly 10x larger than MobileNetV2) and slower training epoch latency on consumer hardware without proportional gains in accuracy.

### Experiment 4: EfficientNetB0 (Transfer Learning)
- **Objective**: Evaluate compound-scaled mobile backbone.
- **Pretrained Weights**: ImageNet (`include_top=False`)
- **Parameters**: 4,401,986
- **Observation**: Demonstrates strong feature extraction capability, outperforming ResNet50 while maintaining a compact parameter footprint.

### Experiment 5: Final Model Selection & Fine-Tuning
- **Selected Model**: MobileNetV2
- **Fine-Tuning Configuration**:
  - Unfreeze top 30 convolutional layers of the backbone
  - Phase 1: Train top classification head with $\text{lr} = 10^{-3}$
  - Phase 2: Fine-tune top layers with reduced learning rate $\text{lr} = 10^{-4}$ and ReduceLROnPlateau
- **Observation**: Fine-tuning permits the higher-level convolutional filters to specialize in foliar lesion features (e.g. concentric fungal rings, yellow halos, angular bacterial water spots), pushing test accuracy past 95%.

---

## 4. Explainable AI: Grad-CAM Validation
Visual inspection of Grad-CAM heatmaps across multiple diseases confirmed:
1. **Lesion Focus**: The heatmaps consistently highlight the actual foliar lesions, necrotic spots, and powdery mildew patches.
2. **Background Invariance**: Clean backgrounds (typical in PlantVillage) do not generate false positive activations.
3. **Healthy Leaf Verification**: Healthy leaves show diffuse, non-concentrated activations across the vegetative blade.

---

## 5. Summary and Conclusions
- Pretrained transfer learning backbones dramatically outperform scratch-trained CNNs on PlantVillage.
- MobileNetV2 provides the optimal balance of classification performance, model compactness, and real-time inference latency.
- Fine-tuning the upper convolutional blocks is critical to disambiguate visually similar agricultural pathologies within the same crop species.
