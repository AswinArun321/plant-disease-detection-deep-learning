# PlantVillage Dataset Information

## Overview
This project uses the public **PlantVillage** dataset, containing 54,305 curated RGB images of plant leaves across 14 crop species and 38 distinct healthy and diseased classes.

## Dataset Structure
The dataset directory contains class-specific subdirectories in the format `Crop___Disease` or `Crop___healthy`:
- Apple (4 classes: Apple scab, Black rot, Cedar apple rust, healthy)
- Blueberry (1 class: healthy)
- Cherry (2 classes: Powdery mildew, healthy)
- Corn / Maize (4 classes: Cercospora leaf spot / Gray leaf spot, Common rust, Northern Leaf Blight, healthy)
- Grape (4 classes: Black rot, Esca / Black Measles, Leaf blight / Isariopsis Leaf Spot, healthy)
- Orange (1 class: Haunglongbing / Citrus greening)
- Peach (2 classes: Bacterial spot, healthy)
- Pepper, Bell (2 classes: Bacterial spot, healthy)
- Potato (3 classes: Early blight, Late blight, healthy)
- Raspberry (1 class: healthy)
- Soybean (1 class: healthy)
- Squash (1 class: Powdery mildew)
- Strawberry (2 classes: Leaf scorch, healthy)
- Tomato (10 classes: Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Spider mites, Target Spot, Tomato Yellow Leaf Curl Virus, Tomato mosaic virus, healthy)

## Acquisition
The dataset is stored in `plantvillage dataset/color/`. Preprocessed splits (training 80%, validation 10%, test 10%) are generated dynamically with fixed random seed 42 to ensure deterministic reproducibility.
