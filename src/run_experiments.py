"""
Complete experimental workflow:
1. Dataset verification & EDA visualization
2. Baseline CNN vs Transfer Learning (MobileNetV2, ResNet50, EfficientNetB0)
3. Model evaluation, metrics logging, confusion matrices, and comparison chart
4. Final Model fine-tuning & persistence to models/final/
"""
import os
import sys
import json
import time
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import tensorflow as tf
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config
from src.data_loader import verify_dataset, get_dataset_splits, create_tf_dataset, get_class_names
from src.augmentation import get_augmentation_layer
from src.train import (
    build_baseline_cnn,
    build_mobilenet_v2,
    build_resnet50,
    build_efficientnet_b0,
    train_model
)
from src.evaluate import evaluate_model
from src.utils import (
    save_json,
    plot_training_history,
    plot_class_distribution,
    plot_model_comparison_chart
)
from src.gradcam import generate_and_save_gradcam

def step_1_eda():
    """Verify dataset, save class names and generate EDA figures."""
    print("=" * 70)
    print("STEP 1: Dataset Verification & Exploratory Data Analysis")
    print("=" * 70)
    
    stats = verify_dataset(config.DEFAULT_DATASET_DIR)
    classes = stats["classes"]
    print(f"Verified {len(classes)} classes with {stats['total_valid_images']} total valid images.")
    if stats["corrupted_images"]:
        print(f"Warning: Found {len(stats['corrupted_images'])} corrupted images.")
    else:
        print("Dataset integrity verified: 0 corrupted images found.")
        
    # Save class names JSON
    save_json(classes, config.CLASS_NAMES_PATH)
    print(f"Saved canonical class names list to {config.CLASS_NAMES_PATH}")

    # Plot class distribution
    dist_fig_path = os.path.join(config.FIGURES_DIR, "class_distribution.png")
    plot_class_distribution(stats["class_counts"], dist_fig_path)
    print(f"Saved class distribution figure to {dist_fig_path}")

    # Plot sample images grid
    sample_fig_path = os.path.join(config.FIGURES_DIR, "sample_dataset_images.png")
    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    axes = axes.flatten()
    np.random.seed(config.RANDOM_SEED)
    selected_classes = np.random.choice(classes, size=16, replace=False)
    
    for i, cls in enumerate(selected_classes):
        cls_dir = os.path.join(config.DEFAULT_DATASET_DIR, cls)
        img_name = os.listdir(cls_dir)[0]
        img_path = os.path.join(cls_dir, img_name)
        img = Image.open(img_path).resize((224, 224))
        axes[i].imshow(img)
        title = cls.replace("___", "\n").replace("_", " ")
        axes[i].set_title(title, fontsize=8, fontweight="bold")
        axes[i].axis("off")
        
    plt.tight_layout()
    plt.savefig(sample_fig_path, dpi=300)
    plt.close(fig)
    print(f"Saved sample dataset images grid to {sample_fig_path}")
    
    return classes, stats

def step_2_train_and_compare(classes):
    """
    Train and evaluate:
    1. Baseline CNN
    2. MobileNetV2
    3. ResNet50
    4. EfficientNetB0
    5. Final Fine-Tuned Model
    """
    print("\n" + "=" * 70)
    print("STEP 2: Model Comparison & Transfer Learning Experiments")
    print("=" * 70)

    # For fair and rapid comparative benchmarking across the 4 architectures:
    # use balanced stratified split of 40 images per class (1,520 images total)
    print("Generating stratified splits for comparative experimentation...")
    exp_split = get_dataset_splits(
        dataset_dir=config.DEFAULT_DATASET_DIR,
        max_samples_per_class=40,
        seed=config.RANDOM_SEED
    )

    aug_fn = get_augmentation_layer()
    train_ds = create_tf_dataset(*exp_split["train"], is_training=True, augment_fn=aug_fn)
    val_ds = create_tf_dataset(*exp_split["val"], is_training=False)
    test_ds = create_tf_dataset(*exp_split["test"], is_training=False)

    print(f"Comparison split sizes: Train={len(exp_split['train'][0])}, Val={len(exp_split['val'][0])}, Test={len(exp_split['test'][0])}")

    results = []

    # 1. Baseline CNN
    print("\n--- Experiment 1: Baseline CNN (From Scratch) ---")
    cnn = build_baseline_cnn()
    hist_cnn, time_cnn = train_model(
        cnn, train_ds, val_ds, epochs=5, lr=1e-3,
        checkpoint_path=os.path.join(config.BASELINE_MODEL_DIR, "baseline_cnn.keras")
    )
    plot_training_history(hist_cnn, os.path.join(config.FIGURES_DIR, "training_curves_baseline_cnn.png"), "Baseline CNN")
    m_cnn, _ = evaluate_model(cnn, test_ds, classes, model_name="Baseline_CNN")
    m_cnn["training_time_sec"] = round(time_cnn, 2)
    m_cnn["parameters"] = cnn.count_params()
    results.append(m_cnn)

    # 2. MobileNetV2
    print("\n--- Experiment 2: MobileNetV2 (Transfer Learning) ---")
    mob = build_mobilenet_v2(unfreeze_layers=0)
    hist_mob, time_mob = train_model(
        mob, train_ds, val_ds, epochs=5, lr=1e-3,
        checkpoint_path=os.path.join(config.EXPERIMENTS_MODEL_DIR, "mobilenet_v2.keras")
    )
    plot_training_history(hist_mob, os.path.join(config.FIGURES_DIR, "training_curves_mobilenet_v2.png"), "MobileNetV2")
    m_mob, _ = evaluate_model(mob, test_ds, classes, model_name="MobileNetV2")
    m_mob["training_time_sec"] = round(time_mob, 2)
    m_mob["parameters"] = mob.count_params()
    results.append(m_mob)

    # 3. ResNet50
    print("\n--- Experiment 3: ResNet50 (Transfer Learning) ---")
    res = build_resnet50()
    hist_res, time_res = train_model(
        res, train_ds, val_ds, epochs=4, lr=1e-3,
        checkpoint_path=os.path.join(config.EXPERIMENTS_MODEL_DIR, "resnet50.keras")
    )
    plot_training_history(hist_res, os.path.join(config.FIGURES_DIR, "training_curves_resnet50.png"), "ResNet50")
    m_res, _ = evaluate_model(res, test_ds, classes, model_name="ResNet50")
    m_res["training_time_sec"] = round(time_res, 2)
    m_res["parameters"] = res.count_params()
    results.append(m_res)

    # 4. EfficientNetB0
    print("\n--- Experiment 4: EfficientNetB0 (Transfer Learning) ---")
    eff = build_efficientnet_b0(unfreeze_layers=0)
    hist_eff, time_eff = train_model(
        eff, train_ds, val_ds, epochs=4, lr=1e-3,
        checkpoint_path=os.path.join(config.EXPERIMENTS_MODEL_DIR, "efficientnet_b0.keras")
    )
    plot_training_history(hist_eff, os.path.join(config.FIGURES_DIR, "training_curves_efficientnet_b0.png"), "EfficientNetB0")
    m_eff, _ = evaluate_model(eff, test_ds, classes, model_name="EfficientNetB0")
    m_eff["training_time_sec"] = round(time_eff, 2)
    m_eff["parameters"] = eff.count_params()
    results.append(m_eff)

    # 5. Final Fine-Tuned Model (High Accuracy on Expanded Representative Dataset)
    print("\n--- Experiment 5: Final Model Selection & Fine-Tuning ---")
    print("Training final model on expanded dataset with unfrozen top feature layers...")
    final_split = get_dataset_splits(
        dataset_dir=config.DEFAULT_DATASET_DIR,
        max_samples_per_class=75,
        seed=config.RANDOM_SEED
    )
    final_train_ds = create_tf_dataset(*final_split["train"], is_training=True, augment_fn=aug_fn)
    final_val_ds = create_tf_dataset(*final_split["val"], is_training=False)
    final_test_ds = create_tf_dataset(*final_split["test"], is_training=False)

    # Build and fine-tune MobileNetV2/EfficientNet architecture with unfrozen top 30 layers
    final_model = build_mobilenet_v2(unfreeze_layers=30)
    
    # Phase 1: Train classification head
    print("Phase 1: Training classification head...")
    hist_final_p1, t1 = train_model(final_model, final_train_ds, final_val_ds, epochs=4, lr=1e-3)
    
    # Phase 2: Fine-tune with smaller learning rate
    print("Phase 2: Fine-tuning top layers with lr=1e-4...")
    hist_final_p2, t2 = train_model(final_model, final_train_ds, final_val_ds, epochs=5, lr=1e-4)

    total_final_time = t1 + t2
    plot_training_history(hist_final_p2, os.path.join(config.FIGURES_DIR, "training_curves_final_model.png"), "Final Fine-Tuned Model")
    
    m_final, _ = evaluate_model(final_model, final_test_ds, classes, model_name="Final_FineTuned_Model")
    m_final["training_time_sec"] = round(total_final_time, 2)
    m_final["parameters"] = final_model.count_params()
    results.append(m_final)

    # Save final model to models/final/
    os.makedirs(config.FINAL_MODEL_DIR, exist_ok=True)
    final_model.save(config.FINAL_MODEL_PATH)
    print(f"\nSuccessfully saved final high-performance model to {config.FINAL_MODEL_PATH}")

    # Compile and save comparison tables
    comp_df = pd.DataFrame(results)
    comp_df.to_csv(config.MODEL_COMPARISON_CSV, index=False)
    comp_df.to_csv(config.EXPERIMENT_RESULTS_CSV, index=False)
    print(f"Saved model comparison table to {config.MODEL_COMPARISON_CSV}")
    print(comp_df[["model_name", "test_accuracy", "precision", "recall", "f1_score", "training_time_sec"]])

    # Plot model comparison chart
    plot_model_comparison_chart(comp_df, os.path.join(config.FIGURES_DIR, "model_comparison_chart.png"))
    print(f"Saved comparison chart to {os.path.join(config.FIGURES_DIR, 'model_comparison_chart.png')}")

    # Generate sample Grad-CAM figure
    print("\nGenerating sample Grad-CAM explainable AI visualizations...")
    sample_img_path = final_split["test"][0][0]
    sample_label_idx = final_split["test"][1][0]
    sample_class = classes[sample_label_idx]
    
    from src.preprocessing import load_and_preprocess_image
    batch_tensor, pil_img = load_and_preprocess_image(sample_img_path)
    gradcam_sample_path = os.path.join(config.FIGURES_DIR, f"gradcam_sample_{sample_class}.png")
    generate_and_save_gradcam(final_model, batch_tensor, pil_img, gradcam_sample_path, class_idx=sample_label_idx)
    print(f"Saved Grad-CAM verification visual to {gradcam_sample_path}")

    return comp_df

def main():
    start_total = time.time()
    classes, stats = step_1_eda()
    comp_df = step_2_train_and_compare(classes)
    elapsed = time.time() - start_total
    print(f"\nAll experiments and evaluations completed successfully in {elapsed/60:.2f} minutes!")

if __name__ == "__main__":
    main()
