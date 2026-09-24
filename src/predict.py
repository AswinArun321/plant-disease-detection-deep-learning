"""
Inference pipeline combining model prediction, Grad-CAM, and advisory lookup.
"""
import os
import sys
import json
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tensorflow as tf
from src import config
from src.preprocessing import load_and_preprocess_image
from src.gradcam import generate_and_save_gradcam, compute_gradcam, overlay_heatmap
from advisory.advisory_service import AdvisoryService

class DiseasePredictor:
    def __init__(self, model_path=config.FINAL_MODEL_PATH, class_names_path=config.CLASS_NAMES_PATH):
        self.model_path = model_path
        self.class_names_path = class_names_path
        self.model = None
        self.class_names = []
        self.advisory_service = AdvisoryService()
        self._load_model_and_classes()

    def _load_model_and_classes(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}. Train the model first.")
        if not os.path.exists(self.class_names_path):
            raise FileNotFoundError(f"Class names file not found at {self.class_names_path}.")
            
        print(f"Loading trained model from {self.model_path}...")
        self.model = tf.keras.models.load_model(self.model_path)
        with open(self.class_names_path, "r", encoding="utf-8") as f:
            self.class_names = json.load(f)
        print(f"Model loaded successfully with {len(self.class_names)} target classes.")

    def predict(self, image_source, gradcam_save_path=None):
        """
        Run end-to-end diagnosis on an image path, bytes, or PIL Image.
        Returns detailed prediction dictionary.
        """
        # Preprocess
        batch_tensor, pil_img = load_and_preprocess_image(image_source, target_size=config.IMAGE_SIZE)
        
        # Inference
        probabilities = self.model.predict(batch_tensor, verbose=0)[0]
        
        # Top-1
        top_idx = int(np.argmax(probabilities))
        top_class = self.class_names[top_idx]
        top_conf = float(probabilities[top_idx])
        
        # Top-3 predictions
        top_indices = np.argsort(probabilities)[::-1][:3]
        top_candidates = []
        for idx in top_indices:
            cls_name = self.class_names[idx]
            top_candidates.append({
                "class_name": cls_name,
                "confidence": float(probabilities[idx]),
                "confidence_percent": f"{float(probabilities[idx]) * 100:.1f}%",
                "crop": cls_name.split("___")[0].replace("_", " "),
                "disease": cls_name.split("___")[1].replace("_", " ") if "___" in cls_name else cls_name
            })

        # Fetch Advisory
        advisory_info = self.advisory_service.get_advisory(top_class, confidence=top_conf)

        # Grad-CAM Generation
        gradcam_path = None
        if gradcam_save_path is not None:
            try:
                generate_and_save_gradcam(
                    model=self.model,
                    img_tensor=batch_tensor,
                    original_image=pil_img,
                    save_path=gradcam_save_path,
                    class_idx=top_idx
                )
                gradcam_path = gradcam_save_path
            except Exception as e:
                print(f"Warning: Grad-CAM generation failed: {e}")

        return {
            "predicted_class": top_class,
            "crop": advisory_info["crop"],
            "disease_name": advisory_info["disease_name"],
            "pathogen": advisory_info["pathogen"],
            "confidence": top_conf,
            "confidence_percent": f"{top_conf * 100:.1f}%",
            "is_healthy": advisory_info["is_healthy"],
            "warning": advisory_info["warning"],
            "description": advisory_info["description"],
            "symptoms": advisory_info["symptoms"],
            "treatment": advisory_info["treatment"],
            "prevention": advisory_info["prevention"],
            "top_candidates": top_candidates,
            "gradcam_path": gradcam_path
        }
