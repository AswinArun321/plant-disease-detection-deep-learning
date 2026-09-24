"""
Advisory service providing agronomic guidance and safety precautions.
"""
import os
import sys
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import config

class AdvisoryService:
    def __init__(self, data_path=config.DISEASE_INFO_PATH):
        self.data_path = data_path
        self.data = self._load_data()

    def _load_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Advisory data file not found at {self.data_path}")
        with open(self.data_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_advisory(self, class_name, confidence=1.0):
        """
        Retrieve structured disease details and tailored recommendations.
        """
        info = self.data.get(class_name)
        
        # Fallback if class not in dictionary
        if not info:
            parts = class_name.split("___")
            crop = parts[0].replace("_", " ")
            disease = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
            info = {
                "crop": crop,
                "disease_name": disease,
                "pathogen": "Unknown",
                "description": f"Diagnosis for {crop} showing signs of {disease}.",
                "symptoms": ["Visible foliar changes"],
                "treatment": ["Consult local agricultural extension service for confirmed diagnosis."],
                "prevention": ["Maintain field hygiene and proper crop rotation."]
            }

        # Format readable crop and disease name
        result = {
            "class_raw": class_name,
            "crop": info.get("crop", "Unknown"),
            "disease_name": info.get("disease_name", class_name),
            "pathogen": info.get("pathogen", "N/A"),
            "description": info.get("description", ""),
            "symptoms": info.get("symptoms", []),
            "treatment": info.get("treatment", []),
            "prevention": info.get("prevention", []),
            "confidence": float(confidence),
            "is_healthy": "healthy" in class_name.lower(),
            "warning": None
        }

        # Apply safety threshold caution
        if confidence < config.CONFIDENCE_THRESHOLD:
            result["warning"] = (
                f"Low prediction confidence ({confidence*100:.1f}%). "
                "This diagnosis is uncertain; please verify leaf symptoms with an agricultural extension officer "
                "or plant pathologist before taking chemical treatment measures."
            )

        return result
