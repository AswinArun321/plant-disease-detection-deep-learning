"""
Flask Web Application for Plant Disease Detection and Intelligent Crop Advisory System.
"""
import os
import sys
import uuid
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash

# Ensure project root is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src import config
from src.predict import DiseasePredictor
from advisory.advisory_service import AdvisoryService

app = Flask(__name__)
app.secret_key = "plant-disease-advisory-secret-key-2026"
app.config["UPLOAD_FOLDER"] = config.UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Global Predictor instance (lazy-loaded or initialized)
_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = DiseasePredictor()
    return _predictor

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS

@app.route("/")
def index():
    """Home landing page with drag-and-drop uploader and crop categories."""
    try:
        advisory_service = AdvisoryService()
        classes = list(advisory_service.data.keys())
        
        # Group diseases by crop for clean directory display
        crops_dict = {}
        for cls in sorted(classes):
            info = advisory_service.data[cls]
            crop = info.get("crop", "Unknown")
            disease = info.get("disease_name", cls)
            if crop not in crops_dict:
                crops_dict[crop] = []
            crops_dict[crop].append({
                "raw": cls,
                "disease": disease,
                "is_healthy": "healthy" in cls.lower()
            })
    except Exception as e:
        crops_dict = {}
        classes = []

    return render_template("index.html", crops_dict=crops_dict, total_classes=len(classes))

@app.route("/predict", methods=["POST"])
def predict():
    """Handle image upload, execute diagnosis pipeline, and render result page."""
    if "file" not in request.files:
        flash("No image file provided in the upload request.", "error")
        return redirect(url_for("index"))
        
    file = request.files["file"]
    if file.filename == "":
        flash("No file was selected. Please choose a leaf image.", "error")
        return redirect(url_for("index"))
        
    if not allowed_file(file.filename):
        flash("Unsupported file format. Please upload a JPG, JPEG, or PNG image.", "error")
        return redirect(url_for("index"))

    try:
        # Generate safe unique filename
        ext = file.filename.rsplit(".", 1)[1].lower()
        unique_id = uuid.uuid4().hex[:10]
        orig_filename = f"leaf_{unique_id}.{ext}"
        gradcam_filename = f"gradcam_{unique_id}.png"
        
        orig_path = os.path.join(app.config["UPLOAD_FOLDER"], orig_filename)
        gradcam_path = os.path.join(app.config["UPLOAD_FOLDER"], gradcam_filename)
        
        file.save(orig_path)
        
        # Execute prediction
        predictor = get_predictor()
        result = predictor.predict(orig_path, gradcam_save_path=gradcam_path)
        
        # Relative URLs for frontend templates
        orig_url = url_for("static", filename=f"uploads/{orig_filename}")
        gradcam_url = url_for("static", filename=f"uploads/{gradcam_filename}")
        
        return render_template(
            "result.html",
            result=result,
            orig_url=orig_url,
            gradcam_url=gradcam_url
        )

    except Exception as e:
        flash(f"Error analyzing image: {str(e)}", "error")
        return render_template("error.html", error_message=str(e)), 500

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """REST API endpoint returning structured JSON diagnosis."""
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file part in request"}), 400
        
    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"status": "error", "message": "Invalid or missing image file"}), 400

    try:
        ext = file.filename.rsplit(".", 1)[1].lower()
        unique_id = uuid.uuid4().hex[:10]
        orig_filename = f"leaf_{unique_id}.{ext}"
        gradcam_filename = f"gradcam_{unique_id}.png"
        
        orig_path = os.path.join(app.config["UPLOAD_FOLDER"], orig_filename)
        gradcam_path = os.path.join(app.config["UPLOAD_FOLDER"], gradcam_filename)
        file.save(orig_path)
        
        predictor = get_predictor()
        result = predictor.predict(orig_path, gradcam_save_path=gradcam_path)
        
        # Clean up absolute filepaths from JSON response
        result["original_image_url"] = f"/static/uploads/{orig_filename}"
        result["gradcam_image_url"] = f"/static/uploads/{gradcam_filename}"
        result["status"] = "success"
        
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return render_template("error.html", error_message="Image file size exceeds the 16MB limit."), 413

@app.errorhandler(404)
def not_found_error(error):
    return render_template("error.html", error_message="The requested page could not be found."), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template("error.html", error_message="An internal server error occurred."), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Plant Disease Detection & Advisory Web Server on http://127.0.0.1:{port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
