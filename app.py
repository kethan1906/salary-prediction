"""Small web demo: a form + JSON API around the saved salary pipeline.

Run:  python app.py   (after `python -m src.train`)  ->  http://127.0.0.1:5000
"""
import os

from flask import Flask, jsonify, render_template, request

from src.config import METADATA_PATH, MODEL_PATH
from src.model_loader import load_metadata
from src.predict import ValidationError, predict_detailed


def create_app(model_path=None, metadata_path=None) -> Flask:
    app = Flask(__name__)
    model_path = model_path or MODEL_PATH
    metadata_path = metadata_path or METADATA_PATH

    @app.after_request
    def headers(resp):
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        return resp

    @app.errorhandler(404)
    def not_found(_):
        return jsonify({"error": "Not found"}), 404

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.route("/api/options")
    def options():
        try:
            meta = load_metadata(metadata_path)
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 503
        cats = meta["known_categories"]
        return jsonify({
            "genders": cats["Gender"],
            "education_levels": cats["Education_Level"],
            "job_titles": cats["Job_Title"],
            "training_ranges": meta["training_ranges"],
            "selected_model": meta["selected_model"],
            "typical_error_mae": meta["test_MAE"],
            "test_r2": meta["test_R2"],
        })

    @app.route("/api/predict", methods=["POST"])
    def predict():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Send a JSON object"}), 400
        try:
            result = predict_detailed(payload, model_path, metadata_path)
        except ValidationError as exc:
            return jsonify({"error": str(exc)}), 400
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 503
        meta = load_metadata(metadata_path)
        return jsonify({
            "predicted_salary": round(result["prediction"], 2),
            "typical_error_mae": round(meta["test_MAE"], 2),
            "warnings": result["warnings"],
            "model": meta["selected_model"],
        })

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "5000")),
            debug=os.getenv("FLASK_DEBUG", "0") == "1")
