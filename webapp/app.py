"""Flask app: web UI at / and JSON API at /api/predict.

    set MODEL_PATH=models/best.pt   (default)
    python -m webapp.app
"""
from __future__ import annotations

import io
import os

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

from skinlesion.config import DISCLAIMER
from skinlesion.triage import build_graph, run_triage

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
Image.MAX_IMAGE_PIXELS = 50_000_000  # guard against decompression bombs


def create_app(predict=None, model_path: str | None = None) -> Flask:
    """``predict`` is injectable (callable(image, top_k) -> list[dict]) for tests."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

    graph = None
    load_error = None
    if predict is None:
        path = model_path or os.getenv("MODEL_PATH", "models/best.pt")
        if os.path.isfile(path):
            from skinlesion.predict import Predictor
            predict = Predictor(path).predict
        else:
            load_error = f"Model file not found at '{path}'. Train one with train.py or set MODEL_PATH."
    if predict is not None:
        graph = build_graph(predict)

    @app.get("/")
    def index():
        return render_template("index.html", disclaimer=DISCLAIMER, ready=graph is not None)

    @app.get("/health")
    def health():
        return jsonify(status="ok" if graph else "model_unavailable", detail=load_error)

    @app.post("/api/predict")
    def api_predict():
        if graph is None:
            return jsonify(error=load_error or "Model unavailable."), 503
        file = request.files.get("image")
        if file is None or file.filename == "":
            return jsonify(error="Send an image in the 'image' form field."), 400
        try:
            img = Image.open(io.BytesIO(file.read()))
            if img.format not in ALLOWED_FORMATS:
                return jsonify(error="Only JPEG, PNG or WEBP images are accepted."), 415
            img.load()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            return jsonify(error="File is not a valid image."), 400

        result = run_triage(graph, img.convert("RGB"))
        if not result.get("quality_ok", False):
            return jsonify(error=result.get("quality_message", "Unusable image."),
                           disclaimer=DISCLAIMER), 422
        return jsonify(result)

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error="Image exceeds the 8 MB limit."), 413

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=False)
