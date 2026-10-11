"""Flask app: web UI at / and JSON API at /api/predict.

    set MODEL_PATH=a.pt,b.pt,c.pt   (default: the 3-model ensemble in DEFAULT_MODEL_PATH)
    python -m webapp.app
"""
from __future__ import annotations

import io
import os
import time
from collections import defaultdict, deque
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

from skinlesion.config import DISCLAIMER, PERFORMANCE_NOTE
from skinlesion.info import CLASS_INFO, PRECAUTIONS
from skinlesion.llm import build_chat_graph, get_llm, known_codes, make_explainer
from skinlesion.triage import build_graph, run_triage

# Three-model ensemble. ConvNeXt-V2 goes FIRST: the Grad-CAM heatmap is drawn from the first model.
DEFAULT_MODEL_PATH = ",".join([
    "models/convnextv2_combined/best.pt",
    "models/effnetv2s_combined/best.pt",
    "models/maxvit_combined/best.pt",
])
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
Image.MAX_IMAGE_PIXELS = 50_000_000  # guard against decompression bombs


CHAT_LIMIT, CHAT_WINDOW = 10, 60  # requests per client per window (seconds)


def create_app(predict=None, model_path: str | None = None, llm=None,
               explainer=None, chat_graph=None, heatmap_fn=None) -> Flask:
    """``predict`` (callable(image, top_k) -> list[dict]), ``explainer`` and ``chat_graph``
    are injectable for tests. Without them the LLM features use ANTHROPIC_API_KEY if set."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

    graph = None
    load_error = None
    if predict is None:
        path = model_path or os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
        paths = [x.strip() for x in path.split(",") if x.strip()]
        if paths and all(os.path.isfile(x) for x in paths):
            from skinlesion.predict import Predictor
            predictor = Predictor(path)
            predict, heatmap_fn = predictor.predict, predictor.heatmap
        else:
            load_error = f"Model file not found at '{path}'. Train one with train.py or set MODEL_PATH."
    llm = llm or get_llm()
    explainer = explainer or make_explainer(llm)
    chat_graph = chat_graph or build_chat_graph(llm)
    if predict is not None:
        graph = build_graph(predict, explainer=explainer)
    hits = defaultdict(deque)

    def page(template, **ctx):
        return render_template(template, disclaimer=DISCLAIMER, performance_note=PERFORMANCE_NOTE, ready=graph is not None,
                               precautions=PRECAUTIONS, info=CLASS_INFO,
                               chat_enabled=chat_graph is not None, **ctx)

    @app.get("/")
    def home():
        return page("home.html")

    @app.get("/test")
    def take_test():
        samples = sorted(p.name for p in (Path(app.static_folder) / "samples").glob("*.jpg"))
        return page("test.html", samples=samples)

    @app.get("/learn")
    def learn():
        return page("learn.html")

    @app.get("/contact")
    def contact():
        return page("contact.html")

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
        top = result["predictions"][0]["code"]
        result["info"] = CLASS_INFO.get(top)
        if heatmap_fn is not None:
            try:
                result["heatmap"] = heatmap_fn(img.convert("RGB"), top)
            except Exception:  # explanation is optional: never fail the prediction because of it
                app.logger.exception("Grad-CAM failed")
        return jsonify(result)

    @app.post("/api/chat")
    def api_chat():
        if chat_graph is None:
            return jsonify(error="The question assistant is not configured (set ANTHROPIC_API_KEY)."), 503
        now, q = time.monotonic(), hits[request.remote_addr]
        while q and now - q[0] > CHAT_WINDOW:
            q.popleft()
        if len(q) >= CHAT_LIMIT:
            return jsonify(error="Too many questions; please wait a minute."), 429
        q.append(now)
        body = request.get_json(silent=True) or {}
        out = chat_graph.invoke({"question": body.get("question"), "codes": known_codes(body.get("codes"))})
        if out.get("error"):
            return jsonify(error=out["error"]), 400 if "limited" in out["error"] or "enter" in out["error"] else 502
        return jsonify(answer=out["answer"], disclaimer=DISCLAIMER)

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error="Image exceeds the 8 MB limit."), 413

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=False)
