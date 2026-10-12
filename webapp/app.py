"""Flask app: web UI at / and JSON API at /api/predict.

    set MODEL_PATH=a.pt,b.pt,c.pt   (default: the 3-model ensemble in DEFAULT_MODEL_PATH)
    python -m webapp.app
"""
from __future__ import annotations

import io
import json
import os
import time
from collections import defaultdict, deque
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

from langgraph.types import Command
from skinlesion.assistant.graph import build_assistant_graph
from skinlesion.assistant.prompts import CONSENT_CHIPS, CONSENT_MESSAGE
from skinlesion.assistant.render import render_report_html, render_report_markdown
from skinlesion.assistant.sessions import SessionManager
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
CALIBRATION_FILE = Path(__file__).with_name("calibration.json")
CARD_FILE = Path(__file__).with_name("model_card.json")


def load_card() -> dict | None:
    return json.loads(CARD_FILE.read_text(encoding="utf-8")) if CARD_FILE.is_file() else None


def performance_note(card: dict | None) -> str:
    """Miss / false-alarm rates quoted in the UI, read from the generated model card (single source of truth)."""
    if not card:
        return PERFORMANCE_NOTE
    r, n = card["ensemble"]["risk_flag"], card["ensemble"]["isic2018"]["n_images"]
    return (f"In testing on {n:,} dermoscopy images, this tool did not flag about {round((1 - r['melanoma_flagged']) * 100)}% "
            f"of melanomas, and flagged about {round(r['nevus_flagged'] * 100)}% of ordinary moles. "
            "A result that is not flagged is not a reassurance.")


def load_calibration(paths: list[str]) -> dict | None:
    """Temperature fitted by scripts/fit_calibration.py, used only if it was fitted for these exact checkpoints."""
    if not CALIBRATION_FILE.is_file():
        return None
    cal = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
    return cal if set(cal.get("members", [])) == set(paths) else None


ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
Image.MAX_IMAGE_PIXELS = 50_000_000  # guard against decompression bombs


CHAT_LIMIT, CHAT_WINDOW = 10, 60  # requests per client per window (seconds)


def create_app(predict=None, model_path: str | None = None, llm=None,
               explainer=None, chat_graph=None, heatmap_fn=None,
               assistant_graph=None, session_manager=None) -> Flask:
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
            import torch
            from skinlesion.predict import Predictor
            cal = load_calibration(paths)
            use_tta = os.getenv("TTA", "1") == "1"  # 8-view averaging, ~8x compute; the documented numbers assume it
            predictor = Predictor(path, tta=use_tta, temperature=cal["temperature"] if cal else 1.0)
            app.logger.info("loaded %d model(s); TTA=%s; temperature=%s", len(paths), use_tta, predictor.temperature)
            predict, heatmap_fn = predictor.predict, predictor.heatmap
        else:
            load_error = f"Model file not found at '{path}'. Train one with train.py or set MODEL_PATH."
    card = load_card()
    note = performance_note(card)
    llm = llm or get_llm()
    explainer = explainer or make_explainer(llm)
    chat_graph = chat_graph or build_chat_graph(llm)
    if predict is not None:
        graph = build_graph(predict, explainer=explainer)
        if assistant_graph is None:
            assistant_graph = build_assistant_graph(
                predict_fn=predict,
                heatmap_fn=heatmap_fn,
                performance_note=note,
            )
    def _wipe_session_data(sid: str) -> None:
        """Delete everything stored for a finished/expired session: graph state (the patient's answers) and any cached photo."""
        from skinlesion.assistant.nodes import _IMAGE_STORE

        _IMAGE_STORE.pop(sid, None)
        saver = getattr(assistant_graph, "checkpointer", None)
        if saver is not None and hasattr(saver, "delete_thread"):
            saver.delete_thread(sid)

    session_mgr = session_manager or SessionManager(on_expire=_wipe_session_data)
    app.extensions["session_manager"] = session_mgr
    app.extensions["assistant_graph"] = assistant_graph
    hits = defaultdict(deque)

    def page(template, **ctx):
        return render_template(template, disclaimer=DISCLAIMER, performance_note=note, ready=graph is not None,
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

    @app.get("/technical")
    def technical():
        return page("technical.html", card=load_card())

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

    # -----------------------------------------------------------------------
    # Clinical Screening Assistant session endpoints
    # -----------------------------------------------------------------------

    @app.post("/api/session")
    def api_session_create():
        if assistant_graph is None:
            return jsonify(error="Screening assistant unavailable (models not loaded)."), 503
        entry = session_mgr.create(request.remote_addr)
        if entry is None:
            return jsonify(error="Too many active sessions or rate limit exceeded. Please wait a moment."), 429

        out = assistant_graph.invoke({}, config=entry.thread_config)
        int_val = out["__interrupt__"][0].value if "__interrupt__" in out and out["__interrupt__"] else {}
        return jsonify({
            "session_id": entry.session_id,
            "type": int_val.get("type", "consent"),
            "message": int_val.get("message", CONSENT_MESSAGE),
            "chips": int_val.get("chips", CONSENT_CHIPS),
        })

    @app.post("/api/session/<sid>/message")
    def api_session_message(sid: str):
        if assistant_graph is None:
            return jsonify(error="Assistant unavailable."), 503
        entry = session_mgr.get(sid)
        if entry is None:
            return jsonify(error="Session expired or not found. Please start a new session."), 404
        if not session_mgr.check_rate(request.remote_addr):
            return jsonify(error="Rate limit exceeded. Please wait a moment."), 429

        body = request.get_json(silent=True) or {}
        text = str(body.get("text", "")).strip()
        if not text:
            return jsonify(error="Missing 'text' in request body."), 400

        out = assistant_graph.invoke(Command(resume=text), config=entry.thread_config)
        snapshot = assistant_graph.get_state(entry.thread_config)

        if "__interrupt__" in out and out["__interrupt__"]:
            int_val = out["__interrupt__"][0].value
            return jsonify({
                "status": "in_progress",
                "type": int_val.get("type"),
                "message": int_val.get("message"),
                "chips": int_val.get("chips", []),
                "field": int_val.get("field"),
                "retry": int_val.get("retry", False),
            })
        else:
            rep = snapshot.values.get("report")
            consent_given = snapshot.values.get("consent_given", False)
            return jsonify({
                "status": "complete" if rep else ("declined" if not consent_given else "ended"),
                "consent_given": consent_given,
                "report_ready": rep is not None,
                "message": "Screening complete." if rep else "Session ended.",
            })

    @app.post("/api/session/<sid>/image")
    def api_session_image(sid: str):
        if assistant_graph is None:
            return jsonify(error="Assistant unavailable."), 503
        entry = session_mgr.get(sid)
        if entry is None:
            return jsonify(error="Session expired or not found. Please start a new session."), 404
        if not session_mgr.check_rate(request.remote_addr):
            return jsonify(error="Rate limit exceeded. Please wait a moment."), 429

        file = request.files.get("image")
        if file is None or file.filename == "":
            return jsonify(error="Send an image in the 'image' form field."), 400

        try:
            img_bytes = file.read()
            probe = Image.open(io.BytesIO(img_bytes))
            if probe.format not in ALLOWED_FORMATS:
                return jsonify(error="Only JPEG, PNG or WEBP images are accepted."), 415
            probe.load()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            return jsonify(error="File is not a valid image."), 400

        out = assistant_graph.invoke(Command(resume=img_bytes), config=entry.thread_config)
        snapshot = assistant_graph.get_state(entry.thread_config)

        if "__interrupt__" in out and out["__interrupt__"]:
            int_val = out["__interrupt__"][0].value
            return jsonify({
                "status": "in_progress",
                "type": int_val.get("type"),
                "message": int_val.get("message"),
                "retry": int_val.get("retry", False),
            })
        else:
            rep = snapshot.values.get("report")
            return jsonify({
                "status": "complete" if rep else "ended",
                "report_ready": rep is not None,
                "final_risk": snapshot.values.get("final_risk"),
            })

    @app.get("/api/session/<sid>/report")
    def api_session_report(sid: str):
        if assistant_graph is None:
            return jsonify(error="Assistant unavailable."), 503
        entry = session_mgr.get(sid)
        if entry is None:
            return jsonify(error="Session expired or not found."), 404

        snapshot = assistant_graph.get_state(entry.thread_config)
        rep = snapshot.values.get("report")
        if not rep:
            return jsonify(error="Report is not ready yet."), 404

        md = render_report_markdown(rep)
        return jsonify({
            "report": rep,
            "markdown": md,
            "html_url": f"/api/session/{sid}/report/html",
        })

    @app.get("/api/session/<sid>/report/html")
    def api_session_report_html(sid: str):
        if assistant_graph is None:
            return jsonify(error="Assistant unavailable."), 503
        entry = session_mgr.get(sid)
        if entry is None:
            return "Session expired or not found.", 404

        snapshot = assistant_graph.get_state(entry.thread_config)
        rep = snapshot.values.get("report")
        if not rep:
            return "Report is not ready yet.", 404

        html_content = render_report_html(rep)
        return html_content, 200, {"Content-Type": "text/html; charset=utf-8"}

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error="Image exceeds the 8 MB limit."), 413

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=False)
