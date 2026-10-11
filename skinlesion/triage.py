"""LangGraph triage pipeline.

    check_quality -> classify -> assess_risk -> explain
          |
          +--(image unusable)--> END

* check_quality : rejects tiny or near-blank images before they reach the model.
* classify      : runs the CNN / transformer and returns the top-k classes.
* assess_risk   : rule-based routing on class and confidence; flags cases for review.
* explain       : plain-language summary from an injected LangChain explainer
                  (skinlesion.llm) when available, otherwise a deterministic template.
                  It never states a diagnosis.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, TypedDict

import numpy as np
from langgraph.graph import END, StateGraph
from PIL import Image

from .config import DISCLAIMER

MALIGNANT_OR_PRECANCEROUS = {"mel", "bcc", "akiec"}
LOW_CONFIDENCE = 0.50
HIGH_RISK_MASS = 0.20  # combined probability of the concerning classes
MIN_SIDE = 64


class TriageState(TypedDict, total=False):
    image: Image.Image
    quality_ok: bool
    quality_message: str
    predictions: list[dict]
    concern_probability: float
    risk: str            # "low" | "elevated" | "high"
    needs_review: bool
    explanation: str
    disclaimer: str


def check_quality(state: TriageState) -> TriageState:
    img = state["image"]
    if min(img.size) < MIN_SIDE:
        return {"quality_ok": False, "quality_message": f"Image is too small (min {MIN_SIDE}px per side)."}
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    if gray.std() < 5:
        return {"quality_ok": False, "quality_message": "Image looks blank or uniform; please retake the photo."}
    return {"quality_ok": True, "quality_message": ""}


def make_classify(predict: Callable[[Image.Image, int], list[dict]]):
    def classify(state: TriageState) -> TriageState:
        return {"predictions": predict(state["image"], 7)}
    return classify


def assess_risk(state: TriageState) -> TriageState:
    preds = state["predictions"]
    concern = sum(p["probability"] for p in preds if p["code"] in MALIGNANT_OR_PRECANCEROUS)
    top = preds[0]
    if top["code"] in MALIGNANT_OR_PRECANCEROUS or concern >= HIGH_RISK_MASS:
        risk = "high"
    elif top["probability"] < LOW_CONFIDENCE:
        risk = "elevated"
    else:
        risk = "low"
    return {
        "concern_probability": round(concern, 4),
        "risk": risk,
        "needs_review": risk != "low" or top["probability"] < LOW_CONFIDENCE,
    }


def _template_explanation(state: TriageState) -> str:
    top = state["predictions"][0]
    parts = [f"The model's closest match is '{top['name']}' ({top['probability']:.0%} confidence)."]
    if state["risk"] == "high":
        parts.append("Some probability falls on lesion types that can be malignant or precancerous, "
                     "so please have this checked by a dermatologist soon.")
    elif state["risk"] == "elevated":
        parts.append("The model is not confident in this result; a clinician should review it.")
    else:
        parts.append("The model did not flag this as high risk, but it misses some skin cancers, so this is "
                     "not reassurance. Have any new, changing or worrying spot checked by a clinician.")
    return " ".join(parts)


def make_explain(explainer: Optional[Callable[[TriageState], Optional[str]]]):
    def explain(state: TriageState) -> TriageState:
        text = explainer(state) if explainer else None
        return {"explanation": text or _template_explanation(state), "disclaimer": DISCLAIMER}
    return explain


def _route_after_quality(state: TriageState) -> str:
    return "classify" if state["quality_ok"] else END


def build_graph(predict: Callable[[Image.Image, int], list[dict]],
                explainer: Optional[Callable[[TriageState], Optional[str]]] = None):
    g = StateGraph(TriageState)
    g.add_node("check_quality", check_quality)
    g.add_node("classify", make_classify(predict))
    g.add_node("assess_risk", assess_risk)
    g.add_node("explain", make_explain(explainer))
    g.set_entry_point("check_quality")
    g.add_conditional_edges("check_quality", _route_after_quality, {"classify": "classify", END: END})
    g.add_edge("classify", "assess_risk")
    g.add_edge("assess_risk", "explain")
    g.add_edge("explain", END)
    return g.compile()


def run_triage(graph: Any, image: Image.Image) -> TriageState:
    result = graph.invoke({"image": image})
    result.pop("image", None)
    result.setdefault("disclaimer", DISCLAIMER)
    return result
