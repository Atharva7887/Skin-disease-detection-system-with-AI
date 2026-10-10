"""LangGraph triage pipeline.

    check_quality -> classify -> assess_risk -> explain
          |
          +--(image unusable)--> END

* check_quality : rejects tiny or near-blank images before they reach the model.
* classify      : runs the CNN / transformer and returns the top-k classes.
* assess_risk   : rule-based routing on class and confidence; flags cases for review.
* explain       : plain-language summary. Uses Claude when ANTHROPIC_API_KEY is set,
                  otherwise a deterministic template. It never states a diagnosis.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Optional, TypedDict

import numpy as np
from langgraph.graph import END, StateGraph
from PIL import Image

from .config import DISCLAIMER

MALIGNANT_OR_PRECANCEROUS = {"mel", "bcc", "akiec"}
LOW_CONFIDENCE = 0.50
HIGH_RISK_MASS = 0.20  # combined probability of the concerning classes
MIN_SIDE = 64
EXPLAIN_MODEL = os.getenv("EXPLAIN_MODEL", "claude-sonnet-5-5")


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
        parts.append("The model considers this likely benign, but only a clinician can confirm that.")
    return " ".join(parts)


def _claude_explanation(state: TriageState) -> Optional[str]:
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic

        facts = "\n".join(f"- {p['name']}: {p['probability']:.1%}" for p in state["predictions"][:3])
        msg = anthropic.Anthropic().messages.create(
            model=EXPLAIN_MODEL,
            max_tokens=300,
            system=("You write short, calm, plain-language summaries of the output of an experimental "
                    "skin-lesion classifier. Never state or imply a diagnosis. Never reassure the user "
                    "that a lesion is safe. Always recommend seeing a dermatologist when risk is "
                    "'high' or 'elevated'. Maximum 4 sentences."),
            messages=[{"role": "user", "content": (
                f"Model output (top classes):\n{facts}\n"
                f"Risk flag: {state['risk']}. Combined probability of malignant/precancerous classes: "
                f"{state['concern_probability']:.1%}.")}],
        )
        return "".join(b.text for b in msg.content if b.type == "text").strip() or None
    except Exception:  # network/auth/quota: fall back rather than fail the request
        return None


def explain(state: TriageState) -> TriageState:
    return {"explanation": _claude_explanation(state) or _template_explanation(state),
            "disclaimer": DISCLAIMER}


def _route_after_quality(state: TriageState) -> str:
    return "classify" if state["quality_ok"] else END


def build_graph(predict: Callable[[Image.Image, int], list[dict]]):
    g = StateGraph(TriageState)
    g.add_node("check_quality", check_quality)
    g.add_node("classify", make_classify(predict))
    g.add_node("assess_risk", assess_risk)
    g.add_node("explain", explain)
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
