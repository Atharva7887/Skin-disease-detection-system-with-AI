"""LangGraph node functions for the Clinical Screening Assistant.

Each function takes ``AssistantState`` and returns partial state updates.
Nodes that need user input call ``interrupt()`` and resume via ``Command(resume=...)``.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import io
import numpy as np
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt
from PIL import Image, ImageFilter

_IMAGE_STORE: dict[str, Image.Image] = {}

from ..config import DISCLAIMER, PERFORMANCE_NOTE
from ..triage import MALIGNANT_OR_PRECANCEROUS, HIGH_RISK_MASS, LOW_CONFIDENCE, MIN_SIDE
from .fusion import compute_final_risk
from .intake_fallback import (
    IntakeQuestion,
    next_unanswered_field,
    parse_chip_response,
    question_for_field,
)
from .prompts import CONSENT_CHIPS, CONSENT_MESSAGE, RED_FLAG_URGENT_MESSAGE, REPORT_LIMITATIONS
from .red_flags import check_red_flags
from .state import INTAKE_FIELD_ORDER, AssistantState, PatientIntake, Report, VisionResult

MAX_INTAKE_TURNS = 6
MAX_FIELD_RETRIES = 2
MAX_IMAGE_RETRIES = 2

# Blur detection: a sharp image loses most of its fine detail when blurred a little more, while an image that
# is already blurry loses little. We compare the Laplacian energy before and after an extra Gaussian blur.
# Measured on 300 real HAM10000 validation images vs artificially defocused copies (256 px working size):
# sharp originals score 0.03 to 0.12 (95th percentile 0.119), mildly blurred ones 0.54 to 0.81, stronger blur
# higher still. 0.5 therefore rejects 0% of the sharp images and about 97% of copies blurred with sigma 1.5 AT THE
# 256 PX WORKING SCALE (about 0.6% of the image width, i.e. sigma ~3.5 on a 600 px photo). Slighter softness is
# NOT caught: sigma 2 on a 600 px image is rejected only ~1% of the time, sigma 5 100% (300 unseen ISIC 2018
# images; all 300 sharp originals were accepted). This is a gate for obviously blurry photos, not a sharpness
# guarantee. Unlike a raw Laplacian-variance cutoff it does not depend on image resolution or skin texture.
# (The earlier cutoff, variance >= 50 at full resolution, rejected about half of real dermoscopy images.)
BLUR_WORK_SIZE = 256
BLUR_EXTRA_SIGMA = 1.5
BLUR_RATIO_MAX = 0.5
BLUR_MIN_ENERGY = 0.5   # below this the image is essentially flat
# Saturation mean threshold for glare detection.
GLARE_SATURATION_THRESHOLD = 15.0


# ---------------------------------------------------------------------------
# consent
# ---------------------------------------------------------------------------

def consent_node(state: AssistantState) -> dict:
    """Show the consent screen and wait for the user's answer."""
    consent_response = interrupt({
        "type": "consent",
        "message": CONSENT_MESSAGE,
        "chips": CONSENT_CHIPS,
    })

    text = str(consent_response).strip().lower()
    accepted = text in (
        "i understand, continue",
        "i understand",
        "continue",
        "yes",
        "accept",
        "ok",
        "okay",
    )
    messages = [AIMessage(content=CONSENT_MESSAGE)]
    if consent_response:
        messages.append(HumanMessage(content=str(consent_response)))
    if accepted:
        messages.append(AIMessage(content="Thank you. Let's begin with a few questions about your skin concern."))
    else:
        messages.append(AIMessage(content="No problem. If you change your mind, you can start a new session anytime."))
    return {
        "consent_given": accepted,
        "messages": messages,
        "intake": PatientIntake().model_dump(),
        "intake_turns": 0,
        "intake_field_retries": {},
        "intake_complete": False,
    }


# ---------------------------------------------------------------------------
# intake (deterministic fallback)
# ---------------------------------------------------------------------------

def _build_chip_message(question: IntakeQuestion) -> str:
    """Format a question + chip labels for display."""
    chips = " · ".join(f"`{label}`" for label in question.chips)
    return f"{question.text}\n\n{chips} · `Skip`"


def intake_node(state: AssistantState) -> dict:
    """Collect one intake field per turn using the deterministic questionnaire.

    Interrupts to wait for user input.  Resumes with the user's chip selection
    or free text.
    """
    intake = PatientIntake.model_validate(state.get("intake", {}))
    turns = state.get("intake_turns", 0)
    retries = dict(state.get("intake_field_retries", {}))

    # Find the next field to ask about.
    field = next_unanswered_field(intake)

    if field is None or turns >= MAX_INTAKE_TURNS:
        # All fields answered or turn limit reached.
        return {"intake": intake.model_dump(), "intake_complete": True}

    question = question_for_field(field)
    if question is None:
        return {"intake": intake.model_dump(), "intake_complete": True}

    # Ask the question and wait for user input.
    question_text = _build_chip_message(question)
    user_response = interrupt({
        "type": "intake",
        "message": question_text,
        "chips": list(question.chips.keys()) + ["Skip"],
        "field": field,
    })

    messages = [
        AIMessage(content=question_text),
        HumanMessage(content=str(user_response)),
    ]

    # Parse the response.
    updates = parse_chip_response(question, str(user_response))

    if not updates and str(user_response).strip().lower() != "skip":
        # Unrecognised input — count as a retry.
        field_retries = retries.get(field, 0) + 1
        retries[field] = field_retries
        if field_retries < MAX_FIELD_RETRIES:
            messages.append(AIMessage(
                content=f"I didn't catch that. Could you pick one of the options above, or say `Skip`?"
            ))
            return {
                "messages": messages,
                "intake_turns": turns + 1,
                "intake_field_retries": retries,
            }
        # Retries exhausted — skip this field.
        messages.append(AIMessage(content="No problem, let's move on."))

    # Apply updates.
    if updates:
        intake_dict = intake.model_dump()
        intake_dict.update(updates)
        intake = PatientIntake.model_validate(intake_dict)

    complete = (next_unanswered_field(intake) is None) or ((turns + 1) >= MAX_INTAKE_TURNS)
    return {
        "messages": messages,
        "intake": intake.model_dump(),
        "intake_turns": turns + 1,
        "intake_field_retries": retries,
        "intake_complete": complete,
    }


# ---------------------------------------------------------------------------
# red_flag_check
# ---------------------------------------------------------------------------

def red_flag_check_node(state: AssistantState) -> dict:
    """Apply deterministic red-flag rules to the intake data."""
    intake = PatientIntake.model_validate(state.get("intake", {}))
    flags = check_red_flags(intake)
    messages = []
    if flags:
        flag_text = "\n".join(f"• {f}" for f in flags)
        messages.append(AIMessage(
            content=f"⚠️ {RED_FLAG_URGENT_MESSAGE}\n\nConcerns noted:\n{flag_text}"
        ))
    return {"red_flags": flags, "messages": messages}


# ---------------------------------------------------------------------------
# image_gate
# ---------------------------------------------------------------------------

def _laplacian_energy(gray: Image.Image) -> float:
    a = np.asarray(gray, dtype=np.float64)
    lap = a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:] - 4 * a[1:-1, 1:-1]
    return float(lap.var())


def _check_blur(image: Image.Image) -> bool:
    """Return ``True`` if the image is acceptably sharp (see the constants above for how this was calibrated)."""
    gray = image.convert("L").resize((BLUR_WORK_SIZE, BLUR_WORK_SIZE), Image.BILINEAR)
    energy = _laplacian_energy(gray)
    if energy < BLUR_MIN_ENERGY:
        return False
    retained = _laplacian_energy(gray.filter(ImageFilter.GaussianBlur(BLUR_EXTRA_SIGMA))) / energy
    return retained <= BLUR_RATIO_MAX


def _check_glare(image: Image.Image) -> bool:
    """Return ``True`` if the image does not have extreme glare (adequate saturation)."""
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32)
    saturation_mean = hsv[:, :, 1].mean()
    return saturation_mean >= GLARE_SATURATION_THRESHOLD


def image_gate_node(state: AssistantState, config: RunnableConfig = None) -> dict:
    """Prompt for a photo, run quality checks, and retry up to MAX_IMAGE_RETRIES times."""
    retries = state.get("image_retries", 0)

    # Ask for the image.
    image_data = interrupt({
        "type": "image_upload",
        "message": "Please upload a clear, well-lit photo of your skin concern.",
        "retry": retries > 0,
    })

    messages = []
    if retries == 0:
        messages.append(AIMessage(content="Please upload a clear, well-lit photo of your skin concern."))

    # Handle bytes or Image
    if isinstance(image_data, (bytes, bytearray)):
        try:
            image_data = Image.open(io.BytesIO(image_data))
            image_data.load()
        except Exception:
            image_data = None

    if not isinstance(image_data, Image.Image):
        messages.append(AIMessage(content="I couldn't read that image. Please try uploading again."))
        return {
            "messages": messages,
            "image_ok": False,
            "image_retries": retries + 1,
        }

    # --- Quality checks ---
    # 1. Existing check_quality logic: size and blankness.
    if min(image_data.size) < MIN_SIDE:
        messages.append(AIMessage(
            content=f"The image is too small (minimum {MIN_SIDE}px per side). Please retake the photo."
        ))
        return {"messages": messages, "image_ok": False, "image_retries": retries + 1}

    gray = np.asarray(image_data.convert("L"), dtype=np.float32)
    if gray.std() < 5:
        messages.append(AIMessage(
            content="The image looks blank or uniform. Please retake the photo."
        ))
        return {"messages": messages, "image_ok": False, "image_retries": retries + 1}

    # 2. Blur check.
    if not _check_blur(image_data):
        messages.append(AIMessage(
            content="The image appears blurry. Please retake with a steadier hand and good lighting."
        ))
        return {"messages": messages, "image_ok": False, "image_retries": retries + 1}

    # 3. Glare check.
    if not _check_glare(image_data):
        messages.append(AIMessage(
            content="The image has too much glare or is washed out. Try avoiding direct flash."
        ))
        return {"messages": messages, "image_ok": False, "image_retries": retries + 1}

    # Cache image in memory for vision node (never placed in state/checkpointer)
    thread_id = (config or {}).get("configurable", {}).get("thread_id", "default")
    _IMAGE_STORE[thread_id] = image_data

    messages.append(AIMessage(content="Image received. Analysing now…"))
    return {"messages": messages, "image_ok": True, "image_retries": retries}


# ---------------------------------------------------------------------------
# vision
# ---------------------------------------------------------------------------

def make_vision_node(
    predict_fn: Callable[[Image.Image, int], list[dict]],
    heatmap_fn: Optional[Callable[[Image.Image, str], str]] = None,
):
    """Create the vision node.  *predict_fn* is ``Predictor.predict``."""

    def vision_node(state: AssistantState, config: RunnableConfig = None) -> dict:
        """Run the existing vision pipeline. Symptom text is never passed to the classifier."""
        thread_id = (config or {}).get("configurable", {}).get("thread_id", "default")
        image = _IMAGE_STORE.pop(thread_id, None)
        if image is None and config:
            image = (config or {}).get("configurable", {}).get("image")
        if image is None:
            raise ValueError("No image available for vision_node")

        preds = predict_fn(image, 7)

        # Replicate assess_risk logic (from triage.py) to get the image risk.
        raw = lambda p: p.get("raw_probability", p["probability"])  # noqa: E731
        concern = sum(raw(p) for p in preds if p["code"] in MALIGNANT_OR_PRECANCEROUS)
        top = preds[0]
        top_raw = max(raw(p) for p in preds)
        if top["code"] in MALIGNANT_OR_PRECANCEROUS or concern >= HIGH_RISK_MASS:
            image_risk = "high"
        elif top_raw < LOW_CONFIDENCE:
            image_risk = "elevated"
        else:
            image_risk = "low"

        heatmap_url = None
        if heatmap_fn is not None:
            try:
                heatmap_url = heatmap_fn(image, top["code"])
            except Exception:
                pass

        vision = VisionResult(
            predictions=preds,
            risk=image_risk,
            concern_probability=round(concern, 4),
            heatmap_url=heatmap_url,
        )

        return {"vision": vision.model_dump()}

    return vision_node


# ---------------------------------------------------------------------------
# fusion
# ---------------------------------------------------------------------------

def fusion_node(state: AssistantState) -> dict:
    """Fuse image and symptom risk: ``final_risk = max(image_risk, symptom_risk)``."""
    intake = PatientIntake.model_validate(state.get("intake", {}))
    vision = VisionResult.model_validate(state.get("vision", {}))
    red_flags = state.get("red_flags", [])

    final = compute_final_risk(vision.risk, red_flags, intake)
    return {"final_risk": final}


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def make_report_node(
    performance_note: str = PERFORMANCE_NOTE,
    report_summary_fn: Optional[Callable[[dict], Optional[str]]] = None,
):
    """Create the report node.

    *report_summary_fn*, if provided, generates an optional LLM summary
    which is then run through the faithfulness verifier.
    """
    from .verify import verify_report

    def report_node(state: AssistantState) -> dict:
        intake = PatientIntake.model_validate(state.get("intake", {}))
        vision = VisionResult.model_validate(state.get("vision", {}))
        red_flags = state.get("red_flags", [])
        final_risk = state.get("final_risk", vision.risk)

        top_classes = [
            {
                "code": p["code"],
                "name": p["name"],
                "calibrated_probability": p["probability"],
                "probability": p["probability"],
                "raw_probability": p.get("raw_probability", p["probability"]),
            }
            for p in vision.predictions[:3]
        ]

        report = Report(
            intake=intake,
            red_flags=red_flags,
            top_classes=top_classes,
            image_risk=vision.risk,
            concern_probability=vision.concern_probability,
            heatmap_data_url=vision.heatmap_url,
            final_risk=final_risk,
            limitations=REPORT_LIMITATIONS,
            performance_note=performance_note,
            disclaimer=DISCLAIMER,
        )

        # Optional LLM summary with faithfulness verification.
        if report_summary_fn is not None:
            try:
                summary = report_summary_fn(report.model_dump())
                if summary:
                    violations = verify_report(report, summary)
                    if not violations:
                        report.generated_summary = summary
                    # else: silently discard the unfaithful summary.
            except Exception:
                pass  # LLM failure — use the structured report only.

        risk_label = {"low": "Low", "elevated": "Elevated", "high": "High"}[final_risk]
        messages = [AIMessage(
            content=f"Your screening report is ready. Overall risk level: **{risk_label}**.\n\n"
                    f"You can view and download the full report now."
        )]

        return {"report": report.model_dump(), "messages": messages}

    return report_node
