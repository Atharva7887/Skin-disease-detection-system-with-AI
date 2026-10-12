"""Pydantic models and LangGraph state for the Clinical Screening Assistant.

PatientIntake      – structured symptom data collected during the intake phase.
VisionResult       – output from the existing Predictor + assess_risk.
Report             – the clinician-ready summary assembled at the end of the pipeline.
AssistantState     – the LangGraph TypedDict that flows through every node.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Optional

from langgraph.graph import add_messages
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Intake
# ---------------------------------------------------------------------------

EvolutionType = Literal["growing", "changing", "stable", "unknown"]
SymptomType = Literal["itching", "bleeding", "pain", "none"]
AnatomyType = Literal["arm", "leg", "back", "face", "scalp", "hand", "foot", "other"]
AgeBandType = Literal["under_20", "20_34", "35_49", "50_64", "65_plus"]
SkinCancerHistoryType = Literal["yes", "no", "unsure"]
SunExposureType = Literal["low", "moderate", "high"]


class PatientIntake(BaseModel):
    """Patient-reported symptom data.  Every field is Optional (user can skip)."""

    duration_weeks: Optional[int] = Field(
        default=None, description="Approximate duration of the lesion in weeks."
    )
    evolution: Optional[EvolutionType] = Field(
        default=None, description="Is the lesion growing, changing, stable, or unknown?"
    )
    symptoms: Optional[list[SymptomType]] = Field(
        default=None, description="Current symptoms: itching, bleeding, pain, or none."
    )
    anatomy: Optional[AnatomyType] = Field(
        default=None, description="Body location of the lesion."
    )
    age_band: Optional[AgeBandType] = Field(
        default=None, description="Patient's age range."
    )
    previous_skin_cancer: Optional[SkinCancerHistoryType] = Field(
        default=None, description="History of skin cancer."
    )
    sun_exposure: Optional[SunExposureType] = Field(
        default=None, description="Level of lifetime sun exposure."
    )
    visual_descriptors: Optional[list[Literal[
        "multiple_colours", "irregular_border", "asymmetric"
    ]]] = Field(
        default=None,
        description="Visual characteristics mentioned by the user.",
    )


# Ordered field list: the intake phase asks in this order.
INTAKE_FIELD_ORDER: list[str] = [
    "duration_weeks",
    "evolution",
    "symptoms",
    "anatomy",
    "age_band",
    "previous_skin_cancer",
    "sun_exposure",
]


# ---------------------------------------------------------------------------
# Vision result (mirrors the existing triage output)
# ---------------------------------------------------------------------------

class VisionResult(BaseModel):
    """Output from the vision pipeline (existing Predictor + assess_risk)."""

    predictions: list[dict] = Field(
        description="Top-k predictions [{code, name, probability, raw_probability}]."
    )
    risk: Literal["low", "elevated", "high"] = Field(
        description="Image-based risk flag from assess_risk."
    )
    concern_probability: float = Field(
        description="Combined raw probability of malignant / precancerous classes."
    )
    heatmap_url: Optional[str] = Field(
        default=None, description="Grad-CAM heatmap as a JPEG data URL."
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

class Report(BaseModel):
    """Clinician-ready structured report."""

    intake: PatientIntake
    red_flags: list[str]

    # Image analysis
    top_classes: list[dict] = Field(
        description="Top predicted classes [{code, name, calibrated_probability}]."
    )
    image_risk: Literal["low", "elevated", "high"]
    concern_probability: float
    heatmap_data_url: Optional[str] = None

    # Fused result
    final_risk: Literal["low", "elevated", "high"]

    # Fixed text (values set by Python, not by the LLM)
    limitations: str
    performance_note: str
    disclaimer: str

    # Optional LLM-generated summary (validated by the verifier)
    generated_summary: Optional[str] = None


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

# Risk ordinal for ``max()`` comparisons.
RISK_ORD: dict[str, int] = {"low": 0, "elevated": 1, "high": 2}

from typing import TypedDict  # noqa: E402 (intentional: after RISK_ORD for readability)


class AssistantState(TypedDict, total=False):
    """The state dict that flows through the assistant LangGraph."""

    messages: Annotated[list[Any], add_messages]

    consent_given: bool

    # Intake
    intake: dict                 # serialised PatientIntake
    intake_turns: int            # max 6
    intake_field_retries: dict   # {field_name: retry_count}
    intake_complete: bool

    # Image
    image_ok: bool
    image_retries: int           # max 2
    image: Optional[Any]         # Temporary PIL Image held only during inference

    # Vision
    vision: dict                 # serialised VisionResult

    # Red flags + fusion
    red_flags: list[str]
    final_risk: str              # "low" | "elevated" | "high"

    # Report
    report: dict                 # serialised Report
