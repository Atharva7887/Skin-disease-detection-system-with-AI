"""Deterministic questionnaire fallback for intake.

Used when there is no LLM API key or when the LLM fails.  Asks a fixed sequence
of multiple-choice questions with quick-reply chips.
"""
from __future__ import annotations

from typing import Optional

from .state import INTAKE_FIELD_ORDER, PatientIntake

# ---------------------------------------------------------------------------
# Question definitions
# ---------------------------------------------------------------------------

_DURATION_CHIPS = {
    "Less than a month": 2,
    "1–3 months": 8,
    "3–12 months": 26,
    "Over a year": 60,
}

_EVOLUTION_CHIPS = {
    "Growing": "growing",
    "Changing shape/colour": "changing",
    "Stable": "stable",
    "Not sure": "unknown",
}

_SYMPTOM_CHIPS = {
    "Itching": ["itching"],
    "Bleeding": ["bleeding"],
    "Pain": ["pain"],
    "None of these": ["none"],
}

_ANATOMY_CHIPS = {
    "Arm": "arm",
    "Leg": "leg",
    "Back": "back",
    "Face": "face",
    "Scalp": "scalp",
    "Hand": "hand",
    "Foot": "foot",
    "Other": "other",
}

_AGE_CHIPS = {
    "Under 20": "under_20",
    "20–34": "20_34",
    "35–49": "35_49",
    "50–64": "50_64",
    "65+": "65_plus",
}

_HISTORY_CHIPS = {
    "Previous skin cancer": ("yes", None),
    "High sun exposure": (None, "high"),
    "Both": ("yes", "high"),
    "Neither": ("no", "low"),
    "Unsure": ("unsure", "moderate"),
}

# ---------------------------------------------------------------------------
# Question model
# ---------------------------------------------------------------------------


class IntakeQuestion:
    """One question in the deterministic questionnaire."""

    __slots__ = ("field", "text", "chips", "multi_select")

    def __init__(
        self,
        field: str,
        text: str,
        chips: dict,
        multi_select: bool = False,
    ):
        self.field = field
        self.text = text
        self.chips = chips          # {label: value}
        self.multi_select = multi_select


QUESTIONS: list[IntakeQuestion] = [
    IntakeQuestion(
        "duration_weeks",
        "How long have you had this skin concern?",
        _DURATION_CHIPS,
    ),
    IntakeQuestion(
        "evolution",
        "Has it been changing recently?",
        _EVOLUTION_CHIPS,
    ),
    IntakeQuestion(
        "symptoms",
        "Are you experiencing any of these?",
        _SYMPTOM_CHIPS,
        multi_select=True,
    ),
    IntakeQuestion(
        "anatomy",
        "Where on your body is it?",
        _ANATOMY_CHIPS,
    ),
    IntakeQuestion(
        "age_band",
        "What is your age range?",
        _AGE_CHIPS,
    ),
    IntakeQuestion(
        "previous_skin_cancer",
        "Any history of skin cancer, or high sun exposure?",
        _HISTORY_CHIPS,
    ),
]


def question_for_field(field_name: str) -> Optional[IntakeQuestion]:
    """Return the ``IntakeQuestion`` for *field_name*, or ``None``."""
    for q in QUESTIONS:
        if q.field == field_name:
            return q
    return None


def next_unanswered_field(intake: PatientIntake) -> Optional[str]:
    """Return the name of the first field in ``INTAKE_FIELD_ORDER`` that is still ``None``."""
    d = intake.model_dump()
    for field in INTAKE_FIELD_ORDER:
        if d.get(field) is None:
            return field
    return None


def parse_chip_response(
    question: IntakeQuestion,
    user_text: str,
) -> dict:
    """Match *user_text* to a chip label (case-insensitive) and return the update dict.

    If the user text is "Skip" or unrecognisable, returns ``{}``.
    For the combined history question, returns updates for two fields.
    """
    text = user_text.strip()
    if text.lower() == "skip":
        return {}

    for label, value in question.chips.items():
        if text.lower() == label.lower():
            if question.field == "previous_skin_cancer":
                # Combined question: value is (cancer_history, sun_exposure)
                cancer, sun = value
                result = {}
                if cancer is not None:
                    result["previous_skin_cancer"] = cancer
                if sun is not None:
                    result["sun_exposure"] = sun
                return result
            return {question.field: value}

    return {}
