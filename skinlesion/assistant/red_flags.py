"""Deterministic red-flag detection from patient-reported intake data.

Every rule is a plain Python predicate on ``PatientIntake`` fields.
The LLM is never involved in deciding red flags.
"""
from __future__ import annotations

from typing import Callable

from .state import PatientIntake

# ---------------------------------------------------------------------------
# Red-flag rules
# ---------------------------------------------------------------------------
# Each entry: (human-readable description, predicate on PatientIntake).
# If the predicate returns True the description is added to the red-flag list.

RED_FLAG_RULES: list[tuple[str, Callable[[PatientIntake], bool]]] = [
    (
        "Rapid growth or change reported",
        lambda i: i.evolution in ("growing", "changing"),
    ),
    (
        "Bleeding or oozing reported",
        lambda i: i.symptoms is not None and "bleeding" in i.symptoms,
    ),
    (
        "A sore not healed in 4+ weeks",
        lambda i: (
            i.duration_weeks is not None
            and i.duration_weeks >= 4
            and i.evolution != "stable"
        ),
    ),
    (
        "New pain reported",
        lambda i: i.symptoms is not None and "pain" in i.symptoms,
    ),
    (
        "Itching with change",
        lambda i: (
            i.symptoms is not None
            and "itching" in i.symptoms
            and i.evolution in ("growing", "changing")
        ),
    ),
    (
        "New lesion after age 50",
        lambda i: (
            i.age_band in ("50_64", "65_plus")
            and i.duration_weeks is not None
            and i.duration_weeks <= 12
        ),
    ),
    (
        "Multiple colours or irregular border described",
        lambda i: bool(
            i.visual_descriptors
            and {"multiple_colours", "irregular_border"} & set(i.visual_descriptors)
        ),
    ),
]

# The message shown immediately when any red flag fires.
RED_FLAG_URGENT_MESSAGE = (
    "Based on what you've described, we recommend you arrange "
    "an in-person dermatology visit soon."
)


def check_red_flags(intake: PatientIntake) -> list[str]:
    """Return a list of red-flag descriptions that apply to *intake*.

    Returns an empty list when nothing is flagged.
    """
    return [desc for desc, test in RED_FLAG_RULES if test(intake)]
