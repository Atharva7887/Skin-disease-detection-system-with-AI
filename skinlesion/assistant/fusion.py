"""Risk fusion: combine image-based and symptom-based risk into a single final risk.

The cardinal rule: symptoms / chat / LLM output can only RAISE concern, never lower it.
``final_risk = max(image_risk, symptom_risk)`` where the ordering is low < elevated < high.
"""
from __future__ import annotations

from .state import RISK_ORD, PatientIntake


def max_risk(a: str, b: str) -> str:
    """Return whichever of *a* and *b* is higher in the risk ordering."""
    return a if RISK_ORD[a] >= RISK_ORD[b] else b


def symptom_risk(red_flags: list[str], intake: PatientIntake) -> str:
    """Derive a risk level from patient-reported data alone.

    * Any red flag → ``"high"``.
    * Previous skin cancer → at least ``"elevated"``.
    * Age 65+ with a growing / changing lesion → at least ``"elevated"``.
    * Otherwise → ``"low"``.
    """
    if red_flags:
        return "high"

    level = "low"

    if intake.previous_skin_cancer == "yes":
        level = max_risk(level, "elevated")

    if intake.age_band == "65_plus" and intake.evolution in ("growing", "changing"):
        level = max_risk(level, "elevated")

    return level


def compute_final_risk(
    image_risk: str,
    red_flags: list[str],
    intake: PatientIntake,
) -> str:
    """Fuse image and symptom risk.  The result is always ≥ *image_risk*."""
    return max_risk(image_risk, symptom_risk(red_flags, intake))
