"""Report faithfulness verifier.

Every number and class name in LLM-generated text must come from the structured
report data.  On any violation the generated summary is discarded and the report
falls back to the fixed template.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ..config import CLASS_NAMES
from .prompts import BANNED_REASSURANCE_WORDS

if TYPE_CHECKING:
    from .state import Report


def _extract_allowed_numbers(report: "Report") -> set[str]:
    """Collect every number that legitimately appears in the structured data."""
    allowed: set[str] = set()
    for cls in report.top_classes:
        # calibrated probability as percentage (e.g. "87%", "87.0%")
        p = cls.get("calibrated_probability", cls.get("probability", 0))
        pct = p * 100
        allowed.add(f"{pct:.0f}%")
        allowed.add(f"{pct:.1f}%")
        allowed.add(f"{p:.0%}")
        allowed.add(f"{p:.1%}")
        allowed.add(f"{p:.2%}")
        # raw float forms
        allowed.add(f"{p:.2f}")
        allowed.add(f"{p:.4f}")
    # concern probability
    cp = report.concern_probability
    allowed.add(f"{cp:.0%}")
    allowed.add(f"{cp:.1%}")
    allowed.add(f"{cp * 100:.0f}%")
    allowed.add(f"{cp * 100:.1f}%")
    allowed.add(f"{cp:.2f}")
    allowed.add(f"{cp:.4f}")
    # duration
    if report.intake.duration_weeks is not None:
        allowed.add(str(report.intake.duration_weeks))
    # Allow single-digit numbers that commonly appear in boilerplate (e.g. "3 sentences")
    for n in range(10):
        allowed.add(str(n))
    return allowed


def verify_report(report: "Report", generated_text: str) -> list[str]:
    """Return a list of violation descriptions.  Empty list means the text is faithful."""
    violations: list[str] = []

    # 1. Numbers in the text must appear in the structured data.
    numbers_in_text = set(re.findall(r"\d+\.?\d*%?", generated_text))
    allowed = _extract_allowed_numbers(report)
    for n in numbers_in_text:
        if n not in allowed:
            violations.append(f"Hallucinated number: {n}")

    # 2. Class names in the text must be among the top predicted classes.
    allowed_classes = set()
    for cls in report.top_classes:
        allowed_classes.add(cls.get("name", "").lower())
        allowed_classes.add(cls.get("code", "").lower())

    text_lower = generated_text.lower()
    for code, name in CLASS_NAMES.items():
        if code.lower() not in allowed_classes and name.lower() not in allowed_classes:
            # Check with word boundaries to avoid false positives (e.g. 'mel' in 'melanocytic')
            if re.search(r"\b" + re.escape(name.lower()) + r"\b", text_lower) or \
               re.search(r"\b" + re.escape(code.lower()) + r"\b", text_lower):
                violations.append(f"Mentions class not in results: {name}")

    # 3. Banned reassurance words.
    # Exclude allowed class names (e.g. 'benign keratosis') from reassurance checks
    sanitized_text = text_lower
    for cls in report.top_classes:
        c_name = cls.get("name", "").lower()
        if c_name:
            sanitized_text = sanitized_text.replace(c_name, "")
    # Also strip common variant 'benign keratosis'
    sanitized_text = re.sub(r"\bbenign keratosis\b", "", sanitized_text)

    for phrase in BANNED_REASSURANCE_WORDS:
        if re.search(r"\b" + re.escape(phrase) + r"\b", sanitized_text):
            violations.append(f"Reassurance language: '{phrase}'")

    return violations
