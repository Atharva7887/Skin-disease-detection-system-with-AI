"""LLM-based adaptive intake: structured extraction from free-text user answers.

The LLM extracts a single field per turn from the user's natural-language response
and generates a follow-up question.  All output is validated through a Pydantic
schema — on any failure the deterministic fallback takes over.

Safety: user text is wrapped in ``<user_input>`` delimiters, the system prompt
forbids mentioning diseases, and the output is post-validated for banned words.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from pydantic import BaseModel, Field

from .prompts import BANNED_DISEASE_NAMES, BANNED_REASSURANCE_WORDS, INTAKE_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Structured output schema for the LLM
# ---------------------------------------------------------------------------

class IntakeExtraction(BaseModel):
    """What the LLM returns when extracting a single intake field."""

    extracted_value: Optional[str] = Field(
        default=None,
        description="The extracted value, must be one of the valid_values or null.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the extraction (0 to 1).",
    )
    follow_up_question: str = Field(
        description="A brief empathetic question to ask next (either a clarification "
                    "or the next intake question).",
    )


# ---------------------------------------------------------------------------
# Post-validation
# ---------------------------------------------------------------------------

def validate_intake_response(text: str) -> bool:
    """Return ``True`` if *text* is safe (no disease names or reassurance)."""
    lower = text.lower()
    for name in BANNED_DISEASE_NAMES:
        if name in lower:
            return False
    for phrase in BANNED_REASSURANCE_WORDS:
        if phrase in lower:
            return False
    return True


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

# Human-readable descriptions for each field.
FIELD_DESCRIPTIONS: dict[str, tuple[str, str]] = {
    "duration_weeks": (
        "How long the user has had this skin concern, in weeks",
        "An integer number of weeks, e.g. 2, 8, 26, 52",
    ),
    "evolution": (
        "Whether the lesion is growing, changing, stable, or unknown",
        "One of: growing, changing, stable, unknown",
    ),
    "symptoms": (
        "Current symptoms the user is experiencing",
        "A comma-separated list from: itching, bleeding, pain, none",
    ),
    "anatomy": (
        "Body location of the lesion",
        "One of: arm, leg, back, face, scalp, hand, foot, other",
    ),
    "age_band": (
        "The user's age range",
        "One of: under_20, 20_34, 35_49, 50_64, 65_plus",
    ),
    "previous_skin_cancer": (
        "Whether the user has a history of skin cancer",
        "One of: yes, no, unsure",
    ),
    "sun_exposure": (
        "The user's level of lifetime sun exposure",
        "One of: low, moderate, high",
    ),
}


def build_extraction_prompt(field_name: str, user_text: str) -> list[dict[str, str]]:
    """Build the chat-style messages for one extraction turn."""
    desc, valid = FIELD_DESCRIPTIONS.get(field_name, (field_name, "any"))
    return [
        {"role": "system", "content": INTAKE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Extract the structured intake field from the user's response.\n"
                f"Field to extract: {field_name}\n"
                f"Field description: {desc}\n"
                f"Valid values: {valid}\n\n"
                f"User's response:\n<user_input>\n{user_text}\n</user_input>\n\n"
                f"If you cannot confidently extract a value, set confidence below 0.7.\n"
                f"Generate a brief, empathetic follow-up question for the NEXT field "
                f"(or a clarification if confidence is low)."
            ),
        },
    ]


def make_intake_extractor(chain: Any = None):
    """Return an extraction function.

    *chain* should accept messages and return an ``IntakeExtraction``,
    or be an LLM supporting ``with_structured_output(IntakeExtraction)``.
    When *chain* is ``None``, returns ``None`` (caller falls back to deterministic).
    """
    if chain is None:
        return None

    if hasattr(chain, "with_structured_output"):
        runnable = chain.with_structured_output(IntakeExtraction)
    else:
        runnable = chain

    def extract(
        field_name: str,
        user_text: str,
    ) -> Optional[IntakeExtraction]:
        """Extract *field_name* from *user_text*.  Returns ``None`` on any failure."""
        try:
            messages = build_extraction_prompt(field_name, user_text)
            result = runnable.invoke(messages)
            if not isinstance(result, IntakeExtraction):
                return None
            # Post-validate: the follow-up question must be safe.
            if not validate_intake_response(result.follow_up_question):
                result.follow_up_question = ""  # strip unsafe text
            return result
        except Exception:
            return None

    return extract


def extract_field_with_llm(
    chain: Any,
    user_message: str,
    current_field: str,
    intake: Optional[PatientIntake] = None,
) -> Optional[tuple[str, Any, str]]:
    """Extract *current_field* from *user_message* using *chain*.

    Returns ``(field_name, value, follow_up_question)`` or ``None`` on extraction failure.
    """
    extractor = make_intake_extractor(chain)
    if extractor is None:
        return None
    res = extractor(current_field, user_message)
    if res is None:
        return None

    val = res.extracted_value
    if val is not None and current_field == "duration_weeks":
        digits = re.sub(r"[^\d]", "", str(val))
        if digits:
            try:
                val = int(digits)
            except ValueError:
                pass
    return (current_field, val, res.follow_up_question)

