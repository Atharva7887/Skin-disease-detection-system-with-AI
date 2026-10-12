"""Build the Clinical Screening Assistant LangGraph.

    consent ─► intake (loop) ─► red_flag_check ─► image_gate (loop) ─► vision ─► fusion ─► report ─► END

Two bounded cycles:
  - intake re-ask: intake → intake (max 6 turns, max 2 retries per field)
  - image retake: image_gate → image_gate (max 2 retries)

Three interrupt points: consent, intake, image_gate.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from PIL import Image

from .nodes import (
    MAX_IMAGE_RETRIES,
    consent_node,
    fusion_node,
    image_gate_node,
    intake_node,
    make_report_node,
    make_vision_node,
    red_flag_check_node,
)
from .state import AssistantState


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def _route_after_consent(state: AssistantState) -> str:
    return "intake" if state.get("consent_given") else END


def _route_after_intake(state: AssistantState) -> str:
    if state.get("intake_complete"):
        return "red_flag_check"
    return "intake"


def _route_after_image_gate(state: AssistantState) -> str:
    if state.get("image_ok"):
        return "vision"
    if state.get("image_retries", 0) >= MAX_IMAGE_RETRIES:
        return END
    return "image_gate"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_assistant_graph(
    predict_fn: Callable[[Image.Image, int], list[dict]],
    heatmap_fn: Optional[Callable[[Image.Image, str], str]] = None,
    performance_note: str = "",
    report_summary_fn: Optional[Callable[[dict], Optional[str]]] = None,
    checkpointer: Any = None,
) -> Any:
    """Compile the assistant LangGraph.

    Parameters
    ----------
    predict_fn : callable
        ``Predictor.predict`` (or a fake for tests).
    heatmap_fn : callable, optional
        ``Predictor.heatmap``.
    performance_note : str
        Performance note read from ``model_card.json``.
    report_summary_fn : callable, optional
        LLM function ``(report_dict) -> str | None`` for the optional report summary.
    checkpointer : optional
        LangGraph checkpointer (``MemorySaver`` for production).
    """
    g = StateGraph(AssistantState)

    # --- Add nodes ---
    g.add_node("consent", consent_node)
    g.add_node("intake", intake_node)
    g.add_node("red_flag_check", red_flag_check_node)
    g.add_node("image_gate", image_gate_node)

    vision_node = make_vision_node(predict_fn, heatmap_fn)
    g.add_node("vision", vision_node)

    g.add_node("fusion", fusion_node)

    report_node = make_report_node(
        performance_note=performance_note,
        report_summary_fn=report_summary_fn,
    )
    g.add_node("report", report_node)

    # --- Entry point ---
    g.set_entry_point("consent")

    # --- Edges ---
    g.add_conditional_edges("consent", _route_after_consent, {
        "intake": "intake",
        END: END,
    })

    g.add_conditional_edges("intake", _route_after_intake, {
        "red_flag_check": "red_flag_check",
        "intake": "intake",
    })

    g.add_edge("red_flag_check", "image_gate")

    g.add_conditional_edges("image_gate", _route_after_image_gate, {
        "vision": "vision",
        "image_gate": "image_gate",
        END: END,
    })

    g.add_edge("vision", "fusion")
    g.add_edge("fusion", "report")
    g.add_edge("report", END)

    # --- Compile ---
    if checkpointer is None:
        checkpointer = MemorySaver()

    return g.compile(checkpointer=checkpointer)
