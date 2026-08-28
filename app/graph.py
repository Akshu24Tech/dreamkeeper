"""DreamKeeper LangGraph pipeline.

Wires the five dream cycle nodes into a state machine:

    scan -> detect -> plan -> [human_review] -> execute -> report

The human_review gate is a conditional edge: if `approved` is True (default,
auto-approve mode), it proceeds directly to execute.  If False, the graph
interrupts and waits for external approval via the API.
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Optional, TypedDict

from langgraph.graph import END, StateGraph

from app.adapters.base import MemoryStoreAdapter
from app.adapters.chroma import ChromaAdapter
from app.models import (
    ActionResult,
    ConsolidationPlan,
    Detection,
    DreamReport,
    Memory,
    MemoryCluster,
)
from app.nodes.detect import detect
from app.nodes.execute import execute
from app.nodes.plan import plan
from app.nodes.report import report
from app.nodes.scan import scan


# ---------------------------------------------------------------------------
# Graph state type
# ---------------------------------------------------------------------------

class DreamGraphState(TypedDict, total=False):
    dream_id: str
    memories: list[Memory]
    clusters: list[MemoryCluster]
    detections: list[Detection]
    plan: Optional[ConsolidationPlan]
    approved: bool
    results: list[ActionResult]
    merges_applied: int
    supersedes_applied: int
    syntheses_applied: int
    report: Optional[DreamReport]
    error: Optional[str]


def _get_default_state() -> dict:
    return {
        "dream_id": str(uuid.uuid4()),
        "memories": [],
        "clusters": [],
        "detections": [],
        "plan": None,
        "approved": True,
        "results": [],
        "merges_applied": 0,
        "supersedes_applied": 0,
        "syntheses_applied": 0,
        "report": None,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Node wrappers (inject the store dependency)
# ---------------------------------------------------------------------------

def _make_scan_node(store: MemoryStoreAdapter):
    async def _scan(state: DreamGraphState) -> dict:
        threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.55"))
        return await scan(state, store=store, similarity_threshold=threshold)

    return _scan


def _make_detect_node():
    async def _detect(state: DreamGraphState) -> dict:
        return await detect(state)

    return _detect


def _make_plan_node():
    async def _plan(state: DreamGraphState) -> dict:
        return await plan(state)

    return _plan


def _make_execute_node(store: MemoryStoreAdapter):
    async def _execute(state: DreamGraphState) -> dict:
        return await execute(state, store=store)

    return _execute


def _make_report_node(store: MemoryStoreAdapter):
    async def _report(state: DreamGraphState) -> dict:
        return await report(state, store=store)

    return _report


# ---------------------------------------------------------------------------
# Conditional edges
# ---------------------------------------------------------------------------

def _should_continue_after_scan(state: DreamGraphState) -> str:
    """Skip the rest if there are fewer than 2 memories."""
    memories = state.get("memories", [])
    if len(memories) < 2:
        return "report"
    return "detect"


def _should_execute(state: DreamGraphState) -> str:
    """Check the HITL gate."""
    plan = state.get("plan")
    approved = state.get("approved", True)

    if not plan or not plan.actions:
        return "report"  # nothing to do
    if not approved:
        return END  # pause for human review
    return "execute"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_dream_graph(
    store: MemoryStoreAdapter | None = None,
) -> Any:
    """Construct the DreamKeeper LangGraph pipeline."""
    if store is None:
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
        store = ChromaAdapter(persist_dir=persist_dir)

    # -- Build graph --
    builder = StateGraph(DreamGraphState)

    builder.add_node("scan", _make_scan_node(store))
    builder.add_node("detect", _make_detect_node())
    builder.add_node("plan", _make_plan_node())
    builder.add_node("execute", _make_execute_node(store))
    builder.add_node("report", _make_report_node(store))

    # -- Edges --
    builder.set_entry_point("scan")

    builder.add_conditional_edges(
        "scan",
        _should_continue_after_scan,
        {"detect": "detect", "report": "report"},
    )

    builder.add_edge("detect", "plan")

    builder.add_conditional_edges(
        "plan",
        _should_execute,
        {"execute": "execute", "report": "report", END: END},
    )

    builder.add_edge("execute", "report")
    builder.add_edge("report", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

async def run_dream(
    *,
    store: MemoryStoreAdapter | None = None,
    auto_approve: bool = True,
) -> dict:
    """Run a full dream cycle and return the final state."""
    graph = build_dream_graph(store=store)
    initial_state = _get_default_state()
    initial_state["approved"] = auto_approve

    result = await graph.ainvoke(initial_state)
    return result
