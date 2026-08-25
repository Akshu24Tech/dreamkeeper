"""REPORT node — Generate the dream report (audit artifact).

Assembles the final DreamReport from the execution results.
This is the output artifact: what changed, why, with full citations.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.adapters.base import MemoryStoreAdapter
from app.models import (
    ActionResult,
    ConsolidationPlan,
    Detection,
    DreamReport,
    Memory,
    MemoryStatus,
)


async def report(
    state: dict,
    *,
    store: MemoryStoreAdapter,
) -> dict:
    """Assemble the dream report from execution results."""

    dream_id: str = state.get("dream_id", "unknown")
    memories: list[Memory] = state.get("memories", [])
    detections: list[Detection] = state.get("detections", [])
    plan: ConsolidationPlan | None = state.get("plan")
    results: list[ActionResult] = state.get("results", [])
    merges_applied: int = state.get("merges_applied", 0)
    supersedes_applied: int = state.get("supersedes_applied", 0)
    syntheses_applied: int = state.get("syntheses_applied", 0)

    # Count active memories after consolidation
    active_after = await store.count(status=MemoryStatus.ACTIVE)

    dream_report = DreamReport(
        dream_id=dream_id,
        completed_at=datetime.now(timezone.utc),
        memories_scanned=len(memories),
        clusters_found=len(state.get("clusters", [])),
        detections=detections,
        actions_planned=plan.total_actions if plan else 0,
        actions_executed=len([r for r in results if r.success]),
        results=results,
        merges_applied=merges_applied,
        supersedes_applied=supersedes_applied,
        syntheses_applied=syntheses_applied,
        memories_before=len(memories),
        memories_after=active_after,
    )

    return {"report": dream_report}
