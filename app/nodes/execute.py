"""EXECUTE node: Apply the consolidation plan to the memory store.

This is the only node that mutates the store.  Every change creates a
traceable diff: nothing is deleted, old memories are flagged with their
new status and linked to their replacement.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.adapters.base import MemoryStoreAdapter
from app.models import (
    ActionResult,
    ConsolidationAction,
    ConsolidationPlan,
    Memory,
    MemoryStatus,
    OperationType,
)


def _build_diff(operation: OperationType, sources: list[Memory], target_content: str) -> str:
    """Build a human-readable diff string."""
    lines = [f"=== {operation.value.upper()} ==="]
    for s in sources:
        lines.append(f"- [{s.id[:8]}] {s.content[:100]}{'...' if len(s.content) > 100 else ''}")
    lines.append(f"+ {target_content[:200]}{'...' if len(target_content) > 200 else ''}")
    return "\n".join(lines)


async def execute(
    state: dict,
    *,
    store: MemoryStoreAdapter,
) -> dict:
    """Apply approved consolidation actions to the store."""

    plan: ConsolidationPlan | None = state.get("plan")
    approved: bool = state.get("approved", True)
    memories: list[Memory] = state.get("memories", [])
    memory_map = {m.id: m for m in memories}

    if not plan or not approved:
        return {
            "report": None,
            "error": "No plan or not approved" if not approved else "No plan",
        }

    results: list[ActionResult] = []
    merges_applied = 0
    supersedes_applied = 0
    syntheses_applied = 0

    for action in plan.actions:
        try:
            source_memories = [memory_map[mid] for mid in action.source_memory_ids if mid in memory_map]
            if not source_memories:
                continue

            if action.operation == OperationType.MERGE:
                # Create canonical memory
                canonical = Memory(
                    content=action.target_content,
                    metadata={"merged_from": action.source_memory_ids},
                    source="dreamkeeper:merge",
                )
                await store.upsert(canonical)

                # Flag sources as merged
                affected = []
                for src in source_memories:
                    await store.update_status(
                        src.id, MemoryStatus.MERGED, linked_to=canonical.id
                    )
                    affected.append(src.id)

                diff = _build_diff(action.operation, source_memories, action.target_content)
                results.append(
                    ActionResult(
                        action_id=action.action_id,
                        operation=OperationType.MERGE,
                        success=True,
                        source_memory_ids=action.source_memory_ids,
                        new_memory_id=canonical.id,
                        affected_memory_ids=affected,
                        diff=diff,
                        citations=action.citations,
                    )
                )
                merges_applied += 1

            elif action.operation == OperationType.SUPERSEDE:
                # Create the current-truth memory
                current = Memory(
                    content=action.target_content,
                    metadata={"supersedes": action.source_memory_ids},
                    source="dreamkeeper:supersede",
                )
                await store.upsert(current)

                # Flag sources as superseded
                affected = []
                for src in source_memories:
                    await store.update_status(
                        src.id, MemoryStatus.SUPERSEDED, linked_to=current.id
                    )
                    affected.append(src.id)

                diff = _build_diff(action.operation, source_memories, action.target_content)
                results.append(
                    ActionResult(
                        action_id=action.action_id,
                        operation=OperationType.SUPERSEDE,
                        success=True,
                        source_memory_ids=action.source_memory_ids,
                        new_memory_id=current.id,
                        affected_memory_ids=affected,
                        diff=diff,
                        citations=action.citations,
                    )
                )
                supersedes_applied += 1

            elif action.operation == OperationType.SYNTHESIZE:
                # Create higher-order memory
                synthesis = Memory(
                    content=action.target_content,
                    metadata={
                        "synthesized_from": action.source_memory_ids,
                        "citations": action.citations,
                    },
                    source="dreamkeeper:synthesize",
                )
                await store.upsert(synthesis)

                # Flag sources as synthesized (but they stay queryable!)
                affected = []
                for src in source_memories:
                    await store.update_status(
                        src.id, MemoryStatus.SYNTHESIZED, linked_to=synthesis.id
                    )
                    affected.append(src.id)

                diff = _build_diff(action.operation, source_memories, action.target_content)
                results.append(
                    ActionResult(
                        action_id=action.action_id,
                        operation=OperationType.SYNTHESIZE,
                        success=True,
                        source_memory_ids=action.source_memory_ids,
                        new_memory_id=synthesis.id,
                        affected_memory_ids=affected,
                        diff=diff,
                        citations=action.citations,
                    )
                )
                syntheses_applied += 1

        except Exception as e:
            results.append(
                ActionResult(
                    action_id=action.action_id,
                    operation=action.operation,
                    success=False,
                    source_memory_ids=action.source_memory_ids,
                    diff=f"ERROR: {e}",
                )
            )

    return {
        "results": results,
        "merges_applied": merges_applied,
        "supersedes_applied": supersedes_applied,
        "syntheses_applied": syntheses_applied,
    }
