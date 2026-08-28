"""DreamKeeper: FastAPI entry point.

Endpoints:
    POST  /dream                   Trigger a dream cycle
    GET   /dream/{dream_id}        Get dream report
    GET   /memories                List all memories with status
    POST  /memories                Add a memory to the store
    GET   /health                  Memory store health metrics
"""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.adapters.chroma import ChromaAdapter
from app.graph import build_dream_graph, _get_default_state
from app.models import DreamReport, Memory, MemoryStatus

load_dotenv()

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

_store: ChromaAdapter | None = None
_dream_reports: dict[str, DreamReport] = {}  # in-memory cache of completed dreams


def _get_store() -> ChromaAdapter:
    global _store
    if _store is None:
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
        _store = ChromaAdapter(persist_dir=persist_dir)
    return _store


@asynccontextmanager
async def lifespan(app: FastAPI):
    _get_store()
    yield


app = FastAPI(
    title="DreamKeeper",
    description="Memory consolidation agent - the 'dreaming' pass for AI agent memory stores.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class AddMemoryRequest(BaseModel):
    content: str
    metadata: dict = {}
    source: str | None = None


class TriggerDreamRequest(BaseModel):
    auto_approve: bool = True


class HealthResponse(BaseModel):
    total_memories: int
    active_memories: int
    merged_memories: int
    superseded_memories: int
    synthesized_memories: int
    dreams_completed: int


class DreamSummary(BaseModel):
    dream_id: str
    status: str
    memories_scanned: int
    actions_executed: int
    merges: int
    supersedes: int
    syntheses: int
    reduction_pct: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/memories", status_code=201)
async def add_memory(req: AddMemoryRequest):
    """Add a memory to the store."""
    store = _get_store()
    memory = Memory(
        content=req.content,
        metadata=req.metadata,
        source=req.source,
    )
    await store.upsert(memory)
    return {"id": memory.id, "status": "stored"}


@app.get("/memories")
async def list_memories(status: str | None = None):
    """List all memories, optionally filtered by status."""
    store = _get_store()
    mem_status = MemoryStatus(status) if status else None
    memories = await store.load_all(status=mem_status)
    return [m.model_dump() for m in memories]


@app.post("/dream")
async def trigger_dream(req: TriggerDreamRequest = TriggerDreamRequest()):
    """Trigger a dream cycle."""
    store = _get_store()
    graph = build_dream_graph(store=store)
    initial_state = _get_default_state()
    initial_state["approved"] = req.auto_approve

    result = await graph.ainvoke(initial_state)

    dream_report: DreamReport | None = result.get("report")
    if dream_report:
        _dream_reports[dream_report.dream_id] = dream_report
        return DreamSummary(
            dream_id=dream_report.dream_id,
            status="completed",
            memories_scanned=dream_report.memories_scanned,
            actions_executed=dream_report.actions_executed,
            merges=dream_report.merges_applied,
            supersedes=dream_report.supersedes_applied,
            syntheses=dream_report.syntheses_applied,
            reduction_pct=dream_report.reduction_pct,
        )

    return {"dream_id": result.get("dream_id"), "status": "no_action_needed"}


@app.get("/dream/{dream_id}")
async def get_dream_report(dream_id: str):
    """Get the full dream report."""
    report = _dream_reports.get(dream_id)
    if not report:
        raise HTTPException(404, f"Dream report {dream_id} not found")
    return report.model_dump()


@app.get("/health")
async def health():
    """Memory store health metrics."""
    store = _get_store()
    total = await store.count()
    active = await store.count(status=MemoryStatus.ACTIVE)
    merged = await store.count(status=MemoryStatus.MERGED)
    superseded = await store.count(status=MemoryStatus.SUPERSEDED)
    synthesized = await store.count(status=MemoryStatus.SYNTHESIZED)

    return HealthResponse(
        total_memories=total,
        active_memories=active,
        merged_memories=merged,
        superseded_memories=superseded,
        synthesized_memories=synthesized,
        dreams_completed=len(_dream_reports),
    )
