"""Pydantic models for DreamKeeper.

Every data structure in the dream cycle from raw memories to consolidation
diffs to the final report is defined here as a typed, validated schema.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MemoryStatus(str, Enum):
    """Lifecycle status of a memory entry."""
    ACTIVE = "active"
    MERGED = "merged"           # collapsed into another memory
    SUPERSEDED = "superseded"   # replaced by a newer fact
    SYNTHESIZED = "synthesized" # absorbed into a higher-order memory


class DetectionType(str, Enum):
    """What the detect node found."""
    DUPLICATE = "duplicate"
    CONTRADICTION = "contradiction"
    STALE = "stale"
    CLUSTER = "cluster"  # group ready for synthesis


class OperationType(str, Enum):
    """The consolidation operation to apply."""
    MERGE = "merge"
    SUPERSEDE = "supersede"
    SYNTHESIZE = "synthesize"


# ---------------------------------------------------------------------------
# Core memory
# ---------------------------------------------------------------------------

class Memory(BaseModel):    # What do we have?
    """A single memory entry in the store."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: MemoryStatus = MemoryStatus.ACTIVE
    access_count: int = 0
    source: Optional[str] = None

    # Set after consolidation
    superseded_by: Optional[str] = None   # id of replacement memory
    merged_into: Optional[str] = None     # id of canonical memory
    synthesized_into: Optional[str] = None  # id of higher-order memory


# ---------------------------------------------------------------------------
# Importance scoring
# ---------------------------------------------------------------------------

class ImportanceScore(BaseModel):       # How important is it?
    """Multi-dimensional importance vector for a memory."""
    relevance: float = Field(0.0, ge=0.0, le=1.0, description="How relevant to recent queries")
    frequency: float = Field(0.0, ge=0.0, le=1.0, description="How often accessed")
    novelty: float = Field(0.0, ge=0.0, le=1.0, description="How unique the information is")
    recency: float = Field(0.0, ge=0.0, le=1.0, description="How recent the memory is")
    composite: float = Field(0.0, ge=0.0, le=1.0, description="Weighted composite score")


# ---------------------------------------------------------------------------
# Detection results
# ---------------------------------------------------------------------------

class Detection(BaseModel):      # What's wrong or interesting about them?
    """A single finding from the detect node."""
    type: DetectionType
    memory_ids: list[str]         # the memories involved
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str              # why this was flagged
    suggested_operation: OperationType


class MemoryCluster(BaseModel):      # Which memories belong together?
    """A group of semantically related memories."""
    cluster_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    memory_ids: list[str]
    centroid_content: str         # representative summary of the cluster
    similarity_scores: list[float] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Consolidation plan (the DIFF you can read)
# ---------------------------------------------------------------------------

class ConsolidationAction(BaseModel):     # What should we do?
    """A single planned change the unit of the preview diff."""
    action_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    operation: OperationType
    source_memory_ids: list[str]  # memories being consolidated
    target_content: str           # the new/merged content
    reason: str                   # why this consolidation is proposed
    citations: list[str] = Field(default_factory=list)  # source memory ids cited


class ConsolidationPlan(BaseModel):     # What are we planning to change?
    """The full set of proposed changes shown in preview before execution."""
    plan_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    actions: list[ConsolidationAction] = Field(default_factory=list)
    total_merges: int = 0
    total_supersedes: int = 0
    total_syntheses: int = 0

    @property
    def total_actions(self) -> int:
        return len(self.actions)


# ---------------------------------------------------------------------------
# Dream report (the audit trail)
# ---------------------------------------------------------------------------

class ActionResult(BaseModel):    # What happened after executing the plan?
    """Result of executing a single consolidation action."""
    action_id: str
    operation: OperationType
    success: bool
    source_memory_ids: list[str]
    new_memory_id: Optional[str] = None   # id of the created memory (merge/synthesize)
    affected_memory_ids: list[str] = Field(default_factory=list)
    diff: str                              # human-readable diff of what changed
    citations: list[str] = Field(default_factory=list)


class DreamReport(BaseModel):    # What happened during the whole dream cycle?
    """Full report of a completed dream cycle the audit artifact."""
    dream_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    memories_scanned: int = 0
    clusters_found: int = 0
    detections: list[Detection] = Field(default_factory=list)
    actions_planned: int = 0
    actions_executed: int = 0
    results: list[ActionResult] = Field(default_factory=list)

    # Summary stats
    merges_applied: int = 0
    supersedes_applied: int = 0
    syntheses_applied: int = 0
    memories_before: int = 0
    memories_after: int = 0     # active memories after consolidation

    @property
    def reduction_pct(self) -> float:
        if self.memories_before == 0:
            return 0.0
        return round((1 - self.memories_after / self.memories_before) * 100, 1)


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class DreamState(BaseModel):      # What information is currently flowing through LangGraph?
    """The state that flows through the LangGraph dream cycle."""
    dream_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    memories: list[Memory] = Field(default_factory=list)
    clusters: list[MemoryCluster] = Field(default_factory=list)
    detections: list[Detection] = Field(default_factory=list)
    plan: Optional[ConsolidationPlan] = None
    approved: bool = True         # default auto-approve; set False for HITL
    report: Optional[DreamReport] = None
    error: Optional[str] = None
