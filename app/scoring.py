"""Importance scoring for memories.

Assigns a multi-dimensional importance vector to each memory based on
relevance, frequency, novelty, and recency.  The composite score is a
weighted sum that the detect/plan nodes use to decide what to keep, merge,
or let fade.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from app.models import ImportanceScore, Memory


# Default weights — tuneable via env or config
WEIGHT_RELEVANCE = 0.30
WEIGHT_FREQUENCY = 0.25
WEIGHT_NOVELTY = 0.25
WEIGHT_RECENCY = 0.20

# Recency half-life in days: a memory loses half its recency score
# every HALF_LIFE_DAYS days since last access.
HALF_LIFE_DAYS = 14.0


def _recency_score(memory: Memory, now: datetime | None = None) -> float:
    """Exponential decay based on age.  Returns 1.0 for brand-new, ~0.5 at
    HALF_LIFE_DAYS, asymptotically approaching 0."""
    now = now or datetime.now(timezone.utc)
    age_days = max((now - memory.updated_at).total_seconds() / 86400, 0)
    return math.exp(-0.693 * age_days / HALF_LIFE_DAYS)  # ln(2) ≈ 0.693


def _frequency_score(memory: Memory, max_access: int = 1) -> float:
    """Normalised access count.  max_access should be the highest access_count
    in the current batch so the score is relative."""
    if max_access <= 0:
        return 0.0
    return min(memory.access_count / max_access, 1.0)


def score_memory(
    memory: Memory,
    *,
    similarity_to_recent: float = 0.0,
    uniqueness: float = 0.5,
    max_access: int = 1,
    now: datetime | None = None,
) -> ImportanceScore:
    """Compute the full importance vector for a single memory.

    Parameters
    ----------
    similarity_to_recent : float
        Cosine similarity of this memory to the most-recent query / session
        context (0-1).  Used as the *relevance* dimension.
    uniqueness : float
        1 minus the max cosine similarity to any other memory in the store.
        High uniqueness → high novelty.
    max_access : int
        The highest access_count in the batch, for normalisation.
    """
    relevance = max(0.0, min(similarity_to_recent, 1.0))
    frequency = _frequency_score(memory, max_access)
    novelty = max(0.0, min(uniqueness, 1.0))
    recency = _recency_score(memory, now)

    composite = (
        WEIGHT_RELEVANCE * relevance
        + WEIGHT_FREQUENCY * frequency
        + WEIGHT_NOVELTY * novelty
        + WEIGHT_RECENCY * recency
    )

    return ImportanceScore(
        relevance=round(relevance, 4),
        frequency=round(frequency, 4),
        novelty=round(novelty, 4),
        recency=round(recency, 4),
        composite=round(composite, 4),
    )


def score_batch(
    memories: list[Memory],
    *,
    similarities: dict[str, float] | None = None,
    uniqueness_map: dict[str, float] | None = None,
    now: datetime | None = None,
) -> dict[str, ImportanceScore]:
    """Score an entire batch.  Returns {memory_id: ImportanceScore}."""
    similarities = similarities or {}
    uniqueness_map = uniqueness_map or {}

    max_access = max((m.access_count for m in memories), default=1) or 1

    return {
        m.id: score_memory(
            m,
            similarity_to_recent=similarities.get(m.id, 0.0),
            uniqueness=uniqueness_map.get(m.id, 0.5),
            max_access=max_access,
            now=now,
        )
        for m in memories
    }
