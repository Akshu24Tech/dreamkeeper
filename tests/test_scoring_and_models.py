"""Unit tests for DreamKeeper detection and scoring logic."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from app.models import (
    Detection,
    DetectionType,
    Memory,
    MemoryStatus,
    OperationType,
    ImportanceScore,
)
from app.scoring import score_memory, score_batch, _recency_score, HALF_LIFE_DAYS


# ---------------------------------------------------------------------------
# Scoring tests
# ---------------------------------------------------------------------------

class TestRecencyScore:
    def test_brand_new_memory_scores_one(self):
        now = datetime.now(timezone.utc)
        mem = Memory(content="test", updated_at=now)
        score = _recency_score(mem, now=now)
        assert abs(score - 1.0) < 0.01

    def test_half_life_scores_half(self):
        now = datetime.now(timezone.utc)
        mem = Memory(content="test", updated_at=now - timedelta(days=HALF_LIFE_DAYS))
        score = _recency_score(mem, now=now)
        assert abs(score - 0.5) < 0.05

    def test_very_old_memory_scores_near_zero(self):
        now = datetime.now(timezone.utc)
        mem = Memory(content="test", updated_at=now - timedelta(days=365))
        score = _recency_score(mem, now=now)
        assert score < 0.01


class TestScoreMemory:
    def test_composite_is_weighted_sum(self):
        now = datetime.now(timezone.utc)
        mem = Memory(content="test", updated_at=now, access_count=5)
        score = score_memory(
            mem,
            similarity_to_recent=0.8,
            uniqueness=0.6,
            max_access=10,
            now=now,
        )
        assert 0.0 <= score.composite <= 1.0
        assert score.relevance == 0.8
        assert score.novelty == 0.6

    def test_clamps_values(self):
        now = datetime.now(timezone.utc)
        mem = Memory(content="test", updated_at=now)
        score = score_memory(mem, similarity_to_recent=1.5, uniqueness=-0.5, now=now)
        assert score.relevance <= 1.0
        assert score.novelty >= 0.0


class TestScoreBatch:
    def test_scores_all_memories(self):
        now = datetime.now(timezone.utc)
        memories = [
            Memory(content="a", updated_at=now, access_count=3),
            Memory(content="b", updated_at=now - timedelta(days=30), access_count=1),
            Memory(content="c", updated_at=now - timedelta(days=90), access_count=0),
        ]
        scores = score_batch(memories, now=now)
        assert len(scores) == 3
        # Newer, more accessed memory should score higher
        ids = [m.id for m in memories]
        assert scores[ids[0]].composite > scores[ids[2]].composite


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestMemoryModel:
    def test_default_status_is_active(self):
        mem = Memory(content="test fact")
        assert mem.status == MemoryStatus.ACTIVE

    def test_unique_ids(self):
        m1 = Memory(content="a")
        m2 = Memory(content="b")
        assert m1.id != m2.id


class TestDetectionModel:
    def test_detection_creation(self):
        det = Detection(
            type=DetectionType.DUPLICATE,
            memory_ids=["a", "b"],
            confidence=0.95,
            explanation="Same fact in different words",
            suggested_operation=OperationType.MERGE,
        )
        assert det.type == DetectionType.DUPLICATE
        assert det.suggested_operation == OperationType.MERGE
        assert len(det.memory_ids) == 2


# ---------------------------------------------------------------------------
# DreamReport tests
# ---------------------------------------------------------------------------

class TestDreamReport:
    def test_reduction_pct_zero_when_no_memories(self):
        from app.models import DreamReport
        report = DreamReport(memories_before=0, memories_after=0)
        assert report.reduction_pct == 0.0

    def test_reduction_pct_calculated(self):
        from app.models import DreamReport
        report = DreamReport(memories_before=10, memories_after=6)
        assert report.reduction_pct == 40.0
