"""Abstract memory store adapter.

Any vector DB or memory backend can be plugged into DreamKeeper by
implementing this interface.  The dream cycle never touches the store
directly — it always goes through an adapter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.models import Memory, MemoryStatus


class MemoryStoreAdapter(ABC):
    """Interface every memory store backend must implement."""

    @abstractmethod
    async def load_all(self, *, status: MemoryStatus | None = MemoryStatus.ACTIVE) -> list[Memory]:
        """Load memories from the store, optionally filtered by status."""
        ...

    @abstractmethod
    async def get(self, memory_id: str) -> Memory | None:
        """Retrieve a single memory by ID."""
        ...

    @abstractmethod
    async def upsert(self, memory: Memory) -> None:
        """Insert or update a memory (and its embedding)."""
        ...

    @abstractmethod
    async def update_status(
        self,
        memory_id: str,
        status: MemoryStatus,
        *,
        linked_to: str | None = None,
    ) -> None:
        """Update a memory's status and optionally link it to a replacement.

        For MERGED → set merged_into = linked_to
        For SUPERSEDED → set superseded_by = linked_to
        For SYNTHESIZED → set synthesized_into = linked_to
        """
        ...

    @abstractmethod
    async def search_similar(
        self,
        text: str,
        *,
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> list[tuple[Memory, float]]:
        """Semantic search.  Returns (memory, similarity_score) pairs."""
        ...

    @abstractmethod
    async def get_pairwise_similarities(
        self,
        memory_ids: list[str],
    ) -> dict[tuple[str, str], float]:
        """Compute pairwise cosine similarities between the given memories.
        Returns {(id_a, id_b): similarity} for every pair where a < b."""
        ...

    @abstractmethod
    async def count(self, *, status: MemoryStatus | None = None) -> int:
        """Count memories, optionally filtered by status."""
        ...
