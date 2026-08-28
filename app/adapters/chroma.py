"""ChromaDB adapter for DreamKeeper.

Zero-setup, pip-installable backend.  Stores memories as ChromaDB documents
with metadata for status tracking and provenance links.  Embeddings are
handled by ChromaDB's built-in fastembed integration.
"""

from __future__ import annotations

import json
from datetime import timezone

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.adapters.base import MemoryStoreAdapter
from app.models import Memory, MemoryStatus


# ChromaDB metadata only supports str/int/float/bool - so we serialise
# datetimes as ISO strings and status as its string value.

def _memory_to_doc(memory: Memory) -> dict:
    """Convert a Memory to ChromaDB document + metadata."""
    meta = {
        "status": memory.status.value,
        "created_at": memory.created_at.isoformat(),
        "updated_at": memory.updated_at.isoformat(),
        "access_count": memory.access_count,
        "source": memory.source or "",
        "superseded_by": memory.superseded_by or "",
        "merged_into": memory.merged_into or "",
        "synthesized_into": memory.synthesized_into or "",
        "extra_metadata": json.dumps(memory.metadata),
    }
    return {
        "id": memory.id,
        "document": memory.content,
        "metadata": meta,
    }


def _doc_to_memory(doc_id: str, document: str, metadata: dict) -> Memory:
    """Reconstruct a Memory from a ChromaDB result."""
    from datetime import datetime

    return Memory(
        id=doc_id,
        content=document,
        metadata=json.loads(metadata.get("extra_metadata", "{}")),
        created_at=datetime.fromisoformat(metadata["created_at"]),
        updated_at=datetime.fromisoformat(metadata["updated_at"]),
        status=MemoryStatus(metadata["status"]),
        access_count=int(metadata.get("access_count", 0)),
        source=metadata.get("source") or None,
        superseded_by=metadata.get("superseded_by") or None,
        merged_into=metadata.get("merged_into") or None,
        synthesized_into=metadata.get("synthesized_into") or None,
    )


class ChromaAdapter(MemoryStoreAdapter):
    """ChromaDB-backed memory store."""

    def __init__(
        self,
        persist_dir: str = "./chroma_data",
        collection_name: str = "dreamkeeper_memories",
    ):
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    async def load_all(self, *, status: MemoryStatus | None = MemoryStatus.ACTIVE) -> list[Memory]:
        where = {"status": status.value} if status else None
        results = self._collection.get(where=where, include=["documents", "metadatas"])

        memories = []
        for i, doc_id in enumerate(results["ids"]):
            memories.append(
                _doc_to_memory(
                    doc_id,
                    results["documents"][i],
                    results["metadatas"][i],
                )
            )
        return memories

    async def get(self, memory_id: str) -> Memory | None:
        results = self._collection.get(
            ids=[memory_id], include=["documents", "metadatas"]
        )
        if not results["ids"]:
            return None
        return _doc_to_memory(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
        )

    async def upsert(self, memory: Memory) -> None:
        doc = _memory_to_doc(memory)
        self._collection.upsert(
            ids=[doc["id"]],
            documents=[doc["document"]],
            metadatas=[doc["metadata"]],
        )

    async def update_status(
        self,
        memory_id: str,
        status: MemoryStatus,
        *,
        linked_to: str | None = None,
    ) -> None:
        memory = await self.get(memory_id)
        if memory is None:
            return

        memory.status = status
        if status == MemoryStatus.MERGED and linked_to:
            memory.merged_into = linked_to
        elif status == MemoryStatus.SUPERSEDED and linked_to:
            memory.superseded_by = linked_to
        elif status == MemoryStatus.SYNTHESIZED and linked_to:
            memory.synthesized_into = linked_to

        from datetime import datetime
        memory.updated_at = datetime.now(timezone.utc)
        await self.upsert(memory)

    async def search_similar(
        self,
        text: str,
        *,
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> list[tuple[Memory, float]]:
        results = self._collection.query(
            query_texts=[text],
            n_results=top_k,
            where={"status": MemoryStatus.ACTIVE.value},
            include=["documents", "metadatas", "distances"],
        )

        pairs = []
        for i, doc_id in enumerate(results["ids"][0]):
            # ChromaDB cosine distance = 1 - similarity
            similarity = 1.0 - results["distances"][0][i]
            if similarity >= threshold:
                mem = _doc_to_memory(
                    doc_id,
                    results["documents"][0][i],
                    results["metadatas"][0][i],
                )
                pairs.append((mem, round(similarity, 4)))
        return pairs

    async def get_pairwise_similarities(
        self,
        memory_ids: list[str],
    ) -> dict[tuple[str, str], float]:
        """Compute pairwise similarities by querying each memory against the
        collection.  Not the most efficient, but correct and simple."""
        if len(memory_ids) < 2:
            return {}

        # Fetch all embeddings
        results = self._collection.get(
            ids=memory_ids,
            include=["embeddings"],
        )

        embeddings = {}
        raw_embeddings = results.get("embeddings")
        if raw_embeddings is not None and len(raw_embeddings) > 0:
            for i, mid in enumerate(results["ids"]):
                if i < len(raw_embeddings) and raw_embeddings[i] is not None:
                    embeddings[mid] = raw_embeddings[i]

        # Compute cosine similarity
        import numpy as np

        pairs: dict[tuple[str, str], float] = {}
        ids = list(embeddings.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a = np.array(embeddings[ids[i]])
                b = np.array(embeddings[ids[j]])
                norm_a = np.linalg.norm(a)
                norm_b = np.linalg.norm(b)
                if norm_a == 0 or norm_b == 0:
                    sim = 0.0
                else:
                    sim = float(np.dot(a, b) / (norm_a * norm_b))
                pairs[(ids[i], ids[j])] = round(sim, 4)

        return pairs

    async def count(self, *, status: MemoryStatus | None = None) -> int:
        if status:
            results = self._collection.get(
                where={"status": status.value}, include=[]
            )
            return len(results["ids"])
        return self._collection.count()
