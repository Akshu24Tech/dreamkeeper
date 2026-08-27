"""SCAN node: Load memories from the store and cluster by similarity.

This is the first node in the dream cycle.  It loads all active memories,
computes pairwise similarities, and groups them into semantic clusters.
"""

from __future__ import annotations

from app.adapters.base import MemoryStoreAdapter
from app.models import DreamState, MemoryCluster, MemoryStatus


# Memories with cosine similarity above this threshold are placed in the
# same cluster.  Tuneable via env SIMILARITY_THRESHOLD.
DEFAULT_SIMILARITY_THRESHOLD = 0.85


async def scan(
    state: dict,
    *,
    store: MemoryStoreAdapter,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> dict:
    """Load active memories and cluster by semantic similarity."""

    # 1. Load all active memories
    memories = await store.load_all(status=MemoryStatus.ACTIVE)
    if len(memories) < 2:
        return {
            "memories": memories,
            "clusters": [],
        }

    # 2. Compute pairwise similarities
    memory_ids = [m.id for m in memories]
    pairs = await store.get_pairwise_similarities(memory_ids)

    # 3. Simple single-linkage clustering
    # Build adjacency: two memories are "linked" if similarity >= threshold
    adjacency: dict[str, set[str]] = {mid: set() for mid in memory_ids}
    for (a, b), sim in pairs.items():
        if sim >= similarity_threshold:
            adjacency[a].add(b)
            adjacency[b].add(a)

    # BFS to find connected components
    visited: set[str] = set()
    clusters: list[MemoryCluster] = []
    memory_map = {m.id: m for m in memories}

    for mid in memory_ids:
        if mid in visited:
            continue
        # BFS
        component: list[str] = []
        queue = [mid]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            queue.extend(adjacency[current] - visited)

        if len(component) >= 2:
            # Gather similarity scores within the cluster
            sims = []
            for (a, b), sim in pairs.items():
                if a in component and b in component:
                    sims.append(sim)

            # Centroid = content of the memory with highest average similarity
            avg_sims: dict[str, float] = {c: 0.0 for c in component}
            for (a, b), sim in pairs.items():
                if a in component and b in component:
                    avg_sims[a] += sim
                    avg_sims[b] += sim
            for c in component:
                count = len(component) - 1
                avg_sims[c] = avg_sims[c] / count if count > 0 else 0.0

            centroid_id = max(avg_sims, key=avg_sims.get)
            centroid_content = memory_map[centroid_id].content

            clusters.append(
                MemoryCluster(
                    memory_ids=component,
                    centroid_content=centroid_content,
                    similarity_scores=sims,
                )
            )

    return {
        "memories": memories,
        "clusters": clusters,
    }
