"""DETECT node: Find duplicates, contradictions, and stale entries.

Uses Gemini to classify whether clustered memories are duplicates,
contradictions, or just related.  Also flags stale entries by age.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from google import genai

from app.models import (
    Detection,
    DetectionType,
    DreamState,
    Memory,
    MemoryCluster,
    OperationType,
)


STALE_DAYS = int(os.getenv("STALE_DAYS", "30"))


DETECT_PROMPT = """You are analysing a cluster of related memories from an AI agent's memory store.
Your job is to classify what kind of issue (if any) exists in this cluster.

## Memories in this cluster:
{memories_text}

## Task
Analyse these memories and determine:
1. Are any of them **duplicates** (same fact stated in different words)?
2. Do any **contradict** each other (conflicting facts about the same topic)?
3. If they are just related facts that could be **synthesized** into one higher-order summary, say so.

Respond in this exact JSON format (no markdown, no extra text):
{{
    "findings": [
        {{
            "type": "duplicate" | "contradiction" | "cluster",
            "memory_ids": ["id1", "id2"],
            "confidence": 0.0 to 1.0,
            "explanation": "why this was flagged"
        }}
    ]
}}

If the memories are genuinely different and unrelated, return {{"findings": []}}.
"""


async def detect(
    state: dict,
    *,
    model_name: str = "gemini-2.5-flash",
) -> dict:
    """Detect duplicates, contradictions, stale entries, and synthesis candidates."""

    memories: list[Memory] = state.get("memories", [])
    clusters: list[MemoryCluster] = state.get("clusters", [])

    detections: list[Detection] = []
    memory_map = {m.id: m for m in memories}

    # --- 1. Stale detection (deterministic, no LLM needed) ---
    now = datetime.now(timezone.utc)
    for mem in memories:
        age_days = (now - mem.updated_at).total_seconds() / 86400
        if age_days > STALE_DAYS and mem.access_count == 0:
            detections.append(
                Detection(
                    type=DetectionType.STALE,
                    memory_ids=[mem.id],
                    confidence=min(age_days / (STALE_DAYS * 2), 1.0),
                    explanation=f"Memory is {int(age_days)} days old with 0 access count",
                    suggested_operation=OperationType.SUPERSEDE,
                )
            )

    # --- 2. Cluster-based detection (LLM-assisted) ---
    if not clusters:
        return {"detections": detections}

    client = genai.Client()

    for cluster in clusters:
        # Build the prompt with memory contents
        memories_text = ""
        for mid in cluster.memory_ids:
            mem = memory_map.get(mid)
            if mem:
                memories_text += f"- ID: {mid}\n  Content: {mem.content}\n  Created: {mem.created_at.isoformat()}\n\n"

        prompt = DETECT_PROMPT.format(memories_text=memories_text)

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                ),
            )

            import json
            result = json.loads(response.text)

            for finding in result.get("findings", []):
                det_type = finding["type"]
                if det_type == "duplicate":
                    dtype = DetectionType.DUPLICATE
                    suggested = OperationType.MERGE
                elif det_type == "contradiction":
                    dtype = DetectionType.CONTRADICTION
                    suggested = OperationType.SUPERSEDE
                else:
                    dtype = DetectionType.CLUSTER
                    suggested = OperationType.SYNTHESIZE

                detections.append(
                    Detection(
                        type=dtype,
                        memory_ids=finding["memory_ids"],
                        confidence=finding.get("confidence", 0.8),
                        explanation=finding.get("explanation", ""),
                        suggested_operation=suggested,
                    )
                )
        except Exception as e:
            # Non-fatal: log and continue with what we have
            print(f"[detect] Warning: LLM detection failed for cluster {cluster.cluster_id}: {e}")
            # Still mark the cluster as synthesis candidate
            if len(cluster.memory_ids) >= 3:
                detections.append(
                    Detection(
                        type=DetectionType.CLUSTER,
                        memory_ids=cluster.memory_ids,
                        confidence=0.5,
                        explanation=f"Cluster of {len(cluster.memory_ids)} similar memories (LLM fallback)",
                        suggested_operation=OperationType.SYNTHESIZE,
                    )
                )

    return {"detections": detections}
