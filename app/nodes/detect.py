"""DETECT node: Find duplicates, contradictions, and stale entries.

Uses Gemini with batch analysis across all clusters to identify duplicates,
contradictions, and synthesis candidates efficiently in a single LLM call.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone

from google import genai
from google.genai import errors as genai_errors

from app.models import (
    Detection,
    DetectionType,
    Memory,
    MemoryCluster,
    OperationType,
)


STALE_DAYS = int(os.getenv("STALE_DAYS", "30"))


BATCH_DETECT_PROMPT = """You are analyzing clusters of semantically related memories from an AI agent's memory store.
Your goal is to detect redundancy, factual contradictions, and synthesis opportunities across these clusters.

## Memory Clusters to analyze:
{clusters_text}

## Task
For each cluster:
1. Identify **duplicate** facts (same fact stated in different words).
2. Identify **contradictions** (conflicting/superseded facts, e.g. location changed or database changed).
3. Identify **clusters** of 2+ related observations that can be synthesized into a higher-order pattern.

Respond in this exact JSON format (no markdown, no preamble):
{{
    "findings": [
        {{
            "type": "duplicate" | "contradiction" | "cluster",
            "memory_ids": ["id_1", "id_2"],
            "confidence": 0.95,
            "explanation": "Brief explanation of why this was flagged",
            "suggested_operation": "merge" | "supersede" | "synthesize"
        }}
    ]
}}

If memories in a cluster are genuinely different and accurate, do not flag them.
"""


def _call_gemini_with_retry(client: genai.Client, model: str, prompt: str, max_retries: int = 3) -> str:
    """Call Gemini with exponential backoff on rate limits."""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                ),
            )
            return response.text
        except genai_errors.APIError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_time = (2 ** attempt) * 3 + 2
                print(f"[detect] Rate limited (429). Retrying in {wait_time}s... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise e
    raise RuntimeError(f"Failed to generate content after {max_retries} attempts.")


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

    # --- 2. Cluster-based detection (single batched LLM call) ---
    if not clusters:
        return {"detections": detections}

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    clusters_text = ""
    for idx, cluster in enumerate(clusters):
        clusters_text += f"\n### Cluster {idx + 1} (Centroid: {cluster.centroid_content[:60]}):\n"
        for mid in cluster.memory_ids:
            mem = memory_map.get(mid)
            if mem:
                clusters_text += f"- ID: {mid}\n  Content: {mem.content}\n  Created: {mem.created_at.isoformat()}\n  Accesses: {mem.access_count}\n"

    prompt = BATCH_DETECT_PROMPT.format(clusters_text=clusters_text)

    try:
        raw_text = _call_gemini_with_retry(client, model_name, prompt)
        result = json.loads(raw_text)

        type_map = {
            "duplicate": (DetectionType.DUPLICATE, OperationType.MERGE),
            "contradiction": (DetectionType.CONTRADICTION, OperationType.SUPERSEDE),
            "cluster": (DetectionType.CLUSTER, OperationType.SYNTHESIZE),
            "stale": (DetectionType.STALE, OperationType.SUPERSEDE),
        }

        for finding in result.get("findings", []):
            det_type_str = finding.get("type", "duplicate").lower()
            dtype, default_op = type_map.get(det_type_str, (DetectionType.DUPLICATE, OperationType.MERGE))

            op_str = finding.get("suggested_operation", "").lower()
            if op_str == "merge":
                suggested_op = OperationType.MERGE
            elif op_str == "supersede":
                suggested_op = OperationType.SUPERSEDE
            elif op_str == "synthesize":
                suggested_op = OperationType.SYNTHESIZE
            else:
                suggested_op = default_op

            detections.append(
                Detection(
                    type=dtype,
                    memory_ids=finding.get("memory_ids", []),
                    confidence=float(finding.get("confidence", 0.85)),
                    explanation=finding.get("explanation", "Detected via memory clustering"),
                    suggested_operation=suggested_op,
                )
            )

    except Exception as e:
        print(f"[detect] Warning: LLM batch detection failed: {e}")
        # Fallback: flag clusters with >= 3 memories as synthesis candidates
        for cluster in clusters:
            if len(cluster.memory_ids) >= 3:
                detections.append(
                    Detection(
                        type=DetectionType.CLUSTER,
                        memory_ids=cluster.memory_ids,
                        confidence=0.5,
                        explanation=f"Cluster of {len(cluster.memory_ids)} similar memories (heuristic fallback)",
                        suggested_operation=OperationType.SYNTHESIZE,
                    )
                )

    return {"detections": detections}
