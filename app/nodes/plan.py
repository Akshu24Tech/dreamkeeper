"""PLAN node: Generate the consolidation plan (the diff preview).

Takes the detections and produces a set of ConsolidationActions that
describe exactly what will change, why, and which sources are cited.
This is the artifact the user reviews before approving.
"""

from __future__ import annotations

import json
import os

from google import genai

from app.models import (
    ConsolidationAction,
    ConsolidationPlan,
    Detection,
    DetectionType,
    Memory,
    OperationType,
)


PLAN_PROMPT_MERGE = """You are consolidating duplicate memories into one canonical version.

## Duplicate memories:
{memories_text}

Write a single, clean canonical version that captures all the information without redundancy.
Respond in this exact JSON format (no markdown):
{{
    "canonical_content": "the merged content",
    "reason": "why these were merged"
}}
"""

PLAN_PROMPT_SUPERSEDE = """You are resolving contradicting or stale memories.

## Conflicting/stale memories (ordered oldest to newest):
{memories_text}

Determine which memory represents the CURRENT truth.
Respond in this exact JSON format (no markdown):
{{
    "current_content": "the current truth",
    "reason": "why the older versions are superseded"
}}
"""

PLAN_PROMPT_SYNTHESIZE = """You are distilling a cluster of related memories into one higher-order insight.

## Related memories:
{memories_text}

Write a single higher-order summary that captures the pattern or conclusion these memories collectively imply. Cite every source memory by ID.
Respond in this exact JSON format (no markdown):
{{
    "synthesis": "the higher-order memory",
    "reason": "why this synthesis was created",
    "cited_ids": ["id1", "id2", "id3"]
}}
"""


async def plan(
    state: dict,
    *,
    model_name: str = "gemini-2.5-flash",
) -> dict:
    """Generate a consolidation plan from detections."""

    memories: list[Memory] = state.get("memories", [])
    detections: list[Detection] = state.get("detections", [])
    memory_map = {m.id: m for m in memories}

    actions: list[ConsolidationAction] = []
    merges = 0
    supersedes = 0
    syntheses = 0

    client = genai.Client()

    for detection in detections:
        # Build memory text for the prompt
        memories_text = ""
        for mid in detection.memory_ids:
            mem = memory_map.get(mid)
            if mem:
                memories_text += (
                    f"- ID: {mid}\n"
                    f"  Content: {mem.content}\n"
                    f"  Created: {mem.created_at.isoformat()}\n\n"
                )

        try:
            if detection.suggested_operation == OperationType.MERGE:
                prompt = PLAN_PROMPT_MERGE.format(memories_text=memories_text)
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        temperature=0.1,
                        response_mime_type="application/json",
                    ),
                )
                result = json.loads(response.text)
                actions.append(
                    ConsolidationAction(
                        operation=OperationType.MERGE,
                        source_memory_ids=detection.memory_ids,
                        target_content=result["canonical_content"],
                        reason=result["reason"],
                        citations=detection.memory_ids,
                    )
                )
                merges += 1

            elif detection.suggested_operation == OperationType.SUPERSEDE:
                prompt = PLAN_PROMPT_SUPERSEDE.format(memories_text=memories_text)
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        temperature=0.1,
                        response_mime_type="application/json",
                    ),
                )
                result = json.loads(response.text)
                actions.append(
                    ConsolidationAction(
                        operation=OperationType.SUPERSEDE,
                        source_memory_ids=detection.memory_ids,
                        target_content=result["current_content"],
                        reason=result["reason"],
                        citations=detection.memory_ids,
                    )
                )
                supersedes += 1

            elif detection.suggested_operation == OperationType.SYNTHESIZE:
                prompt = PLAN_PROMPT_SYNTHESIZE.format(memories_text=memories_text)
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        temperature=0.2,
                        response_mime_type="application/json",
                    ),
                )
                result = json.loads(response.text)
                actions.append(
                    ConsolidationAction(
                        operation=OperationType.SYNTHESIZE,
                        source_memory_ids=detection.memory_ids,
                        target_content=result["synthesis"],
                        reason=result["reason"],
                        citations=result.get("cited_ids", detection.memory_ids),
                    )
                )
                syntheses += 1

        except Exception as e:
            print(f"[plan] Warning: Planning failed for detection {detection.type}: {e}")

    consolidation_plan = ConsolidationPlan(
        actions=actions,
        total_merges=merges,
        total_supersedes=supersedes,
        total_syntheses=syntheses,
    )

    return {"plan": consolidation_plan}
