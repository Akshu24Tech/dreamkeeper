"""PLAN node: Generate the consolidation plan (the diff preview).

Takes the detections and produces a set of ConsolidationActions in a
single batched LLM pass with retry resilience against rate limits.
"""

from __future__ import annotations

import json
import os
import time

from google import genai
from google.genai import errors as genai_errors

from app.models import (
    ConsolidationAction,
    ConsolidationPlan,
    Detection,
    DetectionType,
    Memory,
    OperationType,
)


BATCH_PLAN_PROMPT = """You are generating the consolidation actions for an AI agent's memory store.

Here are the detected issues:
{detections_text}

## Tasks for each detection:
- **MERGE**: Write a single clean canonical version that combines the duplicate facts without redundancy.
- **SUPERSEDE**: Determine the current truth from the conflicting/stale facts and explain why older facts are superseded.
- **SYNTHESIZE**: Distill the related observations into a single higher-order insight/pattern and list all cited memory IDs.

Respond in this exact JSON format (no markdown, no extra preamble):
{{
    "actions": [
        {{
            "operation": "merge" | "supersede" | "synthesize",
            "source_memory_ids": ["id1", "id2"],
            "target_content": "The consolidated or current or synthesized text",
            "reason": "Why this action was taken",
            "citations": ["id1", "id2"]
        }}
    ]
}}
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
                wait_time = (2 ** attempt) * 4 + 2
                print(f"[plan] Rate limited (429). Retrying in {wait_time}s... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise e
    raise RuntimeError(f"Failed to generate plan after {max_retries} attempts.")


async def plan(
    state: dict,
    *,
    model_name: str = "gemini-2.5-flash",
) -> dict:
    """Generate a consolidation plan from detections."""

    memories: list[Memory] = state.get("memories", [])
    detections: list[Detection] = state.get("detections", [])
    memory_map = {m.id: m for m in memories}

    if not detections:
        return {"plan": ConsolidationPlan()}

    actions: list[ConsolidationAction] = []
    merges = 0
    supersedes = 0
    syntheses = 0

    # Separate deterministic stale vs LLM-required detections
    llm_detections = [d for d in detections if len(d.memory_ids) >= 2]
    standalone_stale = [d for d in detections if len(d.memory_ids) == 1 and d.type == DetectionType.STALE]

    # Handle standalone stale memories without consuming LLM quota
    for st in standalone_stale:
        mid = st.memory_ids[0]
        mem = memory_map.get(mid)
        if mem:
            actions.append(
                ConsolidationAction(
                    operation=OperationType.SUPERSEDE,
                    source_memory_ids=[mid],
                    target_content=f"[Archived/Stale] {mem.content}",
                    reason=st.explanation,
                    citations=[mid],
                )
            )
            supersedes += 1

    # If there are cluster/duplicate/contradiction detections, run batched LLM
    if llm_detections:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)

        detections_text = ""
        for i, det in enumerate(llm_detections):
            detections_text += f"\n### Detection #{i+1} ({det.type.value.upper()} -> {det.suggested_operation.value}):\n"
            detections_text += f"Reason: {det.explanation}\nInvolved memories:\n"
            for mid in det.memory_ids:
                mem = memory_map.get(mid)
                if mem:
                    detections_text += (
                        f"  - ID: {mid}\n"
                        f"    Content: {mem.content}\n"
                        f"    Created: {mem.created_at.isoformat()}\n"
                        f"    Accesses: {mem.access_count}\n"
                    )

        prompt = BATCH_PLAN_PROMPT.format(detections_text=detections_text)

        try:
            raw_text = _call_gemini_with_retry(client, model_name, prompt)
            result = json.loads(raw_text)

            for act in result.get("actions", []):
                op_str = act.get("operation", "").lower()
                if op_str == "merge":
                    op = OperationType.MERGE
                    merges += 1
                elif op_str == "supersede":
                    op = OperationType.SUPERSEDE
                    supersedes += 1
                elif op_str == "synthesize":
                    op = OperationType.SYNTHESIZE
                    syntheses += 1
                else:
                    op = OperationType.MERGE
                    merges += 1

                actions.append(
                    ConsolidationAction(
                        operation=op,
                        source_memory_ids=act.get("source_memory_ids", []),
                        target_content=act.get("target_content", ""),
                        reason=act.get("reason", "Consolidated during dream cycle"),
                        citations=act.get("citations", act.get("source_memory_ids", [])),
                    )
                )

        except Exception as e:
            print(f"[plan] Warning: Batch planning failed: {e}")

    consolidation_plan = ConsolidationPlan(
        actions=actions,
        total_merges=merges,
        total_supersedes=supersedes,
        total_syntheses=syntheses,
    )

    return {"plan": consolidation_plan}
