"""DreamKeeper — Interactive Demo

Seeds a ChromaDB store with 15 deliberately messy memories (duplicates,
contradictions, stale entries), then runs a full dream cycle and prints
the dream report.

Usage:
    uv run python demo/demo_dream.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app.adapters.chroma import ChromaAdapter
from app.graph import run_dream
from app.models import Memory, DreamReport


SAMPLE_FILE = Path(__file__).parent / "sample_memories.json"

# Use a separate collection for the demo so we don't pollute real data
DEMO_PERSIST_DIR = "./chroma_demo_data"


def _print_header(text: str):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def _print_memory(mem: Memory, index: int):
    status_icon = {
        "active": "🟢",
        "merged": "🔵",
        "superseded": "🟡",
        "synthesized": "🟣",
    }.get(mem.status.value, "⚪")
    age_days = (datetime.now(timezone.utc) - mem.created_at).days
    print(f"  {status_icon} [{index+1:2d}] {mem.content}")
    print(f"       Status: {mem.status.value} | Age: {age_days}d | Accesses: {mem.access_count}")
    if mem.merged_into:
        print(f"       → merged into {mem.merged_into[:8]}...")
    if mem.superseded_by:
        print(f"       → superseded by {mem.superseded_by[:8]}...")
    if mem.synthesized_into:
        print(f"       → synthesized into {mem.synthesized_into[:8]}...")
    print()


def _print_report(report: DreamReport):
    _print_header("DREAM REPORT")
    print(f"  Dream ID:        {report.dream_id}")
    print(f"  Memories scanned: {report.memories_scanned}")
    print(f"  Clusters found:   {report.clusters_found}")
    print(f"  Detections:       {len(report.detections)}")
    print(f"  Actions planned:  {report.actions_planned}")
    print(f"  Actions executed: {report.actions_executed}")
    print()

    print(f"  📊 Results:")
    print(f"     Merges:     {report.merges_applied}")
    print(f"     Supersedes: {report.supersedes_applied}")
    print(f"     Syntheses:  {report.syntheses_applied}")
    print()
    print(f"  📉 Reduction: {report.memories_before} → {report.memories_after} active memories ({report.reduction_pct}% reduction)")
    print()

    if report.results:
        _print_header("DIFFS (Audit Trail)")
        for r in report.results:
            icon = "✅" if r.success else "❌"
            print(f"  {icon} {r.operation.value.upper()}")
            print(f"     {r.diff}")
            if r.citations:
                print(f"     📎 Cites: {', '.join(c[:8] + '...' for c in r.citations)}")
            print()


async def main():
    _print_header("DreamKeeper — Demo")
    print("  Seeding 15 messy memories (duplicates, contradictions, stale)...")
    print()

    # 1. Load sample memories
    with open(SAMPLE_FILE) as f:
        samples = json.load(f)

    # 2. Seed the store
    store = ChromaAdapter(
        persist_dir=DEMO_PERSIST_DIR,
        collection_name="demo_memories",
    )

    # Clear previous demo data
    try:
        store._client.delete_collection("demo_memories")
        store._collection = store._client.get_or_create_collection(
            name="demo_memories",
            metadata={"hnsw:space": "cosine"},
        )
    except Exception:
        pass

    now = datetime.now(timezone.utc)
    for sample in samples:
        days_ago = sample.get("created_days_ago", 0)
        created = now - timedelta(days=days_ago)
        mem = Memory(
            content=sample["content"],
            metadata={k: v for k, v in sample.get("metadata", {}).items()},
            created_at=created,
            updated_at=created,
            access_count=0 if days_ago > 60 else max(1, 5 - days_ago // 20),
        )
        await store.upsert(mem)

    # 3. Show the messy state
    _print_header("BEFORE DREAMING — Memory Store")
    all_memories = await store.load_all(status=None)
    for i, mem in enumerate(all_memories):
        _print_memory(mem, i)

    print(f"  Total: {len(all_memories)} memories")

    # 4. Run the dream cycle
    _print_header("💤 DREAMING...")
    print("  Running: scan → detect → plan → execute → report")
    print()

    result = await run_dream(store=store, auto_approve=True)

    # 5. Show the report
    report = result.get("report")
    if report:
        _print_report(report)
    else:
        print("  No consolidation was needed.")

    # 6. Show the clean state
    _print_header("AFTER DREAMING — Memory Store")
    all_memories_after = await store.load_all(status=None)
    for i, mem in enumerate(all_memories_after):
        _print_memory(mem, i)

    active_after = [m for m in all_memories_after if m.status.value == "active"]
    print(f"  Total: {len(all_memories_after)} memories ({len(active_after)} active)")

    # Cleanup demo data
    import shutil
    if os.path.exists(DEMO_PERSIST_DIR):
        shutil.rmtree(DEMO_PERSIST_DIR)
        print(f"\n  🧹 Cleaned up demo data at {DEMO_PERSIST_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
