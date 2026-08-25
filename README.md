# DreamKeeper

**The open-source "dreaming" pass for AI agent memory — merges duplicates, supersedes stale facts, synthesizes patterns. Nothing is ever destroyed.**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Pipeline-orange.svg)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com)
[![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-purple.svg)](https://ai.google.dev)

---

## The Problem

A long-running agent gets *worse* at remembering facts over time.

As the memory store grows, the same preference is saved four different ways, last month's decision still contradicts this month's, and every retrieval is now scoring a pile of half-stale, half-duplicate entries. **The store grows and recall degrades with it.**

OpenAI, Anthropic, Google, and Mem0 all ship "dreaming" — a background consolidation pass — but **there is no open-source, pluggable, framework-agnostic version** you can drop into your own agent pipeline.

**DreamKeeper is that missing piece.**

## The Solution

A LangGraph-powered agent that runs a scheduled "dream cycle" over any memory store:

```
┌─────────────────────────────────────────────────────────────┐
│                      DREAM CYCLE                            │
│                                                             │
│  SCAN → DETECT → PLAN → [APPROVE] → EXECUTE → REPORT       │
│                                                             │
│  Three operations:                                          │
│  🔵 MERGE      — collapse duplicates into one canonical     │
│  🟡 SUPERSEDE  — flag stale facts, link to replacement      │
│  🟣 SYNTHESIZE — distill clusters into higher-order memory  │
│                                                             │
│  Every change = a DIFF you can read, trace, and undo        │
│  Every synthesis = CITES its source memories                │
│  Nothing is ever destroyed                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Install

```bash
uv sync
# or: pip install -e ".[dev]"
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and set your GOOGLE_API_KEY
```

### 3. Run the demo

```bash
uv run python demo/demo_dream.py
```

Seeds 15 deliberately messy memories (duplicates, contradictions, stale entries), runs a full dream cycle, and prints the before/after state with the dream report.

### 4. Run the API

```bash
uv run uvicorn app.main:app --reload
```

Then open [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive API.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/memories` | Add a memory to the store |
| `GET` | `/memories` | List all memories (filter by status) |
| `POST` | `/dream` | Trigger a dream cycle |
| `GET` | `/dream/{dream_id}` | Get the full dream report |
| `GET` | `/health` | Memory store health metrics |

### Example: Full workflow

```bash
# 1. Add some memories
curl -X POST http://localhost:8000/memories \
  -H "Content-Type: application/json" \
  -d '{"content": "User lives in Lisbon, Portugal"}'

curl -X POST http://localhost:8000/memories \
  -H "Content-Type: application/json" \
  -d '{"content": "User recently moved to Berlin, Germany"}'

curl -X POST http://localhost:8000/memories \
  -H "Content-Type: application/json" \
  -d '{"content": "The user lives in Lisbon"}'

# 2. Check health (3 memories, potential issues)
curl http://localhost:8000/health

# 3. Trigger a dream cycle
curl -X POST http://localhost:8000/dream

# 4. Check the report
curl http://localhost:8000/dream/{dream_id}
```

---

## How It Works

### Pipeline Architecture

```
scan → detect → plan → [human_review] → execute → report
```

| Node | What it does |
|---|---|
| **Scan** | Load all active memories, compute pairwise similarities, cluster by semantic proximity |
| **Detect** | Find duplicates (same fact, different words), contradictions (conflicting facts), stale entries (old + never accessed), synthesis candidates (related cluster) |
| **Plan** | Generate a consolidation plan — the diff preview with target content, reasons, and citations |
| **Execute** | Apply merges/supersedes/syntheses to the store. **Nothing is ever deleted** — old memories are flagged and linked to their replacement |
| **Report** | Assemble the dream report: what changed, why, with full audit trail |

### Design Principles

1. **Nothing is destroyed.** Superseded memories are flagged, not deleted. Full audit trail.
2. **Never on the hot path.** Consolidation runs async/scheduled, never during user queries.
3. **Importance scoring.** Multi-dimensional: relevance, frequency, novelty, recency (exponential decay).
4. **Provenance.** Every synthesized memory cites its source memories.
5. **Human-in-the-loop.** Optional: preview the plan before approving, or auto-approve for trusted setups.

---

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Orchestration | LangGraph | State machine with conditional edges, HITL support |
| LLM | Gemini 2.5 Flash | Fast, cheap, reliable structured JSON output |
| Vector Store | ChromaDB | Zero-setup, pip-installable, adapter pattern for others |
| Embeddings | ChromaDB default (local) | Free, no API key needed |
| API | FastAPI | Async, auto-docs, production-ready |
| Validation | Pydantic v2 | Schema-level enforcement on every data structure |

---

## Project Layout

```
dreamkeeper/
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── graph.py              # LangGraph dream cycle
│   ├── models.py             # Pydantic schemas (Memory, Detection, DreamReport, etc.)
│   ├── scoring.py            # Multi-dimensional importance scoring
│   ├── nodes/
│   │   ├── scan.py           # Load + cluster memories
│   │   ├── detect.py         # Find dupes / contradictions / stale
│   │   ├── plan.py           # Generate consolidation plan
│   │   ├── execute.py        # Apply merge / supersede / synthesize
│   │   └── report.py         # Generate dream report
│   └── adapters/
│       ├── base.py           # Abstract memory store interface
│       └── chroma.py         # ChromaDB adapter
├── demo/
│   ├── demo_dream.py         # Interactive demo
│   └── sample_memories.json  # 15 messy test memories
├── tests/
│   └── test_scoring_and_models.py
├── pyproject.toml
├── .env.example
└── LICENSE
```

---

## What Makes This Different

| Feature | Mem0 Dream | LangMem | OpenAI | DreamKeeper |
|---|---|---|---|---|
| Open source | ❌ | ✅ | ❌ | ✅ |
| All 3 operations | ✅ | ❌ | Partial | ✅ |
| Audit trail / diffs | ✅ | ❌ | ❌ | ✅ |
| Framework agnostic | ❌ | ❌ | ❌ | ✅ (adapter) |
| Human-in-the-loop | ❌ | ❌ | ❌ | ✅ |
| Importance scoring | ❌ | ❌ | Unknown | ✅ |
| Self-hostable | ❌ | ✅ | ❌ | ✅ |
| Dream reports | ❌ | ❌ | ❌ | ✅ |

---

## Testing

```bash
uv run pytest tests/ -v
```

---

## Inspired By

- [OpenAI: Dreaming — Better memory for a more helpful ChatGPT (June 2026)](https://openai.com/index/chatgpt-memory-dreaming/)
- [Anthropic: New in Claude Managed Agents — Dreaming (May 2026)](https://claude.com/blog/new-in-claude-managed-agents)
- [Google: Gemini Agent Platform Memory Bank](https://docs.cloud.google.com/gemini-enterprise-agent-platform/scale/memory-bank)
- [Mem0: Dream](https://docs.mem0.ai/platform/features/dream)
- [Auto-Dreamer: Learning Offline Memory Consolidation (Ye et al.)](https://arxiv.org/abs/2605.20616)
- [Language Models Need Sleep (Behrouz et al.)](https://arxiv.org/abs/2606.03979)

---

## License

MIT — see [LICENSE](LICENSE).
