# Ask My Docs

Upload a PDF, build a knowledge graph, and ask grounded questions with **hybrid retrieval**, **citation enforcement**, **Langfuse tracing**, and a **CI-gated eval pipeline** 

Works with resumes, research papers, reports, manuals, and other text-based PDFs.

## Features

| Capability | Implementation |
|------------|----------------|
| Document Q&A | **Ask My Docs** — grounded answers over your uploaded PDF |
| Hybrid retrieval | **BM25** + **local vector embeddings** (`all-MiniLM-L6-v2`) |
| Fusion | **Reciprocal Rank Fusion (RRF)** across retrieval legs |
| Reranking | **Cross-encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) |
| Citation enforcement | Numbered sources `[1]`…`[n]`; validation + UI warnings |
| Knowledge graph | Groq entity/relationship extraction + React Flow visualization |
| Observability | **Langfuse** traces, sessions, quality scores; **p50/p95** latency; **cost/request** |
| CI eval gate | `pytest` + offline eval + **regression baselines** in GitHub Actions |

## Architecture

```
PDF upload
  → pypdf (per-page text) → chunks
  → knowledge graph extraction (Groq, traced)
  → passages index (chunks + graph evidence)
  → BM25 + local embeddings (sentence-transformers)
  → save graph.json, chunks.json, passages.json, embeddings.npz

POST /ask
  → BM25 top-K + vector top-K → RRF → cross-encoder rerank
  → numbered sources + Groq answer with [n] citations
  → citation validation
  → Langfuse trace (spans + generations + scores)
  → per-response metrics (latency stages, cost, tokens)
```

## Tech stack

| Layer | Technology |
|-------|------------|
| Backend | FastAPI, Uvicorn, pypdf, Groq SDK |
| Retrieval | rank-bm25, NumPy, sentence-transformers |
| Embeddings | `all-MiniLM-L6-v2` (local; Groq has no embedding API) |
| LLM | Groq `llama-3.3-70b-versatile` |
| Observability | Langfuse SDK + in-process `MetricsStore` |
| Frontend | Next.js 15, React Flow |
| Storage | Local JSON + `.npz` embeddings |
| CI | GitHub Actions, pytest, regression baselines |



## Project structure

```
graph-rag-research-agent/
├── backend/
│   ├── main.py              # FastAPI routes + trace wrappers
│   ├── groq_client.py       # Groq API key + client (shared, no circular imports)
│   ├── groq_llm.py          # Traced Groq chat completions
│   ├── observability.py     # Langfuse, p50/p95, cost, quality scores
│   ├── extractor.py         # PDF extraction + chunking
│   ├── graph_builder.py     # Graph extraction (Groq)
│   ├── passages.py          # Unified retrieval passages
│   ├── embeddings.py        # Local sentence-transformers embeddings
│   ├── retrieval.py         # BM25 + vector + RRF + rerank
│   ├── retrieval_index.py   # Index build/load
│   ├── citations.py         # Numbered citations + enforcement
│   ├── qa.py                # Ask My Docs Q&A orchestration
│   ├── eval/
│   │   ├── run_eval.py      # CI eval + regression gate
│   │   ├── baselines.json   # Regression thresholds
│   │   ├── fixtures/        # Sample document + test cases
│   │   └── tests/           # pytest (retrieval, citations, observability)
│   └── data/                # Generated (gitignored)
├── frontend/app/page.tsx    # Ask My Docs UI + per-answer metrics
├── .cursor/skills/langfuse/ # Langfuse agent skill (optional, for Cursor)
├── .github/workflows/eval.yml
└── README.md
```

## Quick start

### Prerequisites

- Python 3.11+
- Node.js 18+
- [Groq API key](https://console.groq.com/keys)
- (Optional) [Langfuse](https://cloud.langfuse.com) project for tracing

### Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env: GROQ_API_KEY=gsk_...
# Optional: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST

uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

First run downloads the cross-encoder model (~80MB).

Verify:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/metrics
```

### Frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local   # if present
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

### Usage

1. **Upload PDF** — builds graph, chunks, and retrieval index
2. **Explore graph** — click nodes/edges in React Flow
3. **Ask questions** — answers with `[n]` citations; UI shows latency + cost per answer

**Re-upload** after pulling upgrades that change the index format (`passages.json`, `embeddings.npz`).

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health, Groq key status, `langfuse_enabled` |
| `GET` | `/metrics` | p50/p95 latency, cost aggregates, quality rates |
| `POST` | `/upload` | PDF → graph + chunks + retrieval index |
| `GET` | `/graph` | Nodes and edges for React Flow |
| `DELETE` | `/document` | Clear stored document + index |
| `POST` | `/ask` | Hybrid retrieval Q&A with citations + metrics |

### `POST /ask` response

```json
{
  "answer": "Alice Chen founded Acme Corp in 2019 [1].",
  "citations": [{ "citation_id": 1, "source_type": "graph_edge", "text": "..." }],
  "citation_valid": true,
  "sources": { "graph_edges": [], "chunks": [] },
  "metrics": {
    "cost_usd": 0.000734,
    "stages_ms": { "hybrid_retrieval": 450.2, "groq_answer": 2100.1 },
    "groq_tokens": { "prompt": 2100, "completion": 180 }
  }
}
```

### `GET /metrics`

Rolling in-process aggregates (last 500 requests):

- **Latency** — p50/p95 per operation (`ask`, `upload_pdf`)
- **Cost** — total and avg USD per operation (Groq token estimates)
- **Quality** — `citation_valid_rate` across ask requests
- **Tokens** — cumulative Groq prompt/completion counts

## Observability (Langfuse)

Optional but recommended for demos and debugging.

1. Create a project at [cloud.langfuse.com](https://cloud.langfuse.com) or [us.cloud.langfuse.com](https://us.cloud.langfuse.com)
2. **Settings → API Keys** → copy public + secret keys
3. Add to `backend/.env`:

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
# US region (match where you signed up):
LANGFUSE_HOST=https://us.cloud.langfuse.com
# EU region:
# LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_TRACING_ENVIRONMENT=development
```

4. **Restart the backend** (`.env` is only read at startup)
5. Confirm: `curl http://localhost:8000/health` → `"langfuse_enabled": true`
6. Upload + ask → check Langfuse **Tracing** and **Sessions**

**What gets traced**

| Trace | Spans | Scores |
|-------|-------|--------|
| `ask-my-docs/upload_pdf` | `extract_pdf`, `chunk_pages`, `graph_extraction`, `build_retrieval_index` | — |
| `ask-my-docs/ask` | `hybrid_retrieval`, `groq_answer`, `citation_enforcement` | `citation_valid`, `retrieval_hit` |

Groq calls are recorded as **generation** observations (model, tokens, estimated cost, output preview).

Without Langfuse keys, the app still runs — local `/metrics` and UI cost/latency still work.

**Cursor:** Langfuse agent skill is in `.cursor/skills/langfuse/` ([upstream](https://github.com/langfuse/skills)).

## Evaluation pipeline (CI-gated)

```bash
cd backend
source .venv/bin/activate
pytest eval/tests/ -v
python eval/run_eval.py --offline --min-score 0.75
```

GitHub Actions (`.github/workflows/eval.yml`) on push/PR to `main`:

1. Unit tests — retrieval, citations, observability metrics
2. Offline eval — fixture recall (≥75% pass rate)
3. **Regression gate** — `eval/baselines.json` (score, retrieval hit rate, citation fixture rate)
4. Uploads `eval/last_report.json` as a workflow artifact

## Environment variables

| Variable | Where | Default | Description |
|----------|-------|---------|-------------|
| `GROQ_API_KEY` | `backend/.env` | — | **Required** |
| `GROQ_MODEL` | `backend/.env` | `llama-3.3-70b-versatile` | Chat model |
| `LANGFUSE_PUBLIC_KEY` | `backend/.env` | — | Optional tracing |
| `LANGFUSE_SECRET_KEY` | `backend/.env` | — | Optional tracing |
| `LANGFUSE_HOST` | `backend/.env` | `https://cloud.langfuse.com` | Must match your Langfuse region |
| `LANGFUSE_TRACING_ENVIRONMENT` | `backend/.env` | `development` | `development` / `staging` / `production` |
| `GROQ_INPUT_USD_PER_1M` | `backend/.env` | `0.59` | Cost estimate (input tokens) |
| `GROQ_OUTPUT_USD_PER_1M` | `backend/.env` | `0.79` | Cost estimate (output tokens) |
| `NEXT_PUBLIC_API_URL` | `frontend/.env.local` | `http://localhost:8000` | Backend URL |
| `ASK_MY_DOCS_DOMAIN` | `backend/.env` | uploaded document | Prompt domain hint |
| `EMBEDDING_MODEL` | `backend/.env` | `all-MiniLM-L6-v2` | Local embedding model |

Never commit `backend/.env` — use `backend/.env.example` as the template.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `langfuse_enabled: false` | Add keys to `backend/.env`, **save file**, restart uvicorn |
| Traces missing in Langfuse UI | `LANGFUSE_HOST` must match region (US vs EU); re-upload + ask after fix |
| `Retrieval index not found` | Re-upload PDF |
| Slow first question | Cross-encoder cold start; later asks are faster |
| Port 8000 in use | `lsof -ti :8000 \| xargs kill` then restart backend |
| CI eval fails | Run `pytest eval/tests/` and `python eval/run_eval.py --offline` locally |
| Empty graph | Valid `GROQ_API_KEY` in `.env`; check `/health` |

## License

MIT
