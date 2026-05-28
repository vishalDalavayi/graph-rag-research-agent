# Ask My Docs

A domain-specific **“Ask My Docs”** system: upload a PDF, build a knowledge graph, and ask questions with **hybrid retrieval (BM25 + vector search)**, **cross-encoder reranking**, **citation enforcement**, and a **CI-gated evaluation pipeline**.

Works with résumés, research papers, reports, manuals, and other text-based PDFs.

## Features

| Capability | Implementation |
|------------|----------------|
| Document Q&A | **Ask My Docs** — grounded answers over your uploaded PDF |
| Hybrid retrieval | **BM25** + **local vector embeddings** (`all-MiniLM-L6-v2`) |
| Fusion | **Reciprocal Rank Fusion (RRF)** across retrieval legs |
| Reranking | **Cross-encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) |
| Citation enforcement | Numbered sources `[1]`…`[n]`; validation + compliance notes |
| Knowledge graph | Groq entity/relationship extraction + React Flow visualization |
| CI eval gate | `pytest` + offline eval script in GitHub Actions |

## Architecture

```
PDF upload
  → pypdf (per-page text)
  → ~800 char chunks + knowledge graph (Groq)
  → passages index (chunks + graph evidence)
  → BM25 corpus + local embeddings (sentence-transformers)
  → save graph.json, chunks.json, passages.json, embeddings.npz

POST /ask
  → BM25 top-K + vector top-K
  → RRF fusion
  → cross-encoder rerank
  → numbered sources + Groq answer with [n] citations
  → citation validation
```

## Tech stack

| Layer | Technology |
|-------|------------|
| Backend | FastAPI, Uvicorn, pypdf, Groq SDK |
| Retrieval | rank-bm25, NumPy, sentence-transformers (cross-encoder) |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` (local; Groq has no embedding API) |
| LLM | Groq `llama-3.3-70b-versatile` |
| Frontend | Next.js 15, React Flow |
| Storage | Local JSON + `.npz` embeddings |
| CI | GitHub Actions + pytest |

**Not used:** LangChain, LangGraph, Pinecone, Postgres.

## Project structure

```
graph-rag-research-agent/
├── backend/
│   ├── main.py              # FastAPI: /health, /upload, /graph, /ask
│   ├── extractor.py           # PDF extraction + chunking
│   ├── graph_builder.py     # Graph extraction (Groq)
│   ├── passages.py          # Unified retrieval passages
│   ├── embeddings.py        # Groq embeddings
│   ├── retrieval.py           # BM25 + vector + RRF + rerank
│   ├── retrieval_index.py   # Index build/load
│   ├── citations.py           # Numbered citations + enforcement
│   ├── qa.py                  # Ask My Docs Q&A orchestration
│   ├── eval/
│   │   ├── run_eval.py        # CI eval runner
│   │   ├── fixtures/          # Sample document + test cases
│   │   └── tests/             # pytest (retrieval + citations)
│   └── data/                  # Generated (gitignored)
├── frontend/app/page.tsx      # Ask My Docs UI
├── .github/workflows/eval.yml # CI gate
└── README.md
```

## Quick start

### Prerequisites

- Python 3.11+
- Node.js 18+
- [Groq API key](https://console.groq.com/keys)

### Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Set GROQ_API_KEY=gsk_...

uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

First run downloads the cross-encoder model (~80MB).

### Frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

### Usage

1. **Upload PDF** — builds graph, chunks, and retrieval index (BM25 + vectors)
2. **Explore graph** — click nodes/edges
3. **Ask questions** — answers include `[n]` citations; expand **Citations** / **Graph sources** / **Text sources**

**Re-upload required** after upgrading from older versions (needs `passages.json` + `embeddings.npz`).

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health + retrieval stack info |
| `POST` | `/upload` | PDF → graph + chunks + retrieval index |
| `GET` | `/graph` | Nodes and edges for React Flow |
| `POST` | `/ask` | Hybrid retrieval Q&A with citations |

### `POST /ask` response

```json
{
  "answer": "Alice Chen founded Acme Corp in 2019 [1].",
  "citations": [
    {
      "citation_id": 1,
      "source_type": "graph_edge",
      "source_id": "e_founded",
      "text": "Alice Chen --[founded]--> Acme Corp..."
    }
  ],
  "citation_valid": true,
  "sources": {
    "graph_edges": [...],
    "chunks": [...]
  }
}
```

## Evaluation pipeline (CI-gated)

```bash
cd backend
source .venv/bin/activate
pytest eval/tests/ -v
python eval/run_eval.py --offline --min-score 0.75
```

GitHub Actions (`.github/workflows/eval.yml`) runs on every push/PR to `main`:
1. Unit tests — BM25, RRF, hybrid retrieval, citation validation
2. Offline eval — retrieval recall on fixture document (≥75% pass rate)

## Environment variables

| Variable | Where | Default | Description |
|----------|-------|---------|-------------|
| `GROQ_API_KEY` | `backend/.env` | — | Required |
| `NEXT_PUBLIC_API_URL` | `frontend/.env.local` | `http://localhost:8000` | Backend URL |
| `ASK_MY_DOCS_DOMAIN` | `backend/.env` | uploaded document | Domain hint for prompts |
| `EMBEDDING_MODEL` | `backend/.env` | `all-MiniLM-L6-v2` | Local embedding model |

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Retrieval index not found` | Re-upload PDF to rebuild index |
| Slow first question | Cross-encoder model loading; subsequent asks are faster |
| CI fails on eval | Run `pytest eval/tests/` and `python eval/run_eval.py --offline` locally |
| Empty graph | Check `GROQ_API_KEY` via `/health` |

## License

MIT
