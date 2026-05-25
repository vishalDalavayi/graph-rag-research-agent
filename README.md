# Graph RAG Research Agent

A full-stack MVP that turns PDF documents into interactive knowledge graphs. Upload a PDF, extract entities and relationships with **Groq** (`llama-3.3-70b-versatile`), explore the graph in **React Flow**, and ask questions using **hybrid graph + raw chunk** retrieval.

Works with any text-based PDF — résumés, research papers, reports, manuals, etc.

## Features

- **PDF → knowledge graph** — entities and relationships extracted per chunk, merged into one graph
- **Interactive visualization** — pan, zoom, minimap; click nodes/edges for details
- **Hybrid Q&A** — retrieves from both the graph and raw text chunks, then answers with Groq
- **Chat with follow-ups** — last 6 turns sent as history for pronoun resolution
- **Collapsible sources** — graph edges and text chunks shown on demand in the UI
- **No framework overhead** — plain Python + FastAPI + Groq SDK (no LangChain, LangGraph, or vector DB)

## Tech stack

| Layer | Technology |
|-------|------------|
| Backend | FastAPI, Uvicorn, pypdf, Groq SDK |
| Frontend | Next.js 15, React 19, TypeScript, React Flow |
| LLM | `llama-3.3-70b-versatile` via Groq |
| Storage | Local JSON files (`graph.json`, `chunks.json`) |

## Architecture

```
┌─────────────┐     POST /upload      ┌──────────────────────────────────────┐
│   Browser   │ ────────────────────► │  FastAPI backend                     │
│  (Next.js)  │                       │                                      │
│             │     GET /graph          │  1. pypdf → text per page            │
│  React Flow │ ◄──────────────────── │  2. ~800 char chunks (+ page nums)   │
│  + Q&A chat │                       │  3. Groq → entities/relationships    │
│             │     POST /ask         │  4. merge → graph.json + chunks.json │
└─────────────┘ ────────────────────► │  5. keyword retrieval + Groq answer  │
                                      └──────────────────────────────────────┘
                                                    │
                                                    ▼
                                          backend/data/
                                          ├── graph.json
                                          └── chunks.json
```

### Ingestion (upload)

1. Extract text per page with **pypdf**
2. Split into ~800-character chunks (with page numbers)
3. For each chunk, **Groq** returns JSON: entities + relationships with evidence quotes
4. Merge entities by normalized label across chunks
5. Persist graph and raw chunks to disk

### Q&A (hybrid Graph RAG)

1. Load `graph.json` and `chunks.json`
2. **Graph retrieval** — keyword-score nodes/edges; expand 1-hop neighbors
3. **Chunk retrieval** — keyword-score raw text chunks
4. Build combined context (entities, relationships, evidence quotes, chunk text)
5. **Groq** generates a concise answer from context only — prefer graph evidence, use chunks for gaps
6. Return answer + graph edge sources + chunk sources

Retrieval is **keyword-based** (no embeddings or vector DB yet).

## Project structure

```
graph-rag-research-agent/
├── backend/
│   ├── main.py           # FastAPI: /health, /upload, /graph, /ask
│   ├── extractor.py      # PDF text extraction + page-aware chunking
│   ├── graph_builder.py  # Groq entity/relationship extraction + merge
│   ├── qa.py             # Hybrid retrieval + Q&A
│   ├── json_parser.py    # Robust LLM JSON parsing
│   ├── storage.py        # graph.json + chunks.json persistence
│   ├── data/             # Generated on upload (gitignored)
│   ├── .env.example
│   └── requirements.txt
├── frontend/
│   ├── app/page.tsx      # Upload, graph canvas, Q&A chat
│   ├── .env.local.example
│   └── package.json
└── README.md
```

## Quick start

### Prerequisites

- Python 3.11+
- Node.js 18+
- [Groq API key](https://console.groq.com/keys)

### 1. Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env → GROQ_API_KEY=gsk_...

uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Verify: [http://localhost:8000/health](http://localhost:8000/health) should return `"status": "ok"`.

### 2. Frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local     # NEXT_PUBLIC_API_URL=http://localhost:8000

npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

### 3. Use the app

1. Click **Upload PDF** and select a document (processing may take a minute)
2. Explore the graph — click nodes or edges in the details panel
3. Ask a question in the chat box — expand **Graph sources** or **Text sources** to see citations

The graph auto-loads on page refresh if a previous upload exists.

## API reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check + Groq API key status |
| `POST` | `/upload` | Upload PDF → extract graph + chunks |
| `GET` | `/graph` | Return nodes and edges for the UI |
| `POST` | `/ask` | Hybrid graph + chunk Q&A |

### `POST /upload`

Multipart form: `file` (PDF).

**Response:**
```json
{
  "message": "PDF processed successfully",
  "chunks_processed": 12,
  "nodes": 71,
  "edges": 74,
  "elapsed_seconds": 45.2
}
```

### `POST /ask`

**Request:**
```json
{
  "question": "What did they study in college?",
  "history": [
    { "question": "Who is Vishal?", "answer": "Vishal Dalavayi is..." }
  ]
}
```

`history` is optional (max 8 turns). The frontend sends the last 6 for follow-ups.

**Response:**
```json
{
  "answer": "Vishal studied Information Science with a Data Science concentration at the University of Maryland.",
  "sources": {
    "graph_edges": [
      {
        "edge_id": "e_abc123",
        "source": "Vishal Dalavayi",
        "target": "Information Science",
        "relationship": "studied",
        "evidence": "B.S. Information Science Concentration in Data Science"
      }
    ],
    "chunks": [
      {
        "chunk_id": "chunk_003",
        "page": 1,
        "text": "University of Maryland, College Park..."
      }
    ]
  }
}
```

## Environment variables

| Variable | Where | Description |
|----------|-------|-------------|
| `GROQ_API_KEY` | `backend/.env` | Groq API key (required) |
| `NEXT_PUBLIC_API_URL` | `frontend/.env.local` | Backend URL (default `http://localhost:8000`) |

## Data model

**Entity types:** `PERSON`, `ORG`, `PRODUCT`, `TECH`, `CONCEPT`, `LOCATION`, `OTHER`

**Graph node:**
```json
{ "id": "vishal_dalavayi", "label": "Vishal Dalavayi", "type": "PERSON" }
```

**Graph edge:**
```json
{
  "id": "e_abc123",
  "source": "vishal_dalavayi",
  "target": "konfig_ai",
  "label": "worked_at",
  "evidence": "AI/ML Engineer — Konfig AI"
}
```

**Text chunk:**
```json
{ "id": "chunk_001", "text": "...", "page": 1 }
```

Each upload **overwrites** the previous graph and chunks (single-document mode).

## What this project does not include

- LangChain / LangGraph
- Vector database or embedding search
- Neo4j, Postgres, or other databases
- Multi-document persistence
- OCR for scanned/image-only PDFs

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `GROQ_API_KEY not set` | Copy `backend/.env.example` → `backend/.env`, add your key, restart backend |
| `Cannot reach the backend` | Start uvicorn on port 8000; check `NEXT_PUBLIC_API_URL` |
| CORS errors | Backend allows `localhost` and `127.0.0.1` on any port |
| Upload succeeds but graph is empty | Invalid or placeholder API key — check `/health` |
| No text extracted | PDF is likely scanned/image-only; use a text-based PDF |
| 404 on `/ask` | Re-upload — both `graph.json` and `chunks.json` must exist |
| Vague answers | Ask about specific entity names visible in the graph |
| Upload button does nothing | Hard-refresh the page; ensure backend is running |

Backend logs show extraction and Q&A progress in the terminal.

## License

MIT
