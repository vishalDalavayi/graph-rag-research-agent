import logging
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from extractor import chunk_pages, extract_pages_from_pdf
from graph_builder import build_graph_from_chunks
from qa import answer_question
from storage import load_chunks, load_graph, save_chunks, save_graph

# Always load backend/.env (override shell env so old placeholders don't win)
load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

class HistoryTurn(BaseModel):
    question: str
    answer: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    history: list[HistoryTurn] = Field(default_factory=list, max_length=8)


class SourceItem(BaseModel):
    edge_id: str
    source: str
    target: str
    relationship: str
    evidence: str


class ChunkSourceItem(BaseModel):
    chunk_id: str
    page: int | None
    text: str


class SourcesResponse(BaseModel):
    graph_edges: list[SourceItem]
    chunks: list[ChunkSourceItem]


class AskResponse(BaseModel):
    answer: str
    sources: SourcesResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Graph RAG Research Agent", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    logger.debug("Health check")
    key_ok = True
    key_message = None
    try:
        from graph_builder import validate_groq_api_key

        validate_groq_api_key()
    except ValueError as exc:
        key_ok = False
        key_message = str(exc)

    return {
        "status": "ok",
        "llm": "groq",
        "model": "llama-3.3-70b-versatile",
        "groq_api_key_configured": key_ok,
        "groq_api_key_message": key_message,
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    logger.info("Upload received: %s", file.filename)

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    logger.info("PDF size: %d bytes", len(content))
    t0 = time.perf_counter()

    try:
        pages = extract_pages_from_pdf(content)
    except Exception as exc:
        logger.exception("PDF extraction failed")
        raise HTTPException(status_code=400, detail=f"Failed to read PDF: {exc}") from exc

    if not pages:
        raise HTTPException(status_code=400, detail="No text could be extracted from PDF")

    total_chars = sum(len(p["text"]) for p in pages)
    logger.info("Extracted %d characters of text from %d pages", total_chars, len(pages))

    chunks = chunk_pages(pages)
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks produced from PDF text")

    logger.info("Created %d chunks (avg %d chars)", len(chunks), total_chars // max(len(chunks), 1))

    try:
        graph = build_graph_from_chunks([c["text"] for c in chunks])
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Graph extraction failed")
        raise HTTPException(status_code=500, detail=f"Graph extraction failed: {exc}") from exc

    save_graph(graph)
    save_chunks(chunks)
    elapsed = time.perf_counter() - t0

    logger.info(
        "Upload complete in %.1fs — %d nodes, %d edges",
        elapsed,
        len(graph["nodes"]),
        len(graph["edges"]),
    )

    return {
        "message": "PDF processed successfully",
        "chunks_processed": len(chunks),
        "nodes": len(graph["nodes"]),
        "edges": len(graph["edges"]),
        "elapsed_seconds": round(elapsed, 2),
    }


@app.get("/graph")
def get_graph():
    graph = load_graph()
    if graph is None:
        logger.warning("Graph requested but none exists")
        raise HTTPException(status_code=404, detail="No graph found. Upload a PDF first.")

    logger.info("Serving graph: %d nodes, %d edges", len(graph.get("nodes", [])), len(graph.get("edges", [])))
    return graph


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest):
    logger.info("Ask: %s", body.question[:120])

    graph = load_graph()
    if graph is None:
        raise HTTPException(status_code=404, detail="No graph found. Upload a PDF first.")

    chunks = load_chunks()
    if chunks is None:
        raise HTTPException(status_code=404, detail="No chunks found. Upload a PDF first.")

    try:
        history = [{"question": t.question, "answer": t.answer} for t in body.history]
        result = answer_question(graph, body.question, chunks=chunks, history=history)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Q&A failed")
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {exc}") from exc

    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
