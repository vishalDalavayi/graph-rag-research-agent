import logging
from io import BytesIO

from pypdf import PdfReader

logger = logging.getLogger(__name__)

CHUNK_SIZE = 800


def extract_pages_from_pdf(file_bytes: bytes) -> list[dict]:
    """Extract text per page. Returns [{"page": 1, "text": "..."}, ...]."""
    reader = PdfReader(BytesIO(file_bytes))
    pages: list[dict] = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"page": i + 1, "text": text})
            logger.debug("Page %d: %d chars", i + 1, len(text))
    total_chars = sum(len(p["text"]) for p in pages)
    logger.info("PDF: %d pages with text, %d total chars", len(pages), total_chars)
    return pages


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Combined plain text from all pages (for backward compatibility)."""
    pages = extract_pages_from_pdf(file_bytes)
    return "\n\n".join(p["text"] for p in pages).strip()


def _split_text(text: str, chunk_size: int) -> list[str]:
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            boundary = text.rfind("\n\n", start, end)
            if boundary == -1:
                boundary = text.rfind(". ", start, end)
            if boundary != -1 and boundary > start + chunk_size // 2:
                end = boundary + (2 if text[boundary : boundary + 2] == ". " else 0)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end if end > start else start + chunk_size

    return chunks


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    chunks = _split_text(text, chunk_size)
    logger.info("Chunked text into %d chunks (target size ~%d)", len(chunks), chunk_size)
    return chunks


def chunk_pages(pages: list[dict], chunk_size: int = CHUNK_SIZE) -> list[dict]:
    """
    Chunk page text with stable ids and page numbers.
    Returns [{"id": "chunk_001", "text": "...", "page": 1}, ...].
    """
    result: list[dict] = []
    counter = 1

    for page_info in pages:
        page_num = page_info.get("page")
        for text in _split_text(str(page_info.get("text", "")), chunk_size):
            result.append(
                {
                    "id": f"chunk_{counter:03d}",
                    "text": text,
                    "page": page_num if isinstance(page_num, int) else None,
                }
            )
            counter += 1

    logger.info(
        "Chunked %d pages into %d chunks (target size ~%d)",
        len(pages),
        len(result),
        chunk_size,
    )
    return result
