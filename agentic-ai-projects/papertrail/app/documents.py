"""Page-aware parsing and overlapping word chunks."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
import re
from pypdf import PdfReader

MAX_BYTES = 10 * 1024 * 1024
MAX_PAGES = 100
MAX_CHUNKS = 300


class DocumentError(ValueError):
    pass


def parse_document(filename, content):
    if not content or len(content) > MAX_BYTES:
        raise DocumentError("Use a nonempty file up to 10 MB.")
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        try:
            pdf = PdfReader(BytesIO(content), strict=False)
            if pdf.is_encrypted:
                raise DocumentError("Password-protected PDFs are not supported.")
            if len(pdf.pages) > MAX_PAGES:
                raise DocumentError("Use PDFs with at most 100 pages.")
            pages = [
                (i + 1, re.sub(r"\s+", " ", page.extract_text() or "").strip())
                for i, page in enumerate(pdf.pages)
            ]
        except DocumentError:
            raise
        except Exception as exc:
            raise DocumentError("The PDF could not be read.") from exc
    elif suffix in {".md", ".txt"}:
        try:
            pages = [(1, re.sub(r"\s+", " ", content.decode("utf-8-sig")).strip())]
        except UnicodeDecodeError as exc:
            raise DocumentError("Text files must use UTF-8 encoding.") from exc
    else:
        raise DocumentError("Supported formats: PDF, Markdown, and text.")
    pages = [(n, text) for n, text in pages if text]
    if not pages:
        raise DocumentError(
            "No readable text was found. Scanned PDFs require OCR before upload."
        )
    return {
        "name": Path(filename).name[:160],
        "sha256": sha256(content).hexdigest(),
        "pages": pages,
    }


def make_chunks(pages, size=180, overlap=35):
    if not 0 <= overlap < size:
        raise ValueError("Invalid chunk overlap.")
    chunks = []
    for page, text in pages:
        words = text.split()
        for start in range(0, len(words), size - overlap):
            chunks.append({"page": page, "text": " ".join(words[start : start + size])})
            if len(chunks) > MAX_CHUNKS:
                raise DocumentError("Document exceeds the 300-chunk limit.")
            if start + size >= len(words):
                break
    return chunks
