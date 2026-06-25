# Metadata Enrichment at Ingest

## Context

Today's ingestion pipeline (`POST /ingest` → `chunker.py` → `VectorStore.add_documents`) stamps only the four chunker-local fields into `Document.metadata`: `filename`, `total_chunk`, `chunk_num`, `chunk_size` (`src/app/core/pipeline/chunker.py:47-52`). That's enough to identify a chunk inside one document but blind to anything that would help with **filtering, dedup, page-aware UI, and provenance**.

Two signals are currently dropped on the floor:

- **PDF page number.** `pdfminer.high_level.extract_text` returns one string blob — `Document.page_number` (`src/app/models/documents.py:30`) is wired but never populated, so per-page retrieval filters are impossible.
- **Hash + provenance.** No `content_hash` means duplicate ingest is silent; no `uploaded_at` means no time-window retrieval; no `file_type` / `source` means callers can't filter by extension without parsing the filename.

The empty file `src/app/core/orchestration/metadata_enrichment.py` is the pre-allocated slot for this stage. Goal: populate each chunk's metadata with the heuristic set below, with **zero added LLM round-trips** and **no behavior change** at retrieval time beyond richer `Document.metadata` values flowing into traces, retrieval responses, and `rag_traces.context_passed_to_llm`.

## Decisions locked

| Question | Choice |
|---|---|
| Scope | Heuristic only — no LLM |
| Toggle | Always on, no env flag |
| PDF page numbers | Capture per-page via `pdfminer` iteration |

## Target metadata schema

Per-chunk, attached at ingest:

| Key | Type | Source | Always present? |
|---|---|---|---|
| `source` | `str` | filename basename (matches the `Document.source` property — `src/app/models/documents.py:34`) | yes |
| `file_name` | `str` | original filename incl. extension | yes |
| `file_type` | `str` | lowercased extension (`pdf`/`txt`/`md`) | yes |
| `uploaded_at` | `str` (ISO-8601 UTC) | ingest-time timestamp | yes |
| `char_count` | `int` | `len(chunk.page_content)` | yes |
| `content_hash` | `str` | first 16 hex chars of `sha256(chunk.page_content)` | yes |
| `page` | `int` | only when source is PDF and chunk came from a specific page | PDF only |

Existing chunker fields (`filename`, `total_chunk`, `chunk_num`, `chunk_size`) are kept as-is for compatibility.

## Architecture

```
core/pipeline/document_loader.py         + load_pdf_pages(filename, content) -> list[tuple[int, str]]
core/pipeline/chunker.py                * chunk_text: add page_number kw + stamp into metadata
core/orchestration/metadata_enrichment.py   *** NEW module ***
                                           - extract_base_metadata(filename, uploaded_at) -> dict
                                           - enrich_chunks(chunks, base_meta) -> list[Document]
                                           - compute_content_hash(text) -> str  (private)
api/routes/ingestion.py                  * branch on suffix: PDF -> per-page path, else flat path
                                           - call enrich_chunks before passing to VectorStore
```

Layer rule preserved throughout: `core/` stays free of FastAPI/SQLAlchemy imports, `core/orchestration/` is framework-agnostic.

## Critical files

### `src/app/core/orchestration/metadata_enrichment.py` — NEW

Module docstring: "Pure, framework-agnostic metadata derivation for ingested chunks. No LLM, no FastAPI, no DB. Safe to test in isolation."

```python
def extract_base_metadata(
    *,
    filename: str,
    uploaded_at: datetime,        # injected (not now() inside) for testability
) -> dict[str, Any]:
    """Derive the per-document metadata shared by every chunk."""
    base = os.path.basename(filename)
    ext = Path(filename).suffix.lower().lstrip(".") or "unknown"
    return {
        "source": base,
        "file_name": filename,
        "file_type": ext,
        "uploaded_at": uploaded_at.replace(microsecond=0).isoformat(),
    }


def compute_content_hash(text: str) -> str:
    """Stable 16-hex-char SHA-256 fingerprint over chunk content."""
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"


def enrich_chunks(
    chunks: list[Document],
    base_meta: dict[str, Any],
) -> list[Document]:
    """Return new Documents with base metadata + per-chunk char_count/content_hash merged in.

    Does not mutate input chunks. Existing metadata wins on key collision so
    chunker-set fields (filename/total_chunk/page) are preserved verbatim.
    """
    out: list[Document] = []
    for chunk in chunks:
        derived = {
            **base_meta,
            "char_count": len(chunk.page_content),
            "content_hash": compute_content_hash(chunk.page_content),
        }
        merged = {**derived, **dict(chunk.metadata or {})}
        out.append(Document(page_content=chunk.page_content, metadata=merged))
    return out
```

Notes:
- Uses `from langchain_core.documents import Document` to match `chunker.py` and `vector_store.py`.
- `uploaded_at` is injected, not called inside, so tests pin time.

### `src/app/core/pipeline/document_loader.py` — add `load_pdf_pages`

```python
def load_pdf_pages(filename: str, content: bytes) -> list[tuple[int, str]]:
    """Iterate pdfminer per page so each carries its page number.

    Returns ``[(page_number_1_based, page_text), ...]``. Empty pages are skipped.
    """
    if Path(filename).suffix.lower() != ".pdf":
        raise FileTypeNotSupportedException(f"load_pdf_pages expects .pdf, got {filename}")
    try:
        # form-feed heuristic: pdfminer uses \x0c between pages in flat extraction.
        # Fallback path is reliable for ASCII PDFs and zero-dep.
        full = pdf_extract_text(BytesIO(content))
        parts = full.split("\x0c")
        pages: list[tuple[int, str]] = []
        for idx, text in enumerate(parts, start=1):
            if text.strip():
                pages.append((idx, text))
        if not pages:
            # Pure-image or empty PDF — still anchor page=1 so downstream has the field.
            pages = [(1, full or "")]
        return pages
    except Exception as exc:
        raise PDFProcessingException(str(exc)) from exc
```

Decision: form-feed splitter over pdfminer iteration. Same error surface (`PDFProcessingException`), no new dependency surface, accurate enough for the ASCII-heavy PDFs a RAG bot typically sees.

### `src/app/core/pipeline/chunker.py` — add `page_number` kw

```python
def chunk_text(
    text: str,
    *,
    filename: str = "<unknown>",
    chunk_size: int = 1024,
    chunk_overlap: float = 0.2,
    page_number: int | None = None,        # NEW
) -> list[Document]:
    ...
    metadata = {
        "filename": filename,
        "total_chunk": total,
        "chunk_num": i,
        "chunk_size": chunk_size,
    }
    if page_number is not None:
        metadata["page"] = page_number       # NEW
    ...
```

Page routing lives in the route; chunker stays a pure splitter.

### `src/app/api/routes/ingestion.py` — branch on suffix

```python
from datetime import datetime, timezone
from pathlib import Path

from app.core.orchestration.metadata_enrichment import enrich_chunks, extract_base_metadata

@router.post("/", response_model=UploadFileResponse)
async def ingest_file(
    file: UploadFile,
    chunk_size: int = 1024,
    chunk_overlap: float = 0.2,
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    raw_bytes = await file.read()
    uploaded_at = datetime.now(timezone.utc)
    suffix = Path(file.filename).suffix.lower()
    chunks: list[Document] = []

    if suffix == ".pdf":
        try:
            pages = load_pdf_pages(file.filename, raw_bytes)
        except FileTypeNotSupportedException as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except PDFProcessingException as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        for page_num, page_text in pages:
            chunks.extend(
                chunk_text(
                    page_text,
                    filename=file.filename,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    page_number=page_num,
                )
            )
    else:
        try:
            text = load_document_bytes(file.filename, raw_bytes)
        except FileTypeNotSupportedException as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except PDFProcessingException as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        chunks = chunk_text(
            text,
            filename=file.filename,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    if not chunks:
        raise HTTPException(status_code=400, detail="Document produced zero chunks")

    base_meta = extract_base_metadata(
        filename=file.filename, uploaded_at=uploaded_at
    )
    chunks = enrich_chunks(chunks, base_meta)

    ids = [f"{file.filename}_{i}" for i in range(len(chunks))]
    vs = get_vector_store()
    vs.add_documents(chunks, ids=ids)

    logger.info("Ingested %d chunks from '%s'", len(chunks), file.filename)
    return UploadFileResponse(status=200, num_chunk=len(chunks))
```

Key invariants:
- `base_meta` has no `page` — `page` is set per-chunk by the chunker (PDF branch). The `enrich_chunks` merge keeps chunker-set `page` because existing metadata wins on key collision.
- One `uploaded_at` per upload (consistent across chunks from one file).
- Existing exception paths preserved.

## Reused utilities

| Symbol | Path | Why |
|---|---|---|
| `load_document_bytes` | `src/app/core/pipeline/document_loader.py:52` | Existing flat-text loader for `.txt`/`.md` (already used by route) |
| `chunk_text` | `src/app/core/pipeline/chunker.py:17` | Existing chunker; route already routes through it |
| `VectorStore.add_documents` | `src/app/core/retrieval/vector_store.py:84` | Existing sink; accepts arbitrary metadata dicts |
| `Document` (langchain_core) | `src/app/core/retrieval/vector_store.py` | Same flavor chunker uses; new module matches for consistency |
| `PDFProcessingException` / `FileTypeNotSupportedException` | `src/app/utils/exceptions.py` | Already raised by document_loader; route already translates them |
| `UploadFileResponse` | `src/app/models/responses.py` | Unchanged; rich metadata shows up at `GET /retrieve` |

## Verification

1. **Cold-import smoke**
   ```
   cd src/app && uv run python -c "from app.core.orchestration.metadata_enrichment import enrich_chunks, extract_base_metadata"
   ```

2. **Determinism check**
   ```python
   from datetime import datetime, timezone
   from app.core.orchestration.metadata_enrichment import extract_base_metadata, compute_content_hash
   meta = extract_base_metadata(filename="a.txt", uploaded_at=datetime(2026,1,1,tzinfo=timezone.utc))
   print(sorted(meta.items()))
   print(compute_content_hash("hello") == compute_content_hash("hello"))  # True
   ```

3. **End-to-end via running server**
   - `POST /ingest` a `.txt` → `GET /retrieve "{any line}"` shows `metadata.source`, `file_name`, `file_type="txt"`, `uploaded_at`, `char_count`, `content_hash`.
   - `POST /ingest` a 3-page `.pdf` → at least one retrieved chunk carries `metadata.page` in `{1,2,3}`; sum of page-bearing chunks >= 3 (proves per-page split).
   - Re-ingest the same `.txt` → identical `content_hash` per matching chunk (stable fingerprint).

4. **No-regression probe**
   - `rag_traces.context_passed_to_llm` JSON now includes the new fields. Existing columns still populated.
   - `GET /health` still returns `"status": "ok"`.

5. **(Optional)** Inline unit test in `tests/test_metadata_enrichment.py`:
   - `extract_base_metadata` strips path, lowercases extension, omits page when None.
   - `enrich_chunks([]) == []`.
   - Does not mutate input chunks.
   - PDF page metadata survives the merge.

## Out of scope (deliberate)

- LLM-generated `summary` / `keywords` — deferred; needs router-model call per chunk and a cache. Revisit if heuristic fields prove insufficient.
- Auto-dedup on `content_hash` — follow-up; today re-ingest produces duplicate IDs (unchanged).
- Wiring `metadata.page` into retrieval filters — that's a route-level addition; ship independently once metadata is reliable.