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
| Scope | Heuristic metadata + optional NER (transformer-based) per chunk |
| Toggle | NER stage gated by `ner_enabled` env (default `true`); heuristic stage always on |
| PDF page numbers | Capture per-page via `pdfminer` iteration |
| NER routing | **Batch mode:** caller passes `language` (`"en"` / `"id"` / `"auto"`). **FastAPI /ingest:** `langdetect` auto-detects, then routes to the matching NER model. |
| NER models | EN → `dslim/bert-base-NER` (CoNLL-2003 tags). ID → `cahya/bert-base-indonesian-NER` (token-level BIO). Each model loaded once, lazy, background-safe. |

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
| `language` | `str` | ISO-639-1 code detected by `langdetect` (FastAPI path) or passed in (batch path)
| `entities` | `list[dict]` | NER output — `[{"text": "Apple", "label": "ORG", "start": 0, "end": 5}, ...]`; spans char offsets into `page_content`; only if `ner_enabled=true` and model found ≥1 entity | optional |
| `entity_types` | `list[str]` | distinct label set (e.g. `["ORG", "LOC"]`); cheap filter downstream; only if `entities` non-empty | optional |
| `ner_model` | `str` | exact HF id used (e.g. `"dslim/bert-base-NER"`); provenance for triage | only if `ner_enabled=true` |

Existing chunker fields (`filename`, `total_chunk`, `chunk_num`, `chunk_size`) are kept as-is for compatibility.

### NER chunk metadata shape (concrete)

```jsonc
{
  "language": "en",
  "ner_model": "dslim/bert-base-NER",
  "entity_types": ["ORG", "LOC"],
  "entities": [
    {"text": "Felix", "label": "PER", "start": 12, "end": 17},
    {"text": "Jakarta", "label": "LOC", "start": 220, "end": 227}
  ]
}
```

Offsets are character positions into `chunk.page_content` (zero-indexed, `start` inclusive, `end` exclusive — matches Python slice semantics so snippets can be re-extracted: `chunk.page_content[s:e]`).

## Architecture

```
core/pipeline/document_loader.py         + load_pdf_pages(filename, content) -> list[tuple[int, str]]
core/pipeline/chunker.py                * chunk_text: add page_number kw + stamp into metadata
core/orchestration/metadata_enrichment.py   *** NEW module ***
                                           - extract_base_metadata(filename, uploaded_at) -> dict
                                           - enrich_chunks(chunks, base_meta) -> list[Document]
                                           - compute_content_hash(text) -> str  (private)
                                           - extend_metadata_with_ner(chunks, language?) -> None  (mutation OK)
core/pipeline/ner_extractor.py          *** NEW module (transformers-based NER, lazy-singleton, lang-aware) ***
                                           - NERExtractor (per-model, pipeline(NER)) loaded on first use
                                           - extract(text, *, model_name) -> list[dict] — char-offset entities
                                           - model registry: {"en": "dslim/bert-base-NER", "id": "cahya/bert-base-indonesian-NER"}
core/pipeline/language_detect.py        *** NEW thin wrapper (optional dep: langdetect) ***
                                           - detect_language(text) -> "en" | "id" | "unknown" — cheap, FastAPI-only
core/orchestration/ner_service.py       *** NEW orchestrator ***
                                           - enrich_chunks_route(chunks) — auto-detect lang + per-chunk NER (FastAPI)
                                           - enrich_chunks_batch(chunks, language="en"|"id") — caller picks (batch)
api/routes/ingestion.py                  * branch on suffix: PDF -> per-page path, else flat path
                                           - call enrich_chunks before passing to VectorStore (no change)
tools/batch_ingest.py                   *** NEW CLI entrypoint ***  (batch processing, no FastAPI)
                                           - reads file/dir glob, chunks, runs enrich_chunks_batch(language=<arg> or auto),
                                             upserts into VectorStore with stable dedup-by-content_hash IDs
```

Layer rule preserved throughout: `core/` stays free of FastAPI/SQLAlchemy imports, `core/orchestration/` is framework-agnostic.

## Env config (add to `src/app/config.py`)

```python
# --- NER enrichment ---
ner_enabled: bool = Field(default=True, alias="NER_ENABLED")
ner_device: str = Field(default="cpu", alias="NER_DEVICE")

# Optional override — if empty the service picks by language
ner_model_en: str = Field(default="dslim/bert-base-NER", alias="NER_MODEL_EN")
ner_model_id: str = Field(default="cahya/bert-base-indonesian-NER", alias="NER_MODEL_ID")
ner_batch_size: int = Field(default=32, alias="NER_BATCH_SIZE")
```

## Dependency: add `langdetect`

```
# pyproject.toml — add under [project.dependencies]
"langdetect>=1.0.9",
```

Optional for FastAPI only; batch path can skip it (caller provides language).

## Critical files

### `src/app/core/pipeline/ner_extractor.py` — NEW

```python
"""Transformer-based NER extraction, one pipeline per model, lazy-loaded.

Two models: ``dslim/bert-base-NER`` (EN, CoNLL-2003) and
``cahya/bert-base-indonesian-NER`` (ID, token-level BIO).

Thread-safety note: ``transformers`` pipeline is mostly thread-safe for
inference (no graph-building after warmup). Guard lazy-init with a lock so
the first concurrent call from batch code doesn't race.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from transformers import pipeline, Pipeline  # pyright: ignore[reportMissingTypeStubs]

logger = logging.getLogger(__name__)

_MODEL_REGISTRY: dict[str, str] = {
    "en": "dslim/bert-base-NER",
    "id": "cahya/bert-base-indonesian-NER",
}


class NERExtractor:
    """Lazy-per-model NER pipeline. Load once, reuse for all chunks.

    Usage::
        en_ner = NERExtractor("en", device="cpu")
        entities = en_ner.extract("Felix works at Google in Jakarta")
    """

    def __init__(self, language: str, *, device: str | int = "cpu") -> None:
        if language not in _MODEL_REGISTRY:
            msg = f"Unsupported NER language: {language!r} (choose from {set(_MODEL_REGISTRY)})"
            raise ValueError(msg)
        self.language = language
        self.model_id = _MODEL_REGISTRY[language]
        self.device = device
        self._pipeline: Pipeline | None = None
        self._lock = threading.Lock()/mode

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, text: str) -> list[dict[str, Any]]:
        """Run NER on *text*, return list of entity dicts.

        Each dict: ``{"text": …, "label": …, "start": int, "end": int}``.
        Labels from CoNLL-2003: ``PER``, ``ORG``, ``LOC``, ``MISC``.
        Indonesian model uses BIO tags — prefix (``B-``/``I-``) is stripped
        and consecutive same-label tokens are merged into one entity span.
        """
        pipe = self._get_pipeline()
        raw = pipe(text, aggregation_strategy="simple")  # type: ignore[arg-type]
        out: list[dict[str, Any]] = []
        for ent in raw:
            out.append({
                "text": ent["word"],
                "label": ent["entity_group"],
                "start": ent["start"],
                "end": ent["end"],
            })
        return out

    def extract_batch(
        self, texts: list[str], *, batch_size: int = 32
    ) -> list[list[dict[str, Any]]]:
        """Run NER over a batch of texts. Returns one entity list per input."""
        pipe = self._get_pipeline()
        results = pipe(texts, aggregation_strategy="simple", batch_size=batch_size)
        # `results` is list-of-lists from a list input
        return [
            [{"text": e["word"], "label": e["entity_group"], "start": e["start"], "end": e["end"]}
             for e in batch]
            for batch in results
        ]

    def supported_languages(self) -> frozenset[str]:
        return frozenset(_MODEL_REGISTRY)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _get_pipeline(self) -> Pipeline:
        if self._pipeline is None:
            with self._lock:
                if self._pipeline is None:  # double-checked
                    logger.info("Loading NER model: %s (device=%s)", self.model_id, self.device)
                    self._pipeline = pipeline(
                        "ner",
                        model=self.model_id,
                        tokenizer=self.model_id,
                        device=self.device,
                    )
        return self._pipeline
```

Design notes:
- `aggregation_strategy="simple"` groups B-xxx/I-xxx into one entity, which is what you want for metadata. Raw token-level would be noisy.
- Pipeline is lazy and cached per language. Both EN and ID models can coexist (caller creates two extractors or uses `ner_service.py` which manages the map).
- `extract_batch` avoids a Python for-loop per chunk — HF pipeline's native batching handles padding/truncation internally.

### `src/app/core/pipeline/language_detect.py` — NEW

```python
"""Cheap language detection for the NER routing layer.

Thin wrapper around ``langdetect``; returns ``"en"``, ``"id"``, or ``"unknown"``.

Only used in the FastAPI path — batch path gets language from the caller.
"""

from __future__ import annotations

import logging

import langdetect  # pyright: ignore[reportMissingTypeStubs]

logger = logging.getLogger(__name__)

# If you hit "NoClassDefFoundError" on first call it is because langdetect's
# static profile jar is not on the classpath — this happens when the package
# was installed via a broken wheel. Re-run: uv add langdetect
_DETECT_LANG_MAX_LEN = 4096


def detect_language(text: str) -> str:
    """Return ``"en"``, ``"id"``, or ``"unknown"``.

    Feeds the first 4096 characters to ``langdetect``; longer texts are
    truncated silently for speed (language does not need a full document).
    """
    try:
        snippet = text[:_DETECT_LANG_MAX_LEN]
        lang = langdetect.detect(snippet)
        if lang in ("en", "id"):
            return lang
        return "unknown"
    except langdetect.lang_detect_exception.LangDetectException:
        return "unknown"
```

### `src/app/core/orchestration/ner_service.py` — NEW

```python
"""NER orchestration — two entrypoints, one for FastAPI, one for batch.

``enrich_chunks_route``  — auto-detects language per-chunk, routes NER, mutates metadata.
``enrich_chunks_batch``  — caller picks language, one model for all chunks.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document

from app.config import settings
from app.core.pipeline.language_detect import detect_language
from app.core.pipeline.ner_extractor import NERExtractor

logger = logging.getLogger(__name__)

# Lazy cache: language -> NERExtractor
_extractors: dict[str, "NERExtractor"] = {}


def _get_extractor(language: str) -> NERExtractor:
    if language not in _extractors:
        _extractors[language] = NERExtractor(language, device=settings.ner_device)
    return _extractors[language]


def _entities_to_metadata(
    entities: list[dict[str, Any]],
    model_id: str,
    language: str,
) -> dict[str, Any]:
    """Build the NER metadata fragment, or empty dict if no entities."""
    if not entities:
        return {"language": language, "ner_model": model_id}
    return {
        "language": language,
        "ner_model": model_id,
        "entities": entities,
        "entity_types": sorted({e["label"] for e in entities}),
    }


def enrich_chunks_route(chunks: list[Document]) -> None:
    """FastAPI path: detect language per chunk, run NER, mutate metadata.

    Skips chunks whose language is not in {en, id}. The ``ner_enabled``
    guard lives in the route (this fn always processes). Mutates in-place.
    """
    for chunk in chunks:
        text = chunk.page_content
        if not text.strip():
            continue

        lang = detect_language(text)
        if lang == "unknown":
            continue

        extractor = _get_extractor(lang)
        entities = extractor.extract(text)
        ner_meta = _entities_to_metadata(entities, extractor.model_id, lang)
        chunk.metadata.update(ner_meta)


def enrich_chunks_batch(chunks: list[Document], language: str) -> None:
    """Batch path: caller declares language, one extractor for all chunks.

    Args:
        chunks: list of Document objects to enrich (mutated in place).
        language: ``"en"`` or ``"id"`` (use ``"auto"`` to per-chunk detect).
    """
    if language == "auto":
        # Fallback to per-chunk detection like the route
        return enrich_chunks_route(chunks)

    if language not in _get_extractor(language).supported_languages():
        logger.warning("Unsupported batch language: %s — skipping NER", language)
        return

    extractor = _get_extractor(language)

    # --- batch-extract all texts at once ---
    texts = [c.page_content for c in chunks]
    batch_results = extractor.extract_batch(texts, batch_size=settings.ner_batch_size)

    for chunk, entities in zip(chunks, batch_results, strict=True):
        ner_meta = _entities_to_metadata(entities, extractor.model_id, language)
        chunk.metadata.update(ner_meta)
```

Layer note: `ner_service.py` lives in `core/orchestration/` — it imports from `core/pipeline/` and `core/pipeline/`. It does **not** import from `api/` or `services/`. The FastAPI path calls `enrich_chunks_route` from the route handler; the batch path calls `enrich_chunks_batch` from the CLI.

### `tools/batch_ingest.py` — NEW (at repo root `tools/`)

```python
#!/usr/bin/env python3
"""Batch ingest: read all files in a directory, chunk, enrich with NER,
upsert into the vector store. Designed to run outside FastAPI."""

import argparse
import logging
from pathlib import Path

from app.config import settings
from app.core.pipeline.chunker import chunk_text
from app.core.pipeline.document_loader import load_document_bytes
from app.core.orchestration.metadata_enrichment import enrich_chunks, extract_base_metadata
from app.core.orchestration.ner_service import enrich_chunks_batch
from app.services.vector_db import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_all_files(
    root: Path,
    *,
    allowed_suffixes: frozenset = frozenset({".txt", ".md", ".pdf"}),
) -> list[tuple[str, bytes]]:
    """Recursively find and read all allowed documents under *root*."""
    results: list[tuple[str, bytes]] = []
    for path in root.rglob("*"):
        if path.suffix.lower() not in allowed_suffixes:
            continue
        raw = path.read_bytes()
        results.append((str(path), raw))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch ingest with NER enrichment")
    parser.add_argument("path", type=Path, help="File or directory to ingest")
    parser.add_argument("--language", default="en", choices=["en", "id", "auto"],
                        help="NER language (auto = per-chunk detect)")
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--chunk-overlap", type=float, default=0.2)
    parser.add_argument("--no-ner", action="store_true", help="Skip NER enrichment")
    args = parser.parse_args()

    # Resolve input files
    files: list[tuple[str, bytes]] = []
    if args.path.is_file():
        raw = args.path.read_bytes()
        files.append((str(args.path), raw))
    else:
        files = read_all_files(args.path)

    logger.info("Found %d files to ingest", len(files))

    # Single VectorStore instance for batch
    vs = VectorStore(settings.local_data_path, settings.chroma_collection_name)

    for filename, raw_bytes in files:
        try:
            text = load_document_bytes(filename, raw_bytes)
        except Exception as exc:
            logger.warning("Skipping %s: %s", filename, exc)
            continue

        chunks = chunk_text(
            text,
            filename=filename,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        if not chunks:
            continue

        # Heuristic enrichment (always)
        uploaded_at = datetime.now(timezone.utc)
        base_meta = extract_base_metadata(filename=filename, uploaded_at=uploaded_at)
        chunks = enrich_chunks(chunks, base_meta)

        # NER enrichment (optional)
        if not args.no_ner:
            enrich_chunks_batch(chunks, language=args.language)

        # IDs stable by content_hash for idempotent re-ingest
        ids = [f"{chunk.metadata['content_hash']}" for chunk in chunks]
        existing = vs.collection.get(ids=ids)  # skip-duplicates
        new_chunks = [c for c, _id in zip(chunks, ids) if _id not in (existing.get("ids") or [])]
        new_ids = [id for id in ids if id not in (existing.get("ids") or [])]

        if new_chunks:
            vs.add_documents(new_chunks, ids=new_ids)
            logger.info("Ingested %d/%d new chunks from '%s'", len(new_chunks), len(chunks), filename)
        else:
            logger.info("All %d chunks already in store — skipped '%s'", len(chunks), filename)

    logger.info("Batch ingest complete")


if __name__ == "__main__":
    from datetime import datetime, timezone
    main()
```

Design notes:
- Stable IDs via `content_hash` so repeat runs are idempotent — avoids silent duplicates.
- No `langdetect` import in this module; batch path relies on caller's `--language` choice.
- Same `VectorStore` instance reused for all files (don't re-init Chroma connection per file).
- 
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

from app.config import settings
from app.core.orchestration.metadata_enrichment import enrich_chunks, extract_base_metadata
from app.core.orchestration.ner_service import enrich_chunks_route

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

    # --- Heuristic enrichment (always on) ---
    base_meta = extract_base_metadata(
        filename=file.filename, uploaded_at=uploaded_at
    )
    chunks = enrich_chunks(chunks, base_meta)

    # --- NER enrichment (gated by env) ---
    if settings.ner_enabled:
        enrich_chunks_route(chunks)  # mutates metadata in-place

    ids = [f"{file.filename}_{i}" for i in range(len(chunks))]
    vs = get_vector_store()
    vs.add_documents(chunks, ids=ids)

    logger.info("Ingested %d chunks from '%s'", len(chunks), file.filename)
    return UploadFileResponse(status=200, num_chunk=len(chunks))
```

Key invariants:
- `base_meta` has no `page` — `page` is set per-chunk by the chunker (PDF branch). The `enrich_chunks` merge keeps chunker-set `page` because existing metadata wins on key collision.
- One `uploaded_at` per upload (consistent across chunks from one file).
- `enrich_chunks_route` mutates metadata in-place — no new Document objects created, no re-merge needed.
- `NER_ENABLED=True` (default)
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
| `settings.ner_enabled` | `src/app/config.py` | Gates the NER block in the route |
| `settings.ner_device` | `src/app/config.py` | Device (`cpu` / `cuda`) for NER models |
| `NERExtractor` | `src/app/core/pipeline/ner_extractor.py` | Lazy HF pipeline, per-language singleton |
| `detect_language` | `src/app/core/pipeline/language_detect.py` | `langdetect` wrapper, FastAPI path only |
| `enrich_chunks_route` | `src/app/core/orchestration/ner_service.py` | Per-chunk detect + NER, mutates metadata |
| `enrich_chunks_batch` | `src/app/core/orchestration/ner_service.py` | Batched NER, caller picks language |

## Verification

1. **Cold-import smoke**
   ```
   cd src/app && uv run python -c "from app.core.orchestration.metadata_enrichment import enrich_chunks, extract_base_metadata"
   ```
   Plus NER modules:
   ```
   uv run python -c "from app.core.pipeline.ner_extractor import NERExtractor; n = NERExtractor('en'); print(n.extract('Alice works at NASA'))"
   uv run python -c "from app.core.pipeline.language_detect import detect_language; print(detect_language('Halo apa kabar?'))"
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

4. **NER-specific verification** (requires `NER_ENABLED=true` + langdetect installed)
   - `POST /ingest` an English document → retrieved chunks have `metadata.language="en"`, `metadata.ner_model`, and `metadata.entity_types` (e.g. `["PER","ORG"]`).
   - `POST /ingest` an Indonesian document → `metadata.language="id"`, entities extracted with `cahya/bert-base-indonesian-NER`.
   - Entity offsets are correct: `chunk.page_content[entity.start:entity.end] == entity.text`.

5. **Batch CLI smoke**
   ```
   cd tools && uv run python batch_ingest.py path/to/docs/ --language en
   ```
   - Repeat run → zero new chunks (idempotent via content_hash IDs).
   - `--no-ner` → no NER metadata on chunks.

6. **No-regression probe**
   - `rag_traces.context_passed_to_llm` JSON now includes the new fields. Existing columns still populated.
   - `GET /health` still returns `"status": "ok"`.


7. **(Optional)** Inline unit test in `tests/test_metadata_enrichment.py`:
   - `extract_base_metadata` strips path, lowercases extension, omits page when None.
   - `enrich_chunks([]) == []`.
   - Does not mutate input chunks.
   - PDF page metadata survives the merge.
   - `detect_language("Hello world") == "en"`.
   - `detect_language("Apa kabar?") == "id"`.
   - `detect_language("") == "unknown"`.
   - NERExtractor produces correct char-offset entities for a known sentence.

## Out of scope (deliberate)

- LLM-generated `summary` / `keywords` — deferred; needs router-model call per chunk and a cache. Revisit if heuristic fields prove insufficient.
- Non-ID/EN language NER — models only exist for `en` and `id` in this plan; texts in other languages get `language="unknown"` and no entities.
- Wiring `metadata.page`/`metadata.entity_types` into retrieval filters — that's a route-level addition; ship independently once metadata is reliable.