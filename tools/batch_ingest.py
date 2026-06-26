#!/usr/bin/env python3
"""Batch ingest: read all files in a directory, chunk, enrich with NER,
upsert into the vector store. Designed to run outside FastAPI.

Usage::

    cd src/app
    uv run python ../../tools/batch_ingest.py path/to/docs/ --language en
    uv run python ../../tools/batch_ingest.py file.pdf --language id --no-ner
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure src/app is on sys.path so `app.*` imports resolve
_THIS_FILE = Path(__file__).resolve()
_TOOLS_DIR = _THIS_FILE.parent
_SRC_DIR = _TOOLS_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from tqdm import tqdm

from app.config import settings
from app.core.pipeline.chunker import chunk_text
from app.core.pipeline.document_loader import load_document_bytes
from app.core.orchestration.metadata_enrichment import enrich_chunks, extract_base_metadata
from app.core.orchestration.ner_service import enrich_chunks_batch
from app.core.retrieval.vector_store import VectorStore

logging.basicConfig(
    level=logging.WARNING,  # suppress INFO noise behind tqdm bars
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("batch_ingest")

_ALLOWED_SUFFIXES = frozenset({".txt", ".md", ".pdf"})


def read_all_files(root: Path) -> list[tuple[str, bytes]]:
    """Recursively find and read all allowed documents under *root*."""
    results: list[tuple[str, bytes]] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in _ALLOWED_SUFFIXES:
            continue
        raw = path.read_bytes()
        results.append((str(path), raw))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch ingest with NER enrichment")
    parser.add_argument(
        "path", type=Path, help="File or directory to ingest"
    )
    parser.add_argument(
        "--language", default="en", choices=["en", "id"],
        help="NER language for all chunks (default: en)",
    )
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--chunk-overlap", type=float, default=0.2)
    parser.add_argument(
        "--no-ner", action="store_true", help="Skip NER enrichment"
    )
    args = parser.parse_args()

    # Resolve input files
    files: list[tuple[str, bytes]] = []
    if args.path.is_file():
        raw = args.path.read_bytes()
        files.append((str(args.path), raw))
    elif args.path.is_dir():
        files = read_all_files(args.path)
    else:
        logger.error("Path does not exist: %s", args.path)
        sys.exit(1)

    if not files:
        logger.warning("No supported files found (.txt, .md, .pdf)")
        sys.exit(0)

    tqdm.write(f"Found {len(files)} file(s) to ingest\n")

    # Single VectorStore instance for batch — direct construction, no FastAPI singleton
    tqdm.write("Connecting to vector store...")
    vs = VectorStore(
        collection_name=settings.chroma_collection_name,
        persist_path=str(settings.local_data_path),
    )
    tqdm.write("")

    total_added = 0
    file_pbar = tqdm(files, desc="Ingesting files", unit="file", colour="green")
    for filename, raw_bytes in file_pbar:
        file_pbar.set_postfix_str(Path(filename).name)

        try:
            text = load_document_bytes(filename, raw_bytes)
        except Exception as exc:
            tqdm.write(f"[SKIP] {filename}: {exc}")
            continue

        chunks = chunk_text(
            text,
            filename=filename,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        if not chunks:
            tqdm.write(f"[SKIP] {filename}: zero chunks produced")
            continue

        # --- Heuristic enrichment (always on) ---
        uploaded_at = datetime.now(timezone.utc)
        base_meta = extract_base_metadata(filename=filename, uploaded_at=uploaded_at)
        chunks = enrich_chunks(chunks, base_meta)

        # --- NER enrichment (optional) ---
        if not args.no_ner:
            enrich_chunks_batch(chunks, language=args.language)

        ids = [chunk.metadata.get("content_hash", f"{filename}_{i}") for i, chunk in enumerate(chunks)]
        vs.add_documents_bulk(chunks, ids=ids)
        total_added += len(chunks)

    file_pbar.close()
    tqdm.write(f"\n{total_added} chunks staged — rebuilding BM25 index...")

    vs.rebuild_bm25()
    tqdm.write(f"Batch ingest complete — {total_added} total chunks, BM25 rebuilt")


if __name__ == "__main__":
    main()
