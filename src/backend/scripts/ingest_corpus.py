import argparse
import json
import sys
from pathlib import Path
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

from vectordb.core import ChromaDB  # noqa: E402

# Ensure backend package is on path when run from repo root or backend/
_SCRIPT_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_BACKEND_DIR))


CHUNK_SIZE = 1024
CHUNK_OVERLAP = int(0.2 * CHUNK_SIZE)
DEFAULT_BATCH_SIZE = 500


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split text into overlapping chunks using LangChain's recursive splitter with tiktoken."""
    if not text or not text.strip():
        return []
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        model_name="gpt-4",
    )
    documents = text_splitter.create_documents([text])
    return [doc.page_content for doc in documents]


def _discover_jsonl(corpus_dir: Path, recursive: bool = False) -> List[Path]:
    """Find all .jsonl files in the given directory."""
    if recursive:
        return sorted(p for p in corpus_dir.rglob("*.jsonl") if p.is_file())
    return sorted(p for p in corpus_dir.iterdir() if p.is_file() and p.suffix.lower() == ".jsonl")


def _count_lines(jsonl_files: List[Path]) -> int:
    """Count total lines across all JSONL files."""
    total = 0
    for file_path in jsonl_files:
        with file_path.open("r", encoding="utf-8") as f:
            for _ in f:
                total += 1
    return total


def _progress_bar(percentage: float, width: int = 30) -> str:
    filled = int(width * percentage / 100)
    return f"[{'#' * filled}{'-' * (width - filled)}] {percentage:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-ingest JSONL corpus files into the Chroma vector DB."
    )
    parser.add_argument("corpus_dir", help="Directory containing .jsonl corpus files.")
    parser.add_argument(
        "--collection",
        default="file_corpus",
        help="Chroma collection name (default: file_corpus)",
    )
    parser.add_argument(
        "--persist-path",
        default="../../.langchain_chroma/",
        help="Path to Chroma persistence directory (default: ../../.langchain_chroma/)",
    )
    parser.add_argument(
        "--clear-collection",
        action="store_true",
        help="Delete the collection before ingesting (WARNING: irreversible)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search subdirectories for .jsonl files",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help=f"Max tokens per chunk (default: {CHUNK_SIZE})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=CHUNK_OVERLAP,
        help=f"Tokens to overlap between chunks (default: {CHUNK_OVERLAP})",
    )
    parser.add_argument(
        "--vector-batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of chunks per embedding/DB batch (default: {DEFAULT_BATCH_SIZE})",
    )
    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir).resolve()
    if not corpus_dir.exists() or not corpus_dir.is_dir():
        print(f"Error: '{corpus_dir}' does not exist or is not a directory.", file=sys.stderr)
        sys.exit(1)

    # Init ChromaDB
    db = ChromaDB(collection_name=args.collection, persist_path=args.persist_path)

    if args.clear_collection:
        try:
            db.client.delete_collection(name=args.collection)
            print(f"Cleared collection '{args.collection}'.")
            db.get_or_create_collection(args.collection)
        except Exception as exc:
            print(f"Warning: could not clear collection: {exc}", file=sys.stderr)

    jsonl_files = _discover_jsonl(corpus_dir, recursive=args.recursive)
    if not jsonl_files:
        print("No .jsonl files found.")
        sys.exit(0)

    # Count total lines for progress tracking
    print("🔍 Counting entries...")
    total_lines = _count_lines(jsonl_files)
    print(f"   Found {total_lines} entries to process.\n")

    total_docs = 0
    total_chunks = 0
    processed_lines = 0

    batch_docs: List[Document] = []
    batch_ids: List[str] = []

    for file_path in jsonl_files:
        with file_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                processed_lines += 1

                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"   ⚠️  Skipping line {line_num} in {file_path.name}: {exc}")
                    continue

                doc_id = str(entry.get("id", f"{file_path.stem}_L{line_num}"))
                title = entry.get("title", "")
                url = entry.get("url", "")
                text = entry.get("text", "")

                if not text or not text.strip():
                    continue

                chunks = _chunk_text(text, chunk_size=args.chunk_size, overlap=args.chunk_overlap)
                if not chunks:
                    continue

                for i, chunk in enumerate(chunks):
                    metadata = {
                        "corpus_id": doc_id,
                        "title": title,
                        "url": url,
                        "chunk_num": i,
                        "total_chunk": len(chunks),
                    }
                    batch_docs.append(Document(page_content=chunk, metadata=metadata))
                    batch_ids.append(f"{doc_id}_{i}")

                total_docs += 1

                # Flush when we hit the batch threshold
                if len(batch_docs) >= args.vector_batch_size:
                    db.add_documents_bulk(batch_docs, ids=batch_ids)
                    total_chunks += len(batch_docs)
                    batch_docs.clear()
                    batch_ids.clear()

                # Update progress display every ~1%
                if total_lines > 0:
                    progress = (processed_lines / total_lines) * 100
                    update_interval = max(1, total_lines // 100)
                    if processed_lines % update_interval == 0:
                        bar = _progress_bar(min(100, progress))
                        sys.stdout.write(
                            f"\r{bar} | Lines: {processed_lines}/{total_lines} | "
                            f"Docs: {total_docs} | Chunks: {total_chunks}       "
                        )
                        sys.stdout.flush()

    # Flush any remaining documents
    if batch_docs:
        db.add_documents_bulk(batch_docs, ids=batch_ids)
        total_chunks += len(batch_docs)
        batch_docs.clear()
        batch_ids.clear()

    # Rebuild BM25 once at the end
    db.rebuild_bm25()

    # Final progress line
    if total_lines > 0:
        final_bar = _progress_bar(100)
        sys.stdout.write(
            f"\r{final_bar} | Lines: {processed_lines}/{total_lines} | "
            f"Docs: {total_docs} | Chunks: {total_chunks}\n"
        )
        sys.stdout.flush()

    print(f"\n🏁 Done. Ingested {total_docs} docs ({total_chunks} chunks) into collection '{args.collection}'.")


if __name__ == "__main__":
    main()
