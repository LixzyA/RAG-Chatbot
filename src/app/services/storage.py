"""File / blob storage service.

Thin filesystem wrapper — ready to swap for S3, Azure Blob, or GCS later.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class StorageService:
    """Local-filesystem storage adapter.

    Usage::

        storage = StorageService(base_dir="./uploads")
        await storage.save("doc.pdf", file_bytes)
    """

    def __init__(self, base_dir: str | Path = "./storage") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    async def save(self, filename: str, data: bytes) -> Path:
        """Persist *data* to disk and return the resolved path."""
        dest = (self.base_dir / filename).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        logger.info("Saved %s (%d bytes)", dest.name, len(data))
        return dest

    async def load(self, filename: str) -> bytes:
        """Read a previously saved file."""
        return (self.base_dir / filename).read_bytes()

    def exists(self, filename: str) -> bool:
        return (self.base_dir / filename).exists()

    async def delete(self, filename: str) -> None:
        path = self.base_dir / filename
        if path.exists():
            path.unlink()
            logger.info("Deleted %s", path.name)
