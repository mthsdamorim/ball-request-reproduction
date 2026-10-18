"""Small shared helpers: atomic JSON writes and file digests."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any


def atomic_json(path: Path, value: Any) -> None:
    """Write JSON through a temporary file so a failed run leaves no partial output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".json", mode="w", delete=False
    ) as temporary:
        json.dump(value, temporary, indent=2, ensure_ascii=False, allow_nan=False)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()
