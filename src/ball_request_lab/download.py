"""Download the pinned public inputs and verify them by size and SHA-256."""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

CHUNK_BYTES = 1 << 20
RETRIES = 5


def sha256_of(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def is_verified(path: Path, entry: dict[str, Any]) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == entry["size_bytes"]
        and sha256_of(path) == entry["sha256"]
    )


def _fetch(entry: dict[str, Any], partial: Path) -> None:
    """Fetch into ``partial``, resuming from its current size when possible."""
    total = int(entry["size_bytes"])
    for attempt in range(1, RETRIES + 1):
        have = partial.stat().st_size if partial.exists() else 0
        if have > total:
            partial.unlink()
            have = 0
        if have == total:
            return
        request = urllib.request.Request(entry["url"])
        if have:
            request.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if have and response.status != 206:
                    partial.unlink()
                    have = 0
                with partial.open("ab" if have else "wb") as handle:
                    while chunk := response.read(CHUNK_BYTES):
                        handle.write(chunk)
                        have += len(chunk)
                        print(
                            f"\r  {have / 1e6:8.1f} / {total / 1e6:.1f} MB",
                            end="",
                            file=sys.stderr,
                            flush=True,
                        )
            print(file=sys.stderr)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            print(f"\n  attempt {attempt}/{RETRIES} failed: {error}", file=sys.stderr)
            time.sleep(2 * attempt)
    raise RuntimeError(f"Could not download {entry['url']}")


def download(
    sources: Path, data_dir: Path, matches: list[int] | None = None
) -> list[dict[str, Any]]:
    """Download every pinned file for ``matches`` (default: all) into ``data_dir``."""
    manifest = json.loads(sources.read_text())
    wanted = set(matches or manifest["matches"])
    report: list[dict[str, Any]] = []
    for entry in manifest["files"]:
        name = Path(entry["path"]).name
        match_id = int(name.split("_")[0].split(".")[0])
        if match_id not in wanted:
            continue
        target = data_dir / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if is_verified(target, entry):
            print(f"ok       {entry['path']}", file=sys.stderr)
            report.append({"path": entry["path"], "status": "already_verified"})
            continue
        print(f"download {entry['path']}", file=sys.stderr)
        partial = target.with_suffix(target.suffix + ".part")
        _fetch(entry, partial)
        if partial.stat().st_size != entry["size_bytes"] or (sha256_of(partial) != entry["sha256"]):
            partial.unlink()
            raise RuntimeError(f"Checksum mismatch for {entry['path']}; file removed")
        partial.replace(target)
        report.append({"path": entry["path"], "status": "downloaded_and_verified"})
    return report
