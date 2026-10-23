"""Download the documented historical checkpoint with size/hash verification."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from urllib.parse import urlsplit
from urllib.request import urlopen
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


def checksum(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def download(manifest, destination, cache_dir, opener=urlopen):
    size = manifest["size_bytes"]
    expected = manifest["sha256"]
    if type(size) is not int or not 0 < size <= 100 * 1024 * 1024:
        raise ValueError("Invalid checkpoint size")
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("Invalid SHA-256")
    if urlsplit(manifest["url"]).scheme != "https":
        raise ValueError("HTTPS checkpoint URL required")
    destination = Path(destination)
    if destination.exists():
        if destination.stat().st_size == size and checksum(destination) == expected:
            return destination
        raise FileExistsError("Existing checkpoint differs; preserve it before installing another version")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix="checkpoint-", suffix=".download", dir=cache_dir)
    # Retain downloads (including failed ones) in the explicit cache for inspection.
    with os.fdopen(handle, "wb") as output, opener(manifest["url"], timeout=60) as response:
        received = 0
        while True:
            chunk = response.read(min(1024 * 1024, size + 1 - received))
            if not chunk:
                break
            received += len(chunk)
            output.write(chunk)
            if received > size:
                raise ValueError("Checkpoint exceeds manifest size")
    source = Path(name)
    if received != size or checksum(source) != expected:
        raise ValueError("Checkpoint size/SHA-256 mismatch; destination was not changed")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as output, source.open("rb") as input_stream:
        shutil.copyfileobj(input_stream, output)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "checkpoints/checkpoint-manifest.json")
    parser.add_argument("--destination", type=Path, default=ROOT / "checkpoints/emnist_model.pth")
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".codex/cache/tasks" / ("ocr-checkpoint-" + uuid4().hex))
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8") as source:
        manifest = json.load(source)
    result = download(manifest, args.destination, args.cache_dir)
    print("Verified checkpoint:", result)
