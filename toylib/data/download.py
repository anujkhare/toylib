"""Download a file once into the toylib cache, then reuse it."""

import os
import pathlib
import shutil
import tempfile
import urllib.request


def cache_dir(name: str) -> pathlib.Path:
    """Return ``$TOYLIB_CACHE/<name>`` (default ``~/.cache/toylib/<name>``), creating it."""
    root = os.environ.get("TOYLIB_CACHE", pathlib.Path.home() / ".cache" / "toylib")
    path = pathlib.Path(root) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def fetch(url: str, dest: pathlib.Path) -> pathlib.Path:
    """Download ``url`` to ``dest`` unless it is already there.

    Writes to a temporary file first and renames it, so an interrupted download
    never leaves a truncated file that later runs would trust.
    """
    if dest.exists():
        return dest
    print(f"Downloading {url} -> {dest}")
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
        with urllib.request.urlopen(url) as response:
            shutil.copyfileobj(response, tmp)
    os.replace(tmp.name, dest)
    return dest
