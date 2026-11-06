"""Lightweight JSONL log I/O.

``utils/log_analysis.py`` also provides ``open_log``, but importing that module
pulls in pandas (used by the viewer/DataFrame helpers). The pipeline tools only
need to stream JSONL, so they import this module instead and stay usable in a
minimal environment.
"""
from __future__ import annotations

import gzip
from pathlib import Path


def open_log(path):
    """Open a plain or gzipped JSONL log for reading."""
    path = Path(path)
    return gzip.open(path, "rt", encoding="utf-8", errors="replace") \
        if path.suffix == ".gz" else open(path, encoding="utf-8", errors="replace")
