"""Encoding management for file operations.

MOSS vendor patch (see UPSTREAM.md):

Upstream depended on `charset_normalizer` + `cachetools.LRUCache` for
confidence-scored encoding detection.  MOSS avoids both to keep the vendor
subtree dependency-free: try utf-8 first (dominant modern default), fall back
to latin-1 (always succeeds for byte streams, may look garbled — acceptable
because binary files are already rejected by ``validate_file``).  Cache is a
plain dict keyed by ``(path_str, mtime)``, capped at 1000 entries with FIFO
eviction — good enough for the editor's access pattern (one file per command).
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Callable


class EncodingManager:
    """Manages file encodings across operations to ensure consistency."""

    DEFAULT_MAX_CACHE_SIZE = 1000

    def __init__(self, max_cache_size: int | None = None):
        self._max_cache_size = max_cache_size or self.DEFAULT_MAX_CACHE_SIZE
        self._encoding_cache: dict[str, tuple[str, float]] = {}
        self.default_encoding = 'utf-8'

    def detect_encoding(self, path: Path) -> str:
        if not path.exists():
            return self.default_encoding

        sample_size = min(os.path.getsize(path), 1024 * 1024)
        with open(path, 'rb') as f:
            raw = f.read(sample_size)

        try:
            raw.decode('utf-8')
            return 'utf-8'
        except UnicodeDecodeError:
            pass
        return 'latin-1'

    def get_encoding(self, path: Path) -> str:
        path_str = str(path)
        if not path.exists():
            return self.default_encoding

        current_mtime = os.path.getmtime(path)
        cached = self._encoding_cache.get(path_str)
        if cached is not None:
            enc, mtime = cached
            if mtime == current_mtime:
                return enc

        encoding = self.detect_encoding(path)

        if len(self._encoding_cache) >= self._max_cache_size:
            oldest = next(iter(self._encoding_cache))
            del self._encoding_cache[oldest]
        self._encoding_cache[path_str] = (encoding, current_mtime)
        return encoding


def with_encoding(method: Callable) -> Callable:
    """Decorator: auto-detect and pass an ``encoding`` kwarg to file ops."""

    @functools.wraps(method)
    def wrapper(self, path: Path, *args, **kwargs):
        if path.is_dir():
            return method(self, path, *args, **kwargs)

        if not path.exists():
            if 'encoding' not in kwargs:
                kwargs['encoding'] = self._encoding_manager.default_encoding
        else:
            if 'encoding' not in kwargs:
                kwargs['encoding'] = self._encoding_manager.get_encoding(path)

        return method(self, path, *args, **kwargs)

    return wrapper
