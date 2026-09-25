"""Subtitle track cache + time-range query.

A track is a sorted list of ``{from, to, content}`` (seconds, seconds, text),
stored on disk as JSON keyed by bvid.

bvid is a **content cache key**, not a page identity: bilibili auto-plays, so the
same page (label) keeps switching bvid. The page reports its current bvid, and
queries go against that. A new bvid simply has no file yet — availability is
checked per current bvid, stale files stay as cache.

The model-facing surface is ``query(bvid, start, end)`` (lines overlapping an
interval) and ``window(bvid, t, before)`` (recent lines up to ``t``). Never a
whole-file read — a 30-minute track is ~400 lines and must not be dumped into
context.

Privacy is enforced here, not upstream: ``save`` keeps only ``from``/``to``/
``content`` and drops every other field, so uid / nickname / account / cookie
cannot reach disk through this store.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from pathlib import Path

_LINE_KEYS = ("from", "to", "content")


class SubtitleStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._cache: dict[str, list[dict]] = {}

    # -- persistence ------------------------------------------------------

    def path(self, bvid: str) -> Path:
        return self.root / f"{bvid}.json"

    def available(self, bvid: str) -> bool:
        return self.path(bvid).exists()

    def save(self, bvid: str, lines: list[dict]) -> None:
        """Persist a track, keeping only ``from``/``to``/``content`` per line."""
        clean = []
        for line in lines:
            if not isinstance(line, dict):
                continue
            entry = {k: line.get(k) for k in _LINE_KEYS}
            if entry["from"] is None or entry["content"] is None:
                continue
            clean.append(entry)
        clean.sort(key=lambda l: l["from"])
        self.root.mkdir(parents=True, exist_ok=True)
        self.path(bvid).write_text(
            json.dumps(clean, ensure_ascii=False), encoding="utf-8"
        )
        self._cache[bvid] = clean

    def load(self, bvid: str) -> list[dict]:
        if bvid in self._cache:
            return self._cache[bvid]
        try:
            data = json.loads(self.path(bvid).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        self._cache[bvid] = data
        return data

    # -- queries ----------------------------------------------------------

    def query(self, bvid: str, start: float, end: float) -> list[dict]:
        """Lines overlapping ``[start, end]`` (inclusive), sorted by ``from``.

        A line overlaps iff ``line.from <= end`` and ``line.to >= start``.
        """
        if start > end:
            start, end = end, start
        lines = self.load(bvid)
        if not lines:
            return []
        starts = [line["from"] for line in lines]
        head = bisect_right(starts, end)  # lines[:head] all have from <= end
        return [line for line in lines[:head] if line["to"] >= start]

    def at(self, bvid: str, t: float) -> dict | None:
        """The single line covering ``t``, half-open ``[from, to)`` — an exact
        boundary lands in the *following* line, not the one that just ended."""
        lines = self.load(bvid)
        if not lines:
            return None
        starts = [line["from"] for line in lines]
        head = bisect_right(starts, t)  # lines[:head] all have from <= t
        for line in reversed(lines[:head]):
            if t < line["to"]:
                return line
        return None

    def window(self, bvid: str, t: float, before: float = 15.0) -> list[dict]:
        """Recent lines up to ``t``: everything in ``[t - before, t]``, including
        the line currently covering ``t``. Used for the rolling context window."""
        return self.query(bvid, t - before, t)
