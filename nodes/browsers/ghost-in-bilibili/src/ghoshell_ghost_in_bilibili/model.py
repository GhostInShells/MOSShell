"""Shared state between the HTTP server and the model-facing channel.

Single source of truth, like ``ScreenModel`` in screen_manager: the HTTP handler
mutates it (extension events), the channel reads it (notice/named_notices/context)
and writes to it (dispatch commands). GIL-atomic dict ops are enough for the
prototype; a real lock lands when we hit actual races.
"""

from __future__ import annotations

import time
import uuid


class BridgeModel:
    def __init__(self, ghost_name: str = "moss") -> None:
        self.ghost_name = ghost_name
        self.pages: dict[str, dict] = {}        # label (p1/p2/...) -> {bvid,url,title,state}
        self.pending_js: dict[str, list] = {}   # page -> list of {id,page,body}
        self.recent: list[dict] = []
        self._next_id = 1

    # -- page registry -----------------------------------------------------

    def assign_label(self, bvid: str, url: str, title: str) -> str:
        """每个 bvid 一个稳定 label(p1/p2/...)。同一 bvid 复用,新 bvid 自增。"""
        for label, p in self.pages.items():
            if p["bvid"] == bvid:
                p.update(state="on", url=url, title=title)
                return label
        label = f"p{self._next_id}"
        self._next_id += 1
        self.pages[label] = {"bvid": bvid, "url": url, "title": title, "state": "on"}
        return label

    def apply_toggle(self, e: dict) -> None:
        if not e.get("bvid"):
            return
        if e.get("state") == "on":
            self.assign_label(e["bvid"], e.get("url", ""), e.get("title", ""))
        else:
            for p in self.pages.values():
                if p["bvid"] == e["bvid"]:
                    p["state"] = "off"

    def open_pages(self) -> dict[str, dict]:
        """label -> page, only state == 'on'."""
        return {l: p for l, p in self.pages.items() if p["state"] == "on"}

    # -- js command queue --------------------------------------------------

    def dispatch_js(self, page: str, body: str, jid: str | None = None) -> dict:
        jid = jid or uuid.uuid4().hex[:8]
        cmd = {"id": jid, "page": page, "body": body}
        self.pending_js.setdefault(page, []).append(cmd)
        return cmd

    def drain_js(self, page: str) -> list[dict]:
        return self.pending_js.pop(page, [])

    # -- event buffer ------------------------------------------------------

    def add_event(self, parsed: dict) -> None:
        parsed["t"] = round(time.time(), 1)
        self.recent.append(parsed)
        self.recent[:] = self.recent[-50:]
