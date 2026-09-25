"""Shared state between the WS server, the channel, and the audit page.

Single source of truth. The WS server mutates it from extension frames; the
channel reads it (notice / read); the audit page reads the log back out. Runs on
the MOSS event loop — single-threaded, so plain dicts are safe.

Identity (inherited from ghost-in-bilibili, generalized):

- ``session`` — one browser instance. The extension persists it in
  ``chrome.storage.local`` and re-sends it on every ``hello``; it survives
  service-worker restarts.
- ``tab`` — one window, stamped by the SW from ``sender.tab.id``. The content
  script never invents its own identity.
- ``label`` (``p1``/``p2``…) — node-assigned to ``(session, tab)``. The **only**
  persistent identity; never reused after a tab closes.
- ``url``/``title`` — mutable attributes on a page, not identity.

Authorization is one bit per page: ``perceived``. The icon on the page is the
human's switch — clicking it means "the ghost may read this page". Behaviors
beyond reading are **not** authorized here: they are confirmed on the page, per
invocation, and only the outcome travels back on the wire. That is the whole
point of "审批面 = 页面自身" — the node never grants, it mirrors and records.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

__all__ = ["AuditEntry", "Page", "PageModel"]


def _now() -> float:
    return round(time.time(), 3)


@dataclass
class AuditEntry:
    """One line in the audit trail. The audit page shows these verbatim; the
    human's own browsing never appears here — only what the ghost did, what the
    human authorized, and what the human deliberately pushed."""

    label: str
    kind: str  # perception | behavior | event | dialog
    detail: str
    ok: bool | None = None  # None = pending / not applicable
    created: float = field(default_factory=_now)
    image_b64: str | None = None  # screenshot 事件的降采样 JPEG base64,审计页据此渲染 <img>


@dataclass
class Page:
    label: str
    url: str = ""
    title: str = ""
    perceived: bool = False  # 图标:绿 = 人类授权 ghost 读这一页
    reachable: bool = False  # 所属 session 的 WS 是否在线


class PageModel:
    def __init__(self, *, audit_limit: int = 500) -> None:
        self._label_counter = 0
        self.pages: dict[tuple[str, int], Page] = {}  # (session, tab) -> Page
        self._by_label: dict[str, tuple[str, int]] = {}
        self._sessions: dict[str, set[int]] = {}
        self._audit: list[AuditEntry] = []
        self._audit_limit = audit_limit

    # -- labels ----------------------------------------------------------

    def _next_label(self) -> str:
        self._label_counter += 1
        return f"p{self._label_counter}"

    def ensure_page(self, session: str, tab: int) -> Page:
        key = (session, tab)
        if key not in self.pages:
            page = Page(label=self._next_label())
            self.pages[key] = page
            self._by_label[page.label] = key
            self._sessions.setdefault(session, set()).add(tab)
        return self.pages[key]

    # -- transport -------------------------------------------------------

    def on_hello(self, session: str) -> None:
        self._sessions.setdefault(session, set())
        for tab in self._sessions[session]:
            self.pages[(session, tab)].reachable = True

    def on_disconnect(self, session: str) -> None:
        for tab in self._sessions.get(session, ()):
            self.pages[(session, tab)].reachable = False

    # -- extension-driven mutations --------------------------------------

    def update_content(
        self, session: str, tab: int, title: str, url: str
    ) -> Page:
        """A tab opened or navigated. Same (session, tab) → same label; only the
        content attributes change."""
        page = self.ensure_page(session, tab)
        page.title = title
        page.url = url
        page.reachable = True
        return page

    def set_perceived(self, session: str, tab: int, on: bool) -> Page:
        page = self.ensure_page(session, tab)
        page.perceived = on
        self.log(page.label, "perception", "授权感知" if on else "撤销感知", ok=on)
        return page

    def close_tab(self, session: str, tab: int) -> None:
        key = (session, tab)
        page = self.pages.pop(key, None)
        if page is not None:
            self._by_label.pop(page.label, None)
            self.log(page.label, "perception", "页面关闭", ok=False)
        self._sessions.get(session, set()).discard(tab)

    # -- audit -----------------------------------------------------------

    def log(
        self,
        label: str,
        kind: str,
        detail: str,
        ok: bool | None = None,
        image_b64: str | None = None,
    ) -> None:
        self._audit.append(
            AuditEntry(label=label, kind=kind, detail=detail, ok=ok, image_b64=image_b64)
        )
        if len(self._audit) > self._audit_limit:
            del self._audit[: len(self._audit) - self._audit_limit]

    def audit(self, *, limit: int = 200) -> list[AuditEntry]:
        return self._audit[-limit:]

    # -- channel-side queries --------------------------------------------

    def label_of(self, session: str, tab: int) -> str | None:
        page = self.pages.get((session, tab))
        return page.label if page else None

    def page_by_label(self, label: str) -> Page | None:
        key = self._by_label.get(label)
        return self.pages[key] if key else None

    def location_of(self, label: str) -> tuple[str, int] | None:
        """(session, tab) a label maps to — for routing outbound commands."""
        return self._by_label.get(label)

    def all_pages(self) -> dict[str, Page]:
        """label -> Page, including pages the human has not authorized yet."""
        return {page.label: page for page in self.pages.values()}

    def perceived_pages(self) -> dict[str, Page]:
        """label -> Page, only pages the human granted perception on."""
        return {page.label: page for page in self.pages.values() if page.perceived}
