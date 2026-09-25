"""Shared state between the WS server and the channel.

Single source of truth, like ``ScreenModel`` in screen_manager: the WS server
mutates it (extension frames), the channel reads it (notice / context) and writes
to it (dispatch commands). Runs on the MOSS event loop — single-threaded, so
plain dicts are safe.

Identity model (design doc §1) — the thing the probe got wrong:

- ``session``: one browser instance, a stable id the extension persists in
  ``chrome.storage.local`` and re-sends on every ``hello``. Survives SW restarts.
- ``tab``: one window, stamped by the SW from ``sender.tab.id`` — the content
  script never invents its own identity.
- ``label`` (``p1``/``p2``…): node-assigned to ``(session, tab)``. The **only
  persistent identity**. Never reused after a tab closes.
- ``bvid``/``title``/``url``: **mutable attributes** on a page, not identity —
  bilibili auto-plays, so they change under a stable label.

Authorization (design doc §4) is a (page × group) grid. The main ball grants
``presence`` (the ghost may perceive this page); a satellite grants one group
(``sense`` / ``control`` / ``subtitle`` / ``interact``). The node never grants —
it only mirrors what the extension reports. ``reachable`` is transport state:
whether the owning session's WS is currently up.
"""

from __future__ import annotations

from dataclasses import dataclass, field

GROUPS = ("sense", "control", "subtitle", "interact")


@dataclass
class Page:
    label: str
    bvid: str | None = None
    title: str = ""
    url: str = ""
    presence: bool = False  # 主球:绿 = 授权感知本页
    grants: dict[str, bool] = field(default_factory=lambda: {g: False for g in GROUPS})
    reachable: bool = False  # 所属 session 的 WS 是否在线
    # 最近一次实时状态(热数据)
    t: float = 0.0
    paused: bool = True
    rate: float = 1.0
    duration: float = 0.0


class BridgeModel:
    def __init__(self) -> None:
        self._label_counter = 0
        self.pages: dict[tuple[str, int], Page] = {}  # (session, tab) -> Page
        self._by_label: dict[str, tuple[str, int]] = {}  # label -> (session, tab)
        self._sessions: dict[str, set[int]] = {}  # session -> {tab ids}

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
        self, session: str, tab: int, bvid: str | None, title: str, url: str
    ) -> Page:
        """A tab opened, navigated, or auto-played into a new video. Same
        (session, tab) → same label; only the content attributes change."""
        page = self.ensure_page(session, tab)
        page.bvid = bvid
        page.title = title
        page.url = url
        page.reachable = True
        return page

    def set_presence(self, session: str, tab: int, on: bool) -> None:
        self.ensure_page(session, tab).presence = on

    def set_grant(self, session: str, tab: int, group: str, on: bool) -> None:
        page = self.ensure_page(session, tab)
        if group in GROUPS:
            page.grants[group] = on

    def update_state(
        self, session: str, tab: int, *, t: float, paused: bool, rate: float,
        duration: float,
    ) -> None:
        page = self.ensure_page(session, tab)
        page.t = t
        page.paused = paused
        page.rate = rate
        page.duration = duration
        page.reachable = True

    def close_tab(self, session: str, tab: int) -> None:
        key = (session, tab)
        page = self.pages.pop(key, None)
        if page is not None:
            self._by_label.pop(page.label, None)
        self._sessions.get(session, set()).discard(tab)

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

    def open_pages(self) -> dict[str, Page]:
        """label -> Page, only pages whose presence the human granted."""
        return {page.label: page for page in self.pages.values() if page.presence}

    def granted(self, label: str, group: str) -> bool:
        page = self.page_by_label(label)
        return page is not None and page.grants.get(group, False)
