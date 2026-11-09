"""The dsh page as a surface — dsh's own client API, driven from the ghost.

The page is not a picture. A dsh web page is a cordis context running in the browser, and the
services in it — ``layout``, ``sidebarRight``, ``slots``, ``sessions``, ``shortcuts``, ``remote`` —
are the real controls for what it shows and what has focus. So this channel does not paint CSS and
does not scrape the DOM: it runs a piece of code in that context and answers with the JSON the code
returns.

Two consequences worth stating on the surface rather than discovering:

- **``ctx.get(name)`` reaches a service without an ``inject`` declaration.** dsh's own client
  reflection is permissive by design, so "declare what you use" is a convention here, not a fence.
- **The result must be JSON.** Every service hands back live objects; what crosses the bridge is
  whatever ``JSON.stringify`` can carry, and nothing else.

``describe`` exists because the alternative is a hard-coded catalogue that goes stale the moment
the engine changes. It reads the live object, so the surface explains itself from the page.
"""

from __future__ import annotations

import json
from typing import Any

from ghoshell_moss.core.blueprint.channel_builder import MutableChannel, new_channel

__all__ = ["new_page_channel"]

CHANNEL_NAME = "ui"

_INSTRUCTION = """\
The dsh page, from the inside. Code runs in the page's own cordis context, so `ctx.get(name)`
returns a live service — `layout`, `sidebarRight`, `slots`, `sessions`, `shortcuts`, `remote` — and
calling its methods is how the page is driven. That is dsh's own API; the DOM is not the handle.

`describe(name)` reads a service's methods before you call one. `exec` runs the code and answers
with the JSON value it returned, or the exception. Anything the value cannot stringify does not
come back.
"""


def new_page_channel(connection: Any) -> MutableChannel:
    """Build the page channel for one dsh connection."""
    chan = new_channel(name=CHANNEL_NAME, description="the dsh web page — dsh's own client API.")

    @chan.build.instruction
    def _instruction() -> str:
        return _INSTRUCTION

    # The last known attachment state: refreshed by the meta hook, read by the notice. One box,
    # because a notice that asks the bridge per render would pay a round trip for a warm line.
    attached = {"value": False}

    @chan.build.notice
    def _notice() -> str:
        return "page attached" if attached["value"] else "no page attached"

    @chan.build.refresh_meta
    async def _refresh_meta() -> None:
        try:
            attached["value"] = await connection.page_attached()
        except Exception:  # noqa: BLE001 — an unreachable page is a fact, not a notice failure
            attached["value"] = False

    @chan.build.command(name="exec", blocking=False, always_observe=True)
    async def _exec(text__: str, timeout: float = 5.0) -> str:
        """Run code in the page and return what it returns, as JSON.

        :param text__: the body of one async function receiving `ctx`, e.g.
            ``return ctx.get('layout').panelInfo``
        :param timeout: seconds before the page gives up on the call.
        :return: the JSON value, or `failed: <reason>`.
        """
        try:
            value = await connection.page_exec(text__, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 — a failed eval is an answer, not a crash
            return f"failed: {exc}"
        return json.dumps(value, ensure_ascii=False, indent=2)

    @chan.build.command(name="describe", blocking=False, always_observe=True)
    async def _describe(name: str) -> str:
        """Read one client service's own shape — the self-explanation that cannot go stale.

        :param name: a service on the page's context, e.g. `layout`.
        :return: JSON with the service's method names, or `failed: <reason>`.
        """
        literal = json.dumps(name)
        code = (
            f"const service = ctx.get({literal})\n"
            "const proto = Object.getPrototypeOf(service) || {}\n"
            "return {\n"
            f"  service: {literal},\n"
            "  methods: Object.getOwnPropertyNames(proto).filter((k) => k !== 'constructor'),\n"
            "  fields: Object.getOwnPropertyNames(service),\n"
            "}\n"
        )
        try:
            value = await connection.page_exec(code)
        except Exception as exc:  # noqa: BLE001
            return f"failed: {exc}"
        return json.dumps(value, ensure_ascii=False, indent=2)

    @chan.build.command(name="attached", blocking=False, always_observe=True)
    async def _attached_cmd() -> str:
        """Whether a page is polling right now — the bridge's only liveness signal."""
        try:
            return "attached" if await connection.page_attached() else "not attached"
        except Exception as exc:  # noqa: BLE001
            return f"failed: {exc}"

    return chan
