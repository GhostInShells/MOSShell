"""Reusable mock fixtures — a demo scene and a demo-item page.

These exist so a model or a human can populate the stage without standing up real
window services. They are fixtures, not the product: a real item is an iframe pointing
at a live URL (terminal, file_editor, an external page). Nothing in the store or the
surface knows these are mock — they are just ``(id, url)`` pairs.

The demo page carries a keep-alive probe (a timer + an input box) so the stage's
core invariant — no reparent, no reload — is visible to the eye: after a swap or a
group switch, the timer keeps counting and the input keeps its text.
"""

from __future__ import annotations

from urllib.parse import quote

__all__ = ["demo_page", "demo_items"]

_PAGE = """<!DOCTYPE html><html><body style="background:#0a0a0a;color:{color};font:13px ui-monospace;padding:12px">
<b>{label}</b> &nbsp; <span id="t">0</span>s &nbsp; <input placeholder="keep-alive probe" style="width:120px">
<script>let n=0;setInterval(()=>{{document.getElementById('t').textContent=(++n/10).toFixed(1)}},100)</script>
</body></html>"""


def demo_page(label: str, color: str) -> str:
    """A self-contained colored page for one demo item, as a data URL."""
    return "data:text/html;charset=utf-8," + quote(_PAGE.format(label=label, color=color))


def demo_items() -> list[dict]:
    """A pre-built scene mirroring the verified lab pool: three groups, several items.

    Handles are assigned by the store, not named here — the store's monotonic
    index never collides with a real node's item.
    """
    return [
        {"url": demo_page("term", "#5af"), "label": "terminal", "group": "code"},
        {"url": demo_page("edit", "#5fa"), "label": "editor", "group": "code"},
        {"url": demo_page("chat", "#fa5"), "label": "chat", "group": "code"},
        {"url": demo_page("dsh", "#f5a"), "label": "dashboard", "group": "media"},
        {"url": demo_page("cam1", "#5ff"), "label": "cam 1", "group": "media"},
        {"url": demo_page("cam2", "#a5f"), "label": "cam 2", "group": "media"},
        {"url": demo_page("spec", "#ffa"), "label": "spec", "group": "docs"},
        {"url": demo_page("note", "#faa"), "label": "note", "group": "docs"},
    ]
