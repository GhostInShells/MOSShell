"""Text Blocks — Reflex app."""

from __future__ import annotations

import reflex as rx

from ghoshell_text_blocks.pages.index import index

# pulse animation for streaming blocks
_PULSE_KEYFRAMES = """
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
"""


def _global_css() -> rx.Component:
    return rx.html(f"<style>{_PULSE_KEYFRAMES}</style>")


app = rx.App(
    head_components=[_global_css()],
)
app.add_page(index, route="/")
