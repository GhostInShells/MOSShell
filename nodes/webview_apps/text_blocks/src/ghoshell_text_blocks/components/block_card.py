"""Block card — renders a single text block in the list."""

from __future__ import annotations

import reflex as rx

from ghoshell_text_blocks.state import BlockData, TextBlocksState


def _status_color(status: str, lock: str) -> str:
    if status == "error":
        return "#e53e3e"       # red
    if lock == "g":
        return "#38a169"       # green (streaming = model writing)
    if lock == "u":
        return "#d69e2e"       # yellow (human editing)
    if status == "sealed":
        return "#718096"       # gray (done)
    return "#a0aec0"


def _status_style(status: str, lock: str) -> dict:
    base = {
        "width": "8px", "height": "8px", "border-radius": "50%",
        "display": "inline-block", "margin-right": "6px",
        "background-color": _status_color(status, lock),
    }
    if lock == "g":
        base["animation"] = "pulse 1.5s ease-in-out infinite"
    return base


def _source_badge(source: str) -> str:
    return "ghost" if source == "g" else "human"


def _action_button(block: BlockData) -> rx.Component:
    if block.lock == "g":
        return rx.button(
            "View", size="1", variant="soft",
            on_click=TextBlocksState.open_view_dialog(block.id),
        )
    return rx.button(
        "Edit", size="1", variant="solid",
        on_click=TextBlocksState.open_edit_dialog(block.id),
    )


def block_card(block: BlockData) -> rx.Component:
    color = _status_color(block.status, block.lock)
    return rx.box(
        rx.hstack(
            # status dot
            rx.box(style=_status_style(block.status, block.lock)),
            # header
            rx.text(
                f"#{block.id} · {_source_badge(block.source)}",
                font_size="12px", color="#a0aec0",
            ),
            rx.spacer(),
            rx.cond(
                block.lock != "",
                rx.text(f"lock={block.lock}", font_size="11px", color="#e2e8f0"),
            ),
            rx.cond(
                block.status == "streaming",
                rx.text("streaming", font_size="11px", color="#38a169"),
            ),
            rx.text(f"v{block.version_count}", font_size="11px", color="#718096"),
            _action_button(block),
            align="center",
            padding="8px 12px 4px",
        ),
        # title
        rx.text(
            block.title or "(untitled)",
            font_size="14px", font_weight="600",
            color="#e2e8f0", padding="0 12px",
        ),
        # content preview
        rx.box(
            rx.text(
                block.content[:200] + ("..." if len(block.content) > 200 else ""),
                font_size="13px", color="#a0aec0",
                white_space="pre-wrap",
            ),
            padding="4px 12px 10px",
            border_left=f"2px solid {color}",
            margin="4px 12px 12px",
            background="#1a202c",
            border_radius="0 4px 4px 0",
        ),
        border=f"1px solid {color}" if block.lock else "1px solid #2d3748",
        border_radius="8px",
        margin="8px 0",
        background="#171923",
    )
