"""Main page — block list with top bar, bottom bar, and dialogs."""

from __future__ import annotations

import reflex as rx

from ghoshell_text_blocks.components.block_card import block_card
from ghoshell_text_blocks.components.dialogs import (
    new_block_dialog,
    edit_block_dialog,
    view_block_dialog,
)
from ghoshell_text_blocks.state import TextBlocksState


def top_bar() -> rx.Component:
    return rx.hstack(
        rx.heading("text-blocks", size="5", color="#e2e8f0"),
        rx.spacer(),
        rx.button(
            "+ New Block",
            variant="solid",
            color_scheme="green",
            on_click=TextBlocksState.open_new_block_dialog,
        ),
        padding="12px 16px",
        border_bottom="1px solid #2d3748",
        background="#171923",
    )


def bottom_bar() -> rx.Component:
    return rx.hstack(
        rx.text(TextBlocksState.summary, font_size="12px", color="#718096"),
        rx.spacer(),
        rx.cond(
            TextBlocksState.pending_diff_count > 0,
            rx.text(
                f"Submit {TextBlocksState.pending_diff_count} edit(s)",
                font_size="12px", color="#d69e2e",
            ),
        ),
        padding="8px 16px",
        border_top="1px solid #2d3748",
        background="#171923",
    )


def dev_bar() -> rx.Component:
    """S1 dev-only: simulate model actions."""
    return rx.hstack(
        rx.text("dev:", font_size="11px", color="#718096"),
        rx.button(
            "sim: model create", size="1", variant="soft",
            on_click=TextBlocksState.sim_model_create,
        ),
        rx.text("seal id:", font_size="11px", color="#718096"),
        rx.input(
            value="",
            placeholder="block id",
            size="1", width="80px",
            # dummy — just show the concept
        ),
        rx.button(
            "seal", size="1", variant="soft",
            on_click=TextBlocksState.sim_model_seal(
                TextBlocksState.dialog_block_id,
            ),
        ),
        padding="4px 16px",
        border_top="1px solid #2d3748",
        background="#1a202c",
        display="none",  # hidden by default, toggle via reflex state for dev
    )


def index() -> rx.Component:
    return rx.box(
        top_bar(),
        # block list
        rx.box(
            rx.foreach(
                TextBlocksState.sorted_blocks,
                block_card,
            ),
            padding="16px",
            min_height="calc(100vh - 140px)",
            overflow_y="auto",
        ),
        # dialogs
        new_block_dialog(),
        edit_block_dialog(),
        view_block_dialog(),
        # dev bar
        dev_bar(),
        # bottom bar
        bottom_bar(),
        background="#0d1117",
        min_height="100vh",
    )
