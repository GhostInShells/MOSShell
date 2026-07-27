"""Dialogs — new block, edit block, view block."""

from __future__ import annotations

import reflex as rx

from ghoshell_text_blocks.state import TextBlocksState


def new_block_dialog() -> rx.Component:
    return rx.dialog.root(
        rx.dialog.content(
            rx.dialog.title("New Block"),
            rx.vstack(
                rx.text("Title", font_size="12px", color="#a0aec0"),
                rx.input(
                    placeholder="optional — defaults to first sentence",
                    value=TextBlocksState.dialog_title,
                    on_change=TextBlocksState.set_new_title,
                    width="100%",
                ),
                rx.text("Content", font_size="12px", color="#a0aec0"),
                rx.text_area(
                    placeholder="write here...",
                    value=TextBlocksState.dialog_content,
                    on_change=TextBlocksState.set_new_content,
                    width="100%",
                    min_height="200px",
                    font_family="monospace",
                ),
                rx.hstack(
                    rx.button(
                        "Cancel", variant="soft", color_scheme="gray",
                        on_click=TextBlocksState.cancel_new_block,
                    ),
                    rx.button(
                        "Create", variant="solid", color_scheme="green",
                        on_click=TextBlocksState.create_block,
                    ),
                    justify="end",
                    width="100%",
                ),
                spacing="2",
                align="start",
            ),
            max_width="640px",
        ),
        open=TextBlocksState.dialog_mode == "new",
    )


def edit_block_dialog() -> rx.Component:
    return rx.dialog.root(
        rx.dialog.content(
            rx.dialog.title(
                f"Editing #{TextBlocksState.dialog_block_id} · "
                f"{TextBlocksState.dialog_title}" if TextBlocksState.dialog_title
                else f"Editing #{TextBlocksState.dialog_block_id}",
            ),
            rx.vstack(
                rx.text_area(
                    value=TextBlocksState.dialog_content,
                    on_change=TextBlocksState.set_edit_content,
                    width="100%",
                    min_height="300px",
                    font_family="monospace",
                ),
                rx.text("Note (optional, sent with diff)", font_size="12px", color="#a0aec0"),
                rx.input(
                    placeholder="what did you change?",
                    value=TextBlocksState.dialog_human_note,
                    on_change=TextBlocksState.set_edit_human_note,
                    width="100%",
                ),
                rx.hstack(
                    rx.button(
                        "Cancel", variant="soft", color_scheme="gray",
                        on_click=TextBlocksState.cancel_edit,
                    ),
                    rx.button(
                        "Submit", variant="solid", color_scheme="green",
                        on_click=TextBlocksState.submit_edit,
                    ),
                    justify="end",
                    width="100%",
                ),
                spacing="2",
                align="start",
            ),
            max_width="700px",
        ),
        open=TextBlocksState.dialog_mode == "edit",
    )


def view_block_dialog() -> rx.Component:
    return rx.dialog.root(
        rx.dialog.content(
            rx.dialog.title(
                f"#{TextBlocksState.dialog_block_id} · "
                f"{TextBlocksState.dialog_title}" if TextBlocksState.dialog_title
                else f"#{TextBlocksState.dialog_block_id}",
            ),
            rx.vstack(
                rx.text("model is writing...", font_size="12px", color="#38a169"),
                rx.box(
                    rx.text(
                        TextBlocksState.dialog_content,
                        font_size="13px", color="#e2e8f0",
                        white_space="pre-wrap",
                        font_family="monospace",
                    ),
                    padding="12px",
                    background="#1a202c",
                    border_radius="6px",
                    width="100%",
                    min_height="200px",
                ),
                rx.hstack(
                    rx.button(
                        "Close", variant="soft",
                        on_click=TextBlocksState.close_view_dialog,
                    ),
                    justify="end",
                    width="100%",
                ),
                spacing="2",
                align="start",
            ),
            max_width="700px",
        ),
        open=TextBlocksState.dialog_mode == "view",
    )
