"""Sidebar — command list with status lights and stale toggle."""

import reflex as rx

from ghoshell_desktop_gui.state import DesktopState, CommandStatus
from ghoshell_desktop_gui.components.status_light import status_light


def command_item(command) -> rx.Component:
    return rx.hstack(
        status_light(command.status),
        rx.text(command.summary, font_size="14px", truncate=True),
        rx.spacer(),
        rx.text(command.channel_path, font_size="11px", color_scheme="gray"),
        padding="8px 12px",
        border_radius="6px",
        bg=rx.cond(
            DesktopState.selected_id == command.id,
            rx.color("accent", 3),
            "transparent",
        ),
        cursor="pointer",
        on_click=DesktopState.select_command(command.id),
        width="100%",
    )


def sidebar() -> rx.Component:
    return rx.vstack(
        rx.heading("Desktop", size="4", padding="12px"),
        rx.switch(
            checked=DesktopState.show_stale,
            on_change=DesktopState.toggle_stale(),
            text="Show stale",
        ),
        rx.divider(),
        rx.foreach(
            DesktopState.visible_commands,
            command_item,
        ),
        rx.spacer(),
        padding="8px",
        min_width="280px",
        max_width="360px",
        height="100vh",
        border_right="1px solid",
        border_color=rx.color("gray", 5),
        overflow_y="auto",
    )
