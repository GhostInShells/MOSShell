"""Sidebar — command list, mock control panel, stale toggle."""

import reflex as rx

from ghoshell_desktop_gui.state import DesktopState
from ghoshell_desktop_gui.components.status_light import status_light


# -- mock control panel (dev only) --

def mock_control() -> rx.Component:
    return rx.vstack(
        rx.text("Mock Commands", font_weight="bold", font_size="12px", color_scheme="gray"),
        rx.text("bash", font_size="11px", color_scheme="gray", padding_top="4px"),
        rx.hstack(
            rx.input(
                value=DesktopState.mock_exec_cmd,
                on_change=DesktopState.on_mock_exec_cmd_change,
                placeholder="shell command",
                size="1",
                width="100%",
            ),
            rx.checkbox(
                "approval",
                checked=DesktopState.mock_exec_approval,
                on_change=DesktopState.on_mock_exec_approval_change,
                size="1",
            ),
            rx.button(
                "exec",
                size="1",
                on_click=lambda: DesktopState.inject_mock_command(
                    "desktop.bash", "exec",
                    '{"text__": "' + DesktopState.mock_exec_cmd + '"}',
                    DesktopState.mock_exec_approval,
                ),
            ),
            spacing="1",
            width="100%",
        ),
        rx.text("file_editor", font_size="11px", color_scheme="gray", padding_top="8px"),
        rx.text("str_replace", font_size="10px", color_scheme="gray"),
        rx.hstack(
            rx.input(
                value=DesktopState.mock_str_replace_path,
                on_change=DesktopState.on_mock_str_replace_path_change,
                placeholder="path",
                size="1",
                width="40%",
            ),
            rx.button(
                "edit",
                size="1",
                color_scheme="orange",
                on_click=lambda: DesktopState.inject_mock_command(
                    "desktop.file_editor", "str_replace",
                    '{"path": "' + DesktopState.mock_str_replace_path + '", '
                    '"old_str": "port: 8080", "new_str": "port: 9090"}',
                    True,
                ),
            ),
            spacing="1",
            width="100%",
        ),
        rx.text("view", font_size="10px", color_scheme="gray"),
        rx.hstack(
            rx.input(
                value=DesktopState.mock_view_path,
                on_change=DesktopState.on_mock_view_path_change,
                placeholder="path",
                size="1",
                width="100%",
            ),
            rx.button(
                "view",
                size="1",
                color_scheme="blue",
                on_click=lambda: DesktopState.inject_mock_command(
                    "desktop.file_editor", "view",
                    '{"path": "' + DesktopState.mock_view_path + '"}',
                    False,
                ),
            ),
            spacing="1",
            width="100%",
        ),
        rx.button(
            "write: /etc/nginx.conf",
            size="1",
            variant="soft",
            on_click=lambda: DesktopState.inject_mock_command(
                "desktop.file_editor", "write",
                '{"path": "/etc/nginx.conf", "text__": "server { listen 80; }"}',
                True,
            ),
            width="100%",
        ),
        rx.divider(),
    )


# -- command list --

def _status_badge(status) -> rx.Component:
    return rx.badge(
        rx.cond(
            status == "awaiting_approval", "approval",
            rx.cond(status == "running", "running",
                rx.cond(status == "approved", "approved",
                    rx.cond(status == "rejected", "rejected",
                        rx.cond(status == "completed", "done",
                            rx.cond(status == "error", "error",
                                rx.cond(status == "stale", "stale",
                                    status,
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
        color_scheme=rx.cond(
            status == "awaiting_approval", "orange",
            rx.cond(status == "running", "blue",
                rx.cond(status == "approved", "green",
                    rx.cond(status == "rejected", "red",
                        rx.cond(status == "completed", "green",
                            rx.cond(status == "error", "red",
                                "gray",
                            ),
                        ),
                    ),
                ),
            ),
        ),
        variant="soft",
        size="1",
    )


def command_item(command) -> rx.Component:
    return rx.hstack(
        status_light(command.status),
        rx.vstack(
            rx.text(
                rx.cond(
                    command.summary != "",
                    command.summary,
                    command.channel_path + ":" + command.command_name,
                ),
                font_size="13px", truncate=True, width="100%",
            ),
            rx.hstack(
                rx.text(command.channel_path + ":" + command.command_name,
                        font_size="10px", color_scheme="gray"),
                _status_badge(command.status),
                spacing="1",
            ),
            spacing="0",
            width="100%",
        ),
        rx.spacer(),
        padding="6px 10px",
        border_radius="6px",
        bg=rx.cond(
            DesktopState.selected_id == command.id,
            rx.color("accent", 3),
            "transparent",
        ),
        cursor="pointer",
        on_click=lambda: DesktopState.select_command(command.id),
        width="100%",
    )


def sidebar() -> rx.Component:
    return rx.vstack(
        rx.heading("Desktop", size="4", padding="8px 12px 4px"),
        rx.hstack(
            rx.switch(
                checked=DesktopState.show_stale,
                on_change=DesktopState.toggle_stale,
                text="Stale",
                size="1",
            ),
            rx.cond(
                DesktopState.pending_approval_count > 0,
                rx.badge(
                    DesktopState.pending_approval_count,
                    color_scheme="orange",
                    variant="solid",
                    size="1",
                ),
            ),
            spacing="2",
            padding="0px 12px",
        ),
        rx.divider(),
        rx.scroll_area(
            mock_control(),
            max_height="360px",
            padding="0px 8px",
        ),
        rx.foreach(
            DesktopState.visible_commands,
            command_item,
        ),
        rx.spacer(),
        padding="8px 4px",
        min_width="300px",
        max_width="380px",
        height="100vh",
        border_right="1px solid",
        border_color=rx.color("gray", 5),
        overflow_y="auto",
    )
