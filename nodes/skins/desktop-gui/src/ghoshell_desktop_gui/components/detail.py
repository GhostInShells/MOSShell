"""Detail panel — command content, approval dialog, and results."""

import reflex as rx

from ghoshell_desktop_gui.state import DesktopState


def _no_selection() -> rx.Component:
    return rx.center(
        rx.text("Select a command from the sidebar", color_scheme="gray"),
        width="100%",
        height="100%",
    )


def _command_view(command) -> rx.Component:
    return rx.vstack(
        rx.heading(command.summary, size="3"),
        rx.text(
            command.channel_path + ":" + command.command_name,
            color_scheme="gray",
            font_size="12px",
        ),
        rx.divider(),
        # payload section
        rx.text("Command", font_weight="bold", font_size="13px"),
        rx.code_block(
            rx.cond(
                command.payload,
                command.payload.to_string(),
                "(no payload)",
            ),
            language="json",
            width="100%",
        ),
        # approval section
        rx.cond(
            command.status == "awaiting_approval",
            rx.vstack(
                rx.divider(),
                rx.text("Awaiting Approval", font_weight="bold", color="orange"),
                rx.text(
                    rx.cond(command.approval_prompt, command.approval_prompt, "Approve this action?")
                ),
                rx.hstack(
                    rx.button("Approve", color_scheme="green"),
                    rx.button("Reject", color_scheme="red"),
                ),
            ),
        ),
        # result section
        rx.cond(
            (command.status == "completed") | (command.status == "error"),
            rx.vstack(
                rx.divider(),
                rx.text("Result", font_weight="bold", font_size="13px"),
                rx.code_block(
                    rx.cond(command.result, command.result, "(no output)"),
                    width="100%",
                ),
            ),
        ),
        # rejected note
        rx.cond(
            command.status == "rejected",
            rx.vstack(
                rx.divider(),
                rx.text("Rejected", font_weight="bold", color="red"),
                rx.text(
                    rx.cond(command.human_reply, command.human_reply, "No reason given")
                ),
            ),
        ),
        padding="16px",
        width="100%",
        overflow_y="auto",
    )


def detail_panel() -> rx.Component:
    return rx.box(
        rx.cond(
            DesktopState.selected_id != "",
            _command_view(DesktopState.selected_command),
            _no_selection(),
        ),
        width="100%",
        height="100vh",
    )
