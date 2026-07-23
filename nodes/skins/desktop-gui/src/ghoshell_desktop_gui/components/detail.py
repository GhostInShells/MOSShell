"""Detail panel — per-command-type rendering, approval controls, dialogue thread."""

import reflex as rx

from ghoshell_desktop_gui.state import DesktopState
from ghoshell_desktop_gui.components.status_light import status_light


# -- no selection --

def _no_selection() -> rx.Component:
    return rx.center(
        rx.vstack(
            rx.icon("panel-left", size=32, color=rx.color("gray", 7)),
            rx.text("Select a command from the sidebar", color_scheme="gray", size="3"),
            rx.text(
                "Use the Mock Commands panel to inject test commands",
                color_scheme="gray", size="1",
            ),
            spacing="3",
        ),
        width="100%",
        height="100%",
    )


# -- command header --

def _command_header() -> rx.Component:
    cmd = DesktopState.selected_command
    return rx.vstack(
        rx.hstack(
            status_light(cmd.status),
            rx.heading(cmd.summary, size="4"),
            spacing="2",
            align="center",
        ),
        rx.hstack(
            rx.code(cmd.channel_path + ":" + cmd.command_name, font_size="11px"),
            rx.badge(cmd.status, variant="soft", size="1"),
            rx.text(cmd.id, font_size="10px", color_scheme="gray"),
            spacing="2",
        ),
        rx.divider(),
        width="100%",
    )


# -- per-command-type rendering --

def _render_bash_exec() -> rx.Component:
    cmd = DesktopState.selected_command
    cmd_text = cmd.params.get("text__", "")
    cwd = cmd.params.get("cwd", "")
    return rx.vstack(
        rx.text("Shell Command", font_weight="bold", font_size="13px"),
        rx.box(
            rx.code(cmd_text, font_size="13px"),
            padding="8px 12px",
            border_radius="6px",
            bg=rx.color("gray", 2),
            width="100%",
        ),
        rx.cond(
            cwd != "",
            rx.hstack(
                rx.text("cwd:", font_size="11px", color_scheme="gray"),
                rx.code(cwd, font_size="11px"),
                spacing="1",
            ),
            rx.fragment(),
        ),
        width="100%",
    )


def _render_file_view() -> rx.Component:
    cmd = DesktopState.selected_command
    path = cmd.params.get("path", "")
    return rx.vstack(
        rx.text("File Path", font_weight="bold", font_size="13px"),
        rx.code(path, font_size="12px"),
        width="100%",
    )


def _render_file_str_replace() -> rx.Component:
    cmd = DesktopState.selected_command
    path = cmd.params.get("path", "")
    return rx.vstack(
        rx.text("Edit File", font_weight="bold", font_size="13px"),
        rx.code(path, font_size="12px"),
        rx.cond(
            cmd.diff_opcodes != None,
            _diff_view(cmd.diff_opcodes),
            rx.text("(no diff data)", font_size="11px", color_scheme="gray"),
        ),
        width="100%",
    )


def _render_file_write() -> rx.Component:
    cmd = DesktopState.selected_command
    path = cmd.params.get("path", "")
    text = cmd.params.get("text__", "")
    return rx.vstack(
        rx.text("Write File", font_weight="bold", font_size="13px"),
        rx.code(path, font_size="12px"),
        rx.cond(
            text != "",
            rx.code_block(text, width="100%", font_size="12px"),
            rx.fragment(),
        ),
        width="100%",
    )


def _render_generic() -> rx.Component:
    return rx.vstack(
        rx.text("Parameters", font_weight="bold", font_size="13px"),
        rx.text(
            "See command details in sidebar",
            font_size="12px", color_scheme="gray",
        ),
        width="100%",
    )


def _render_command_body() -> rx.Component:
    cmd = DesktopState.selected_command
    return rx.match(
        cmd.command_name,
        ("exec", _render_bash_exec()),
        ("run", _render_bash_exec()),
        ("view", _render_file_view()),
        ("str_replace", _render_file_str_replace()),
        ("write", _render_file_write()),
        ("create", _render_file_write()),
        _render_generic(),
    )


# -- diff view --

def _diff_block(op) -> rx.Component:
    return rx.hstack(
        rx.box(
            rx.code(op["old_text"], font_size="11px"),
            padding="4px 8px",
            border_radius="4px",
            bg=rx.cond(
                (op["tag"] == "replace") | (op["tag"] == "delete"),
                rx.color("red", 2),
                "transparent",
            ),
            width="48%",
            overflow_x="auto",
        ),
        rx.box(
            rx.code(op["new_text"], font_size="11px"),
            padding="4px 8px",
            border_radius="4px",
            bg=rx.cond(
                (op["tag"] == "replace") | (op["tag"] == "insert"),
                rx.color("green", 2),
                "transparent",
            ),
            width="48%",
            overflow_x="auto",
        ),
        width="100%",
    )


def _diff_view(opcodes) -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.text("Old", font_size="11px", color_scheme="gray", width="48%"),
            rx.text("New", font_size="11px", color_scheme="gray", width="48%"),
            width="100%",
        ),
        rx.foreach(
            opcodes,
            _diff_block,
        ),
        width="100%",
    )


# -- approval section --

def _approval_section() -> rx.Component:
    approval = DesktopState.selected_approval
    return rx.vstack(
        rx.divider(),
        rx.hstack(
            rx.icon("shield-alert", size=16, color="orange"),
            rx.text("Awaiting Approval", font_weight="bold", color="orange", size="3"),
            spacing="2",
        ),
        rx.text(approval.prompt, font_size="13px"),
        rx.text_area(
            placeholder="Reason (optional)",
            value=DesktopState.approval_reason,
            on_change=DesktopState.set_approval_reason,
            size="1",
            width="100%",
        ),
        rx.hstack(
            rx.button(
                rx.icon("check"),
                "Approve",
                color_scheme="green",
                on_click=DesktopState.approve,
            ),
            rx.button(
                rx.icon("x"),
                "Reject",
                color_scheme="red",
                variant="soft",
                on_click=DesktopState.reject,
            ),
            spacing="2",
        ),
        width="100%",
    )


# -- dialogue section --

def _dialogue_section() -> rx.Component:
    return rx.vstack(
        rx.divider(),
        rx.text("Dialogue", font_weight="bold", font_size="13px"),
        rx.foreach(
            DesktopState.selected_dialogues,
            lambda msg: rx.hstack(
                rx.badge(
                    msg.sender,
                    color_scheme=rx.cond(
                        msg.sender == "human", "blue", "purple",
                    ),
                    variant="soft",
                    size="1",
                ),
                rx.text(msg.message, font_size="12px"),
                spacing="2",
            ),
        ),
        rx.hstack(
            rx.input(
                placeholder="Ask Ghost about this command…",
                value=DesktopState.dialogue_input,
                on_change=DesktopState.set_dialogue_input,
                width="100%",
                size="1",
            ),
            rx.button(
                "Send",
                size="1",
                on_click=DesktopState.send_dialogue,
            ),
            spacing="2",
            width="100%",
        ),
        width="100%",
    )


# -- result section --

def _result_section() -> rx.Component:
    cmd = DesktopState.selected_command
    result_text = rx.cond(
        cmd.result != None,
        cmd.result,
        "(no output)",
    )
    return rx.vstack(
        rx.divider(),
        rx.text("Result", font_weight="bold", font_size="13px"),
        rx.code_block(result_text, width="100%", font_size="12px"),
        width="100%",
    )


# -- resolved approval (read-only) --

def _resolved_approval_section() -> rx.Component:
    approval = DesktopState.selected_approval
    return rx.cond(
        approval != None,
        rx.vstack(
            rx.divider(),
            rx.hstack(
                rx.cond(
                    approval.status == "approved",
                    rx.icon("check", size=16, color="green"),
                    rx.icon("x", size=16, color="red"),
                ),
                rx.cond(
                    approval.status == "approved",
                    rx.text("Approved", font_weight="bold", size="3"),
                    rx.text("Rejected", font_weight="bold", size="3"),
                ),
                spacing="2",
            ),
            rx.cond(
                approval.human_reply != "",
                rx.hstack(
                    rx.text("Reason:", font_size="12px", color_scheme="gray"),
                    rx.text(approval.human_reply, font_size="12px", color_scheme="gray"),
                    spacing="1",
                ),
                rx.fragment(),
            ),
            width="100%",
        ),
        rx.fragment(),
    )


# -- main detail panel --

def detail_panel() -> rx.Component:
    return rx.box(
        rx.cond(
            DesktopState.selected_id != "",
            rx.vstack(
                _command_header(),
                _render_command_body(),
                # approval: show controls when awaiting_approval
                rx.cond(
                    DesktopState.selected_command.status == "awaiting_approval",
                    _approval_section(),
                    rx.fragment(),
                ),
                # resolved approval: show when approved or rejected
                rx.cond(
                    (DesktopState.selected_command.status == "approved")
                    | (DesktopState.selected_command.status == "rejected"),
                    _resolved_approval_section(),
                    rx.fragment(),
                ),
                # dialogue: always available when a command is selected
                _dialogue_section(),
                # result: when completed or error
                rx.cond(
                    (DesktopState.selected_command.status == "completed")
                    | (DesktopState.selected_command.status == "error"),
                    _result_section(),
                    rx.fragment(),
                ),
                rx.cond(
                    DesktopState.selected_command.status == "rejected",
                    rx.vstack(
                        rx.divider(),
                        rx.text(
                            "Command was rejected and did not execute.",
                            font_size="12px", color_scheme="gray",
                        ),
                        width="100%",
                    ),
                    rx.fragment(),
                ),
                padding="16px 20px",
                width="100%",
                overflow_y="auto",
            ),
            _no_selection(),
        ),
        width="100%",
        height="100vh",
    )
