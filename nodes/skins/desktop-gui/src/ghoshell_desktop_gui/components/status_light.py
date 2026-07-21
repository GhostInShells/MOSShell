"""Status light — breathing dot indicator for each command status."""

import reflex as rx

from ghoshell_desktop_gui.state import CommandStatus

_STATUS_COLORS: dict[CommandStatus, str] = {
    CommandStatus.PENDING: "gray",
    CommandStatus.RUNNING: "blue",
    CommandStatus.AWAITING_APPROVAL: "orange",
    CommandStatus.APPROVED: "green",
    CommandStatus.REJECTED: "red",
    CommandStatus.COMPLETED: "green",
    CommandStatus.ERROR: "red",
}

_ANIMATION_CLASS: dict[CommandStatus, str] = {
    CommandStatus.PENDING: "",
    CommandStatus.RUNNING: "status-pulse-slow",
    CommandStatus.AWAITING_APPROVAL: "status-pulse-fast",
    CommandStatus.ERROR: "status-pulse-fast",
    CommandStatus.APPROVED: "",
    CommandStatus.REJECTED: "",
    CommandStatus.COMPLETED: "",
}


def pulse_keyframes() -> str:
    return """
    @keyframes pulse-opacity {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }
    .status-pulse-slow {
        animation: pulse-opacity 2s ease-in-out infinite;
    }
    .status-pulse-fast {
        animation: pulse-opacity 0.8s ease-in-out infinite;
    }
    """


def status_light(status: CommandStatus) -> rx.Component:
    return rx.box(
        width="10px",
        height="10px",
        border_radius="50%",
        background=_STATUS_COLORS.get(status, "gray"),
        class_name=_ANIMATION_CLASS.get(status, ""),
        flex_shrink="0",
    )
