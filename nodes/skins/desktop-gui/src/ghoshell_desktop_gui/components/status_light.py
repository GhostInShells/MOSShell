"""Status light — breathing dot indicator for each command status."""

import reflex as rx

_STATUS_COLORS: dict[str, str] = {
    "pending": "gray",
    "running": "blue",
    "awaiting_approval": "orange",
    "approved": "green",
    "rejected": "red",
    "completed": "green",
    "error": "red",
}

_ANIMATION_CLASS: dict[str, str] = {
    "pending": "",
    "running": "status-pulse-slow",
    "awaiting_approval": "status-pulse-fast",
    "error": "status-pulse-fast",
    "approved": "",
    "rejected": "",
    "completed": "",
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


def status_light(status: str) -> rx.Component:
    return rx.box(
        width="10px",
        height="10px",
        border_radius="50%",
        background=_STATUS_COLORS.get(status, "gray"),
        class_name=_ANIMATION_CLASS.get(status, ""),
        flex_shrink="0",
    )
