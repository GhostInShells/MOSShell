"""Status light — breathing dot indicator for each command status."""

import reflex as rx


def status_light(status) -> rx.Component:
    return rx.box(
        width="10px",
        height="10px",
        border_radius="50%",
        background=rx.cond(
            status == "running", "blue",
            rx.cond(status == "awaiting_approval", "orange",
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
        class_name=rx.cond(
            status == "running", "status-pulse-slow",
            rx.cond(status == "awaiting_approval", "status-pulse-fast",
                rx.cond(status == "error", "status-pulse-fast",
                    "",
                ),
            ),
        ),
        flex_shrink="0",
    )


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
