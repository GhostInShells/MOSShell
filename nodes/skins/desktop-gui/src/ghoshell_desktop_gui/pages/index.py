"""Main page — dual-pane layout."""

import reflex as rx

from ghoshell_desktop_gui.state import DesktopState
from ghoshell_desktop_gui.components.sidebar import sidebar
from ghoshell_desktop_gui.components.detail import detail_panel


def index() -> rx.Component:
    return rx.hstack(
        sidebar(),
        rx.divider(orientation="vertical"),
        detail_panel(),
        width="100vw",
        height="100vh",
    )
