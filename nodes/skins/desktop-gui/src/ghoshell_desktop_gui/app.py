"""Desktop GUI — Reflex app with Matrix lifespan."""

import reflex as rx

from ghoshell_desktop_gui.pages.index import index
from ghoshell_desktop_gui.components.status_light import pulse_keyframes


def _global_style() -> rx.Component:
    return rx.html(
        f"<style>{pulse_keyframes()}</style>",
    )


def main():
    app = rx.App(style=_global_style())
    app.add_page(index, route="/")
    app.run()
