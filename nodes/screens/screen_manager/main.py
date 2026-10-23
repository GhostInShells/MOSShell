"""MOSS node cell entry point.

Start:  moss nodes run nodes/screens/screen_manager
Debug:  python main.py

Port:   --port N, else MOSS_SCREEN_MANAGER_PORT, else 20381 (fixed).

A screen is a *place*, not a service: the human's window onto the ghost has to be
findable at a known address, so this node claims a fixed port from the 2038 series
(20380 is the zenoh mesh; this is its neighbour) instead of an ephemeral one. Pass
``--port`` to run a second screen beside the first.

Open:   the surface opens in the system browser on start (``--no-open`` for a
headless run). A windowless screen is one nobody can see; and because the port is
fixed, a restart re-points an already-open tab rather than stranding it. Pass
``--debug`` to keep the on-screen instruments (status line, frame rate, geometry,
activity log) docked at the bottom of the sidebar — off by default.

One process, two faces over one store: the channel (the ghost's control surface) and
the web surface (the human's window view and steering). The human's moves reach the
ghost through ``matrix.send_signal_to_ghost``.

On the mesh, the screen is also a **consumer**: it watches the ``webview`` service
kind and adopts every live view onto the desktop, so a node that serves a page
appears on screen with no manual ``open``.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import secrets
import sys
import webbrowser
from pathlib import Path

_NODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_NODE_DIR / "src"))

from ghoshell_moss.core.blueprint.matrix import Matrix  # noqa: E402
from ghoshell_moss.services.webview import WebViewClient  # noqa: E402

from ghoshell_screen_manager.audio import (  # noqa: E402
    AudioSource,
    MockAudioSource,
    TopicAudioSource,
)
from ghoshell_screen_manager.bridge import WebViewBridge  # noqa: E402
from ghoshell_screen_manager.clauses import TopicClauseFeed, TopicEventFeed  # noqa: E402
from ghoshell_screen_manager.channel import build_screen_channel  # noqa: E402
from ghoshell_screen_manager.model import ScreenModel  # noqa: E402
from ghoshell_screen_manager.surface import ScreenSurface  # noqa: E402

_INDEX_HTML = _NODE_DIR / "index.html"
_THEMES = _NODE_DIR / "themes"

HOST = "127.0.0.1"
DEFAULT_PORT = 20381
"""2038 series: 20380 is the zenoh mesh endpoint, so the screen sits next to the
thing that carries it. Fixed, not ephemeral — the human's window must live at an
address that survives a restart (and a second screen opts in via ``--port``)."""


def resolve_port() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--port", type=int, default=0)
    args, _ = parser.parse_known_args()
    if args.port:
        return args.port
    return int(os.getenv("MOSS_SCREEN_MANAGER_PORT", DEFAULT_PORT))


def wants_open() -> bool:
    """The browser opens by default — pass ``--no-open`` for a headless run."""
    return "--no-open" not in sys.argv[1:]


def wants_mock_audio() -> bool:
    """``--mock-audio`` swaps the live bus for the fabricated demo feed."""
    return "--mock-audio" in sys.argv[1:]


def wants_debug() -> bool:
    """``--debug`` keeps the on-screen instruments: the status line, frame rate,
    geometry and the activity log, docked at the bottom of the sidebar.

    Off by default — a screen the human watches shows the product, not the
    instruments behind it. The page's self-report to stdout is *not* this switch:
    it never reaches the screen, and it is how the model sees what the page does.
    """
    return "--debug" in sys.argv[1:]


def theme_name() -> str:
    """``--theme NAME`` — which palette the desk wears (``themes/NAME.json``)."""
    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg == "--theme" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--theme="):
            return arg.split("=", 1)[1]
    return "violet"


def presence_path() -> str:
    """``--presence PATH`` — the page that inhabits the desk, set before the first frame.

    The point of it being a start argument rather than a command: a screen that
    belongs to someone should *open* occupied. Sending the ghost in afterwards means
    the human watches an empty desk for as long as it takes the model to notice, and
    that gap is exactly what "it's slow" means.
    """
    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg == "--presence" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--presence="):
            return arg.split("=", 1)[1]
    return ""


_SESSION = secrets.token_hex(4)
"""这一轮起屏的窗口口令。

只有这个节点刚开的那扇窗带着它 —— 于是只有它能不加点击就成为屏。上一轮留下的
标签还在往同一个端口重连，它们带的 token 已经作废：它们只会是遮罩，不会突然
变成屏幕。人按按钮不受这条约束。
"""


async def main(matrix: Matrix) -> None:
    model = ScreenModel()
    model.set_theme(theme_name())

    # The live bus is the default: the background is driven by real speech, and
    # speech arriving *is* the switch. ``--mock-audio`` is the demo/headless escape.
    audio: AudioSource
    # The stage's ear on the conversation: the same bus that carries the audio
    # samples carries every finished clause. Subscribed only when stage(talk=...)
    # asks for it — see the channel's `stage` command.
    clauses = TopicClauseFeed(matrix.session.topics)
    # The other half of the mirror: what the ghost is doing (a state light, its
    # tool calls). Same bus, same switch — ``stage(talk=...)`` runs both.
    events = TopicEventFeed(matrix.session.topics)
    if wants_mock_audio():
        audio = MockAudioSource()
        print("[webview_screen] audio: mock (--mock-audio)", flush=True)
    else:
        audio = TopicAudioSource(matrix.session.topics)
        await audio.start()
        print("[webview_screen] audio: live (AudioSampleTopic)", flush=True)

    surface = ScreenSurface(
        model,
        audio,
        send_signal=matrix.send_signal_to_ghost,
        self_identity=matrix.this.address,
        host=HOST,
        port=resolve_port(),
        html_path=_INDEX_HTML,
        session=_SESSION,
        themes_dir=_THEMES,
    )
    await surface.start()
    print(f"[webview_screen] surface at {surface.url}", flush=True)

    # The desk is occupied before the first page ever loads: whoever lives on this
    # screen is part of it, not a window opened onto it. A bad path is a fact to
    # report and never a reason to die — the screen still works with an empty desk.
    occupant = presence_path()
    if occupant:
        try:
            model.set_presence_default(surface.add_local(Path(occupant)))
            print(f"[webview_screen] presence: {occupant}", flush=True)
        except ValueError as e:
            print(f"[webview_screen] presence rejected: {e}", flush=True)

    if wants_open():
        # Open just after the surface is listening, so the first request finds a
        # live server. A failure here is a fact to report, never a reason to die:
        # the screen is still usable from the url in the notice.
        # `claim=1` marks the window *we* opened as the screen: this node just told
        # the human to look here, so it takes the screen on hello instead of showing
        # the mask. Tabs left over from earlier runs carry no claim and stay masked.
        url = (
            surface.url
            + f"?claim=1&session={_SESSION}"
            + ("&debug=1" if wants_debug() else "")
        )
        try:
            opened = await asyncio.to_thread(webbrowser.open, url)
            if not opened:
                print(
                    f"[webview_screen] no browser accepted the url — "
                    f"open {url} by hand",
                    flush=True,
                )
        except Exception as e:  # noqa: BLE001 — never let a window kill the body
            print(f"[webview_screen] could not open a browser: {e!r}", flush=True)

    # Mesh consumer: adopt live web views onto the desktop. Optional — a screen
    # with no mesh peers just never adopts anything.
    client = await matrix.connect_service(WebViewClient)
    bridge = WebViewBridge(model, client, emit=surface.broadcast)
    await bridge.start()

    channel = build_screen_channel(
        model,
        surface=surface,
        audio=audio,
        clauses=clauses,
        events=events,
        surface_url=lambda: surface.url,
    )
    await matrix.provide_channel(channel)


if __name__ == "__main__":
    Matrix.discover().run(main)
