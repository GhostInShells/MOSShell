"""Serve-side probe for the webview service kind.

Provides a WebViewServer with a synthetic url and drives the kind's two
self-claims — ``notify`` (unread + activity) and ``touch`` (activity only) — on
a timer.  A consumer node watches discovery, the stream, and ordering.

Start:  moss nodes run .moss/system_test_nodes/webview_server --mode system_test
Debug:  python main.py
"""

import asyncio

from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.services.webview import WebViewDeclaration, WebViewServer


async def main(matrix: Matrix):
    server = await WebViewServer.serve(
        matrix,
        WebViewDeclaration(
            url="http://127.0.0.1:9001/",
            title="webview_server",
            description="serve-side probe",
        ),
    )
    print(f"[webview_server] provided at {matrix.this.address}", flush=True)

    tick = 0
    while True:
        await asyncio.sleep(2)
        tick += 1
        if tick % 3 == 0:
            server.touch()  # activity without a badge
            print(f"[webview_server] touch #{tick}", flush=True)
        else:
            server.notify(f"tick {tick}")  # activity + something to read
            print(f"[webview_server] notify #{tick}", flush=True)


if __name__ == "__main__":
    Matrix.discover().run(main)
