"""Consume-side probe for the webview service kind.

Connects a WebViewClient, prints a one-line summary on every change, and after
the first unread appears, activates that view once and prints the settled
condition — proving discovery, the change stream, ordering, and focus
convergence.

Start:  moss nodes run .moss/system_test_nodes/webview_consumer --mode system_test
Debug:  python main.py
"""

import asyncio

from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.services.webview import WebViewClient


async def main(matrix: Matrix):
    client = await matrix.connect_service(WebViewClient)
    activated = {"done": False}

    def summarize() -> str:
        parts = []
        for item in client.items():
            s = item.state
            parts.append(
                f"{item.address}::{item.declaration.title}"
                f"(unread={s.unread_count}, focused={s.focused_at is not None}, "
                f"last={s.last_message!r})"
            )
        return " | ".join(parts) or "(none)"

    def on_change():
        print(f"[webview_consumer] {summarize()}", flush=True)

    client.on_change(on_change)
    print(f"[webview_consumer] started: {summarize()}", flush=True)

    while True:
        await asyncio.sleep(1)
        items = client.items()
        if activated["done"] or not items:
            continue
        if not any(item.state.unread_count for item in items):
            continue
        address = items[0].address
        settled = await client.activate(address)
        activated["done"] = True
        print(
            f"[webview_consumer] activate {address} -> "
            f"unread={settled.unread_count} focused={settled.focused_at is not None}",
            flush=True,
        )


if __name__ == "__main__":
    Matrix.discover().run(main)
