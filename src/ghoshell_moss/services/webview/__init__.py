"""``webview`` service kind — a cell that serves a page for a human.

Self-contained by design: this package owns its declaration, its condition, its
server and its client, and is imported directly rather than re-exported through
``ghoshell_moss.services``.

    from ghoshell_moss.services.webview import WebViewServer, WebViewDeclaration

    server = await WebViewServer.serve(
        matrix,
        WebViewDeclaration(url='http://127.0.0.1:8768/', title='Terminal'),
    )
    server.notify('build finished')

A consumer keeps every live view and its condition, and decides for itself what
to do with them:

    client = await matrix.connect_service(WebViewClient)
    for item in client.items():          # most recently active first
        render(item.declaration, item.state)
    await client.activate(items[0].address)
"""

from .declaration import (
    ACTIVATE_KEY,
    KIND,
    STATE_KEY,
    WebViewDeclaration,
    WebViewState,
)
from .client import WebViewClient, WebViewItem
from .server import WebViewServer

__all__ = [
    'KIND',
    'STATE_KEY',
    'ACTIVATE_KEY',
    'WebViewDeclaration',
    'WebViewState',
    'WebViewServer',
    'WebViewClient',
    'WebViewItem',
]
