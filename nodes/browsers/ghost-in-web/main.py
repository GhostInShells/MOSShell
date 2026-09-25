"""ghost-in-web node entry point.

Start:  moss nodes run nodes/browsers/ghost-in-web
Debug:  python main.py

Ports:  WS   --port N, else MOSS_GHOST_IN_WEB_PORT, else 23890 (fixed — the
        extension hardcodes the node URL, so an ephemeral port would desync).
        Audit --audit-port N, else MOSS_GHOST_IN_WEB_AUDIT_PORT, else 23891.

One process, three faces over one store: the channel (the ghost's surface), the WS
server (the extension's edge), and the audit page (the human's). Human panel input
reaches the ghost as an ``input`` signal; a satellite click (screenshot) reaches it
as an ``aside`` signal carrying the image.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_NODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_NODE_DIR / "src"))

from ghoshell_moss.core.blueprint.matrix import Matrix  # noqa: E402

from ghoshell_ghost_in_web.audit import AuditServer  # noqa: E402
from ghoshell_ghost_in_web.channel import build_channel  # noqa: E402
from ghoshell_ghost_in_web.model import PageModel  # noqa: E402
from ghoshell_ghost_in_web.server import WebServer  # noqa: E402

HOST = "127.0.0.1"
DEFAULT_PORT = 23890
DEFAULT_AUDIT_PORT = 23891


def _arg(name: str) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(f"--{name}", type=int, default=0)
    args, _ = parser.parse_known_args()
    return getattr(args, name.replace("-", "_"))


def resolve_port() -> int:
    return _arg("port") or int(os.getenv("MOSS_GHOST_IN_WEB_PORT", DEFAULT_PORT))


def resolve_audit_port() -> int:
    return _arg("audit-port") or int(
        os.getenv("MOSS_GHOST_IN_WEB_AUDIT_PORT", DEFAULT_AUDIT_PORT)
    )


def resolve_origins() -> list[str] | None:
    """Pin the extension id (comma-separated) to keep other web pages out of the
    localhost WS. Empty = allow all (dev); the id is known only after install."""
    raw = os.getenv("MOSS_GHOST_IN_WEB_ORIGINS", "")
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    return origins or None


def _print_install_guide() -> None:
    """First-run guidance — the extension is the whole point of this body, so
    point the human at it instead of leaving two bare URLs and hoping."""
    guide = (
        "[ghost-in-web] 装扩展: chrome://extensions → 开发者模式 → 加载已解压的扩展程序\n"
        f"[ghost-in-web]   目录: {_NODE_DIR}/extension\n"
        "[ghost-in-web]   装好后把扩展 id pin 进 MOSS_GHOST_IN_WEB_ORIGINS"
        " (chrome-extension://<id>), 否则任意网页都能连本地 WS\n"
        "[ghost-in-web]   装完打开任意非 local 网页, 点右上角图标授权感知"
    )
    print(guide, flush=True)


async def main(matrix: Matrix) -> None:
    model = PageModel()

    audit = AuditServer(model, host=HOST, port=resolve_audit_port())
    audit.start()

    server = WebServer(
        model,
        send_signal=matrix.send_signal_to_ghost,
        host=HOST,
        port=resolve_port(),
        origins=resolve_origins(),
    )
    await server.start()
    print(f"[ghost-in-web] ws at {server.url}", flush=True)
    print(f"[ghost-in-web] audit at {audit.url}", flush=True)
    _print_install_guide()

    await matrix.provide_channel(build_channel(model, dispatch=server))


if __name__ == "__main__":
    Matrix.discover().run(main)
