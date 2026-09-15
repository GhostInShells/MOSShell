"""Stream vision node entry point.

Start:  moss nodes run nodes/visions/stream -- --address rtmp://127.0.0.1/live --label desk
Debug:  ../.venv/bin/python main.py --address <url>

The stream address is the node's identity (launch argument); behavior knobs
come from cell-level env (copy `.env.example`). The stream is bound at launch;
`watch` only gates whether a fresh frame rides each round of context — it does
not touch the connection.
"""
from __future__ import annotations

import logging
import os
import pathlib
import sys

_NODE_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_NODE_DIR / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_NODE_DIR / ".env")

from ghoshell_moss.core.blueprint.matrix import Matrix  # noqa: E402

from stream_node.source import FfmpegSource  # noqa: E402
from stream_node.stream import StreamController  # noqa: E402


def _parse_args(argv: list[str]) -> dict[str, str | None]:
    out: dict[str, str | None] = {"address": None, "label": None}
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("--address", "--label") and i + 1 < len(argv):
            out[arg.lstrip("--")] = argv[i + 1]
            i += 2
            continue
        i += 1
    return out


def _config(argv: list[str]) -> dict:
    args = _parse_args(argv)
    address = args["address"] or os.getenv("STREAM_ADDRESS")
    if not address:
        raise SystemExit("--address <url> required (or STREAM_ADDRESS env)")
    return {
        "address": address,
        "label": args["label"] or os.getenv("STREAM_LABEL", ""),
        "fps": float(os.getenv("STREAM_FPS", "2")),
        "max_edge": int(os.getenv("STREAM_MAX_EDGE", "1568")),
        "max_bytes": int(os.getenv("STREAM_MAX_BYTES", str(512 * 1024))),
        "jpeg_quality": int(os.getenv("STREAM_JPEG_QUALITY", "80")),
        "stale_seconds": float(os.getenv("STREAM_STALE_SECONDS", "5")),
    }


async def main(matrix: Matrix) -> None:
    logger = matrix.logger or logging.getLogger("moss.visions.stream")
    cfg = _config(sys.argv[1:])
    logger.info("stream node starting (address=%s label=%r)", cfg["address"], cfg["label"])

    source = FfmpegSource(cfg["address"], fps=cfg["fps"], quality=4, logger=logger)
    controller = StreamController(
        source=source,
        address=cfg["address"],
        label=cfg["label"],
        fps=cfg["fps"],
        max_edge=cfg["max_edge"],
        max_bytes=cfg["max_bytes"],
        jpeg_quality=cfg["jpeg_quality"],
        stale_seconds=cfg["stale_seconds"],
        logger=logger,
    )
    # Failure to connect does not block launch — it surfaces via the channel
    # health short-circuit, so the ghost learns the stream is broken.
    await source.start()

    try:
        await matrix.publish_event(
            f"stream vision node alive; watching {cfg['label'] or cfg['address']}"
        )
    except Exception as e:
        logger.debug("publish_event failed: %s", e)

    try:
        await matrix.provide_channel(controller.as_channel())
    finally:
        await source.stop()


if __name__ == "__main__":
    Matrix.discover().run(main)
