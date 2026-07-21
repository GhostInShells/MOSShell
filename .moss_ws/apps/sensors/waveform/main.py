"""Waveform visualization app — subscribes to audio/pcm Zenoh stream, renders terminal bars.

Usage:
    moss apps test sensors/waveform
    moss apps start sensors/waveform
"""
import asyncio
import json
import struct
import sys
import time

from ghoshell_moss.core.blueprint.matrix import Matrix

_STREAM_KEY = "audio/pcm"
_FLOOR_DB = -60.0
_CEILING_DB = 0.0
_BAR_WIDTH = 40
_PARTIALS = " ▏▎▍▌▋▊▉█"


def _db_to_ratio(rms_db: float) -> float:
    clamped = max(_FLOOR_DB, min(_CEILING_DB, rms_db))
    return (clamped - _FLOOR_DB) / (_CEILING_DB - _FLOOR_DB)


def _render_bar(value_db: float, width: int) -> str:
    ratio = _db_to_ratio(value_db)
    filled = ratio * width
    full = int(filled)
    rem = filled - full
    bar = "█" * full
    if full < width:
        bar += _PARTIALS[int(rem * 8)]
        bar += " " * (width - full - 1)
    return bar


def _unpack_meta(data: bytes) -> dict:
    meta_len = struct.unpack(">I", data[:4])[0]
    return json.loads(data[4:4 + meta_len])


def _render_frame(meta: dict) -> str:
    bands = meta.get("bands", {})
    lines = []
    for band in ["bass", "mid", "high"]:
        db = bands.get(band, -96)
        bar = _render_bar(db, _BAR_WIDTH)
        lines.append(f"  {band:5s} {bar} {db:+.1f} dB")
    rms = meta.get("rms_db", -96)
    is_silent = meta.get("is_silent", True)
    status = "(silent)" if is_silent else "(active)"
    lines.append(f"  {'rms':5s} {_render_bar(rms, _BAR_WIDTH)} {rms:+.1f} dB {status}")
    return "\n".join(lines)


async def main(matrix: Matrix) -> None:
    session = matrix.session
    logger = matrix.logger
    logger.info("Waveform app starting, subscribing to %s", _STREAM_KEY)

    # Print header once
    print("\033[2J\033[H", end="")
    print("  Audio Waveform [bass | mid | high]\n")
    # Reserve 5 lines for bars + 1 blank
    for _ in range(5):
        print()

    stream = session.get_stream(_STREAM_KEY, maxsize=64)
    frame_count = 0
    start_time = time.time()

    try:
        async with stream:
            async for sample in stream:
                meta = _unpack_meta(sample.payload)
                # Render bars starting at line 4
                lines = _render_frame(meta).split("\n")
                for i, line in enumerate(lines):
                    print(f"\033[{4 + i};1H\033[K{line}", end="", flush=True)
                frame_count += 1

    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"\n\n  Stopped. {frame_count} frames in {elapsed:.1f}s ({fps:.0f} fps)")


if __name__ == "__main__":
    try:
        Matrix.discover().run(main)
    except KeyboardInterrupt:
        print("\nWaveform app stopped.")
        sys.exit(0)
