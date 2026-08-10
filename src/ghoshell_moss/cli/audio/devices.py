"""device command — enumerate input/output audio endpoints.

设备面: miniaudio 枚举输入 (capture) 与输出 (playback) 端点.
miniaudio 是重依赖, 惰性 import (depend_host 守卫).
"""

from __future__ import annotations

from ghoshell_moss.cli.audio import audio_app
from ghoshell_moss.cli.utils import print_error, print_simple_table, print_warning


@audio_app.command("device")
def device() -> None:
    """List available audio input and output devices."""
    try:
        from ghoshell_moss.depends import depend_host
        depend_host()
        import miniaudio
    except Exception:
        print_error("miniaudio not available — install ghoshell-moss[host]")
        return

    try:
        devs = miniaudio.Devices()
    except Exception as e:
        print_error(f"failed to enumerate devices: {e}")
        return

    rows = []
    for d in devs.get_captures():
        ch = max(f["channels"] for f in d["formats"]) if d["formats"] else "?"
        rows.append(["input", d["name"], str(ch)])
    for d in devs.get_playbacks():
        ch = max(f["channels"] for f in d["formats"]) if d["formats"] else "?"
        rows.append(["output", d["name"], str(ch)])

    if not rows:
        print_warning("no audio devices found")
        return

    print_simple_table(rows, headers=["Type", "Name", "Max Ch"], title="audio devices")
