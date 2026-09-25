"""serve — 常驻启动 dsh web + plugin, 监听 tool/call, 回调 touch 文件.

拓扑:
  dsh web 进程 (挂 plugin.ts, 监听 :3083)
    ├─ plugin 注册 tool 'moss_shell_observe' + /plugin-api/callback
    └─ session/event (含 tool/call) 广播到 /api/events.mux WS
  本脚本 (python):
    ├─ 连 /api/events.mux, 监听 tool/call 帧
    ├─ 收到 moss_shell_observe 的 tool/call → 调 /plugin-api/callback
    └─ 打印事件, 常驻运行, ctrl+c 关闭

运行 (skill 目录下):
    python3 serve.py
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import urllib.request
from pathlib import Path

import websockets

SKILL_DIR = Path(__file__).resolve().parent
DSH_HOME = SKILL_DIR / "home"
PORT = 3083
HTTP_BASE = f"http://127.0.0.1:{PORT}"
MUX_URL = f"ws://127.0.0.1:{PORT}/api/events.mux"


def start_dsh() -> subprocess.Popen[str]:
    env = {**os.environ, "DSH_HOME": str(DSH_HOME)}
    return subprocess.Popen(
        ["dsh", "--profile", "web", "--port", str(PORT)],
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(SKILL_DIR),
    )


def http_post(path: str, body: str = "") -> dict:
    data = body.encode()
    req = urllib.request.Request(
        f"{HTTP_BASE}{path}", data=data, method="POST",
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read().decode())


async def watch_mux(proc: subprocess.Popen[str]) -> None:
    """连 /api/events.mux, 打印所有 frame, tool/call 时触发回调."""
    print(f"[serve] connecting {MUX_URL} ...")
    async with websockets.connect(MUX_URL) as ws:
        print("[serve] connected to mux stream")
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if msg.get("type") != "server-request":
                continue
            payload = msg.get("payload", {})
            method = msg.get("method")
            print(f"[mux] method={method}")

            # 会话事件: 打印 event type
            if method == "session/event":
                ev = payload.get("event", {})
                print(f"  event.type={ev.get('type')} seq={ev.get('seq')} sessionId={payload.get('sessionId')}")

                # 命中特殊 tool 的 tool/call → 回调
                if ev.get("type") == "tool/call" and ev.get("data", {}).get("name") == "moss_shell_observe":
                    print("  >>> moss_shell_observe tool/call detected, invoking callback ...")
                    r = http_post("/plugin-api/callback", json.dumps({
                        "callId": ev.get("data", {}).get("callId"),
                        "arguments": ev.get("data", {}).get("arguments"),
                    }))
                    print("  callback resp:", r)


async def main() -> None:
    proc = start_dsh()
    stop = asyncio.Event()

    def on_sig(signum, frame):  # noqa: ARG001
        print("\n[serve] signal received, shutting down ...")
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: on_sig(s, None))

    try:
        await asyncio.sleep(8)  # 让 dsh web + plugin 起来
        print("[serve] dsh web up, watching mux stream (ctrl+c to stop)")
        watcher = asyncio.create_task(watch_mux(proc))
        await stop.wait()
        watcher.cancel()
    except Exception as exc:
        print("[serve] FAIL", type(exc).__name__, exc)
        if proc.stderr is not None:
            print("--- dsh stderr tail ---")
            print("\n".join(proc.stderr.read().splitlines()[-30:]))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("[serve] dsh stopped")


if __name__ == "__main__":
    asyncio.run(main())
