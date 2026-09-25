"""probe — 验证 plugin api + session.event 通过 dsh web 内置 WS 下行流.

拓扑:
  dsh web 进程 (挂 plugin.ts, 监听 :3083)
    ├─ plugin 注册 /plugin-api/emit (append 事件) + /plugin-api/result (回调)
    └─ dsh web 内置 client-connection 把 session/event 广播到 /api/events.mux (WS 下行)
  python 父进程:
    ├─ 用 websockets 连 ws://127.0.0.1:3083/api/events.mux 收 session/event
    ├─ 调 /plugin-api/emit 触发 session.event
    └─ 调 /plugin-api/result 验证反向回调

运行 (在 skill 目录下):
    python3 probe.py

输出 *_OK / *_FAIL, 退出码 0 = 通过.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
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


async def collect_session_events(queue: list, stop: asyncio.Event) -> None:
    """连 /api/events.mux, 把 session/event frame 收进 queue."""
    async with websockets.connect(MUX_URL) as ws:
        while not stop.is_set():
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            except asyncio.TimeoutError:
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            # 外层 envelope: { type: 'server-request', method, payload }
            if msg.get("type") == "server-request" and msg.get("method") == "session/event":
                queue.append(msg.get("payload", {}))
            elif msg.get("type") == "server-request":
                queue.append({"__method__": msg.get("method"), "payload": msg.get("payload")})


async def main() -> None:
    proc = start_dsh()
    stop = asyncio.Event()
    queue: list[dict] = []

    collector = asyncio.create_task(collect_session_events(queue, stop))
    try:
        print("[probe] waiting for dsh web + plugin ...")
        await asyncio.sleep(8)  # 用 async sleep, 让 collector task 能 connect

        # 1. 调 api1 触发 session.event
        r1 = http_post("/plugin-api/emit", json.dumps({"msg": "hello-from-python"}))
        print("EMIT_RESP", r1)

        # 2. 等 WS 收到 session/event
        deadline = time.monotonic() + 10.0
        event = None
        while time.monotonic() < deadline:
            if queue:
                event = queue.pop(0)
                break
            await asyncio.sleep(0.05)

        if event is None:
            print("SESSION_EVENT_FAIL (no session/event via mux)")
        else:
            print("SESSION_EVENT_OK", json.dumps(event, ensure_ascii=False)[:400])

        # 3. 调 api2 验证反向回调
        r2 = http_post("/plugin-api/result", json.dumps({"answer": 42}))
        print("RESULT_RESP", r2)
        print("RESULT_OK" if r2.get("ok") else "RESULT_FAIL")

    except Exception as exc:
        print("PROBE_FAIL", type(exc).__name__, exc)
        if proc.stderr is not None:
            print("--- dsh stderr tail ---")
            print("\n".join(proc.stderr.read().splitlines()[-30:]))
    finally:
        stop.set()
        collector.cancel()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    asyncio.run(main())
