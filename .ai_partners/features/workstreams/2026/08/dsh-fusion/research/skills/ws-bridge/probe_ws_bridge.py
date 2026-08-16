"""probe_ws_bridge — 验证 dsh plugin 注册 ws endpoint, 启动 dsh 的 python 进程做 ws client 双向通讯.

运行方式 (在 research/ 目录下):
    cd research
    python3 skills/ws-bridge/probe_ws_bridge.py

零参数 — cwd = research/, 定位:
- dsh 源码:      ./source/deepseek-harness  (node --import tsx 起 harness)
- cordis.yml:    ./home/cordis.yml           (加载 ./ws-echo.ts)

流程: subprocess 起 dsh(tsx) -> 等 ws://127.0.0.1:8765 就绪 -> 连上发 ping -> 收 echo -> 验证.
结果: 打印 SENT/RECV 与 ECHO_OK (或 ECHO_MISMATCH).
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from pathlib import Path

import websockets

RESEARCH = Path.cwd()
DSH_SRC = RESEARCH / "source" / "deepseek-harness"
CORDIS = RESEARCH / "home" / "cordis.yml"
WS_URL = "ws://127.0.0.1:3081/echo"


def start_harness() -> subprocess.Popen[str]:
    cmd = [
        "node", "--import", "tsx/esm",
        "packages/examples/jsonrpc-demo/src/bin.ts",
        str(CORDIS),
    ]
    return subprocess.Popen(
        cmd,
        cwd=DSH_SRC,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


async def connect_with_retry(url: str, timeout: float = 20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            return await websockets.connect(url)
        except OSError:
            await asyncio.sleep(0.3)
    raise TimeoutError(f"ws server did not come up at {url}")


async def main() -> None:
    proc = start_harness()
    try:
        async with await connect_with_retry(WS_URL) as ws:
            await ws.send("ping")
            reply = await ws.recv()
            print("SENT", "ping")
            print("RECV", reply)
            print("ECHO_OK" if reply == "ping" else "ECHO_MISMATCH")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        if proc.stderr is not None:
            stderr = proc.stderr.read()
            if stderr.strip():
                print("--- dsh stderr tail ---")
                print("\n".join(stderr.strip().splitlines()[-20:]))


if __name__ == "__main__":
    asyncio.run(main())
