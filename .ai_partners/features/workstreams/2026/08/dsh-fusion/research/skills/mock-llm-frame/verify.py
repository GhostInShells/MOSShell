"""verify — mock LlmAdapter 产出 tool-use 机制验证.

拓扑:
  dsh web 进程 (挂 plugin.ts, :3085)
    └─ plugin:
         FrameMockAdapter (moss-frame 产 moss_frame tool-call / moss-real 产文本)
         agent/request 路由: step1 → moss-frame, 之后 → moss-real
         moss_frame tool (execute 返回占位帧)
         RPC: GET  /plugin-api/frame-log     (读 session log)
              POST /plugin-api/frame-trigger (建 agent + steer 唤醒 turn)
  本脚本 (纯标准库, 无第三方依赖):
    ├─ 创建 home/workspace (agent cwd)
    ├─ 启动 dsh web (DSH_HOME=home)
    ├─ 轮询 frame-log 直到可用
    ├─ POST frame-trigger → 触发帧交付 turn
    ├─ 轮询 frame-log 直到 turn/end 出现 (带超时)
    └─ 断言: ① tool/call moss_frame ② tool/result 成对
            ③ request/header 里 moss-frame 在 moss-real 之前 (路由切换)
            ④ turn/end completed ⑤ 切回后产了文本 assistant/message (续走)

运行 (无需 MOSS venv, 纯标准库; 需要 `dsh` 在 PATH):
    python3 verify.py
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent
DSH_HOME = SKILL_DIR / "home"
WORKSPACE_DIR = DSH_HOME / "workspace"
PORT = 3085
HTTP_BASE = f"http://127.0.0.1:{PORT}"
TRIGGER_URL = f"{HTTP_BASE}/plugin-api/frame-trigger"
LOG_URL = f"{HTTP_BASE}/plugin-api/frame-log"


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


def http_json(url: str, method: str = "GET", timeout: float = 5.0):
    req = urllib.request.Request(url, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def wait_for_plugin(proc: subprocess.Popen[str], timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            http_json(LOG_URL, timeout=1.0)
            return
        except (urllib.error.URLError, OSError, ConnectionError):
            time.sleep(0.3)
    raise TimeoutError(f"dsh web 未在 {timeout}s 内就绪 (plugin rpc 不可达: {LOG_URL})")


def fetch_log() -> list[dict]:
    return http_json(LOG_URL).get("events", [])


def wait_for_turn_end(timeout: float = 30.0) -> list[dict]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        events = fetch_log()
        if any(e["type"] == "turn/end" for e in events):
            return events
        time.sleep(0.3)
    raise TimeoutError(f"turn/end 未在 {timeout}s 内出现")


def assert_mechanism(events: list[dict]) -> list[tuple[str, bool]]:
    tool_calls = [e for e in events if e["type"] == "tool/call"]
    tool_results = [e for e in events if e["type"] == "tool/result"]
    headers = [e for e in events if e["type"] == "request/header"]
    turn_ends = [e for e in events if e["type"] == "turn/end"]
    assistant_msgs = [e for e in events if e["type"] == "assistant/message"]

    providers = [h["data"]["header"]["config"]["provider"] for h in headers]

    frame_tool_call = any(tc["data"].get("name") == "moss_frame" for tc in tool_calls)
    text_continuation = any(
        any(b.get("type") == "text" and "[mock text]" in (b.get("text") or "")
            for b in am["data"]["message"]["content"])
        for am in assistant_msgs
    )
    switch_order = (
        "moss-frame" in providers and "moss-real" in providers
        and providers.index("moss-frame") < providers.index("moss-real")
    )

    return [
        ("① tool/call moss_frame 出现 (mock 产出可调度)", frame_tool_call),
        ("② tool/result 成对出现 (frame tool resolve)", len(tool_results) >= 1),
        ("③ 路由切换: moss-frame 出现在 moss-real 之前", switch_order),
        ("④ turn/end completed (turn 正常收线)", any(te["data"].get("reason", {}).get("kind") == "completed" for te in turn_ends)),
        ("⑤ 切回后产文本 assistant/message (续走)", text_continuation),
    ]


def main() -> None:
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    proc = start_dsh()
    try:
        wait_for_plugin(proc)
        print("[verify] dsh web up, triggering frame turn")
        http_json(TRIGGER_URL, method="POST")
        events = wait_for_turn_end()
        print(f"[verify] turn/end observed, {len(events)} events in log")

        trace = [e["type"] for e in events]
        print("[verify] event trace:", " → ".join(trace))

        results = assert_mechanism(events)
        failures = [label for label, ok in results if not ok]
        for label, ok in results:
            print(f"  [{'PASS' if ok else 'FAIL'}] {label}")

        if failures:
            print(f"\n[verify] RESULT: {len(failures)} FAILED — mock LLM tool-use 机制未通过")
            raise SystemExit(1)
        print("\n[verify] RESULT: ALL PASS — mock LlmAdapter 可确定性产出 tool-use, 路由切换成立")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("[verify] dsh stopped")


if __name__ == "__main__":
    main()
