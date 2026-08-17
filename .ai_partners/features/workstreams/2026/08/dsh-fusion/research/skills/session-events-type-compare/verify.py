"""verify — TS↔Python session event 类型数据比较 (长期可验证哨兵).

拓扑:
  dsh web 进程 (挂 plugin.ts, :3084)
    └─ plugin 注册 HTTP rpc GET /plugin-api/session-events-mock
         → 返回 TS 侧构造的 13 种 session event mock 数组 (ground truth)
  本脚本 (python, MOSS repo venv):
    ├─ 启动 dsh web
    ├─ 轮询 rpc 路由直到可用
    ├─ 拉取 mock events
    ├─ 逐条喂给 ghoshell_moss.agents.deepseek_harness.session_events 的强类型模型:
    │     SessionEvent.from_dict → 按 type 分发到具体 SessionEventModel
    │     断言 ① 分发到正确具体类  ② to_dict() == 原始 mock (round-trip)
    │            ③ seq/time/type 信封字段借道正确
    └─ 打印每事件 PASS/FAIL 表; 任一失败 → 非零退出码; 关停 dsh

运行 (需要 MOSS 仓库 venv, 因为 import ghoshell_moss):
    <repo>/.venv/bin/python verify.py
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from ghoshell_moss.agents.deepseek_harness.session_events import (
    SessionEvent,
    SessionEventModel,
    AssistantChunk,
    AssistantMessageEvent,
    RequestContextEvent,
    RequestHeader,
    SessionEndSeed,
    StepEnd,
    StepStart,
    TodoWrite,
    ToolCallEvent,
    ToolResultEvent,
    TurnEnd,
    TurnStart,
    UserMessageEvent,
)

SKILL_DIR = Path(__file__).resolve().parent
DSH_HOME = SKILL_DIR / "home"
PORT = 3084
HTTP_BASE = f"http://127.0.0.1:{PORT}"
MOCK_URL = f"{HTTP_BASE}/plugin-api/session-events-mock"


def build_registry() -> dict[str, type[SessionEventModel]]:
    models: list[type[SessionEventModel]] = [
        TurnStart,
        TurnEnd,
        StepStart,
        StepEnd,
        UserMessageEvent,
        AssistantChunk,
        AssistantMessageEvent,
        ToolCallEvent,
        ToolResultEvent,
        TodoWrite,
        RequestHeader,
        RequestContextEvent,
        SessionEndSeed,
    ]
    return {m.event_type(): m for m in models}


REGISTRY = build_registry()


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


def http_get_json(url: str, timeout: float = 5.0):
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def wait_for_rpc(proc: subprocess.Popen[str], timeout: float = 30.0) -> None:
    """轮询 rpc 路由直到可用, 不 sleep 死等."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            http_get_json(MOCK_URL, timeout=1.0)
            return
        except (urllib.error.URLError, OSError, ConnectionError):
            time.sleep(0.3)
    raise TimeoutError(f"dsh web 未在 {timeout}s 内就绪 (rpc 不可达: {MOCK_URL})")


def check_event(raw: dict) -> list[str]:
    """校验一条 mock 事件, 返回失败消息列表 (空 = 通过)."""
    failures: list[str] = []
    ev_type = raw.get("type", "")

    container = SessionEvent.from_dict(raw)
    if container.meta.type != ev_type:
        failures.append(f"容器 type 不匹配: {container.meta.type!r} != {ev_type!r}")

    # ① 分发到正确具体类
    cls = REGISTRY.get(ev_type)
    if cls is None:
        failures.append(f"无对应具体类: {ev_type!r}")
        return failures
    concrete = cls.from_session_event(container)
    if concrete is None:
        failures.append(f"{cls.__name__}.from_session_event 返回 None")
        return failures
    if not isinstance(concrete, cls):
        failures.append(f"分发结果非 {cls.__name__}: {type(concrete).__name__}")

    # ③ 信封字段借道正确
    if concrete.seq != raw.get("seq"):
        failures.append(f"seq 借道错误: {concrete.seq} != {raw.get('seq')}")
    if concrete.time != raw.get("time"):
        failures.append(f"time 借道错误: {concrete.time} != {raw.get('time')}")
    if concrete.type != ev_type:
        failures.append(f"type 借道错误: {concrete.type!r} != {ev_type!r}")

    # ② round-trip: to_dict() == 原始 mock
    if concrete.to_dict() != raw:
        failures.append("round-trip to_dict() != 原始 mock")

    # 反向断言: 其它类型 from_session_event 均返回 None (不错配)
    for other_type, other_cls in REGISTRY.items():
        if other_type == ev_type:
            continue
        if other_cls.from_session_event(container) is not None:
            failures.append(f"错配: {other_cls.__name__} 对 {ev_type!r} 返回了实例")
            break

    return failures


def main() -> None:
    proc = start_dsh()
    try:
        wait_for_rpc(proc)
        print(f"[verify] dsh web up, fetching mock events from {MOCK_URL}")
        body = http_get_json(MOCK_URL)
        events = body.get("events", [])
        print(f"[verify] received {len(events)} mock events")

        total_fail = 0
        for i, raw in enumerate(events, 1):
            failures = check_event(raw)
            if failures:
                total_fail += 1
                print(f"  [{i:>2}] FAIL  type={raw.get('type')!r}")
                for f in failures:
                    print(f"        - {f}")
            else:
                print(f"  [{i:>2}] PASS  type={raw.get('type')!r}")

        if total_fail:
            print(f"\n[verify] RESULT: {total_fail}/{len(events)} FAILED — dsh 类型漂移或 Python 模型缺陷")
            raise SystemExit(1)
        print(f"\n[verify] RESULT: ALL {len(events)} PASS — Python 模型与 TS session event 类型同步")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("[verify] dsh stopped")


if __name__ == "__main__":
    main()
