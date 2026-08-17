"""verify — 验证 DshRpcClient 的强类型动词对真实 apiproxy 的调用.

拓扑:
  DshLauncher (python)
    ├─ spawn dsh web profile (DSH_HOME = 本 skill home)
    ├─ __aenter__ 等 ws 连上 (push 式就绪)
    └─ client.session_list / workspace_list → 强类型 value

运行 (MOSS 仓库 venv, 因 import ghoshell_moss):
    cd <repo>/.ai_partners/features/workstreams/2026/08/dsh-fusion/research/skills/launcher-prototype
    <repo>/.venv/bin/python verify.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from ghoshell_moss.agents.deepseek_harness.launcher import DshLauncher, DshLauncherConfig
from ghoshell_moss.agents.deepseek_harness.client import DshRpcClient
from ghoshell_moss.agents.deepseek_harness.types import domains, sessions

SKILL_DIR = Path(__file__).resolve().parent
HOME = SKILL_DIR / "home"
PORT = 3085


async def main() -> None:
    config = DshLauncherConfig(home=HOME, port=PORT)
    async with DshLauncher(config) as launcher:
        client = DshRpcClient(launcher)

        # session.list — 只读, 无副作用
        value: sessions.SessionListValue = await client.session_list(sessions.SessionListParams())
        print("SESSION_LIST items=%d" % len(value.items))

        # workspace.list — 只读, 无副作用
        wvalue: domains.WorkspaceListValue = await client.workspace_list()
        print("WORKSPACE_LIST items=%d" % len(wvalue.items))

    print("VERIFY_DONE")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print("VERIFY_FAIL", type(exc).__name__, exc)
        sys.exit(1)
