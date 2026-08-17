"""verify — 验证 DshLauncher 的 rpc() 原语对真实 apiproxy 动词的调用.

拓扑:
  DshLauncher (python)
    ├─ spawn dsh web profile (DSH_HOME = 本 skill home)
    ├─ __aenter__ 等 ws 连上 (push 式就绪)
    └─ rpc('session.list' / 'workspace.list') → RpcResult

运行 (MOSS 仓库 venv, 因 import ghoshell_moss):
    cd <repo>/.ai_partners/features/workstreams/2026/08/dsh-fusion/research/skills/launcher-prototype
    <repo>/.venv/bin/python verify.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from ghoshell_moss.agents.deepseek_harness.launcher import DshLauncher, DshLauncherConfig

SKILL_DIR = Path(__file__).resolve().parent
HOME = SKILL_DIR / "home"
PORT = 3085


async def main() -> None:
    config = DshLauncherConfig(home=HOME, port=PORT)
    async with DshLauncher(config) as launcher:
        # session.list — 只读, 无副作用
        r = await launcher.rpc("session.list", {})
        print("SESSION_LIST ok=%s value=%s" % (r["ok"], r.get("value")))

        # workspace.list — 只读, 无副作用
        r = await launcher.rpc("workspace.list", {})
        print("WORKSPACE_LIST ok=%s value=%s" % (r["ok"], r.get("value")))

    print("VERIFY_DONE")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print("VERIFY_FAIL", type(exc).__name__, exc)
        sys.exit(1)
