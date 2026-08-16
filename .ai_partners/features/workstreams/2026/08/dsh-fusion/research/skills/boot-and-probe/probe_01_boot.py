"""probe_01_boot — 用 Python SDK 启动 dsh, 验证加载 hello-plugin.

运行方式 (在 research/ 目录下):

    cd research
    export DSH_HOME=./home
    python3 skills/boot-and-probe/probe_01_boot.py

脚本零参数 — cwd 与 env 提供一切上下文:
- cwd = research/, session 默认落 ./.sessions
- DSH_HOME = ./home, 隔离运行, 不污染 ~/.dsh
- cordis 指向 home/cordis.yml: bundled 默认配置 + hello-plugin 绝对路径.
  runtime 启动时打印 '[hello-plugin] plugin loaded!' 即验证通过.

结果: 打印 FINAL_RESPONSE 与 BOOT_OK.
"""

from __future__ import annotations

import signal
from pathlib import Path

from deepseek_harness import DeepSeekHarness


def main() -> None:
    signal.alarm(60)  # 60 秒兜底超时, 防止 exe 挂起无限卡住 (macOS 无 timeout 命令)
    cordis = (Path.cwd() / "home" / "cordis.yml").resolve()
    harness = DeepSeekHarness(cordis=str(cordis))  # DSH_CORDIS_CONFIG 指向 home/cordis.yml
    try:
        result = harness.run("只回复 OK 两个字。")
        print("SESSION_ID", result.session_id)
        print("FINAL_RESPONSE", result.final_response)
        print("FINISH_REASON", result.finish_reason)
        print("BOOT_OK")
    finally:
        harness.close()


if __name__ == "__main__":
    main()
