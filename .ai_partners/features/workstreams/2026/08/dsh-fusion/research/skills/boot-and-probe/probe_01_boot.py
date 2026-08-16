"""probe_01_boot — 用 Python SDK 启动 dsh, 验证目录安排可用.

运行方式 (在 research/ 目录下):

    cd research
    export DSH_HOME=./home
    python3 skills/boot-and-probe/probe_01_boot.py

脚本零参数 — cwd 与 env 提供一切上下文:
- cwd = research/, session 默认落 ./.sessions
- DSH_HOME = ./home, 隔离运行, 不污染 ~/.dsh
- 不传 cordis: SDK runtime 需要 bundled 默认配置 (sdk-jsonrpc-server +
  agent spine), web profile 的空 cordis.yml 会令 runtime 启动即退
  (见 research/2026-08-14_dsh_source_survey.md 的 SDK 默认配置极简).

结果: 打印 FINAL_RESPONSE 与 BOOT_OK.
"""

from __future__ import annotations

from deepseek_harness import DeepSeekHarness


def main() -> None:
    harness = DeepSeekHarness()  # cwd 默认 = 当前目录, 环境继承 DSH_HOME
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
