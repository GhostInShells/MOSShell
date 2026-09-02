# Main channel — the single entry point for this mode's CTML shell.
#
# This is the ONLY channel file: it remains a flat module (not a package).
# Matrix scans for name() == '__main__' Channel instances.
#
# Meta mode — MOSS 自省 / dogfood 模式:
#   matrix (节点/网络治理) + mcp hub (外部工具接入) + grounds (项目认知场).
#   echo 经 ghost-bridge 回话需要 matrix/mcp 绑定; grounds 提供认知场.

from ghoshell_moss import new_moss_main_channel
from ghoshell_moss.channels.moss_cli import build_moss_cli_channel
from ghoshell_moss.channels.runtime_debug_channel import build_runtime_debug_channel

main = new_moss_main_channel()

main.import_channels(build_runtime_debug_channel())

# -- moss_cli: 去授权的 moss CLI 自举 ---------------------------
main.import_channels(build_moss_cli_channel(name="moss_cli"))
