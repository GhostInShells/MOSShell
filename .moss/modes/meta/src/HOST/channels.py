# Main channel — the single entry point for this mode's CTML shell.
#
# This is the ONLY channel file: it remains a flat module (not a package).
# Matrix scans for name() == '__main__' Channel instances.
#
# Meta mode — MOSS 自省 / dogfood 模式:
#   matrix (节点/网络治理) + mcp hub (外部工具接入) + grounds (项目认知场).
#   echo 经 ghost-bridge 回话需要 matrix/mcp 绑定; grounds 提供认知场.

from ghoshell_moss import new_shell_main_channel
from ghoshell_moss.core.ctml.shell.ctml_main import inject_system_primitives
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.channels.moss_cli import build_moss_cli_channel
from ghoshell_moss.channels.matrix_channel import build_matrix_channel
from ghoshell_moss.channels.ground_channel import build_grounds_channel
from ghoshell_moss.channels.mcp_channel import build_mcp_hub_channel

main = new_shell_main_channel()

# -- 系统原语 --------------------------------------------------
inject_system_primitives(main, extended=True)

# -- matrix: 节点/网络治理 --------------------------------------
main.import_channels(build_matrix_channel())

# -- grounds: MOSS 项目认知场 -----------------------------------
main.import_channels(build_grounds_channel())

# -- mcp hub: 外部工具接入 (需要 Matrix 实例, 薄包装解容器) ------
def _mcp_hub(container):
    matrix = container.force_fetch(Matrix)
    return build_mcp_hub_channel(matrix, name='mcp', scopes=['ghost', 'mode'])

main.import_channels(_mcp_hub)

# -- moss_cli: 去授权的 moss CLI 自举 ---------------------------
main.import_channels(build_moss_cli_channel(name="moss_cli"))
