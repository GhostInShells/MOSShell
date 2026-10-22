# Main channel — the single entry point for this mode's CTML shell.
#
# This is the ONLY channel file: it remains a flat module (not a package).
# Matrix scans for name() == '__main__' Channel instances.
#
# Develop mode — iterate MOSS with MOSS.
#   bringup (HOST.md): screen_manager / file_editor / terminal — the worktable is
#   there the moment the shell is up; playwright and the camera are opened on demand.
#   channel face: same as default (system primitives + matrix + moss_cli + macro
#   store), plus the runtime-error self-diagnosis. No runtime_debug: develop is
#   NOT root — meta keeps the super-user doors. The difference here is not
#   permission but the loop: edit in file_editor, run in terminal, see on screen,
#   and never reach for a harness tool to close it.
# --
# 主 Channel — 当前 mode 的 CTML shell 唯一入口。
# develop 模式：用自己的工具迭代自己。闭环只在 MOSS 内。

from ghoshell_moss import new_moss_main_channel
from ghoshell_moss.channels.macro_store import MacroStoreModule
from ghoshell_moss.channels.moss_cli import build_moss_cli_channel
from ghoshell_moss.channels.runtime_error_channel import new_runtime_error_channel

main = new_moss_main_channel()

# -- moss_cli: 去授权的 moss CLI 自举 ---------------------------
main.import_channels(build_moss_cli_channel(name="moss_cli"))

# -- Macro Store: 程序性记忆 (root = project path, 权限边界) --
main.with_module(MacroStoreModule.new_from_moss_project())

# -- Runtime Error: 运行时错误自诊断 (模型 pull 背景/启动期降级) --
main.import_channels(new_runtime_error_channel())
