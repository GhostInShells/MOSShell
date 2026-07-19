# Main channel — the single entry point for this mode's CTML shell.
#
# This is the ONLY channel file: it remains a flat module (not a package).
# Matrix scans for name() == '__main__' Channel instances.
#
# Two construction patterns:
#   1. From scratch (independent mode):
#      from ghoshell_moss import new_default_shell_main_channel
#      main = new_default_shell_main_channel(description="...")
#      # append commands or compose sub-channels on main...
#
#   2. Extend global main (mode as incremental customization):
#      from MOSS.manifests.channels import main
#      # customize on top...
#
# --
# 主 Channel — 当前 mode 的 CTML shell 唯一入口。
# 保持为单文件模块 (非 package)。Matrix 扫描 name() == '__main__' 的 Channel 实例。

from ghoshell_container import IoCContainer

from ghoshell_moss import new_default_shell_main_channel
from ghoshell_moss.channels.file_editor_channel import build_file_editor_channel
from ghoshell_moss.channels.terminal_channel import build_terminal_channel
from ghoshell_moss.core.blueprint.channel_builder import MutableChannel
from ghoshell_moss.core.concepts.channel import Channel


def _build_bash_with_file_editor(container: IoCContainer) -> Channel:
    """system_test only: mount file_editor as a sub-channel of bash so
    compound-scenario dogfooding (create/edit files + run tests) exercises
    the parent-child channel tree in one workstream."""
    bash = build_terminal_channel()(container)
    assert isinstance(bash, MutableChannel)
    bash.import_channels(build_file_editor_channel())
    return bash


main = new_default_shell_main_channel()
main.import_channels(_build_bash_with_file_editor)
