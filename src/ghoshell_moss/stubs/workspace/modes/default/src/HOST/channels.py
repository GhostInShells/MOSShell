# Main channel — the single entry point for this mode's CTML shell.
#
# This is the ONLY channel file: it remains a flat module (not a package).
# Matrix scans for name() == '__main__' Channel instances.
#
# 轻注入: 系统原语 + moss_cli (去授权的 moss CLI 自举通道)。
# Speech/AppStore 等重通道不属于降级后的默认 mode。
# --
# 主 Channel — 当前 mode 的 CTML shell 唯一入口。
# 保持为单文件模块 (非 package)。Matrix 扫描 name() == '__main__' 的 Channel 实例。

from ghoshell_moss import new_shell_main_channel
from ghoshell_moss.core.ctml.shell.ctml_main import inject_system_primitives
from ghoshell_moss.channels.moss_cli import build_moss_cli_channel

main = new_shell_main_channel()

# -- 系统原语 --------------------------------------------------
inject_system_primitives(main, extended=True)

# -- moss_cli: 去授权的 moss CLI 自举 ---------------------------
main.import_channels(build_moss_cli_channel(name="moss_cli"))
