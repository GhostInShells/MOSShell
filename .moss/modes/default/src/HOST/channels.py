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

from ghoshell_moss import new_default_shell_main_channel

main = new_default_shell_main_channel()
