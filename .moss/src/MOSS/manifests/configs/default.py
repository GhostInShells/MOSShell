# Project default config manifest — inherit the openbox baseline.
#
# Re-exports the canonical openbox configs.  Project-specific configs are
# added here or in sibling modules; the scanner walks this whole package.
#
# --
# Project 默认 Config 清单 — 继承 openbox 基线。
# 重导出 canonical openbox configs。项目专属 config 在此追加或另建模块，
# 扫描器会遍历整个包。

from ghoshell_moss.matrix.openbox.configs import *
