# Project default provider manifest — inherit the openbox baseline.
#
# Re-exports the canonical openbox providers.  Project-specific providers are
# added here or in sibling modules; the scanner walks this whole package.
#
# --
# Project 默认 Provider 清单 — 继承 openbox 基线。
# 重导出 canonical openbox providers。项目专属 provider 在此追加或另建模块，
# 扫描器会遍历整个包。

from ghoshell_moss.matrix.openbox.providers import *
