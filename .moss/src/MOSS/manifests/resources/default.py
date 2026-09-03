# Project default resource manifest — inherit the openbox baseline.
#
# Re-exports the canonical openbox resources.  Project-specific resources are
# added here or in sibling modules; the scanner walks this whole package.
#
# --
# Project 默认 Resource 清单 — 继承 openbox 基线。
# 重导出 canonical openbox resources。项目专属 resource 在此追加或另建模块，
# 扫描器会遍历整个包。

from ghoshell_moss.matrix.openbox.resources import *
