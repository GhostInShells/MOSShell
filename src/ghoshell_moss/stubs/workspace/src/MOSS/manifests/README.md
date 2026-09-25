# MOSS.manifests — project manifests

Workspace 级基线声明。在 moss workspace 里**所有模式**都会装线（wired in every mode）。

承载通讯必需 / workspace 级基线能力（providers / configs / topics / signals / parameters / resources / nuclei）。

- 每个类目一个子包，`default.py` 重导出 canonical openbox 基线（`ghoshell_moss.matrix.openbox`）。
- 项目专属声明在 `default.py` 追加，或新建同名子模块（扫描器遍历整个包）。
- 默认内容随 `moss project overwrite-stubs` 同步，未改动时自动跟随上游。
