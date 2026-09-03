# MATRIX.manifests — mode 级环境能力声明

在指定 mode 下启动 node-cell 体系时**分享**的声明（cross-cell）。

用 moss 启动一个**不在当前仓库内**的 node 时，仍会分享这些环境配置。

- 初始全空，mode 按需追加。
- 每个类目一个子包；声明任何 cell 在此 mode 下可共享的能力。
- host（`HOST/`）可覆盖此层。
