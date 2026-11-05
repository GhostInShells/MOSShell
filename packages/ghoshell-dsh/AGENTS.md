# AGENTS.md

本包用**自己的 features 系统**迭代，独立于仓库根。

- 治理：在本包目录下 `moss features --dir features ...`（缺省根即 `packages/ghoshell-dsh/features`）。
- 每个 dsh 代际（`vX_Y_Z`）是一棵自包含的兼容单元，包含该代际的 protocol、plugin、stubs 与 research。
- 依赖方向单向：本包不依赖 `ghoshell-moss`。
- vendor 源码在 `.vendors/`，源码 gitignore、结论入库，约定见 `.vendors/dsh-research/README.md`。
