---
name: ghoshell-dsh
description: dsh 多版本兼容层 — 用 python 启动并管理多个不同的 dsh 版本
---

# ghoshell-dsh

dsh 版本兼容包。dsh 上游在破坏性迭代（破坏到 patch 位），本包用**一个发行版 + 多个代际目录**
（`vX_Y_Z`）让多个 dsh 版本可以被 python 启动、共存，并让历史数据仍然可读。

依赖方向单向：本包不依赖 `ghoshell-moss`，只依赖机制层（如 `ghoshell-processes`）。
迭代方式见 [AGENTS.md](AGENTS.md)，vendor 源码见 `.vendors/`。
