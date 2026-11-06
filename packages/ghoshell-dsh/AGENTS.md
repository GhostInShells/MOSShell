# AGENTS.md

本包用**自己的 features 系统**迭代，独立于仓库根。

- 治理：在本包目录下直接 `moss features ...`（自动找到 `.features/`）；显式指定用 `--dir .features`。
- 每个 dsh 代际（`vX_Y_Z`）是一棵自包含的兼容单元，包含该代际的 protocol、plugin、stubs 与 research。
- 依赖方向单向：本包不依赖 `ghoshell-moss`。
- vendor 源码在 `.vendors/`，源码 gitignore、结论入库，约定见 `.vendors/dsh-research/README.md`。

## 两种验证，两个家

- **单元测试** → `tests/unit/`，`pytest` 即可，不需要 dsh、不起进程。
- **对着真实 dsh 的验证** → `src/ghoshell_dsh/skills/`，一个领域一个可运行样例，同时是给使用者的
  说明文档（随包分发，装在 wheel 里，消费者也读得到）。平铺一层，不分类——这个池子不会大到需要分类，
  分类会让每个类目只有一个成员。
  skill 的写法照 `src/ghoshell_moss/cli/skills/AGENTS.md` 的四段（Scenario / Method /
  Exploration path / How to trust this），`How to trust this` 里就是那条跑得起来的命令。
  跑不了、说不清的验证，不算验证。
