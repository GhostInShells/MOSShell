---
created: 2026-06-28
depends:
- matrix-cell-governance
description: moss cells 命令行体系 — 统一取代 apps/script/runtime 三个旧 CLI 组。 静态发现、生命周期管理、运行时治理、认知引导全部归入
  cells 命令组。
milestone: 0.1.0
priority: P0
status: completed
title: Cells CLI
updated: '2026-07-21'
---

# Cells CLI

> 人类架构师 + claude-opus-4-7。从 matrix-cell-governance 的设计结论出发，
> 定义 `moss cells` 的完整命令体系。

## Motivation

旧 CLI 有三个独立命令组 (`moss apps`, `moss script`, `moss runtime`)，
对应三种不同的 cell 管理模型。matrix-cell-governance 将它们统一为 cell
概念后，需要一个对应的命令行入口：静态发现、进程启动、运行时治理、认知引导。

核心原则：**模型是第一开发者**。CLI 不需要覆盖所有操作——specification 定义约定，
模型读 specification 后自己改 CELL.md。CLI 只做最薄的约定执行层。

## Key Decisions

### 命令树 (10 个命令)

```
moss cells
├── specification     # 认知入口: CELL.md 格式 + 五类 cell + launcher + 依赖策略
├── create <name>     # 从 stub 生成最简模板 (CELL.md + main.py + README.md + INSTALL.md)
│   --group <group>
├── list              # CELL.md 扫描，mode-aware
│   --installed       #   只显示已安装
│   --include/exclude #   fnmatch 筛选
├── show <path>       # manifest 全部字段 + instruction 全文
├── register <file>   # 外部脚本快捷方式 → workspace/cells/{group}/{name}/
│   --name --group
├── run <target>      # 目录/脚本/名称 三种模式 → spawn + 启动前信息反馈
│   [args...]         #   透传
├── install <name>    # 标记已安装 (touch .installed)，不做自动化
├── status [address]  # 运行时: 无参=列表, 带 address=单 cell 详情
├── kill <address>    # 强杀 + 清 runtime file
└── kill-all          # 强杀全部
```

### 五类 cell (launcher 配置差异，不是类型差异)

| 标签 | interpreter | 依赖 | 例子 |
|------|------------|------|------|
| standalone | python (sys.executable) | 零外部依赖 | moss_self |
| project | python (sys.executable) | 依赖 project (import MOSS.*) | playwright |
| isolated | .venv/bin/python | 独立 pyproject.toml, 最小依赖 ghoshell_moss[matrix] | feishu |
| script | /bin/bash | 无关 ghoshell，纯 shell | install 脚本 |
| remote | N/A | 独立 .moss workspace，靠 network/scope 通讯 | G1 PC2 |

全部是 CellType.worker。差异在 CELL.md 的 launcher 字段。

### specification 是认知入口

`moss cells specification` 输出 cell 开发完整约定，等同 cell 世界的 `moss start`。
涵盖: CELL.md 格式、五类 cell 详解、launcher 配置、依赖隔离策略、cell 发现规则。
每个 CELL.md stub 的 instruction body 首行写 `moss cells specification — cell development guide.`
作为必然被读到的 hint。

### CELL.md + README.md 文档分工

- CELL.md body (instruction): 模型入口 — 能力清单 + CTML 命令 + 怎么调用
- README.md: 人类开发者入口 — 环境准备 + 依赖 + 开发流程 + 已知问题
- 不创建 CLAUDE.md (旧 app 的 AI 开发者上下文)

### create 不加 --type

specification 定义约定，模型读后自己改 CELL.md。CLI 只生成一种最简 stub。
独立 venv、pyproject.toml、复杂依赖——模型自己加。

### enable 概念删除

无 enable/disable 命令。筛选通过 CellRegistry.list_cell_manifests(include=, exclude=)
的 fnmatch 模式 + 模式的 exclude_cell_paths。

### install 是标记，不是自动化

`moss cells install <name>` 只创建 .installed 空文件。INSTALL.md 存在表示需要
安装——Ghost/模型读 INSTALL.md 用 bash:exec 跑安装步骤。CLI 不替 cell 跑脚本。

### run 反馈启动信息，不替代 python main.py

`moss cells run` 走完整 spawn 协议: resolve manifest → 写 runtime file → spawn。
`python main.py` 是开发者的快捷路径 (Cell.from_proc 自发现)。CLI 不对后者负责。

## Implementation Notes

- 复用 `Project.discover()` → `project.cells` (CellRegistry) 做静态发现
- 复用 `CellRegistry.spawn_cell()` 做进程启动
- 复用 `CellRegistry.local_runtime_cells()` + `Cell.is_alive()` 做运行时查询
- 复用 `CellRegistry.recursively_kill_process()` + `kill_all_runtime_cells()` 做进程杀死
- `show` 的入参是 project-relative 目录路径，不做向上查找
- `register` 创建的 CELL.md 放在 workspace/cells/ 下
- 命令间无状态的 CLI 模式——每个命令独立 `Project.discover()`，不跨命令共享

---

## §Rewrite 2026-07-17 (opus-4.7-1m + 人类架构师)

> 上文 (06-28 版) 的 API 命名 (`CellRegistry` / `spawn_cell` /
> `local_runtime_cells` / `recursively_kill_process` / `kill_all_runtime_cells`)
> 在当前抽象层**全部不存在**. matrix-cell-governance §UU 三域拆分后, 治理
> API 是 `NodeManager` / `NodeManifest` / `matrix.run_node` / `Subprocesses` /
> `CellRuntimeInfo` (blueprint/cell.py 权威). 命令树核心动作虽然大部分保留,
> 但**实现路径与命名全部要按新抽象重来** (§AAA-3 步骤 10 判决 "cli 我宁愿重做").
>
> **本节以后为权威**. 上文只作历史轨迹, 请以 `plan.md` (同目录) + 本节为准.

### §R-1. 决策差异一览 (vs 06-28)

| 项 | 06-28 | 2026-07-17 |
|---|---|---|
| 命令组 | `moss cells` | `moss nodes` (抽象层 node 化, code as prompt 对齐) |
| Manifest 文件 | `CELL.md` | `NODE.md` (`NodeManifest.MANIFEST_FILENAME`) |
| target 参数 | 目录 / 脚本 / **name** 三模式 | **path only** 三合一 (dir / NODE.md / .py). 无 name 反查 (name 属运行时) |
| `specification` 命令 | 认知入口, 强 hint | **删除**. 探索路径指向 `moss codex get-interface`/`codex blueprint`/`ctml read`/`howtos`, 不誊抄 |
| 五类 cell 分类 | standalone/project/isolated/script/remote | **删除**. 抽象层无此分类, 是旧 launcher.interpreter 时代产物 |
| `register` | 快捷方式命名 (语义不清) | 改名 `link`. 参数 = A 目录 (cell workspace) + B 脚本 (绝对路径), 无自动检测 (`--command` 必填, WW-2 判决) |
| `run:` frontmatter 糖 | fable 版 stub 引入 (WW-3) | **彻底删除**. NODE.md 与 `NodeManifest` pydantic 1:1, `exec: {command, args, env}` 直书, 无翻译层 |
| stub 目录 | `stubs/cell/` | `stubs/node/`. 加 `.gitignore` (`.installed`/`__pycache__`/`.venv`/`*.log`/`runtime/logs/`, 有注释解释为什么). README 极简骨架 |
| stub `singleton` 默认 | `false` | `true` (与 `NodeManifest.singleton` field default 一致) |
| run 实现 | `CellRegistry.spawn_cell` | CLI 独立 launcher: `NodeLauncher.from_manifest` + `subprocess.Popen(start_new_session=True)` + signal handler + `killpg` 兜底. 100% Project 层, 不起 matrix |
| status | `CellRegistry.local_runtime_cells` | `CellRuntimeInfo.iter_runtime_info(env.cell_runtimes_dir)` + `is_alive()` |
| kill | 立即杀 | 默认 `SIGTERM + 3s → SIGKILL`, `--force` 立即 `SIGKILL` |
| prune (新) | 无 | 孤儿 killer, 默认统统杀 (孤儿会锁 singleton), `--keep-alive` 只删死的 |
| stdout/stderr | 记录到文件 | CLI 前台 = inherit 终端 (直接看); 后台记录归 Jobs 层 (未定, 与 channel 一起讨论) |

### §R-2. 定位钉子 (最重要, 未来化身必读)

**CLI = cell 开发生命周期的地面站 (操作员/维护动作面)**. 严守边界:

- **100% Project 层** (0% Matrix, 除 subprocess.Popen 本身外). CLI 命令秒级
  返回, 不起 zenoh session, **网络挂掉时仍可用**.
- **五件事**: 发现 (list/show) / 创建 (create/link/install) / 启动 (run) /
  debug (status) / 清理 (kill/prune). 只做这些.
- **运行时智能面完全不塞**: 无 accept/deny (agent 在 channel 内通过
  `CommandUtil` 拿 mesh 自决), 无 mesh view (需要就写 debug cell), 无 attach
  (跨到 channel 层归 moss-repl).
- **深度 debug = 写 debug cell**: channel 交互 / mesh 观察 / 命令调用属
  "临时创建一个 cell 里面 Matrix.discover() 自己看" 的路径, 不塞 CLI.
- **target 只走 path** (不走 name): name 是运行时 (address / mesh view /
  agent) 的东西, path 是文件层原生货币. tab completion 天然 file completion.
- **stdout/stderr 不做文件记录**: 前台 inherit 到终端, 后台归 Jobs 层未来命题.

### §R-3. 待 dogfood 验证点

以下决策基于讨论时的最佳猜想, 实现完毕后跑 `.ai_partners/regressions/nodes-cli/`
(待建) baseline 验证:

- `show` 命令输出格式 (verbatim NODE.md + 目录列表) 是否操作员用着顺
- `run` launch debug 段字段是否覆盖 debug 所需, 有无漏项
- `prune` 默认统统杀在真孤儿场景是否好用, 是否需要 `--dry-run`
- `kill` grace = 3s 是否足够 (cell 关 zenoh session / cancel task 需时)
- `run` grace = 5s 是否足够
- `link` 命令的绝对路径策略在 cell 移动场景是否合理 (还是改相对更好)
- README stub 的骨架结构是否够用 (还是应该更简/更繁)

### §R-4. 后续 (超出本 workstream)

- Jobs 层设计 (stdout/stderr 后台记录的命题归宿) — 与 channel workstream 结合讨论
- nodes CLI regression set (`.ai_partners/regressions/nodes-cli/`) — dogfood 期建立
- 未来若增运行时可视化需求, 走 `moss-repl` 或 `debug cell` 路径, 不塞 CLI