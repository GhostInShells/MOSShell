# Nodes CLI Dogfood — 最小认知包 + 验证清单

> 给下一化身:
> 你用最小上下文进入 dogfood. 目标是跑 `moss nodes` 全命令, 记下哪个不 work,
> 哪个手感差, 哪些字段/参数/输出该改. 手感反馈比通过率更重要 — 因为通过率
> 只说"能跑", 手感说"值不值得留".
>
> 本轮 CLI (2026-07-17 opus-4.7-1m) 是**猜想版**: 讨论时保留了若干未定项,
> 你的 dogfood 就是把这些拍板. 敢改的都可以改, 只要留下改动理由.

## 最小认知包 (读这些就够开始)

**必读三份**:
1. **`plan.md`** (同目录) — 五件事定位 + 命令树 + 挡板 (防偏航)
2. **`FEATURE.md §Rewrite 2026-07-17`** (同目录) — 决策 delta + 判决理由
3. **`moss codex get-interface ghoshell_moss.core.blueprint.cell:NodeManifest`**
   — 结构真相 (NodeManifest 是所有 CLI 命令的第一性输入)

**遇到接口不明再查**:
- `moss codex get-source ghoshell_moss.core.blueprint.cell` (build_node_from_manifest / discover_this_node / enter_cell_lifecycle 的实现)
- `src/ghoshell_moss/cli/nodes_cli.py` (本轮 CLI 实现全文)
- `src/ghoshell_moss/matrix/matrix_impl.py:run_node` (matrix 侧咽喉参照, CLI run 与之共用 `NodeLauncher.from_manifest`)
- `src/ghoshell_moss/stubs/node/` (stub 骨架 — 每次 `moss nodes create` 出的东西)

**上下文包**: matrix-cell-governance workstream 的 §UU-4 六动词代数 + §UU-6 ledger 单写者
+ §AAA-3 步骤 10 判决 — 只作背景, 不是必读.

## 调试空间约定

`.moss/` 是本项目为**新版 cell 体系**准备的 workspace. dogfood 时不要动
真的 apps/, 全部 debug artifacts 建在 `.moss/cells/debug/` 下.

**两种布局, 你按需选**:

**布局 A** — 每个 case 一个 node (`.moss/cells/debug/{caseXX}/`):
```
.moss/cells/debug/
├── case-simplest/       # 最简 node: 只 provide 一个 ping channel
├── case-singleton/      # 测 singleton 冲突
├── case-crash/          # 主动 raise 测 exit code 传导
├── case-slow-shutdown/  # 关闭时 sleep(10) 测 grace 兜底
└── case-with-install/   # 带 INSTALL.md 测 install 命令
```

**布局 B** — 一个 debug node + 多个 debug 脚本 (`.moss/cells/debug/dev/`):
```
.moss/cells/debug/dev/
├── NODE.md
├── main.py              # 主 entrypoint
└── scripts/             # 一堆调试脚本, 直接写 matrix 实现验证
    ├── check_mesh.py
    ├── force_singleton_conflict.py
    └── ...
```

B 的优势: 一个 NODE.md 就能验各种 matrix 集成场景 (直接 `Matrix.discover()`
连上, 看 CellRuntimeInfo/mesh/subprocess 三个视角). 缺点: CLI 五件事的
verification case 不够独立.

**推荐**: A 布局跑 CLI 五件事 baseline; B 布局做深度 matrix 集成验证.

## 验证清单

### 发现 (list / show)

| Case | 命令 | Expected | 手感反馈项 |
|------|------|---------|-----------|
| L-01 | `moss --ai nodes list` | 列出 debug/ 下所有 case, 显示 name/path/description | 列输出信息量够不够? |
| L-02 | `moss --ai nodes list --include 'debug/*'` | 只显示 debug/ 下 nodes | fnmatch 语法直觉? |
| L-03 | `moss --ai nodes list --installed` | 只显示装了的 | filter 语义清楚? |
| S-01 | `moss --ai nodes show .moss/cells/debug/case-simplest/` | verbatim NODE.md + 目录列表 | verbatim 好用还是结构展开好用? |
| S-02 | `moss --ai nodes show .moss/cells/debug/case-simplest/NODE.md` | 同上 | 与 dir 模式一致? |
| S-03 | `moss --ai nodes show .moss/cells/debug/case-simplest/main.py` | from_script ad-hoc identity, 无 file 时提示如何 create | 提示是否明确? |

### 创建 (create / link / install)

| Case | 命令 | Expected | 手感反馈项 |
|------|------|---------|-----------|
| C-01 | `moss --ai nodes create case-new` | 在 `.moss/cells/case-new/` 生成 stub 全套 | 生成后的 next-step hint 够不够? |
| C-02 | `moss --ai nodes create case-new --group debug` | 在 `.moss/cells/debug/case-new/` 生成 | group 语义直觉? |
| C-03 | `moss --ai nodes create case-existing` (已存在) | 报错拒绝, 不覆盖 | 错误信息够清楚? |
| C-04 | 编辑生成的 NODE.md 手动填 `exec.command/args/singleton` | `moss nodes show` 能看到 verbatim | 字段结构够 code-as-prompt? |
| K-01 | `moss --ai nodes link .moss/cells/debug/ /path/to/external/script.py --name external-test --command python` | 在 debug/external-test/ 建 NODE.md 指向绝对路径 | link 参数顺序直觉? |
| K-02 | `moss --ai nodes link ... --command ''` (空 command) | 报错要求 --command 明确, 不自动检测 | 错误信息给了 --command 例子? |
| I-01 | 手建带 INSTALL.md 的 node, `moss --ai nodes install <path>` | touch .installed | install 是否要在 hint 里说"不跑 install 步骤"? |
| I-02 | `moss --ai nodes install <无 INSTALL.md 的 path>` | 提示 "无需 install" 而非报错 | 提示够温和? |

### 启动 (run) — 最重头戏

| Case | 命令 | Expected | 手感反馈项 |
|------|------|---------|-----------|
| R-01 | `moss --ai nodes run .moss/cells/debug/case-simplest/` | launch debug 段打印 → 子进程 stdout 直接可见 → Ctrl+C 干净退出 (exit 0) | launch debug 段字段完整? 有没有漏 debug 需要的? |
| R-02 | `moss --ai nodes run .moss/cells/debug/case-simplest/main.py` | 同上, 用脚本模式解析 | 从 script 起有没有丢 NODE.md 上下文? |
| R-03 | `moss --ai nodes run` (无参, cwd 认亲) | find_upward(cwd) → 找到最近 NODE.md 起 | 无参手感是否顺? 找不到时报错够清楚? |
| R-04 | `moss --ai nodes run <case> -- --debug --port 8000` | `-- ` 后 args append 到 argv | `-- ` 分隔是否符合直觉? extra args 在 launch debug 段可见? |
| R-05 | 两个 shell 同时 run 同一 singleton case | 第二个立即报 singleton 冲突, 显示第一个 address+pid | 错误信息够 code-as-prompt (说 "kill it first")? |
| R-06 | Ctrl+C 时子进程主动 sleep(10) | SIGTERM 后等 5s → SIGKILL 兜底 → CLI 退出 (exit code 非 0) | 5s grace 够不够? kill 兜底 log 清楚? |
| R-07 | 子进程主动 exit(3) | CLI 退出 code = 3 (bash pipe 友好) | exit code 传导对? |
| R-08 | 子进程 raise 异常 | CLI 退出 code = 1, stderr 可见 traceback | traceback 直接 inherit 可见? |
| R-09 | run 未 install 的 node | 报错指向 INSTALL.md + `moss nodes install` | 提示链完整? |

### debug (status)

| Case | 命令 | Expected | 手感反馈项 |
|------|------|---------|-----------|
| ST-01 | 无 running 时 `moss --ai nodes status` | 提示 "No runtime entries found" | 空态提示够温和? |
| ST-02 | 一个在跑时 `moss --ai nodes status` | 列表显示 address/name/role/pid/state/singleton | 列信息量够操作员用? |
| ST-03 | `moss --ai nodes status <address>` | detail 全字段 (含 spawn_cwd, ledger, description reverse-lookup) | 字段有没有多/少? |
| ST-04 | `moss --ai nodes status <partial-address>` | endswith 匹配 (uid 尾段) | 部分匹配语义直觉? |
| ST-05 | kill -9 子进程后 `moss --ai nodes status <addr>` | state = "stale" (ledger 残留, pid 不存活) | stale 语义清楚? |

### 清理 (kill / prune)

| Case | 命令 | Expected | 手感反馈项 |
|------|------|---------|-----------|
| KL-01 | `moss --ai nodes kill <address>` (正常 running) | SIGTERM → 3s → 干净退 | 3s 是否够? |
| KL-02 | `moss --ai nodes kill <address> --force` | 立即 SIGKILL | force 语义直觉? |
| KL-03 | `moss --ai nodes kill <非 running address>` | 找不到 entry 报错 | 报错够清楚? |
| KL-04 | kill 后 `status` | ledger 已清 | 清理动作是否幂等? |
| PR-01 | 有 alive + 有 stale 时 `moss --ai nodes prune` | 全清 (含 alive), 报告 killed X removed Y | 默认统统杀是否 alarming (需要 dry-run)? |
| PR-02 | `moss --ai nodes prune --keep-alive` | 只清 stale | 参数名直觉? |
| PR-03 | `moss --ai nodes prune --force` | 立即 SIGKILL alive orphans | 与 kill --force 一致? |
| PR-04 | 无 entry 时 `prune` | 提示 "No runtime entries to prune" | 空态提示? |

### 深度集成 (布局 B, 可选)

在 `.moss/cells/debug/dev/` 里写 debug node, 里面用 Matrix.discover() 做:

| Case | 场景 | Expected |
|------|------|---------|
| M-01 | debug node 起来后 `matrix.this` | Cell 身份 = 父 CLI 通过 `MOSS_CELL_ADDRESS` 注入的那个 |
| M-02 | debug node 内 `matrix.mesh().view()` | 看到自己 presence, 看到 host presence (如果有 host 在跑) |
| M-03 | debug node 内 `matrix.processes.execute(...)` | 起孙子进程, ctrl+C 传导整链 |
| M-04 | debug node crash → CLI 侧 stale ledger → 另一个 shell prune 清 | CLI 与 matrix 侧 ledger 单写者语义一致 |

## 手感反馈填空 (跑完写)

以下决策是本轮猜想版, 需要你实测拍板:

- [ ] **`show` 命令输出格式**: verbatim NODE.md + 目录列表 vs 结构化字段展开. 哪个用着爽?
- [ ] **`run` launch debug 段字段**: 缺什么? 多什么? 顺序对不对?
- [ ] **`prune` 默认统统杀**: 是否需要 `--dry-run` (展示会杀什么再确认)?
- [ ] **`kill` grace = 3s / `run` grace = 5s**: 够不够? 太长?
- [ ] **`link` 绝对路径策略**: 脚本移动就断. 需要改相对路径? 还是保持绝对+接受断链?
- [ ] **README stub 骨架**: 太简/够用/太繁?
- [ ] **NODE.md stub `exec:` 结构**: 与 pydantic 1:1 好懂? 还是想要 sugar?
- [ ] **`--include/--exclude` fnmatch pattern 匹配对象**: 是 project-relative 路径? 是 name? 明确写在 help 里?
- [ ] **`create --group` vs 直接手建 `.moss/cells/x/y/z/NODE.md`**: --group 有没有实用价值? 是否只该允许一层?
- [ ] **status 详情 `endswith` 部分匹配**: 是否会误匹配 (uid 冲突)? 要不要改前缀匹配?
- [ ] **launch debug 段的 `MOSS_*` 环境变量清单**: 需不需要遮罩敏感值 (project_id 等)?

## 报告方式

推荐两种:

1. **日记形式**: 同目录建 `dogfood-round-1.md`, 按 case 走一遍留意评, 手感反馈填空.
2. **regression baseline**: 在 `.ai_partners/regressions/nodes-cli/` 建 REGRESSION.md
   + `baselines/YYYY-MM-DD_v1.md` (参照 `.ai_partners/regressions/ghost-runtime/`
   的形式). 每 case 抓 `moss --ai` 输出快照, diff 友好.

第 1 种适合"手感"反馈, 第 2 种适合建立可 diff 的 CI 基线. dogfood 期先做第 1 种,
稳定后再抬升到第 2 种.

## 挡板 (dogfood 时不要越界)

以下决策**已经碰死**, dogfood 时如果诱惑修改, 先 stop 找人类:

- **`run:` frontmatter 糖不要复活**: NODE.md 与 `NodeManifest` pydantic 1:1,
  `exec: {command, args, env}` 直书. 想加糖 = 加融合, 违反 §TT-1.
- **`specification` 命令不要复活**: 探索路径指向 codex/blueprint/ctml/howtos.
  想集中 hint = 誊抄别处内容.
- **`link` / `run` 的自动检测不要加**: 按扩展名猜 command 是 WW-2 判决的死路.
- **CLI 不要引 matrix 生命周期**: 除 `subprocess.Popen` 本身外, 100% Project 层.
  想 join mesh view = 越到运行时智能面.
- **无 name 反查 target**: name 是运行时东西, path 是文件层原生货币.
- **kill/prune 默认无防御**: 维护动作正确姿态. 加防御 = 加 `--` 参数.
