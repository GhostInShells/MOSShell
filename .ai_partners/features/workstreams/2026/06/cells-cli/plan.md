# Nodes CLI 开工计划

> 依赖 matrix-cell-governance §AAA-3 步骤 10 判决 ("cli 我宁愿重做").
> 本 plan 是 2026-07-17 opus 1m + 人类架构师讨论收敛记录, 取代
> 06-28 版 FEATURE.md 的过时 API 引用 (`CellRegistry`/`spawn_cell`/
> `local_runtime_cells` 等在新抽象中不存在).

## 定位钉子 (先钉边界, 后谈实现)

**CLI = cell 开发生命周期的地面站 (操作员/维护动作面)**. 严守边界:

- 100% Project 层. 除 `subprocess.Popen` 本身外, 0% Matrix.
- 五件事: 发现 / 创建 / 启动 / debug / 清理.
- 运行时智能面 (agent 决策) 完全不塞: 无 accept/deny/mesh view/attach.
  - 这些是 agent 在 channel 内通过 `CommandUtil` 拿 mesh 自己决策, 不是 CLI 命令.
- 深度 debug (channel 交互/命令调用) 通过 "写一个 debug cell 自己 Matrix.discover"
  完成, 不塞 CLI. 这是 §UU-11 telos "身体长出新器官" 的自然延伸.

## T0. 命名统一 (cells → nodes)

**判据**: 抽象层已 100% node 化 (`NodeManager` / `NodeManifest` / `NODE.md` /
`matrix.run_node` / `project.nodes`), CLI 层跟上, 消除 code as prompt 断裂.

- 命令组: `moss cells` → `moss nodes`
- 文件: `src/ghoshell_moss/cli/cells_cli.py` → `nodes_cli.py`
- 引用点: `src/ghoshell_moss/cli/main.py` app.add_typer 名字
- stub 目录: `src/ghoshell_moss/stubs/cell/` → `stubs/node/`
- specification 文本里 "cell" → "node" (保留 Cell 作为网络实体统称的表述)

## T1. Stub 重做 (最短路径先发)

**判据**: `moss nodes create foo` 出的是符合当前抽象的 NODE.md, 能直接 `run`.

- `stubs/node/NODE.md`: 与 `NodeManifest` pydantic **1:1** —
  `exec: {command, args, env}` 直书, **不用 `run:` 糖** (fable 版 WW-3 引入的糖
  层与 pydantic 结构不 1:1, 本轮讨论碰死删除). `singleton: true` 默认.
- `stubs/node/main.py`: `Matrix.discover().run(main)`. body 极简, 只留探索路径
  指针 (指向 codex/channel_builder/ctml read), 不誊写内容.
- `stubs/node/README.md`: 极简骨架 (~10 行), 定位 = 开发时给协作者读的
  CLAUDE.md 视角 (运行时模型不读). 各 section 空白让用户自己填.
- `stubs/node/INSTALL.md`: 默认放 (存在 = 需装, 用户不需要就删), 带 hint 提示删除.
- `stubs/node/.gitignore`: 每行一注释解释为啥, 教使用者需要 gitignore 啥.
  最小清单: `.installed` / `__pycache__/` / `.venv/` / `*.log` / `runtime/logs/`.
  **不 blanket ignore `runtime/`** (cell 可能有自持意图状态).

## T2. 命令树 (五件事)

```
moss nodes
├── list                          # 发现: project.nodes.list_nodes()
├── show <path>                   # 发现: verbatim NODE.md + 目录列表 (保留文件真相)
│
├── create <name> [--group G]     # 创建: 从 stub 生成 {workspace}/cells/[G/]{name}/
├── link <workspace_dir> <script> # 创建: A workspace + B 脚本, 绝对路径, 无自动检测
│                                 #      --command 必填 (WW-2 判决: 无自动检测)
├── install <path>                # 创建: touch {node_dir}/.installed
│
├── run <path> [-- args...]       # 启动: 前台阻塞子进程, CLI 是 owner
│
├── status [address]              # debug: 无参 ps 列表; 带参 单 CellRuntimeInfo 详情
│
├── kill <address> [--force]      # 清理: SIGTERM+3s→SIGKILL, --force 立即 SIGKILL
└── prune [--keep-alive] [--force] # 清理: 孤儿 killer, 默认统统杀
```

**无 `specification` 命令** — node 开发路径已有更好的信源
(`moss codex get-interface`/`codex blueprint`/`ctml read`/`howtos`). specification
只会是"上述信源目录", 誊抄违反 "指向信源不誊抄" 纪律. 各命令的 hint 只指信源不誊写.

**Target = path only**, 无 name 反查. list 里若展示 name 是给操作员看的信息,
不作 CLI 输入货币. `list --include/--exclude` 用 fnmatch 走路径匹配.

## T3. run 咽喉机制 (讨论最深, 独立小节)

### 分工

| 维度 | CLI run (调试路径) | matrix.run_node (运行时) |
|---|---|---|
| owner | CLI 进程 (同步阻塞) | 父 matrix (async, 后台) |
| spawn API | `subprocess.Popen` | `Subprocesses.execute` (async) |
| lifecycle | signal handler + wait + killpg | on_exit callback + reclaim polling |
| stdout/stderr | inherit 到终端 (直接看) | Subprocesses 内部策略 |

**共享**: `NodeLauncher.from_manifest(env, manifest)` (blueprint 层 spawn
preparation). 两条路径同一个 preparation entry, 差异只在 owner + subprocess API.

### Target 解析 (path only, 三合一)

- 路径是目录 → `NodeManifest.read_from_directory(path)`
- 路径是 `NODE.md` → `NodeManifest.read_from_file(path)`
- 路径是 `.py` 脚本 → `NodeManifest.from_script(path)` (upward 认亲 → 找不到降级 ad-hoc)
- 无参 → `NodeManifest.find_upward(Path.cwd())` (裸调试入口)

**裸 script 不 auto-generate NODE.md**: 副作用严重, from_script 的 ad-hoc 降级
已经能跑. `moss nodes register <file>` 是显式创建入口, 分工清晰 (run 只跑,
create/register 才写文件). 错误信息里明确写出 "script/xxx 是临时身份, 每次不同 uid;
稳定引用请先 register" (code as prompt).

### 启动前 debug 段 (spawn 前先打印)

```
[run] Starting node cell
  address:   node/my-cell/AbCd1234
  cwd:       /workspace/.moss/runtime/cells/node_my_cell_AbCd1234/
  argv:      /path/to/python main.py --debug
  ledger:    .moss/runtime/cells/node_my_cell_AbCd1234.json
  singleton: true (holding lock 'my_cell')
  env:
    MOSS_WORKSPACE:            /workspace/.moss
    MOSS_PROJECT_DIR:          /workspace
    MOSS_PROJECT_ID:           abc-123-...
    MOSS_MODE_NAME:            dev
    MOSS_NETWORK:              local
    MOSS_NETWORK_SCOPE:        default
    MOSS_CELL_ADDRESS:         node/my-cell/AbCd1234
    MOSS_PARENT_CELL_ADDRESS:  (none, CLI is not a cell)
--- child stdout/stderr below ---
```

`--ai` 模式一样打印 (纯文本, 无颜色). 操作员/agent 一眼看清 "谁, 起在哪,
用啥环境, 写哪份 ledger".

### 骨架 (对齐 blueprint 心智)

```python
def run(target: str, args: list[str]):
    project = Project.discover()
    env = project.env
    manifest = resolve_target(project, target)  # 三合一, path only
    if not manifest.installed:
        raise <指向 INSTALL.md, 错误即 prompt>

    launcher = NodeLauncher.from_manifest(env, manifest)
    # launcher.runtime.cell.address 已生成, launcher.env 已含 MOSS_CELL_ADDRESS
    # extra args append 到 launcher.run 尾
    launcher.run.extend(args)

    # singleton 查重 (与 matrix.run_node 同源: 遍历 project.cell_runtimes)
    if launcher.runtime.cell.singleton:
        for existing in project.cell_runtimes():
            if existing.is_alive() and existing.cell.fullname == launcher.runtime.cell.fullname:
                raise DuplicatedError(<clear 信息>)

    spawn_cwd = env.cell_runtimes_dir / normalize(launcher.runtime.address)
    spawn_cwd.mkdir(parents=True, exist_ok=True)

    _print_launch_debug(launcher, spawn_cwd, env)   # T3 debug 段

    with contextlib.ExitStack() as stack:
        if launcher.runtime.cell.singleton:
            stack.enter_context(env.workspace.lock(launcher.runtime.locker_name()))

        proc = subprocess.Popen(
            launcher.run,
            cwd=spawn_cwd,
            env=launcher.env,
            start_new_session=True,
            # stdout/stderr = inherit (默认)
        )
        launcher.runtime.pid = proc.pid
        launcher.runtime.pgid = os.getpgid(proc.pid)
        launcher.runtime.write_to_runtime_dir(env.cell_runtimes_dir)

        try:
            signal.signal(signal.SIGINT, lambda *_: proc.send_signal(signal.SIGTERM))
            signal.signal(signal.SIGTERM, lambda *_: proc.send_signal(signal.SIGTERM))
            exit_code = proc.wait()
        finally:
            if proc.poll() is None:
                # 5s hardcode 优雅期 (父 Ctrl+C 后给子 async cleanup 窗口)
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    os.killpg(launcher.runtime.pgid, signal.SIGKILL)
                    proc.wait()
            launcher.runtime.delete_invalid(env.cell_runtimes_dir)

    sys.exit(proc.returncode)
```

**不复用 `enter_cell_lifecycle`**: 它是子进程 own 自己 lifecycle 的 helper
(内含 host 分支 clear_cell_runtimes, 是子进程本人才有的语义). CLI 是启动者, 手工
写 ledger + try/finally 删 ledger, 与 matrix.run_node 同款结构.

### extra args 传递

`moss nodes run <target> [-- <args>...]`. Unix 传统, `--` 分割 CLI flag 与
子进程 argv. Typer:
```python
context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
```
`--` 后 args 直接 append 到 `launcher.run` 尾部 (声明的 ExecSpec.args 是默认,
本次追加是覆写/加)。

## T4. Status 展开 (dogfood 后定)

初版默认 project-level join:
- CellRuntimeInfo 主体 (address / pid / pgid / start_time / is_alive)
- 关联 NodeManifest.description (从 project.nodes 反查, 让操作员知道 "这是啥")
- spawn cwd 推导路径 (`env.cell_runtimes_dir / normalize(address)`, bash 兜底入口)
- singleton 锁状态 (若持有, 标 "locked")

字段最终形状**待 dogfood 定** — 第一次跑起来看操作员视角哪些字段真的用得上,
再收敛。无 `--mesh` (过度设计, 需要 mesh 视角就写 debug cell)。

## T5. Kill / Prune 语义

### kill
- 默认: `SIGTERM → wait 3s → SIGKILL` 兜底 (给 cell 一点 async cleanup 机会)
- `--force`: 立即 `SIGKILL` (windows "结束进程" vs "强制结束进程" 同构)
- 无 `start_time` 防御 (维护动作默认无防御, 参考进程管理器纪律)

### prune (孤儿 killer)
- 默认: **统统杀** (`--ai` 无交互). 孤儿 = 有 ledger + 有 pid + 父不再治理,
  会锁 singleton, 必须能杀活的.
- 保守 (人类交互模式): 发现 alive 时 dialog 确认.
- `--keep-alive`: 只删死的 (`!is_alive`), 保留活的.

共享 helper: `graceful_terminate(pgid, grace_seconds)`. kill/run 关闭传导/prune
统一走一条路, 差异只在 grace 值.

## T6. 挡板 (防偏航)

- **`run:` 糖彻底删**: NODE.md 与 `NodeManifest` pydantic 1:1. `exec: {command,
  args, env}` 直书. 未来任何模型看到 `run:` sugar 冲动都要过融合检验.
- **无 `specification` 命令**: 探索路径指向 codex/channel_builder/ctml/howtos,
  不誊抄. specification 若做只会是目录.
- **无 name 反查**: target 只 path. list 输出显示 name 是给操作员看的信息,
  不作 CLI 输入货币.
- **无 stdout/stderr 文件记录**: 前台 inherit 到终端, 后台 (matrix 拉起) 归
  Jobs 层命题 (等 channel 那轮讨论清楚再动).
- **无 mesh 视图 / accept / deny / attach**: 运行时智能面, agent 在 channel
  内自决. CLI 越界即漂移.
- **无自动检测**: `link` 的 `--command` 必填, 拒绝按扩展名猜 (`.py` → python,
  `.sh` → bash). WW-2 判决 "自动检测是死路", 隐式检测就是 fable 版 v1 六坑
  重演的温床.
- **五类 cell 分类等 stale 信息一点都不留**: 未来化身看到会照单全收. 老 CLI
  里 standalone/project/isolated/script/remote 分类删干净.
- **`kill` / `prune` 默认无防御**: 维护动作正确姿态. 需要防御走 `--force` /
  `--keep-alive` opt-in.
- **status 无 `--mesh`**: 过度设计. 需要 mesh 视角 = 写 debug cell.

## T7. 编码顺序 (小 PR 纪律)

拓扑, 非严格时序:

1. **T0 + T1 首发** (stub node 化 + 命名统一): 最短路径, 无副作用,
   `moss nodes create foo && moss nodes run foo` 冒烟通.
2. **T3 run 咽喉**: 独立小 PR, 与 T2 其余命令解耦. 编码同时对齐启动前 debug 段.
3. **T2 其余命令** (list / show / create / register / install): 纯 project 层,
   无副作用, 可并行发.
4. **T4 status + T5 kill/prune**: T3 跑通后, 有真实 CellRuntimeInfo 可读才有意义.
5. **regression set 建立** (`.ai_partners/regressions/nodes-cli/`): 五件事作
   P0 case, 每命令一 test case, baseline 抓 `moss --ai nodes ...` 输出快照
   (--ai 模式纯文本, diff 友好).

## T8. cells-cli FEATURE.md 处置

06-28 版 FEATURE.md API 命名 (`CellRegistry`/`spawn_cell`/`local_runtime_cells`)
在新抽象中不存在. 处置选项:

- **A**: 本 plan.md 落地后, FEATURE.md rewrite 为决策摘要 (匹配当前抽象),
  历史留 git log.
- **B**: FEATURE.md 加 `## §Rewrite 2026-07-17` 段追加, 明示旧 API 段作废,
  以本 plan 为准.

倾向 **B** (轨迹保留). rewrite 是下一化身的活, 不阻塞开工.

## 未决 (dogfood 后定)

- Status 字段最终形状 (T4).
- 人类模式 prune / kill --safe 的 dialog 具体形状 (`--ai` 已定, 交互模式待定).
- Kill 是否显示子进程 exit code (若能 wait 到).
- register 生成的 NODE.md 是否要 `run:` 默认值 (脚本入口), 还是留给用户填.
