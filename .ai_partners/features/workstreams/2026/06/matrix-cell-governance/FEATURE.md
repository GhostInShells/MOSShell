---
title: Matrix Cell Governance
status: in-progress
priority: P0
created: 2026-06-09
updated: 2026-06-28
depends:
  - cell-discovery-refactor
  - cell-session-bootstrap
milestone: 0.1.0
description: >-
  Matrix cell 体系治理总任务。从 circusd 死胡同到 node 体系：
  重新定义 cell 类型、发现模型、最小依赖闭包。Node 取代 app/script，
  fractal 回归 cell 身份，host 成为拓扑聚合者。环境变量三个变一个。
  进程管理三件套：start_new_session + pipe fencing + polling。
  不碰现有 apps 代码，先建 parallel node 线。
status_note: >-
  2026-06-28 claude-opus-4-7 + 人类架构师讨论会话:
  (1) 补完前任未显式记录的 L0→L3 跃迁认知 (§MM) — matrix 成为最小通讯依赖, cells 真相源上升到 network.
  (2) 二元真相显式承认 (§NN) — status 变更不广播, live_cells 是延迟视图. 选 (c) 不加回 pub/sub.
  (3) 三重身份问题 framing (§OO) — type 字段一人干三份活的概念漂移定位, 字段拆解候选方案. 核心待决, 本轮未敲定.
  (4) cell type 升格 / 多 network 升格的 beta1 保守路径 (§PP).
  (5) channel proxy 根 channel 数量两个候选 (§QQ).
  (6) FEATURE.md 滞后修正 (§RR) — §L bridge_address 二分, §N HostCellNetwork ABC 分离, 均已合并/不做.
  (7) §SS 开工契约 — 三重身份 + log 三处零漂移 + 自动 proxy + cells/network 两个 channel + Matrix 启动流程. 决策全部收敛, 可开工.
  下一实例认知重建支点: 读本文 §SS (含 §MM-§RR 所有上文).
---

# Matrix Cell Governance

> 人类做代码架构，模型 review。此 feature 记录动机、共识和推进方法。

## Motivation

当前 MOSS 的 cell 体系有三套互相竞争的机制：circusd 进程守护（AppStore）、
zenoh queryable 发现（cell discovery）、以及裸 asyncio subprocess（ManagedProcess）。
加之 cell 类型混乱——app/script/fractal/host 的边界从未被严格定义——
导致开发者在"如何启动一个 cell"这件事上各自为政。

circusd 被设计为独立系统守护进程，它的核心能力（重启、监控、web dashboard）
面向部署场景，而非 per-session 的进程图。把它塞进 host 进程的子进程位置，
造成双层通讯（host → ZMQ → circusd → watcher → 子进程）、
孤儿泄漏（host SIGKILL 后 circusd 不知道宿主已死）、
以及两套监控体系（circus status vs zenoh queryable）。

**真命题**：MOSS 是通讯总线，不是进程管理器。Cell 的"存活"由网络查询决定，
不由本地守护进程决定。需要一套统一的 cell 治理框架：身份定义、发现机制、
生命周期契约、最小依赖闭包。

## 共识结论

### Cell 类型的重新定义

旧四类 → 新三类：

- **host** — 保留。由 MODE.md 定义，是 session 入口。负责聚合拓扑、提供 queryable。
- **node** — 新增。worker cell。App 和 script 归一为此类型。
  被 host spawn → 角色是 app（host 管理生命周期）。
  外部进程 → 角色是 node（外部管理生命周期）。
  同一份 CELL.md / NODE.md 描述，启动者决定角色。
- **fractal** — 保留为 cell 类型。不是"不是 cell"——它是拥有 address、
  可在 Matrix 总线上被寻址的参与者。只是自己不提供 channel，
  由 host 通过 FractalHub 注入到 queryable。

`CellType.app` 消失。App 是 node 被 host spawn 后的运行时角色，不是静态类型。
`CellType.script` 消失。被 node 取代。

### Cell 的唯一定义

拥有 address、可在 Matrix 总线上被寻址的参与者。
不问"你是什么类型"，只问"你的 address 是什么"。

### 发现模型

host 是拓扑聚合者。三种路径注入 cell 到 queryable：
1. node 自宣告（自己的 queryable）
2. host spawn（host 自然知道它的存在）
3. fractal peer（FractalHub 注入）

在 host 的 queryable 里，所有 cell 扁平化为 `address → Cell`。
观察者（UI 面板、REPL）看到统一的网络拓扑。

### 两轴状态模型

活跃轴和可用轴正交：

```
活跃轴（liveness）           可用轴（availability）
──────────────────          ──────────────────────
在线 / 离线                  已安装 / 未安装 / 停用
Zenoh queryable 判定         文件系统状态
"进程在跑吗"                 "模型该看见它吗"
```

可用轴的三个状态：

- **未安装** — 注册表里有记录（`moss nodes register` 写入），代码未在环境里。
  模型不可见。只是一段元数据。
- **已安装** — 代码就位。模型可见，可被 bringup。
- **停用** — 代码在但标记 disabled。文件在，cell 可能在跑，但模型不可见。
  `moss nodes disable/enable` 切换。

cell meta 文件（`cells/cell-{address}.json`）已经承载 PID 验活和僵尸杀灭——
它本质上是**离线声明**。加一个 `enabled: bool` 字段即可同时表达可用状态。
host 在构建 queryable 结果时按可用轴过滤。

当前 apps 的痛点：`moss apps list` 列出的是静态目录遍历，代码未安装的 app
模型也能看见甚至尝试调用。这是混淆了注册（存在）和可用（可调用）。

### 最小依赖闭包

node 加入 Matrix 网络的最小依赖：`ghoshell_moss[cell]` → `eclipse-zenoh>=1.8.0`。
core（blueprint + concepts + contracts + duplex）+ bridges/zenoh_bridge。
不需要 host、cli、speech、circus。

### 环境变量回归一个

`MOSS_WORKSPACE` 是唯一必须的环境变量。
`MOSS_SESSION_SCOPE` 默认 `"default"`，需要隔离时显式传。
`MOSS_CELL_ADDRESS` 从 CELL.md 推导或自动生成。
`MOSS_PARENT_PID` 由启动器自动注入，node 用 `== 0` 判断是否需要跟随生命周期。

### 进程生命周期

三件套（全覆盖正常退出和 SIGKILL）：
1. `start_new_session` + `killpg` — 优雅退出时父进程一刀切
2. Pipe fencing — 父进程 SIGKILL 时子进程零延迟自检
3. Polling（`_ensure_parent_process_exists`）— 已有的兜底，不留

不做进程编排（restart / health check / bringup）。那是 Channel 的能力，不是框架的职责。

### 启动 node 的三种方式

1. 纯 Python：`eval $(moss shell-init) && python my_cell.py`
2. moss 命令：`moss nodes run <target>`（目录/脚本/名称三种模式，见解析规则）
3. cell 间：`matrix.spawn(...)`（ProcessNursery 已实现）

### CELL.md 最小格式

```yaml
name: web-fetch
group: tools
description: "抓取网页内容并提取正文"
executable: python          # 默认启动器，缺省 = sys.executable
script: main.py             # 默认入口
arguments: ""               # 默认额外 CLI 参数
```

- `executable` 默认 `python`（`sys.executable`）。不做 `uv run` 等环境管理魔法——
  实际使用中 `uv run` 隐式约定太多、pep 737 不稳定。复杂环境需求由 `install.sh` 处理。
- type 字段不在 CELL.md 中。类型由启动者赋予。
- CELL.md 向上查找，行为同 MOSS.md——找到第一个就停。一个 cell 是一个闭合目录，不嵌套查找。

### moss nodes run 解析规则

`moss nodes run <target> [args...]` 是 cell 启动的统一入口。三种模式：

**目录模式** — `target` 是目录：
```
moss nodes run ./my-node/ --flag
```
1. 找 `./my-node/CELL.md` → 没有则报错
2. 读 CELL.md 的 `executable` + `script` 作为默认启动参数
3. `--flag` 透传给子进程

**脚本模式** — `target` 是文件：
```
moss nodes run ./my-node/scripts/debug.py --verbose
```
1. 判断文件类型：
   - `.py` → `executable = sys.executable`（当前 Python 解释器）
   - `.sh` → `executable = /bin/bash`
   - 其他（有 +x）→ `executable = 文件本身`
2. 从文件所在目录向上查找 CELL.md（同 MOSS.md 规则，找到第一个即停）：
   - 找到 → 读 CELL.md 元数据（name, group），CELL.md 所在目录设为 cwd
   - 找不到 → cell 身份退化为 `script/{uuid}`，无 name/group 元数据
3. 命令行参数覆盖 CELL.md 的 executable/script
4. `--verbose` 透传

**名称模式** — `target` 是裸名称：
```
moss nodes run web-fetch
```
1. 扫描 `nodes/` 目录树，匹配 CELL.md 中 `name` 字段
2. 找到 → 走目录模式（用 CELL.md 的 executable + script）
3. 找不到 → 报错

**一个 cell 多种启动方式**：
```
# 默认启动（名称查找）
moss nodes run web-fetch

# 默认启动（目录）
moss nodes run ./nodes/tools/web-fetch/

# 跑 scripts 里的调试脚本（覆盖 executable/script）
moss nodes run ./nodes/tools/web-fetch/scripts/debug.py --port 9999

# 跑 install（实际就是跑约定脚本）
moss nodes run ./nodes/tools/web-fetch/scripts/install.sh
```

`moss nodes install web-fetch` 本质是 `moss nodes run ./nodes/tools/web-fetch/scripts/install.sh`
+ exit code 检查 + 写 `runtime/nodes/{name}/state.json`。

### Skills 与 Meta Channel

Node 的 dev/debug 脚本不需要反射为 virtual channel。方案极简：

- meta channel 接受 `paths` 配置项，指向已安装 node 的 scripts 目录
- 扫描 `--help` 输出，聚合到 channel instruction 里
- ghost 看到 skill 列表，通过 `bash:exec`（AI Terminal 已提供）调用

这本质上是 `moss all-commands` 的 channel 化——meta channel 就是 CTML 语境下的
能力目录。无需 virtual channel、动态命令挂载。

## 迭代路径

1. **不碰 apps**：现有 App 体系保留，circusd 不去。bug fix 照常。
2. **建 node 并行线**：`CellType.node`、CELL.md、`moss nodes` CLI、注册表。← 当前
3. **Matrix 接口补齐**：Process Nursery、`matrix.spawn()`、pipe fencing。✅ done
4. **运行时可观测性**：Matrix 查询接口、TopicWindow 事件广播、运行时文件布局。← 设计完成
5. **环境变量契约文档化**：`moss shell-init`、最小启动示例。
6. **验证完毕后才讨论 apps 迁移**：当 node 机制覆盖了 apps 的所有场景。

## 实现记录

### 2026-06-09: ProcessNursery + Matrix.spawn() + pipe fencing

**scope**: 仅 ProcessNursery。不碰 CellType、不建 NODE.md、不改 apps。

**设计收敛**:
- 方法名 `spawn()` 而非 `run_cell()`——它是进程 spawn 原语，是不是 cell 由调用方通过 `cell_address` 参数决定
- Nursery 不维护子进程列表——pipe fencing 让子进程在父进程 SIGKILL 时零延迟自检
- 正常退出路径：父进程主动发 SIGTERM 给子进程 pgid，等 timeout，未退的 SIGKILL
- pipe 由调用方创建，nursery 不持有 fd。`nursery_fd` 参数可选
- start_new_session 隔离进程组，防止 Ctrl+C 误杀子进程
- 不耦合注册表——注册表是运行时 debug/强杀工具，和 nursery 正交

**方法签名**:
```python
async def spawn(
    self,
    *args: str,
    cell_address: str | None = None,
    cwd: str | Path | None = None,
    extra_env: dict | None = None,
    nursery_fd: int | None = None,
) -> asyncio.subprocess.Process:
```

**文件**:
- `src/ghoshell_moss/host/nursery.py` — ProcessNursery 独立模块
- `src/ghoshell_moss/core/blueprint/matrix.py` — Matrix.spawn() 抽象方法
- `src/ghoshell_moss/host/matrix.py` — MatrixImpl 集成
- `tests/ghoshell_moss/host/test_nursery.py` — 单元 + 集成测试
- `tests/fixtures/nursery_child.py` — 测试用子进程脚本

## 2026-06-09 晚间设计会话：可观测性与 install 约定

> 人类架构师 + deepseek-v4-pro。三大设计域：可用状态解耦、install 约定、
> 运行时文件布局 + Matrix 接口 + TopicWindow 事件广播。

### 1. 可用状态与 NODE.md 解耦

**原则**：NODE.md 是源码声明（进 git），installed/enabled 是运维状态（gitignored）。
两者生命周期不同——NODE.md 随 commit 变，可用状态随 `moss nodes install/enable` 变。

**两个文件，两个写者**：

| 文件 | 写者 | 内容 |
|------|------|------|
| `runtime/nodes/{name}/state.json` | CLI (`moss nodes install/enable/disable`) | installed, enabled, installed_at, install_exit_code |
| `runtime/cells/cell-{address}.json` | cell 自身（运行时启动时写） | pid, session_id, session_scope, mode, ghost, role, started_at |

写者唯一原则：CLI 不写 cell meta，cell 进程不写 node state。路径通过 name/address 推导，
不需要注册表索引。

**`runtime/nodes/{name}/state.json` 格式**：
```json
{
  "name": "web-fetch",
  "group": "tools",
  "installed": true,
  "enabled": true,
  "installed_at": "2026-06-09T20:00:00Z",
  "install_exit_code": 0
}
```

**`runtime/cells/{address}/meta.json` 格式**：
```json
{
  "address": "node/tools/web-fetch",
  "pid": 12345,
  "parent_pid": 12344,
  "role": "node",
  "session_id": "01KT...",
  "session_scope": "default",
  "mode_name": "desktop",
  "ghost_name": "echo",
  "started_at": "2026-06-09T20:00:00Z",
  "python_version": "3.12.4"
}
```

`role` 是启动者赋予的运行时角色——同一个 node 代码，被 host spawn 时 role=app，
被外部启动时 role=node。`pid_alive` 在查询时通过 `os.kill(pid, 0)` 实时判定，不存文件。

### 2. Install 作为约定，不做重实现

**原则**：框架只提供最薄的执行封装。每个 node 的依赖复杂度差异巨大（纯 Python / 系统库 /
编译工具链），框架不可能穷举。约定脚本入口，让 node 作者自己表达。

**目录约定**：
```
nodes/{group}/{name}/
  NODE.md
  scripts/
    install.sh         # 约定：运行我即安装
    check.sh           # 约定：exit 0 = 已安装，可作为 prereq 检查
    ...                # 其余 skills
```

**CLI 做的事极薄**：
1. 找 `nodes/{group}/{name}/scripts/install.sh`
2. 跑它
3. exit 0 → 写 `runtime/nodes/{name}/state.json` → `{"installed": true}`
4. exit != 0 → 报失败，不写标记

**模型驱动路径**：Ghost 可以直接 `bash:exec nodes/tools/web-fetch/scripts/install.sh`，
也可以走 CLI `moss nodes install web-fetch`。两条路径等价。Ghost 不需要学"安装协议"——
它就是跑 shell 脚本。check.sh 和 install.sh 的关系 ghost 能自己推理（先 check，失败则 install）。

**Skills 收敛**：install.sh 和 check.sh 是特殊约定的脚本名，其余 scripts/ 下的都是普通 skill。
meta channel 扫描 `--help` 的逻辑不变，路径改为 `nodes/{name}/scripts/`。

**`moss nodes enable/disable`** 只翻转 `runtime/nodes/{name}/state.json` 的 `enabled` 字段。
和 `install` 正交——可以先 disable 再 update 再 enable。

### 3. 运行时文件布局

**目录聚合**：
```
runtime/cells/{address}/
  meta.json         # cell 自身写（见上）
  stdout.log        # ProcessNursery 重定向
  stderr.log        # ProcessNursery 重定向
  moss.log          # LoggerProvider handler
```

**路径推导，不存路径**：知道 `MOSS_WORKSPACE` + `MOSS_CELL_ADDRESS` 就能推导所有文件路径。
meta.json 不需要存 paths 字段。

**ProcessNursery stdout/stderr 策略**：
- Nursery 内部从 `cell_address` 推导日志路径
- 默认：`open(path, 'a')` 传 fd 给 `create_subprocess_exec`，同步写文件
- 调用方可覆盖 `stdout` / `stderr` 参数（REPL 场景打到终端）
- 不需要 asyncio pipe 包装，不需要独立 watchdog task

**LoggerProvider 时序**：
- 依赖隐式契约：`MOSS_CELL_ADDRESS` 必须在 LoggerProvider 初始化前设置
- 当前通过环境变量传入，时序正确
- `shell-init` 将此约定文档化

**全部使用 Python 标准库，零平台差异**：
- 读日志 → `open()` + `readlines()` + offset/limit 分页（不调 `tail`）
- 验活 → `os.kill(pid, 0)`
- 扫描 cell → `os.listdir()`
- 路径推导 → `Path` 拼接

### 4. Matrix 接口

三种角色共享同一套 Matrix Python 方法：

| 角色 | 读 cell 运行时 | 读 node 状态 | 改 node 状态 |
|------|---------------|-------------|-------------|
| Developer (CLI) | `moss runtime cells list` | `moss nodes list` | `moss nodes install/enable` |
| Ghost (CTML) | `matrix:list_cells` | `matrix:list_nodes` | `bash:exec nodes/x/scripts/install.sh` |
| UI/App (Matrix API) | `matrix.list_cells()` | `matrix.list_nodes()` | — |

**查询接口**（全部异步，全部文件 I/O）：

```python
class Matrix(ABC):

    # -- Cell 运行时查询 --
    async def list_cells(self) -> list[CellRuntimeInfo]: ...
    async def get_cell_info(self, address: str) -> CellRuntimeInfo: ...
    async def get_cell_output(
        self, address: str, stream: str = "stdout",
        offset: int = 0, limit: int = 200
    ) -> CellOutput: ...

    # -- Node 运维查询 --
    async def list_nodes(self) -> list[NodeInfo]: ...
    async def get_node_info(self, name: str) -> NodeInfo: ...

    # -- Matrix 自身 --
    async def get_matrix_info(self) -> MatrixInfo: ...
```

**数据类**：

```python
@dataclass
class CellRuntimeInfo:
    address: str
    role: str              # host | node | fractal
    pid: int
    parent_pid: int
    session_id: str
    session_scope: str
    mode_name: str
    ghost_name: str
    started_at: str
    pid_alive: bool         # 查询时实时判定
    has_stdout: bool
    has_stderr: bool
    has_log: bool

@dataclass
class CellOutput:
    address: str
    stream: str             # stdout | stderr | moss
    lines: list[str]
    offset: int
    total_lines: int
    has_more: bool          # ghost 判断"还有更多吗"

@dataclass
class NodeInfo:
    name: str
    group: str
    address: str            # 推导: node/{group}/{name}
    description: str        # 从 NODE.md 读
    installed: bool         # 从 runtime/nodes/{name}/state.json
    enabled: bool
    installed_at: str | None
    is_running: bool        # 推导: cells/ 里有此 address 且 pid 存活
    running_pid: int | None

@dataclass
class MatrixInfo:
    session_id: str
    session_scope: str
    cells_count: int
    cells: list[str]        # address 列表
    uptime_seconds: float
```

**get_cell_output 的 Python tail 实现**：
```python
def _read_tail(path: Path, limit: int, offset: int = 0) -> tuple[list[str], int, bool]:
    if not path.exists():
        return [], 0, False
    with open(path) as f:
        all_lines = f.readlines()
    total = len(all_lines)
    start = max(0, offset)
    end = min(total, start + limit)
    return [l.rstrip('\n') for l in all_lines[start:end]], end, end < total
```

后续可加 ring buffer 或 log rotate，现阶段直接读即可。

### 5. TopicWindow 全局事件广播

**原则**：Matrix 在关键生命周期节点向 well-known topic 发布事件。
Channel 通过 TopicWindow 订阅，看到运行时历史——不轮询、不读文件。

**三层信息索引**：

| 层 | 来源 | 访问方式 | 适合查什么 |
|------|------|------|---------|
| 实时事件流 | TopicWindow | `window.values()` | 最近发生了什么、现在谁活着 |
| 文件快照 | `cells/{addr}/meta.json` | `matrix.get_cell_info()` | 某个 cell 的静态身份 |
| 日志内容 | `cells/{addr}/*.log` | `matrix.get_cell_output()` | 具体输出内容 |

TopicWindow 和文件互补——"面"查询 vs "点"查询。Ghost 先 `window.values()` 看全局，
再按需 `get_cell_output()` 深入具体 cell。

**事件模型**：
```python
class RuntimeEvent(TopicModel):
    event: str              # cell_started | cell_stopped | cell_died | session_started | session_stopping
    address: str
    role: str
    pid: int
    timestamp: str
    detail: dict            # exit_code, reason, etc.
```

**发布点**：
- `cell_started` — Nursery.spawn() 子进程成功启动后
- `cell_stopped` — 正常退出（exit code 0）
- `cell_died` — 异常退出（exit code != 0 或被 SIGKILL）
- `session_started` — Matrix session 初始化完成
- `session_stopping` — Matrix 开始 shutdown

**Topic**: `moss/runtime/events`

**Matrix 接口**：
```python
def create_runtime_window(self, max_size: int = 200) -> TopicWindow[RuntimeEvent]:
    """创建观察 Matrix 运行时事件的滑动窗口."""
    ...
```

**Channel 使用**：
```python
window = matrix.create_runtime_window(max_size=200)
window.values()           # 最近的事件列表
window.on_change(cb)      # 实时回调
```

### 6. moss nodes run 解析规则

> 2026-06-10 人类架构师 + deepseek-v4-pro。追加 `moss nodes run` 的完整解析状态机。

**核心原则**：
- CELL.md 是 cell 的身份文件，统一替代 NODE.md 命名
- CELL.md 向上查找，行为同 MOSS.md——找到第一个就停，不嵌套
- `executable` 默认 `sys.executable`，不碰 `uv run`（一周实际使用验证坑太多，pep 737 不稳定，隐式约定过多）
- 找不到 CELL.md 时身份退化为 `script/{uuid}`，不拒绝运行

**解析状态机**（见上文"moss nodes run 解析规则"章节）。

**与 ProcessNursery.spawn() 的关系**：
- CLI 的 `moss nodes run` 解析完 executable + script + args + cwd → 调用 `Nursery.spawn()`
- `spawn()` 不关心 CELL.md——它只接受 `*args` + `cell_address` + `cwd` + `extra_env`
- 解析逻辑在 CLI 层，spawn 逻辑在 Nursery 层，分工清晰

**与 install 约定的关系**：
- `moss nodes install` 本质是 `moss nodes run scripts/install.sh` + exit code 检查 + 写 state.json
- 两条路径底层都是同一个 spawn 调用

### 7. Matrix.spawn() JSON-line 支持与上下文透传

> 2026-06-10 人类架构师 + deepseek-v4-pro。基于 Playwright channel (module-eval-channel) 的实际验证，
> 发现当前 spawn 缺失 PIPE 支持，导致需要 JSON-line 通信的 channel 被迫绕开 Matrix 用裸 subprocess.Popen。

**痛点**：

Playwright channel 的 `eval_server.py` 子进程用 JSON-line 协议（一行请求、一行响应）与父进程通信。
这要求 `stdin=PIPE, stdout=PIPE`。但当前 Nursery.spawn() 不传 stdio 参数——子进程继承父进程终端，
无法做结构化通信。更关键的是，**裸 `subprocess.Popen` 的子进程拿不到 MOSS 上下文**——不知道
workspace、session、cell address，完全瞎的。

**解决方案：Nursery 加透传参数**

```python
# Nursery.spawn() — 三个新增参数，全部透传给 create_subprocess_exec
async def spawn(
    self,
    *args: str,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    nursery_fd: int | None = None,
    stdin: int | None = None,          # None = 继承父进程
    stdout: int | None = None,         # asyncio.subprocess.PIPE = 异步流
    stderr: int | None = None,         # int = 自定义 fd
) -> asyncio.subprocess.Process:
```

`asyncio.create_subprocess_exec` 原生支持这三个参数。Nursery 只透传，零额外逻辑。

**Matrix.spawn() 同样透传，上下文自动注入**：

```python
async def spawn(
    self,
    *args: str,
    cell_address: str | None = None,
    cwd: str | Path | None = None,
    extra_env: dict | None = None,
    nursery_fd: int | None = None,
    stdin: int | None = None,
    stdout: int | None = None,
    stderr: int | None = None,
) -> asyncio.subprocess.Process:
```

上下文注入（已有，不改）：`dump_moss_env(for_child_process=True)` 自动注入全部运行时上下文
（workspace, session_scope, session_id, mode, ghost, cell_address, parent_pid）。

**Playwright channel 改造后**：

```python
proc = await matrix.spawn(
    sys.executable, "-u", server_script,
    cell_address="node/browsers/playwright",
    stdin=asyncio.subprocess.PIPE,
    stdout=asyncio.subprocess.PIPE,
)
# proc.stdin → StreamWriter, proc.stdout → StreamReader
# 子进程自动拿到 MOSS_WORKSPACE, MOSS_SESSION_ID 等
jsonline = JsonLineProcess(proc)
```

**JSON-line 协议层（Nursery 之外）**：

```python
class JsonLineProcess:
    """JSON line protocol adapter. 一行一个 JSON."""
    def __init__(self, proc: asyncio.subprocess.Process): ...
    async def send(self, msg: dict) -> None: ...
    async def recv(self) -> dict: ...
    async def request(self, msg: dict, timeout: float = 30.0) -> dict: ...
```

JSON line 是协议，不是 spawn 的职责。Nursery 提供 pipe，JsonLineProcess 消费 pipe。

**关键约束**：
- stdout 走 PIPE 时，不进 Nursery 文件日志。两者互斥（默认模式 vs PIPE 模式），第一版不做 tee
- stderr 独立：stdout 走 PIPE 时，stderr 仍可走文件（默认）或 PIPE（调用方传）
- JSON-line 只用 stdout，stderr 是纯人类日志

### 更新后的 Open Questions

- fractal 连接到 host 的 channel proxy 语义：透明转发还是显式声明？
- `moss shell-init` 的完整契约文档化——env var 注入顺序、必须保证的时序
- Node script 的超时策略：install.sh 跑多久算失败？框架层定义默认 timeout 还是交给 node 作者声明？
- CELL.md 的 `executable` 字段在脚本模式下被命令行覆盖——还需要显式声明吗？保留的理由：它是 cell 作者的意图声明，目录模式和名称模式时作为默认值。
- Nursery 默认 stdout/stderr 落文件 vs 继承终端：二阶封装问题，人类架构师后续抽象

## 2026-06-15 协议层重新对齐：address 自由度、type 注册、worker 与 cell 分立

> 2026-06-15 人类架构师 + claude-opus-4-7。沿 06-09/06-10 设计闭环往下推一层，
> 解决 module_eval channel 暴露的 "address 自由度" 摩擦点，把协议层的承诺与
> 实现层的选择分清。

### 1. address 与 type 的协议位置

**Address 是 cell 在 Matrix 上的唯一标识**。今天的实际形态：

- `MOSS_CELL_ADDRESS` 是 free-form 字符串，Environment 不做结构校验
- 副作用发生在 Matrix instantiation 时点：`workspace.lock("moss_cell_{type}_{name}")`
  是真正的 address 唯一性保证；host 用 `moss_host_{scope}` 同形锁
- zenoh CellDiscovery 的 `announce_cell` 紧随 flock 之后；同 address 的第二个
  Matrix 实例在 flock 失败、永远到不了 zenoh 声明
- ScopeMeta / CellMeta 文件层是 PID 验尸的事后清扫，不参与 lock

**Type 是 first-class 协议字段，但不是封闭枚举**。设计闭环阶段提的 "host/node/fractal
三类" 是 framework 保留集，不是 type 全集。type 命名空间开放，由 owner channel 在
运行时声明所有权。

### 2. Open type namespace with owner-channel registration

Type 的合法性由 "有 owner channel 正在 announce 我管这个 type" 这件事证实。

- **保留 type**：`host` / `node` / `fractal` —— framework 占用，由对应内置 owner 管理；
  `worker` —— framework 占用，**无 owner**（见 §4 方案 C，用于无语义兜底）
- **开放 type**：任何 channel 可以通过 `matrix.announce_type(type, owner_address)` 声明所有权
- **announce-time 校验**：spawn_cell 时 type 不在 registered 集合则拒绝；address 必须
  `{type}/{name}` 形式
- **owner 退场**：type 注册随 owner channel 的 queryable 消失而失效

形态约束：type 是 flat 字符串，不允许 `a/b` 形态嵌套（与 address 内的 `/` 分隔语义解耦）。

### 3. spawn_worker / spawn_cell 二分 API

`Matrix.spawn` 演化为两个 API，表达调用者的语义意图，不是机制差异：

```python
matrix.spawn_cell(*args, type: str, name: str, ...)
    # 声明新身份。type 必须已注册。address = f"{type}/{name}"
    # 经 announce 校验 + flock，落 CellMeta

matrix.spawn_worker(*args, ...)
    # 无身份诉求的 OS 子进程
    # 自动注入 worker/{uuid} 作为 address —— 拒绝承诺语义，但不拒绝入网
```

API 形态上不允许调用者写字面字符串 `cell_address="module_eval/foo"`。type 与 name
拆开传，type 走 registry 强校验。

### 4. 方案 C：worker 入网不被拒，semantic 不被承诺

worker 与 cell 的分立不靠 policy 禁止（拒绝构造、剥离 env、防蠢 sentinel），靠
substrate 兜底：

- worker 拿到 `worker/{uuid}` 这种 framework 保留的 "无语义 address" —— uuid 唯一，
  flock 永不冲突；announce 走得通；bus 看得见；但 type registry 里没有 `worker` 的
  owner，所以 ghost 视角下找不到 owner channel，无法被打开为 channel proxy
- session_scope / workspace / mode / ghost_name 等元信息照常透传，worker 可以读
  scope-shared resource、观察 topic
- worker 不被承诺管理：framework 不重启、不健康检查、不暴露为 ghost 的可控制 surface
- 父 cell 自己负责 worker 进程的回收（OS 层 `await proc.wait()`、pipe fencing 已覆盖
  父死子死）

历史轨迹：此前的 `script/` `task/` 都在尝试同一件事 —— 给 "不该被当作 cell 但确实
存在的子进程" 一个安身之所。`worker/{uuid}` 是这条轨迹的形式化收口。

### 5. cell meta 是否记录 type 字段 —— 留给迭代

worker 每次 spawn 都会产生一个 CellMeta 文件。两个候选：

- **type 不写**：worker 的 meta 文件结构与 cell 一致，少一字段。日后无法区分
  worker 与 cell
- **type 写**：清晰区分，但高频 spawn worker 的 channel（例如批量计算）会让
  `runtime/cells/` 文件数量爆表

判断推迟到真实压力点出现：当真有 channel 产生 worker 数量上的痛点时，结合那时
workload 形态裁决。在此之前 meta 字段写法以最简实现为准。

### 6. 跨进程异步基底

MOSS 的跨进程异步通讯走 **process + Matrix 异步回调机制**。回调媒介（topic /
mindflow signal / 其他）由 channel 实现层自行选择，协议层不预设。

implication：channel 作者写长跑逻辑时，async 心智模型是 "起一个 OS 子进程 +
在 Matrix 上等回调"，而非 "启动 Python coroutine + register callback"。后者
是单进程实现细节。

### 7. Fractal 维度的命名空间

Fractal 的 channel 抽象本身是分形的（已有 unit test 验证）：本地
`apps.bodies_g1.arm:wave`，远端通过 fractal provide 后变为
`fractal.moss_xxx.apps.bodies_g1.arm:wave`，两层嵌套变为
`fractal.moss_xxx.fractal.moss_yyy.apps.bodies_g1.arm:wave`。

- **命名空间分形保留是协议正确性** —— type registry 在 fractal 下天然带前缀，
  `node` 与 `fractal.B.node` 永远不冲突
- **扁平化是 alias 命题** —— ghost UX 上希望短名字时，用 alias 表，不动 protocol 结构

### 8. operations channel pattern 在实现层的体现

每个 declared type 的发现 / 启动 / 生命周期归 owner channel。pattern 与现有
AppStoreChannel 同形：

| 角色 | 现有 | 新增 |
|---|---|---|
| 数据层 (Store) | AppStore | NodeStore |
| ghost 面向 (Channel) | AppStoreChannel | NodeStoreChannel |

NodeStoreChannel 承接 `nodes:run` / `nodes:list` / `nodes:install` / `nodes:enable` /
`nodes:disable`，通过 `get_virtual_children()` 将活的 node 暴露为 channel proxy，
通过 `get_context_messages()` 反映两轴状态。framework 不在 Matrix 层定义重启 /
健康检查 —— 这是 channel 的政策空间。

无需 framework 提供通用 `CellStore` protocol，因为不同 type 的发现机制本质异构
（CELL.md 文件遍历 / app.yml / fractal peer announce），强抽象会逼出最小公分母。

### 9. 协议层 vs 实现层

| 标 | 项目 |
|---|---|
| 协议 | type 是开放命名空间，必须 owner channel announce 才合法 |
| 协议 | framework 保留 `host` / `node` / `fractal` / `worker` 四个 type |
| 协议 | spawn_cell 在 Matrix bootstrap 时做 type registered 校验 + address 一致性校验，flock 兜底 |
| 协议 | spawn_worker 自动注入 `worker/{uuid}` address，session_scope / workspace 透传 |
| 协议 | 跨进程异步基底是 process + Matrix 异步回调机制 |
| 协议 | type registry 在 fractal 下天然带前缀，扁平化是 alias 而非协议 |
| 实现 | Per-type Store (NodeStore / AppStore 等)，由 owner channel 内部使用 |
| 实现 | ProcessNursery 的 pipe fencing + flock 是当下进程生命周期实现 |
| 实现 | CellMeta md5 哈希文件名 |
| 实现 | Environment 的 "default to host" fallback 形态 —— 后续独立步骤消除 |

### 10. 遗留的迭代标记

预决会过早闭合可能性的事，留给压力点出现后再裁：

- **type unregister 时 type 下 cell 的回收政策**：owner channel 退场（含异常退出）时，
  已经活着的同 type cell 如何处置 —— 自尽 / 转交 / 孤儿 / 框架强制 reap。等真实场景
  出现再决
- **worker cell meta type 字段是否记录**：见 §5
- **worker 对外通讯方式**：worker 是 hack 空间，MOSS 不承诺接口。父 cell 自行决定
  如何与 worker 通讯
- **Environment "default to host" 消除的具体形态**：sentinel 注入 vs entry point
  显式声明 vs 其他。独立步骤处理

### 推翻的设计与原因

这一轮推翻了几个 06-10 设计闭环阶段提的判断，记录原因供下一个实例参考：

- **CellType 5 类过渡**：推翻。理由：污染概念层心智模型，不消除迁移破坏面。改为新三类
  直接成立，旧 app/script 作为 deprecated alias 在实现层兼容
- **spawn 强制 cell_address**：推翻。理由：把 "加入 Matrix 网络" 的合同强加给进程
  原语层，破坏了 substrate 与 semantic 的层分。改为方案 C 的 `worker/{uuid}` 注入
- **role=app / role=node 写 cell meta**：推翻。理由：与 mode 体系的权限 / 资源语义
  重复。运行时角色由 origin_address 推导即可，不进 meta 字段
- **owner channel 退场时的 finalize 政策预决**：推翻。理由：过早闭合可能性，留待
  迭代逼出
- **address mandatory 与 worker env 防蠢拒绝**：推翻。理由：与 "Matrix 是总线，不审
  用意" 的设计方向不兼容；改为兜底注入 + 无语义承诺

### 与 Open Questions 的合并

原 06-10 Open Questions 中以下条目在本轮已经决：

- ~~fractal 连接到 host 的 channel proxy 语义~~ —— 命名空间分形保留即正确性，扁平化
  是 alias 命题
- ~~CELL.md `executable` 字段在脚本模式下被命令行覆盖还需要显式声明吗~~ —— 保留作
  默认值，命令行覆盖是正常优先级，无歧义
- 仍未决：`moss shell-init` 完整契约 / install.sh timeout 政策 / Nursery 默认
  stdout/stderr 处置 —— 三项与本次协议固熵正交，后续单独裁决

## 2026-06-15 续：实施纲领与下一个实例的认知重建支点

> 紧接 §1-§10 协议层重新对齐。本节面向下一个推进实例 —— 记录推进分工、待最终
> 拍板的决策、上下文恢复时的高优探索线索与 audit 指标。

### A. 推进分工

- 整体推进由人类架构师主导
- 拆 branch 治理 workstream，**branch 内不拆上下文**，只拆并行验收点
- 单模型上下文无法推完整个 matrix 治理已经被反复验证为事实 —— 不要试图反证
- 模型职责：协议固熵的 review、决策轨迹保真、branch 内具体片段的实现协作

### B. 本节新增决策（§1-§10 之外）

1. **Nodes 进 manifests** —— 推荐进，等最终拍板。理由：
   - 概念一致性 —— nodes 与 channel / provider / config 同等地位的能力声明
   - mode 隔离的真实场景（desktop vs outdoor 启用不同 nodes 集）
   - 一次性付的破开成本：让 manifests 抽象容纳非包扫描类型，建立 skill / macro
     / 其他未来声明类型进 manifests 的范式
   - PackageManifests 当前不持 mode，nodes 进入意味着 manifests 抽象首次集成
     env.current_mode

2. **Mode 上 nodes 字段的覆盖语义** —— 候选三种，等最终拍板：
   - 完全替代（K5 形态，类比 `__main__` channel）
   - 叠加（providers 形态）
   - 叠加 + name 冲突时 mode 覆盖（推荐，对应 providers 显式继承 + 追加的实际语义）

3. **命名分层** —— 推荐三层 + 域运维正交：
   - `moss codex` — 源码层 introspection，跟环境无关
   - `moss manifests` — workspace 静态声明（nodes 进则归此）
   - `moss runtime` — 进程/实例运行时状态：cells / topics / channels / events / sessions
   - 域运维入口：`moss nodes` / `moss workspace` / `moss modes` / `moss ghosts`，
     与三层正交
   - `moss apps` 在迁移期保留，第 14 项删除时一起去

4. **Script 体系整体删除** —— `CellType.script` 与 `moss script` CLI
   入口（`src/ghoshell_moss/cli/main.py:41`）一并清理。worker 接管原 script
   的兜底身份位置

5. **ChannelName 是入参不是类名约束** —— `AppStoreChannel(name='apps')` 的
   `name` 是 ChannelName 入参；改默认值即可改 ghost 视角的 channel 名，类名独立决策

6. **Worker 是否写 cell meta 的决策位置** —— 在 Matrix 进/退创建/删除 cell meta
   的时点决定，而非文件结构层。worker 的 `worker/{uuid}` 是否落 CellMeta 文件归
   Matrix bootstrap 路径裁决

### C. 推进的 14+ 项拓扑

人类架构师方案 14 项 + 本轮反馈整合后真实拓扑（顺序按依赖，非严格线性）：

1. cell 模板 `host.stubs.node`
2. node 数据结构 + NodeManager 抽象
3. env 发现/删除逻辑稳定化（含 Environment "default to host" fallback 消除）
4. Matrix.spawn 改造（spawn_cell / spawn_worker 二分 + announce-time 校验）
5. 高阶 node 运行接口（决策：是否提升到 host 抽象层）
6. Matrix cell 逻辑归纳分组（cell 运行时是否上移到 host / nodes 声明 / nodes 运行时管理）
7. provider / proxy 暂不动
8. Node 环境治理（参考 app，但日志/运行时 debug 机制更完善一致）
9. CLI: create / register / run / status 等
10. `moss runtime` 命令集（含 cell manifest 等运行时调试机制）
11. Mode 完成环境发现约束 + AppStoreChannel-like NodesChannel
12. 试点迁移若干 `.moss_ws` 内 app，走运行时开发吃狗粮
13. nodes 文档套件（dogfooding 反向产出）
14. 全面迁移 apps，删除 apps + script + 配套 pyproject 更新

跨步骤的协议级注入点（不单独成步、贯穿）：
- type registry 协议接口（`announce_type` / `find_owner` / `list_types`）
- RuntimeEvent topic 广播 —— 本 workstream 仅记录扩展点，不在本轮做
- 测试体系作为贯穿任务，协议层测试随实现同时落
- fractal alias 不做（迭代核心动机已完成，不是强 feature）

### D. 高优探索线索（上下文恢复时必读）

下一个实例进入时，按优先级读这些文件理解技术现实，不要重新 grep：

| 文件 / 位置 | 用意 |
|---|---|
| `src/ghoshell_moss/channels/module_eval_channel.py` + `tools/module_eval.py` | "address 自由度" 反例标本。看 `cell_address=f"module_eval/{module_name}"` 暴露的 bus 干净 / 概念污染不对称 |
| `src/ghoshell_moss/channels/app_store_channel.py` | operations channel pattern 范式，NodesChannel 同形参考 |
| `src/ghoshell_moss/host/nursery.py` | ProcessNursery 实现，pipe fencing + start_new_session 是当下进程生命周期承诺 |
| `src/ghoshell_moss/host/matrix.py:170-176` | `workspace.lock("moss_cell_{type}_{name}")` 是真实 address 唯一性保证；host 用 `moss_host_{scope}` 同形锁 |
| `src/ghoshell_moss/host/matrix.py:566-585` | `Matrix.spawn` 当前签名，演化为 spawn_cell / spawn_worker 二分的起点 |
| `src/ghoshell_moss/core/blueprint/environment.py:224-227` | `MOSS_CELL_ADDRESS` 默认 `host/{mode}` fallback —— worker 安全性陷阱来源 |
| `src/ghoshell_moss/host/manifests/impl.py` | PackageManifests 纯包扫描，nodes 进入需破开 |
| `src/ghoshell_moss/core/blueprint/manifests.py:280` | `Manifests` 基类，加 `nodes()` 方法的位置 |
| `src/ghoshell_moss/host/cell_discovery.py` | zenoh announce / query portal，type registry 协议物理落点 |
| `src/ghoshell_moss/host/stubs/mode/providers.py` 等 | mode 通过 `from MOSS.manifests.* import *` 继承全局的范式 |
| `src/ghoshell_moss/core/blueprint/matrix.py:105` | `class Mode(BaseModel)` 加 nodes 字段的位置 |
| `src/ghoshell_moss/core/blueprint/matrix.py:22` | `CellType` 枚举，host/app/fractal/script → host/node/fractal + worker 改造点 |

### E. Audit 指标（下个实例如何自我校准）

- 是否引入了 "policy 拒绝" 而非 "substrate 兜底"（例：worker 不应该 raise，应该
  `worker/{uuid}` 注入；type 不在 registry 应该 announce-time 拒绝而不是 spawn-time 拒绝）
- 是否在协议层预设了未必发生的政策（owner channel 退场的 finalize 流程 / 跨进程
  回调媒介选择 / worker 对外通讯方式 —— 这些都被本轮明确推翻为不预设）
- 是否预测了未发生的压力点（worker cell meta type 字段、type 重名碰撞策略、fractal 跨域校验等
  —— 见 §10 迭代标记）
- 命名上是否引入了第四套体系（B.3 三层 + 域运维之外不要再加）
- 协议项 [P] 与实现项 [I] 是否混淆 —— §9 表格是真相

### F. 推翻路径的 audit

下一个实例如果想推翻本节或 §1-§10 的某个决策，先确认：

1. 推翻目标是 [P] 协议项还是 [I] 实现项 —— 实现项可以自由换，协议项需完整论证
2. 推翻理由是出现了真实压力点，还是设计偏好替换
3. 推翻后 §10 的迭代标记中是否有项被迫立即决 —— 若是，说明推翻链触发了过早闭合

### G. 不属于本 workstream 的事

明确划走，避免下个实例越界：

- RuntimeEvent topic 事件广播 —— 已设计（§5），独立 workstream 推进
- Skills market channel —— 独立 feature，验收后反向集成
- Fractal alias 扁平化 —— 不强需求，迭代核心动机已完成
- `moss shell-init` 完整契约 / install.sh timeout / Nursery 默认 stdio 处置 ——
  三项与本轮协议固熵正交


## 2026-06-21 认知交接：实现层设计收敛 (claude-opus-4-7)

> 本轮经过多轮 review 循环，将 FEATURE.md 的协议设计落地为具体数据模型和 ABC。
> 以下是与原 FEATURE.md 不同的决策点，以及下一个实例进入时的导航地图。

### 数据模型 (src/ghoshell_moss/core/blueprint/cell.py)

```
CellType         — host / worker / fractal  (app/script 删除)
CellMetadata     — type, name, singleton, description
CellLauncher     — interpreter, cmd, args, cwd, extra_env  (日志路由字段删除, 名词 executable→interpreter, script→cmd)
CellManifest     — 组合模型 (非继承 CellMetadata): type, name, singleton, description + launcher + instruction + installed
                   CELL.md 平铺 frontmatter, launcher 为嵌套 YAML key.
                   installed 默认 True. 有 INSTALL.md 时从 .installed 文件推导.
CellStatus       — uid, state(starting/alive/stopped), pid(int|None), failure
Cell             — meta + launcher + status. set_alive(pid) / set_failed(reason) 语法糖.
                   address 依赖 meta.singleton: True→type/name, False→type/status.uid
                   bridge_address = address/status.uid  (网络唯一)
CellRegistry(ABC) — 静态发现 + 本地运行时注册. spawn_cell() 为 concrete code-as-prompt.
CellNetwork(ABC)  — 网络发现 + provider/proxy + detection loop. provider/proxy 入参统一用 CellBridgeAddress.
```

### 寻址体系

```
address         — type/name (人可读, 本地唯一). 例: worker/camera
bridge_address  — type/name/uid (线唯一, 网络反查). 例: worker/camera/01KVG93...
normalize()     — / → __  (文件系统友好). 例: worker__camera
runtime file    — cell-{normalized_address}.json  (注册目录下)
```

### RuntimeScope 与环境 (src/ghoshell_moss/core/blueprint/environment.py)

```
RuntimeScope     — session 身份唯一信源. mode/ghost/session_scope/session_id/host_pid.
                   进程内不可变. write_to_directory / read_from_directory 独立于 Workspace.
Environment      — 接受 (Workspace, RuntimeScope). 不再管理 cell address.
                   不再推导 cell 身份. dump_moss_env 包含 workspace+scope+cell keys.
```

### 已删除的概念

- `CellProvision` — 删除. installed 回到 CellManifest, enabled 由 manifest include/exclude 替代.
- `CellType.app`, `CellType.script` — 删除. app 是 node 的 runtime role, script 被 worker 取代.
- `Environment` 上的 cell_meta / cell_address / kill_cell — 全部移除, 由 CellRegistry 接管.

### 实现层 (src/ghoshell_moss/host/cell_registry.py)

```
EnvCellRegistry(CellRegistry) — 基于 Environment + 文件系统的实现.
  list_cell_manifests → cells/{group}/{name}/CELL.md 遍历.
  local_runtime_cells  → runtime_registry_dir 下 cell-*.json 扫描.
  discover_current_cell → MOSS_CELL_ADDRESS env → runtime file → fallback Cell.from_proc().
```

### 当前分支破坏性改动范围

- `src/ghoshell_moss/core/blueprint/cell.py` ✅ 抽象完成
- `src/ghoshell_moss/core/blueprint/environment.py` ✅ 重构完成
- `src/ghoshell_moss/host/cell_registry.py` ✅ 实现完成
- `tests/ghoshell_moss/blueprint/test_cell_design.py` ✅ 77 tests
- `tests/ghoshell_moss/blueprint/test_environment_design.py` ✅ 12 tests

**待推进 (按依赖顺序)**:
1. 修复旧模块 import 链 (app.py 引用 matrix.Cell 等)
2. 删除 AppStore / AppStoreChannel
3. 创建 cell stub (替代 app stub)
4. 重做 CLI (`moss cells` 替代 `moss apps` + `moss script`)
5. 删除 TUI 中 app inspector
6. Matrix impl 集成 CellRegistry + CellNetwork
7. workspace stubs 大面积调整

### 单测原则

只测数据模型行为 + 纯函数, 不测 `Environment.discover()` 等有全局状态的路径.
用 temp dir + monkeypatch env var, 不 mock Workspace.
测试文件: `tests/ghoshell_moss/blueprint/test_cell_design.py` + `test_environment_design.py`.

---

## 2026-06-22 设计收敛 + 实现: liveness listener / hub 重构 / cell announce (claude-opus-4-7)

> 人类架构师 + claude-opus-4-7。从 ZenohChannelHub 重构切入，提取通用 liveness 抽象，
> 收敛 cell announce 机制，确立 host/non-host 二分 Network 模型。

### I. ZenohLivenessListener — 通用 liveness 监听

**定位**: zenoh liveness 的最佳实践封装 —— 全量查询 + subscribe 变更 + 慢周期 reconcile。
替代此前每个组件手写的 subscribe/query/key 解析逻辑。

```
文件: src/ghoshell_moss/tools/zenoh_helper.py

ZenohLivenessListener(liveness_prefix, session, logger,
                      on_online=None, on_offline=None,
                      reconcile_interval=10.0)

__aenter__:
  1. get_liveness_keys() 全量 seed 缓存
  2. declare_subscriber 监听 liveness 变更
  3. fire on_online 给所有初始 key (让调用方感知完整状态)
  4. 启动 reconcile loop

__aexit__:
  unsubscribe + 清缓存 + 停 reconcile

属性:
  live_keys          → list[str]  零延时缓存快照
  get_liveness_keys()             同步阻塞全量查询
  get_liveness_keys_async()       线程池异步版
```

单元测试 13 条 (`tests/ghoshell_moss/tools/test_zenoh_liveness.py`):
全量查询 / 初始 seed / PUT 感知 / DELETE 感知 / on_online/on_offline 回调 /
reconcile 补漏 / reconcile 清过期 / exit 清缓存 / 重复 enter 幂等 / 多 token 独立 /
key 中 `/` 完整保留。

### J. ZenohChannelHub 重构 — 委托 listener

**改动** (`src/ghoshell_moss/bridges/zenoh_bridge/_hub.py`):

- 删掉 `_liveness_subscriber`、`_on_provider_liveness`、手写 `get_liveness_provider_address`
- 替换为一个 `ZenohLivenessListener` 实例，`on_online` / `on_offline` 回调处理业务逻辑
- `__aenter__` / `__aexit__` 一行委托，`get_liveness_provider_address` 一行委托
- **on_online 不再自动建 proxy** —— 只记录 record。proxy 创建由 CellNetwork 驱动
- `proxy()` 用 `normalize(address)` 自动推导 name
- liveness prefix 从 `_hub_expr.new_expr('**').provider_liveness_prefix` 推导

**关键分离**:
- Listener: 纯粹的存活感知 (subscribe, query, reconcile)
- Hub: 业务政策 (proxy 管理, record, name 规范化)

原 4 tests 全部通过。

### K. MOSSNamespace 中心化

**文件**: `src/ghoshell_moss/tools/zenoh_helper.py`

zenoh key 体系收敛到 `MOSSNamespace`:

```
MOSSNamespace(namespace)
  ├── channels_namespace    # {ns}/channels
  ├── cells_namespace       # {ns}/cells
  ├── topics_namespace      # {ns}/topics
  ├── signals_namespace     # {ns}/signals
  ├── streams_namespace     # {ns}/streams
  └── outputs_namespace     # {ns}/outputs

MOSSEnvNamespace(env)         → MOSS/{moss_name}/scope/{session_scope}
MOSSScopeNamespace(scope)     → MOSS/{scope}
```

`HubKeyExpr.__init__` 不再追加 `/channels` 后缀（namespace 已经包含）。

### L. bridge_address 简化为 `type/{uid}`

> ⚠️ 2026-06-28 修正: 本节方案已被实现层推翻 — 当前 `Cell.address` 是
> `type/normalized_name/uid` 三段, 合并了 address 与 bridge_address.
> 见 §RR 与 §OO 三重身份问题. 本节保留作为决策轨迹.

**推翻**: `type/name/uid` 三层 bridge address。
**理由**: name 在 bridge 中不提供唯一性 (uid 已保证)，但使 `split_bridge_address` 语义模糊
（name 可能含 `/`）。

**新规范**:
```
address         = type/name          (人可读, 本地唯一)
bridge_address  = type/{status.uid}  (线唯一, uid 生成)
```

`normalize()` 统一处理 `/`, `.`, `-`, `\\` → `_`。

### M. cell announce 收敛: PUT + queryable (不用 liveness token)

**动机**: cell 变化低频 (启动/停止/状态变更)，用 liveness token 持续宣告浪费资源。
PUT + queryable 两个 key 完成所有需求。

**Key 布局**:
```
{ns}/cells/{bridge}          ← PUT (change notify) + DELETE (revoke)
{ns}/cells/{bridge}/info     ← queryable (被动返回 Cell JSON)
```

**变更通知流**:
1. cell 上线/状态变更 → PUT `cells/{bridge}` (value = updated_at timestamp)
2. host subscribe `cells/**` → 收到 PUT, key 中含 bridge
3. host 拼 `/info` 后缀 queryable 查询 → 拿到完整 Cell 数据
4. DELETE → 立刻清理缓存 + proxy

**死检**: detection loop 慢周期逐个 query 已知 cell 的 `/info` queryable。
无响应 → stale 计数 → 超时清理。比 liveness DELETE 慢 (stale_timeout 级别 vs 秒级),
但 cell 数量小 (< 100)、变化低频，可接受。

### N. CellNetwork 二分: CellNetwork vs HostCellNetwork

> ⚠️ 2026-06-28 修正: 本节 ABC 分离方案不做 — 当前只有一个 `CellNetwork` ABC,
> host/non-host 区分通过 `allow_create_proxy: bool` 构造参数实现.
> matrix.py 的 `only_allowed_in_host_cell` 参数是同一约束的第三次表达, 待 §OO/§QQ
> 决策后一并收敛. 本节保留作为决策轨迹.

| 能力 | CellNetwork (通用) | HostCellNetwork |
|------|---------------------|-----------------|
| announce_cell | ✅ | ✅ |
| live_cells (query) | ✅ | ✅ |
| list_providers | ✅ | ✅ |
| provide / create_proxy | ✅ | ✅ |
| subscribe `cells/**` | | ✅ |
| 缓存 cells/providers | | ✅ |
| 自动 proxy 管理 | | ✅ (按 providing_channel) |
| 变更列表 (maxsize) | | ✅ (模型可见) |
| 封装为 CellsChannel | | ✅ |

`proxies()` 返回 `dict[name, ChannelProxy]` — key 是 proxy name 而非 bridge_address。

**ABC 简化方向** (下一轮实施):
- 从 CellNetwork ABC 移除 `cached_*`, `start/stop_detection_loop`, `start_cell`
- 新增 HostCellNetwork ABC (继承 CellNetwork, 加 `proxies` + 变更列表 + `start_detection_loop`)
- `start_cell` 留在 Matrix，不在 CellNetwork

### O. providing_channel 字段 — cell 驱动的 proxy 管理

`CellStatus` 加 `providing_channel: bool = False` 字段。运行时快照，非静态声明。

**建 proxy 的唯一路径**: cell info 中 `providing_channel = True` → CellNetwork 建 proxy。
取代此前 hub `_on_provider_online` 自动建 proxy 的逻辑。

**与 channel liveness 的关系**: provider/proxy 通讯层有自己的保活机制。cell 层不要求
"cell 宣告" 与 "channel provider 上线" 严格有序。proxy 创建可能短暂失败 (provider 尚未 ready)，
retry 由 reconcile 或下一轮 detection loop 兜底。

### P. start_cell 阻塞语义

`Matrix.spawn_cell(cell, providing_channel=True)`:
1. 强制立即建 proxy (不等 detection loop)
2. `proxy.wait_connected()` 阻塞到联通
3. 返回联通后模型立即可用

解决 "启动 cell → 两轮 detection loop → 模型才看见" 的延迟问题。

### Q. Matrix 三层模型

未来 Matrix 网络拓扑三分:

| 层 | 关系 | 发现策略 | 现有实现 |
|---|---|---|---|
| 上 | MOSS ↔ MOSS (peer) | 双向协商, 显式 accept | 尚未有独立抽象 |
| 中 | host ↔ node/worker | 同 scope 内自动发现 | CellNetwork (正在做) |
| 下 | host → 子 MOSS (fractal) | host subscribe manifest | FractalHub |

中层是当前 workstream 目标。三层在 zenoh 上靠 scope 隔离，互不干扰。
proxy 创建策略三层不同: 中层 cell 上线可 auto-proxy (host 信任自己的 cells)，
上层 peer MOSS 的 channel 需显式 accept。

### R. 本轮实现清单

**已完成**:
- `src/ghoshell_moss/tools/zenoh_helper.py` — MOSSNamespace + ZenohLivenessListener
- `tests/ghoshell_moss/tools/test_zenoh_liveness.py` — 13 tests
- `src/ghoshell_moss/bridges/zenoh_bridge/_hub.py` — 重构为 listener 驱动, on_online 不建 proxy
- `src/ghoshell_moss/bridges/zenoh_bridge/_utils.py` — HubKeyExpr 简化 (去掉 double /channels)
- `src/ghoshell_moss/contracts/configs.py` — LocalConfigStore mode-aware 重写
- `tests/ghoshell_moss/contracts/test_local_configs.py` — 39 tests
- `src/ghoshell_moss/host/cell_network.py` — ZenohCellNetwork 原型 (需按 host/non-host 二分重做)
- `src/ghoshell_moss/host/fractal/zenoh_fractal.py` — accept/ignore 从 Cell 剥离
- `tests/ghoshell_moss/host/test_zenoh_fractal.py` — 5 tests 全部通过

**待推进 (下一会话)**:
1. 完成 CellNetwork ABC 精简: 移 `cached_*`, `start/stop_detection_loop`, `start_cell`
2. 新增 HostCellNetwork ABC: `proxies()`, changes (`maxsize` 环形列表), `start_detection_loop`
3. CellStatus 加 `providing_channel: bool` 字段
4. bridge_address 改为 `type/{uid}`, normalize 支持更多分隔符
5. 重做 ZenohCellNetwork: PUT + queryable announce, subscribe `cells/**`, reconcile
6. 实现 ZenohHostCellNetwork: extends ZenohCellNetwork, proxy 自动管理
7. Hub 增加 `remove_proxy(address)`
8. cell announce 两 key 实现: PUT on change + queryable
9. 修改 `CellKeyExpr` 与 namespace 对齐
10. `Matrix.cell_channel_proxy` 用 cell.normalized_name 做 name, cell.description 做 description
11. CellsChannel 封装 HostCellNetwork 为 ghost-visible channel
12. 修复 test_matrix_init.py 等旧 import 链 (DEFAULT_CELL_ADDRESS, ScopesKey 等)
13. CellNetwork 集成测试套件

### S. 上下文恢复线索 (下一会话必读)

按优先级:

1. **本文件 §I-§R** — 本次设计收敛
2. `src/ghoshell_moss/tools/zenoh_helper.py` — ZenohLivenessListener + MOSSNamespace
3. `src/ghoshell_moss/bridges/zenoh_bridge/_hub.py` — 重构后的 hub
4. `src/ghoshell_moss/host/cell_network.py` — 当前 ZenohCellNetwork 原型
5. `src/ghoshell_moss/core/blueprint/cell.py:693-801` — CellNetwork ABC (待精简)
6. `tests/ghoshell_moss/tools/test_zenoh_liveness.py` — listener 测试 (参考模式)

### T. audit 指标 (下个实例自我校准)

- Cell announce 是否只用 PUT + queryable, 没有引入 liveness token?
- on_online 是否只记 record, 不自动建 proxy?
- proxy 创建是否由 `providing_channel` 字段驱动, 而非 channel liveness 事件?
- HostCellNetwork 和 CellNetwork 的职责边界是否清晰?
- bridge_address 是否只有 `type/{uid}`, name 不在其中?
- `proxies()` 返回的 key 是否是 proxy name 而非 bridge address?
- 三套命名体系是否收敛到 MOSSNamespace?

## 2026-06-25 大规模抽象重构 — L4 OS 架构跃迁 (claude-opus-4-7)

> 人类架构师 + claude-opus-4-7。从 L2 (单 project 内) 跃迁到 L4 (跨实例组网)。
> Matrix 独立组网, Project 与 Mode 解耦, 四元语义确立, blueprint 层 6 文件重做。

### U. 架构跃迁: 从 L2 到 L4

三层跃迁路径:
1. moss 仅在 `.moss_ws` 内运作
2. moss 扩展到 workspace 所在 project
3. moss 路径无关, 同 OS 内组网
4. moss 跨越 network, 跨实例组网 ← **当前目标**

Matrix 必须独立于 Mode 运行。`Matrix.discover()` 不再依赖 Host/Mode,
只依赖 `env → project → matrix`。无 mode 时 matrix 仍可启动,
cell registry 为 null。

### V. 命名定案: 四元语义 + transport

| 概念 | 名字 | 位置 | 职责 |
|------|------|------|------|
| 连接配置 | network | `networks/{name}.json` | driver + transport params + 默认 scope |
| 传输驱动 | driver | NetworkMetadata.driver | zenoh / mqtt / ws |
| 通讯子空间 | scope | NetworkMetadata.scope | zenoh key 隔离, 可 CLI --scope 覆盖 |
| 本地依赖隔离 | mode | `modes/{name}/MOSS.md` | providers/channels/configs |
| 运行时身份 | ghost | GHOST.md | ghost 空间 |
| 会话作用域 | session_scope | `mode-{m}-ghost-{g}-network-{n}` | 可重入的组合标识, 不含 scope |

zenoh key 空间:
```
MOSS/matrix/
  hosts/                              ← all_hosts() 跨 scope
  scopes/{scope}/
    host-alive                        ← get_host() scope 内 host 唯一标记
    cells/                            ← cell liveness + queryable
    channels/                         ← channel provider/proxy
    topics/ | signals/ | streams/ | outputs/
```

端口约定: 2038-n 系列 (来自 Detroit: Become Human 2038 时间点)。
默认 `127.0.0.1:20380`。

### W. 六文件重做摘要

**environment.py** — RuntimeScope 消除。字段回归 Environment 平铺属性。
bootstrap 后所有属性读 os.environ — 运行时 `os.environ['MOSS_MODE_NAME']=...` 立即生效。
新增 `fixture()` 上下文管理器做测试隔离, `reset()`/`set_instance()` 控制单例。
`DEFAULT_NETWORK_SCOPE = 'default'` (替代空串)。`MOSS_SESSION_ID` 不传给子进程。

**cell.py** — CellType 收敛为 host/worker。discover() 简化为纯函数:
读 runtime file 或 from_proc(), 不做 kill/takeover。
Bug 修复: spawn_cell env 传错 (env 参数 vs env_data),
is_alive pid is None 死判断 (pid 已是 int), uid 覆写移除,
to_json 副作用移除。DuplicatedError 保留给 Matrix 层。

**project.py** — MossMeta → ModeMeta。新增 `project_cell_paths` /
`workspace_cell_paths` / `exclude_cell_paths` 做 cell 发现路径。
NetworkMetadata (旧 SessionScopeMetadata): driver/scope/config 字段。
NetworkConfig ABC: driver_name() + to_metadata() + from_metadata() 序列化桥。
Project 不再持有 cells (cell 发现无 mode 时无意义);
HostMode 上 cells() 返回 CellRegistry。

**matrix.py** — Matrix ABC 加 project 属性。discover() 走 `env → project → matrix`。
cells 属性返回 CellRegistry, network 属性返回 CellNetwork。
移除 Mode 依赖。RuntimeScopeKey 更新: scope/network/mode/ghost/cell。
移除 scope_home (scope 是纯通讯概念, 不需要 storage)。

**session.py** — session_scope 不含 scope (`mode-{m}-ghost-{g}-network-{n}`)。
session_id 是进程级身份。pub_logos/get_logos stream_id 参数替代 session_id,
默认用 session_scope 共享 logos 流。

**host.py** — MossHost.discover() 走 `env → project → host`。工厂模式 ghoshell_moss.factory。

### X. 新增模块

**NetworkNamespace** (`matrix/zenoh_impl/_helper.py`) — 替代旧 MOSSNamespace。
从 Environment 构造, 产出 `MOSS/matrix/scopes/{scope}/...` key 空间。

**ZenohNetworkConfig** (`matrix/networks/zenoh_network.py`) — NetworkConfig 的 zenoh 实现。
按 cell type 组织: `host: ZenohNodeConfig(listen=..., multicast=...)`,
`worker: ZenohNodeConfig(connect=...)`。for_cell(type) 分发方法。
`create_zenoh_session_from_metadata()` 一站式: metadata → config → session。

**Network stubs** (`stubs/workspace/networks/`):
- `local.json` — Tier 1 默认: host listen 127.0.0.1:20380
- `lan.json` — Tier 2: host listen 0.0.0.0:20380 + multicast

**session 实现搬迁** (`matrix/session/zenoh_session.py`) — MossSessionWithZenoh
用 MatrixNamespace 组装 zenoh key。storage/tmp_storage 在 session_scope 级别。

### Y. 测试现状

| 文件 | tests | 备注 |
|------|-------|------|
| test_environment_design.py | 37 | fixture() 隔离, 测行为不测字面值 |
| test_cell_design.py | 85 | discover 简化后全部通过 |
| test_zenoh_network.py | 12 | config 默认值/for_cell/往返/driver mismatch |
| test_zenoh_session.py | 33 | MatrixNamespace 入参适配, key 断言用 endswith |

### Z. 待推进 (按依赖)

1. CellRegistry 新实现 (project_cell_registry) — glob/iterdir 替代两层目录
2. HostCellNetwork / CellNetwork ABC 分离
3. Matrix impl 重写 (对齐新 ABC)
4. ProcessManager 集成 (替代 spawn_cell 底层)
5. CLI 重建 (`moss cells` 替代 apps/script)
6. 旧代码清理 (AppStore, circusd, script CLI)
7. 文档体系 (howtos, docs, tutorials)

已删除:
  - `src/ghoshell_moss/host/cell_network.py` — 旧 ZenohCellNetwork, 与新版 CellNetwork ABC 不兼容
  - `src/ghoshell_moss/host/cell_registry.py` — 旧 EnvCellRegistry, 已被 ProjectCellRegistry 替代

### 上下文恢复支点 (下一实例)

1. **本文件 §AA** — manifests 体系重建
2. `src/ghoshell_moss/project/manifests/` — 8 种 scanner + 2 个 concrete impl
3. `src/ghoshell_moss/core/blueprint/project.py` — Manifest[T] ABC + MatrixManifest + ModeManifests
4. `src/ghoshell_moss/stubs/workspace/src/MOSS/manifests/` — 全局 stub (package 结构)
5. `src/ghoshell_moss/stubs/workspace/modes/default/src/HOST/` — mode stub (package 结构)
6. `tests/ghoshell_moss/matrix/project/test_manifests.py` — 49 matrix tests
7. `tests/ghoshell_moss/matrix/project/test_mode_manifests.py` — 10 mode tests

---

## 2026-06-27 manifests 体系全面重建 (deepseek-v4-pro)

> 人类架构师 + deepseek-v4-pro。从 3 个 scanner 扩展到 8 个, ABC 修正,
> concrete impl 组装, stub 文件迁移为 package 结构, 双语头规范。

### AA. Manifest[T] 通用抽象

`Manifest[T]` 是扫描到的 Python 对象的自描述包装:

- `name()` / `description()` / `found_at()` / `import_path()` — 自描述字段
- `value() -> T` — 纯类型返回值 (非 `T | Exception`)
- `is_error() -> bool` / `error() -> Exception | None` — 显式错误查询, 替代静默失败
- `update_container()` — 可选, 仅 Provider 需要覆盖

`ScannedManifest[T]` 是通用实现。`source` 字段从 ABC 移除 (不重要), 保留在实现层可选。

### BB. 8 种 scanner 矩阵

| Scanner | T | 检测方式 | 返回 | 测试 |
|------|---|------|------|------|
| `providers` | `Provider` | `isinstance` 实例 | `Iterable` | 8 |
| `configs` | `ConfigType` | `isinstance` 实例 | `Iterable` | 5 |
| `signals` | `SignalSchema` | `issubclass(SignalMeta)` + `isinstance` | `Iterable` | 8 |
| `topics` | `TopicSchema` | `issubclass(TopicModel)` + `isinstance` | `Iterable` | 6 |
| `parameters` | `ParameterSchema` | `issubclass(ParameterModel)` + `isinstance` | `Manifest` (单值) | 5 |
| `nuclei` | `NucleusMeta` | `isinstance` 实例 | `Iterable` | 5 |
| `resources` | `ResourceStorageMeta` | `isinstance` 实例 | `Iterable` | 3 |
| `channel` | `PrimeChannel` | `isinstance(Channel)` + `name() == '__main__'` | `Manifest` (单值) | 5 |

`parameters` 和 `channel` 保持单值返回 — 一个 mode 只有一个主 channel 和一组 parameters。

所有 scanner 共享同一模式: `scan_package(max_depth=2)` → `iter_members` → `isinstance`/`issubclass` → wrap → yield。每条 ~50 行, 未到需要抽象 helper 的重复阈值。

错误处理: `strict` flag (raise) + `errors` 收集器 + error manifest (不静默跳过)。package 不存在时不抛异常 (无匹配时 `Iterable` 返回空, 单值返回 error manifest)。

### CC. ABC 修正

`resources()` 补回 — 旧 `Manifests` 有但新 ABC 遗漏, 加入 `MatrixManifest` 和 `ModeManifests`。

`nuclei()` 改为 `Iterable[Manifest[NucleusMeta]]` — 原为单值, 但一个 mode 可挂多个 NucleusMeta。与 `parameters` (单值) 不同 — parameters 是全局唯一声明。

`configs()` 改为 `Manifest[ConfigType]` (原 `ConfigSchema`) — `ConfigStore.get_or_create()` 需要完整实例, 不是 schema 片段。

### DD. Concrete impl

`ScannedMatrixManifest(MatrixManifest)` — 接收 root package (默认 `MOSS.manifests`), 6 个方法各自委托 scanner。

`ScannedModeManifests(ScannedMatrixManifest, ModeManifests)` — 继承前者所有 scanner, 追加 `channel()`, `nuclei()`, `resources()`。默认根包 `HOST.manifests`。

### EE. Stub 文件重构

所有 stub manifest 从 `{type}.py` 迁移为 `{type}/__init__.py` package 结构:

```
MOSS/manifests/                    HOST/
  __init__.py                        __init__.py
  providers/__init__.py              channels.py (单文件, 非 package)
  configs/__init__.py               providers/__init__.py
  signals/__init__.py               configs/__init__.py
  topics/__init__.py                signals/__init__.py
  parameters/__init__.py            topics/__init__.py
  resources/__init__.py             nuclei/__init__.py
                                    resources/__init__.py
                                    parameters/__init__.py
```

理由: package 结构允许扩展 (`providers/database.py` 而不改 `__init__.py`), `from MOSS.manifests.providers import *` 语义不变。channel 保持单文件 — 它是入口点, 不需要子模块。

所有文件头部统一为双语格式: English description block + `--` 分隔 + 中文说明。

### FF. 测试

| 文件 | tests | 覆盖 |
|------|-------|------|
| `test_manifests.py` | 49 | ScannedManifest + 7 种 scanner + ScannedMatrixManifest 集成 |
| `test_mode_manifests.py` | 10 | ChannelManifest + ScannedModeManifests + package-not-exists 容错 |

总计 59 tests, < 1s。

### 上下文恢复支点

1. `src/ghoshell_moss/project/manifests/` — 完整 tree
2. `tests/ghoshell_moss/matrix/project/test_manifests.py` — scanner 测试范式
3. `tests/ghoshell_moss/matrix/project/test_mode_manifests.py` — mode 测试范式
4. `src/ghoshell_moss/stubs/workspace/src/MOSS/manifests/` — 全局 stub 参考
5. `src/ghoshell_moss/stubs/workspace/modes/default/src/HOST/` — mode stub 参考


## 2026-06-27 会话 — MossMeta + LocalHostMode + LocalProject + CLI 重建 (claude-opus-4-7)

> 人类架构师 + claude-opus-4-7。从 manifests 体系完成后的下一步出发，
> 重新锚定项目认知体系：MossMeta 回归、Host 模式 concrete 化、Project 发现、
> CLI 命令体系彻底重排。

### GG. MossMeta 回归

MOSS.md 作为 workspace 内唯一配置入口，承载项目级默认值：

| 字段 | 说明 |
|------|------|
| name | 项目名 |
| default_mode/ghost/network/scope | 四项默认值 |
| project_id | 持久化 UUID |
| system_project | 追加到 CTML 后的项目级提示词 |
| cell_paths | cell 发现路径 |

MOSS.md 只在 workspace 内查找，不递归。frontmatter 读写，
`write_to_file` 排除 `file` 字段避免绝对路径进版本控制。

`MOSS_NAME_PATTERN = r'^[a-zA-Z_][a-zA-Z0-9_]*$'` —
与 `ChannelNamePattern` 软对齐。作用于 MossMeta 全部 name 字段 +
`HostModeMeta.name`。

### HH. HostModeMeta + LocalHostMode

`ModeMeta` → `HostModeMeta`。`MODE.md` → `HOST.md`（`modes/{name}/HOST.md`）。

`LocalHostMode` 构造：
```
LocalHostMode(env, meta, workspace_dir, *, logger=None)
```
- `workspace_dir` 显式传入，不依赖 `meta.file`
- `logger` fallback `logging.getLogger('moss')`
- `bootstrap()` 显式调用后才允许访问 `manifests()` / `cells()`
- `_check_bootstrapped()` 守卫，防止静默失效
- 懒加载 + 缓存（`_manifests`, `_cells`）

### II. LocalProject 原型

完整实现 `Project` ABC 的 7 个抽象方法：

| 方法 | 实现 |
|------|------|
| `ghosts()` | 委托 `search_ghost_manifests()` → `MOSS.ghosts` package |
| `get_ghost(name)` | 缓存查找 + `LookupError` |
| `list_modes()` | 遍历 `modes/` 目录，`from_file(file, name=dir_name)` |
| `get_mode(name)` | 缓存 → `LocalHostMode(env, meta, workspace_dir)` |
| `cells` | mode 优先 → `env.cell_dirs()` fallback |

### JJ. CLI 重建

| 变化 | 内容 |
|------|------|
| 新增根命令 | `moss init` (project dir 目标, .moss 在其下), `moss shell-init` |
| 重命名 | `workspace` → `project` |
| 归入 project | modes, ghosts (list-modes/show-mode/list-ghosts/show-ghost) |
| 删除 | workspace, runtime, script, apps, modes, ghosts 独立命令组 (6 个文件) |
| 新全局参数 | `--ghost/-g`, `--network`, `--scope` |
| 删除参数 | `--session-scope` |
| 删除文件 | apps_cli.py, ghosts_cli.py, modes_cli.py, runtime_cli.py, scripts_cli.py, workspace_cli.py |

`moss project` 子命令：`where`, `env-init`, `list-modes`, `show-mode`,
`list-ghosts`, `show-ghost`。全部走 `Project.discover()`。

### KK. 常量与 stub 改造

- `DEFAULT_WORKSPACE_DIR_NAME`: `.moss_ws` → `.moss`
- `WORKSPACE_STUB_PACKAGE`: `ghoshell_moss.host.stubs.workspace` → `ghoshell_moss.stubs.workspace`
- `MODE_STUB_PACKAGE`: `ghoshell_moss.host.stubs.mode` → `ghoshell_moss.stubs.workspace.modes.default`
- 删 `APP_STUB_PACKAGE`
- stub 模板：`workspace/MOSS.md` + `modes/default/HOST.md`（全注释，零强值约定）
- `logging.yml`：只配格式/等级，`Project.bootstrap()` 动态注入 `TimedRotatingFileHandler`
- 所有路径收敛到 workspace 内部：`modes_home`/`network_configs_dir`/`configs_dir` 统一用 `workspace_dir`

### LL. 测试

| 文件 | 测试数 | 覆盖 |
|------|------|------|
| `test_environment_design.py` | 50 | MossMeta默认值/优先级/命名校验 |
| `test_local_host_mode.py` | 12 | 构造/bootstsrap守卫/缓存/副作用 |
| `test_local_project.py` | 10 | modes发现/ghosts/cells/属性 |

### 上下文恢复支点

1. **本文件 §GG-KK** — 本次会话
2. `src/ghoshell_moss/core/blueprint/environment.py` — MossMeta + MOSS_NAME_PATTERN
3. `src/ghoshell_moss/core/blueprint/project.py` — HostModeMeta + Project ABC
4. `src/ghoshell_moss/project/local_host_mode.py` — LocalHostMode 实现
5. `src/ghoshell_moss/project/local_project.py` — LocalProject 实现
6. `src/ghoshell_moss/cli/main.py` — init, shell-init, 全局参数
7. `src/ghoshell_moss/cli/project_cli.py` — project 命令组
8. `src/ghoshell_moss/cli/start.md` — 认知入口（已重写）

---

## 2026-06-28 三重身份问题与 L3 跃迁认知补完 (claude-opus-4-7)

> 与人类架构师讨论会话. 本节做四件事:
> (1) 补完前任未显式记录的 L0→L3 跃迁认知 — 后续所有设计的基线
> (2) 二元真相问题的设计裁决 — 选 (c) 显式承认, 不加回 pub/sub
> (3) 三重身份问题 framing — 核心待决, 本轮未敲定具体字段方案
> (4) 修正 §I-§T 中已被推翻/合并的设计 (§L, §N)
>
> 代码本轮不动. 字段方案敲定后, 与 §RR 末尾的"确定性修复清单"一并落地.

### MM. L0→L3 跃迁认知 (基线补完)

前任 §U 提过 "L2 → L4 跃迁" 但未把层次完整展开, 也未记录跃迁的核心 implication.
本节补完认知地图.

| 层 | "我是谁"的边界 | 可见性边界 | cells 真相源 |
|---|---|---|---|
| L0 | apps in workspace | workspace 内 | workspace runtime file |
| L1 | apps in project | project 内 | project runtime file |
| **L2** | cells in OS | 同 network 的 cells (跨目录 / 底层依赖不一) | network liveness |
| **L3** | cells in LAN | 同 network 的 cells (跨机器) | 同 L2 |

当前重构目标: **直接跃迁到 L3** (不做 web 级 L4).

关键 implications:

- **matrix 是最小通讯依赖** — `pip install ghoshell-moss[matrix]` 仅需 zenoh,
  任何 Python 进程可加入 cell network, 不需要 host mode / CLI / 文件系统约定.
  印证: pyproject.toml 中 `matrix = ["eclipse-zenoh>=1.8.0"]`, `host = [...]`
  在 matrix 之上追加 TUI / 音频 / MCP 等本地能力.
- **底层依赖不一致是协议层问题, 不是实现层问题** — A cell Python 3.11 + numpy 2.x,
  B cell Python 3.12 + numpy 1.x. L3 不要求一致, 只要求 network 上消息形态
  (CTML, TopicModel 序列化, channel proxy interface) 一致.
- **cell runtime file 的三层真相**:
  - **network 级 (真相源)**: cells_ns liveness + queryable
  - **project 级 (本地观察缓存)**: workspace/runtime/cells/
  - **process 级 (自我认知)**: MOSS_CELL_ADDRESS + runtime file 恢复

  原 workspace runtime file 是 single source of truth (进程清理基线),
  现在降级为本地缓存, **network 才是真相**. 但 network 看不见本机其它
  project 的进程清理 (L2 跨目录), project runtime 必须保留作为
  **本机清扫边界**.
- **L2 不是被跳过** — 一个 zenoh network 默认 `tcp/127.0.0.1:20380` 时就是 L2;
  改 `0.0.0.0:20380 + multicast` 就是 L3 LAN. 同一抽象, 不同 NetworkConfig 切换.

### NN. 二元真相显式承认 — status 变更不广播 (设计裁决)

zenoh 通讯原本三件套 (liveness 推上下线 + queryable 拉 + pub/sub 推变更),
在 §M 阶段拿掉 pub/sub, 理由 "cell 生命周期不变".

**真问题**: `failure` / `degraded` 这种**进程还活着但状态变了**的中间态,
liveness 不动, subscriber 看不见. cell 本地真相与 network 真相在中间态分叉.
这是 L3 跨网络的物理现实.

**裁决**: 选 (c) — 显式承认二元真相, **不加回 pub/sub**.

理由:

- liveness 本质就是 best-effort online presence, 不是 status truth source.
  zenoh 也不保证 SIGKILL 后立即 DELETE (靠 keep-alive timeout).
- 加回 pub/sub 的代价是全网广播 status delta. 真正需要 act on failure 的
  只有 host 与 cell 自身. 全网代价对低频事件不值.
- queryable handler 每次返回 ann.cell 最新 snapshot — 拉就是最新.

**落地约束** (待 §OO 字段方案敲定后随确定性修复清单执行):

- **CellNetwork docstring 必须显式承诺**: "live_cells() 返回的是延迟视图,
  只承诺 online/offline 准确, status 字段可能滞后到下次 reconcile.
  需要实时 status 的调用方必须 get_live_cells(refresh=True)."
- **subscriber 路径只触发 online/offline**, 不感知 status 变更.
- **失败传播的逃生口**: cell 自己想让 network 知道 degraded, 可主动
  `revoke_cell + update_cell`. liveness 的 DELETE+PUT 就传出去. 这是 cell
  有意识的传播, 不是框架默认行为.
- **version 字段当前只有语义价值** — 缺 pub/get 配套机制 (没有 "广播带 version
  后缀的 address" 这种实现), reconcile 也不用. **beta1 不强推 version 进
  reconcile loop**. 留待 status 同步成为真实痛点时再补.

### OO. 三重身份问题 framing (核心待决)

cell 当前的字段在做三件互相牵连的事, 概念漂移已经发生.

**漂移现象**:

- `type` 字段一个人干三份活:
  - `host/worker` (framework 角色, 枚举有限)
  - `sensors/bodies/tools` (project 分类, 用户自由)
  - `local/remote/fractal` (network 视角, **应由观察者推断**)
- `name` 既是 project 内唯一性也是 network address 段
- `address` 既是 process 唯一也是 network 唯一. `bridge_address` 概念在 §L
  已被合并 (见 §RR 修正).

**身份归属表** (重新对齐):

| 身份 | 真相源 | 唯一性范围 | 用途 |
|------|--------|----------|------|
| process self | env + runtime file | 进程内 | 进程内 "我是谁", discover 还原 |
| project | runtime file | project 内 | 本机清扫、本地单例锁 |
| network | liveness + queryable | network scope 内 | 跨进程发现、proxy 目标 |

**重要观察**: network origin (local/remote/fractal) **是相对的**.
同一 cell, 我自己 announce 的是 local, subscriber 见到别人 announce 的是
remote, FractalHub 注入的是 fractal. **不该是 cell 自己的字段**.

**字段拆解候选方案 (未决, 仅作讨论基础)**:

```python
class CellMetadata:
    role: Literal['host', 'worker']     # network 框架角色 (framework 保留)
    group: str = 'default'              # project 分类 (用户自由, sensors/bodies/...)
    name: str                           # project 内 + group 内唯一
    singleton: bool                     # 保留
    description: str                    # 保留
    channel: bool                       # 保留

class CellStatus:
    uid: str                            # process / network 级唯一 id

class Cell:
    @property
    def project_id(self) -> str:        # group/name
        ...
    @property
    def network_address(self) -> str:   # role/group/name/uid
        ...
    @property
    def self_uid(self) -> str:          # status.uid
        ...
```

`network_origin` 不进字段, 由 CellNetwork 观察上下文标记.

**未决问题 (留给人类架构师裁)**:

1. 字段方案采纳粒度 — **完整拆解** vs **beta1 保守** (保留 type 二态 = host/worker,
   但加 group, docstring 严格收敛只表达 "framework role")?
2. `type` 字段是否消失? 若消失, 迁移面较大 (manifests / CLI / zenoh key 全要改).
3. uid 在 address 中的位置 — singleton cell 是否也带 uid? singleton 的
   project 唯一性 vs network 唯一性的边界 (这一条是 §4 worker dup 检查
   问题的本源).

### PP. cell type 升格 / 多 network 升格 — beta1 保守路径

人类架构师指出: cell type 终极升格危险 — network 可能多实例创建,
local network 可能根本不是 zenoh. 这条思路对 beta1 工期危险, 本轮明确不做.

**beta1 保守路径** (留扩展点, 不锁死未来):

1. **type 二态保留 + 加 group** (见 §OO 候选方案) — 不做 type registry, 不做
   `announce_type`. group 用户自由, 不参与 framework 行为. 未来若真需 type 升格,
   group 自然演化为 type namespace.

2. **`Matrix.network` 接口为多准备** — 当前返回单 CellNetwork. 接口声明为
   "primary network", 但留下 `Matrix.networks() -> dict[name, CellNetwork]`
   的位置. beta1 只用 primary, 实际只接 zenoh. 未来加 local IPC backend
   或 gRPC 不破协议.

3. **CellNetwork 不预决准入** — network 层见到什么就是什么. 过滤策略在
   channel 层 (cells channel 决定本地显示哪些, fractal channel 决定接受
   哪些远程). 对应 §1-§10 反复强调的 "政策不进 substrate".

### QQ. channel proxy 给模型的根 channel 数量 (待决)

人类架构师指出: **cell 启动与 proxy 接入是两个命题**:

- 本地 cell: 启动 + proxy 可原子化
- 远程 cell: 关键是 "自动准入" vs "主动 accept"

**老代码参考** (`src/ghoshell_moss/channels/` 下):

- `app_store_channel.py` — 单根 channel + `own_commands` (start/stop/list) +
  `virtual_children` (本地 alive app 反射为 proxy). AppStore 是数据源.
- `fractal_hub.py` — 单根 channel + 准入策略 (`allow_all` / `auto_start`).
  FractalHub 是数据源.

两个老 channel 的命名空间与控制策略完全分离, 共存于 main channel 树下.

**两个候选方案 (未决)**:

**候选 A — 多根 (语义分离)**:

```
main/
  cells/        ← own_commands: start/stop/list
                 virtual_children: 本机 alive cell
  fractal/      ← own_commands: accept/list_remote
                 virtual_children: 准入的远端 host
```

论点: "本地启停" 与 "远程准入" 语义不同. 合一根可能模糊模型对操作目标的理解.
对应老代码现有 pattern, 迁移最便宜.

**候选 B — 单根 matrix channel + 子树分组**:

```
matrix/
  local/        ← virtual subtree: 本机 cell
  remote/       ← virtual subtree: 同 network 远端 host
  fractal/      ← virtual subtree: FractalHub 注入
  cells/        ← own_commands
```

论点: 统一 cell 命名空间, 管理入口集中, 与 "matrix 是最小通讯依赖" 命名对齐.

两候选数据源都是 `Matrix.network.live_cells()`, 视角和操作不同.

**未决问题**:

1. A 还是 B?
2. 准入策略 (远程 cell 何时变成本地可见 channel proxy) 的默认行为
   — allow_all / explicit accept / 二者皆有配置项?
3. 在 L3 视角下 fractal 还有独立必要吗? 还是 fractal 只是 "远端 host
   接入" 的别名? (这条与 §OO 的 network origin 由观察者推断密切相关 —
   如果 origin 不进字段, fractal vs remote 的差异只在准入策略, 不在身份.)

### RR. FEATURE.md 滞后修正

前任模型在大型重构中阶段性遗漏了关键变更记录. 已在文中内联标记 ⚠️ 修正
(见 §L 与 §N), 此处汇总:

- **§L (bridge_address = type/uid) — 已被合并**: 当前 `Cell.address` 实现为
  `type/normalized_name/uid`, 合并了 address 与 bridge_address. 合并是否为最终
  方案待 §OO 字段拆解决议后定型.
- **§N (HostCellNetwork vs CellNetwork ABC 二分) — 不做**: 当前只有一个
  CellNetwork ABC, host/non-host 区分通过 `allow_create_proxy: bool` 构造参数.
  `matrix.py::channel_proxy` 上的 `only_allowed_in_host_cell` 参数是同一约束的
  第三次表达. 三层重复待 §OO/§QQ 决策后一并收敛.
- **§Z 待推进清单中的 "HostCellNetwork / CellNetwork ABC 分离" — 已废**: 见 §N 修正.

### 待决策项 (Open Decisions, 按依赖顺序)

以下决策待人类架构师裁决. 不裁决前不动代码.

1. **三重身份字段拆解粒度** (§OO) — 完整拆解 vs beta1 保守 (type 二态 + group)
2. **`type` 字段去留** (§OO) — 完全消失 vs 保留并严格收敛语义
3. **channel proxy 根数量** (§QQ) — 候选 A (多根分离) vs 候选 B (单根分组)
4. **远程 cell 准入默认策略** (§QQ) — allow_all / explicit accept / 配置项
5. **`only_allowed_in_host_cell` + `allow_create_proxy` + 隐式 host 检查的三层重复
   收敛位点** (§RR / §N)
6. **L2 vs L3 NetworkConfig 出厂默认** — `127.0.0.1` (L2 启动友好) vs
   `0.0.0.0 + multicast` (L3 ready)
7. **singleton 在 network 视角的语义** — singleton cell 的 address 是否带 uid
   (§OO 第 3 子问)

### 确定性修复清单 (Pending Fixes)

以下条目不强依赖待决问题, 但部分会被字段方案影响. 待 §OO 敲定后统一执行:

1. **MossMeta.write_to_file bug**: `Environment.create_meta_file` 时设置
   `moss_meta.file = str((directory / MOSS_META_FILE).absolute())`;
   `project_id` 副作用从 `write_to_file` 移到 `Environment.__init__` 统一处理.
   配单测覆盖 "首次 discover 无 MOSS.md 文件" 路径.
2. **HostModeMeta.cell_dirs 重复定义** (`project.py` line 101 vs 143): 删一份.
3. **CellNetwork ABC 补 @abstractmethod**: `live_cells` / `on_change` 补抽象标记,
   或提供默认空实现. 当前 body 是 `pass` 但无 @abstractmethod, 子类不实现也不报错.
4. **CellNetwork docstring 显式承诺二元真相** (§NN 落地约束):
   `live_cells` 延迟视图声明; `update_cell` docstring 修正 worker dup 检查的实际
   语义 (待 §OO 第 3 子问决策后定文).
5. **CellLauncher cwd 默认补全**: `CellManifest.read_from_file` 中无论
   launcher.cwd 原值为何, 空就填 `found_dir`. 统一在 Matrix 启动时
   `Cell.discover` 阶段获取 cwd (人类架构师定的位点).
6. **FEATURE.md §L / §N 内联推翻标记** ✅ 已完成 (本节附属).

### 上下文恢复支点 (下一会话/实例必读)

按优先级:

1. **本文件 §MM-§RR** — 本次讨论, 三重身份 framing + 二元真相裁决 + L3 跃迁
2. **本文件 §AA-§LL** — 前两轮 (manifests 体系 + MossMeta/LocalProject)
3. **本文件 §1-§10** — 协议层最早奠基 (type 开放命名空间 / spawn 二分 / worker 兜底)
4. `src/ghoshell_moss/core/blueprint/cell.py` — 当前 Cell 字段实现 (待 §OO 改造)
5. `src/ghoshell_moss/matrix/networks/zenoh_cell_network.py` — §NN 二元真相落地点
6. `src/ghoshell_moss/channels/app_store_channel.py` — §QQ channel proxy 根 pattern
7. `src/ghoshell_moss/channels/fractal_hub.py` — §QQ 准入策略 pattern
8. `pyproject.toml` — `matrix = ["eclipse-zenoh>=1.8.0"]` 印证 §MM matrix 最小依赖
9. **本节 "待决策项" 清单** — 推进起点

---

## 2026-06-28 §SS 开工契约 (claude-opus-4-7 + 人类架构师)

> 紧接 §MM-§RR. 本节是 beta1 收敛后的最终施工契约. 决策全部敲定, 不留待决项.
> 推翻本节任何决议前, 必须先论证驱动场景已变.

### SS-1. 三重身份字段定型

```python
class CellMetadata:
    type: str = 'worker'      # 开放命名空间. 'host' / 'worker' 是框架保留字
    name: str                  # type 内唯一. 与 type 组成 identity
    singleton: bool = True
    description: str = ''
    # 删: group (并入 type 开放定义)
    # 删: channel (provide channel 是 runtime 事实, manifest 不预声明)

class Cell:
    # 命名定型 — 三个 property, 不存储, 显式生成规则 (code as prompt)

    @property
    def identity(self) -> str:
        """type/name. project-internal stable id, 跨 cell 重启不变."""
        return f"{self.meta.type}/{self.meta.name}"

    @property
    def address(self) -> str:
        """type/name/uid. network 唯一, 含 uid 防 stale."""
        return f"{self.identity}/{self.status.uid}"

    @property
    def channel_name(self) -> str:
        """proxy channel name.
        - singleton: normalize(identity).lower()
        - non-singleton: normalize(identity).lower() + '_' + uid[:8] 防碰撞
        """
        base = normalize(self.identity).lower()
        if not self.meta.singleton:
            return f"{base}_{self.status.uid[:8]}"
        return base
```

**模型只需记一个心智锚: `identity = type/name`**. address 在 collision 时偶尔出现,
channel_name 由 channel tree 自动派生, 模型读到就知道怎么调.

### SS-2. CellLog 信号量约定

```python
class CellLog(BaseModel):
    address: CellAddress       # 含 uid, 网络精确
    content: str               # 自由文本, 不约定 enum (隐藏 type 留在心里, 第一期不暴露)
    timestamp: float
    terminal: bool = False     # True: cell 已没了, 接收方别 query, 删自己 cache
```

`terminal=True` 触发场景:
- `revoke_cell` 内部 broadcast 时
- liveness DELETE → CellNetwork 自动 broadcast (`"cell-gone"`, terminal=True)
- ProcessManager on_exit 回调 → CellsManager broadcast (`"exited"`, terminal=True)

接收方行为:
- terminal=False → append log + 触发 follow-up query 刷 cache
- terminal=True  → append log + pop cache entry, 不 query

### SS-3. CellNetwork API 定型

CellNetwork 是 cell 发现 + proxy/provider 单一真相源. 屏蔽 zenoh hub 差异性.

```python
class CellNetwork(ABC):
    # -- 发现与广播 --
    async def update_cell(self, cell: Cell, *, log: str = '') -> None: ...
    async def revoke_cell(self, cell: Cell, *, log: str = '') -> None: ...
    async def broadcast_log(self, address: str, content: str, *, terminal: bool = False) -> None: ...
    async def check_unique(self, cell: Cell) -> None:
        """singleton: 查 identity wildcard; non-singleton: 查 address exact.
        raise DuplicatedError if collide."""

    async def get_live_cells(self, *, type: str | None = None, local: bool | None = None,
                              refresh: bool = False) -> dict[CellAddress, Cell]: ...
    def recent_logs(self, *, limit: int = 20, local: bool | None = None) -> list[CellLog]: ...

    # -- proxy / provider (代理 zenoh hub) --
    def proxies(self) -> dict[CellAddress, ChannelProxy]: ...     # 自动构建的全部
    def get_proxy(self, address: str) -> ChannelProxy | None: ...
    async def wait_connected(self, address: str, *, timeout: float = 30) -> bool: ...
    async def provide(self, channel: Channel, *, address: str | None = None) -> ChannelProvider: ...

    # -- 回调 (zenoh 后台线程触发, 调用方负责线程安全) --
    def on_provider_online(self, callback: Callable[[CellAddress], None]) -> Callable[[], None]: ...
    def on_provider_offline(self, callback: Callable[[CellAddress], None]) -> Callable[[], None]: ...
```

CellNetwork 自动 build proxies 的触发链:
```
zenoh hub liveness PUT (channels_ns/{address})
  → on_provider_online callback
  → CellNetwork 查 cells_ns 拿 Cell snapshot
  → CellNetwork 用 cell.channel_name 建 proxy
  → proxies dict 加入
  → broadcast_log(address, "channel-ready")

zenoh hub liveness DELETE
  → on_provider_offline callback
  → CellNetwork remove proxy (释放, 防泄漏)
  → broadcast_log(address, "channel-gone", terminal=True)
```

### SS-4. CellsManager API 定型

CellsManager 是本地 cell 进程的二阶状态管理. 与 CellNetwork 解耦, 只关心
"我自己启动过哪些, process 状态如何".

```python
class CellsManager(ABC):
    def list_manifests(self) -> dict[RelativePath, CellManifest]: ...
    async def start(self, path: RelativePath) -> Cell: ...    # fire-and-forget
    async def stop(self, path: RelativePath, *, timeout: float = 5.0) -> None: ...
    def opened_cells(self) -> dict[Identity, Cell]: ...        # 当前本地启动过的
    def recent_logs(self, *, limit: int = 20) -> list[ProcessLog]: ...
                # process 视角: started / exited / start-failed / bringup-failed
```

实现层:
- 持有 `dict[RelativePath, ManagedProcess]` 关联进程
- 通过 `process_manager.execute(..., on_exit=self._on_cell_exit)` 注册死亡回调
- 订阅 `network.on_provider_offline`, 自己的 cell offline 时从 opened_cells 里 pop

### SS-5. ZenohChannelHub 补强

```python
class ZenohChannelHub:
    # provider / proxy 各两个状态 (declared / running). is_running 不允许重复创建.

    def provider(self, address: str, channel: Channel) -> ZenohChannelProvider:
        """创建并 declare. 已 running 则 raise. 查 channels_ns liveness 一次去重.
        declare 后立即二次查 confirm, 撞到别人 declare 则自我撤销."""

    def get_provider(self, address: str) -> ZenohChannelProvider | None: ...

    def proxy(self, address: str, *, name: str, description: str) -> ZenohProxyChannel:
        """proxy 的 channel uid = address (含 uid). 已 running 则 raise.
        proxy 也 own liveness, 让对端可感知 watcher 死亡."""

    def get_proxy(self, address: str) -> ZenohProxyChannel | None: ...

    # 回调钩子 (zenoh 后台线程 fire)
    def on_provider_online(self, cb: Callable[[str], None]) -> Callable[[], None]: ...
    def on_provider_offline(self, cb: Callable[[str], None]) -> Callable[[], None]: ...
    def on_proxy_online(self, cb: Callable[[str], None]) -> Callable[[], None]: ...
    def on_proxy_offline(self, cb: Callable[[str], None]) -> Callable[[], None]: ...
```

约束:
- hub 自己不主动建/拆 proxy / provider, 只持有句柄
- 同名上线撞击 (provider 同名 / proxy 同名) 第一期不区分 id, 不处理
- 第一期约束: **只有 host 可以 create_proxy** — CellNetwork 内部 enforce, worker 调用 raise
- declare 前唯一性 check 一次. declare 后二次查 confirm 作为 TOCTOU 兜底

### SS-6. Matrix 接口定型

```python
class Matrix(ABC):
    @property
    def sub_processes(self) -> ProcessManager: ...  # 暴露给开发者. 带下划线防 IDE
    @property
    def cells(self) -> CellsManager: ...            # 本地 cell 治理
    @property
    def network(self) -> CellNetwork: ...           # proxy/provider/cell 发现真相源
    @property
    def project(self) -> Project: ...

    @property
    def this(self) -> Cell: ...                     # 自己

    async def provide(self, channel: Channel, *, address: str | None = None) -> ChannelProvider:
        """三层调用: matrix.provide → network.provide → hub.provider."""

    # 删: Matrix.spawn (移到 sub_processes.execute / spawn_task)
    # 删: Matrix.cell_workspace 等不相关属性 (cell 自己有 workspace 推导)
```

模型/开发者不能从 Matrix 直接拿到 `CellRegistry` — 要走 `Matrix.project.cells`.

### SS-7. Matrix 启动 / 关闭流程

```python
async def __aenter__(self):
    # Step 0: project-level flock (防本机重启)
    if self.this.meta.singleton:
        self._cell_lock = self._project.workspace.lock(self.this.cell_locker_name)
        if not self._cell_lock.acquire():
            raise DuplicatedError(f"local lock held: {self.this.identity}")

    # Step 1: network startup (要让 check_unique 能查)
    await self.network.__aenter__()

    # Step 2: network 唯一性校验
    await self.network.check_unique(self.this)

    # Step 3: 其它组件 lifecycle
    await self._sub_processes.__aenter__()
    await self._cells.__aenter__()

    # Step 4: host announce
    self.this.set_alive()
    await self.network.update_cell(self.this, log="started")

    # NOTE: bringup 不在 __aenter__ 内 — 见 Step 5

    return self

async def post_enter_bringup(self):
    """matrix entered 后调用 (matrix.run / 上层 runtime 内). bringup 失败不阻断 matrix."""
    mode = self.project.current_mode()
    if not mode or not mode.meta.bringup:
        return
    manifests = self.cells.list_manifests()
    for path, manifest in self._registry.match_cells(manifests, include=mode.meta.bringup):
        try:
            await self.cells.start(path)
        except Exception as e:
            self.logger.error("bringup %s failed: %s", path, e)
            # CellsManager 自己 broadcast "bringup-failed" 到 process log

async def __aexit__(self, *args):
    # 反向 + 必清理 proxy 防泄漏
    await self.network.update_cell(self.this, log="stopping")
    await self.network.revoke_cell(self.this, log="stopped")   # 含 terminal=True
    await self._cells.__aexit__(*args)
    await self._sub_processes.__aexit__(*args)
    await self.network.__aexit__(*args)
    if self._cell_lock:
        self._cell_lock.release()
```

**关键约束**: cells 操作不能让 matrix 爆炸. bringup 任何失败 catch 住, 写日志.

### SS-8. 第一期 channels — cells + network 两个都做

mac 控制 g1 依赖 network channel 本期可见, 不延期.

```
cells channel (本地进程治理):
  own_commands: start / stop / list
  virtual_children: opened cells 的 proxies (network.get_proxy(cell.address))
  context_messages: CellsManager.recent_logs (process 视角)
  log 焦点: started / exited / start-failed / bringup-failed

network channel (cell 发现 + 自动 proxy):
  own_commands: list / list_logs
  virtual_children: network.proxies() 自动构建的全部 (第一期 accept all = ['*'])
  context_messages: CellNetwork.recent_logs (network 视角)
  log 焦点: cell-announced / channel-ready / channel-gone / cell-gone (terminal)
```

**关键: 两个 channel 的虚拟子节点会重叠** — 本地 cell 的 proxy 既在 cells.{channel}
(因为 cells_manager opened) 又在 network.{channel} (因为 network 自动 build).
channel tree 用 channel.id() = cell.address dedupe, **同 id 只有一个 runtime**.
这是有意的 — 两个父 channel 是两个视角, runtime 实例只有一个.

模型路径: 调用时按 channel tree 实际路径走 (`<cells.web_fetch:fetch ... />` 或
`<network.web_fetch:fetch ... />` 都行, channel id 相同).

### SS-9. 三处独立 log, 零混淆

| 来源 | 内容 | 入 context 的 channel |
|------|------|-------------------|
| CellsManager.recent_logs | "started apps/web-fetch" / "exited apps/web-fetch code=0" / "bringup-failed sensors/camera: ..." | cells channel |
| CellNetwork.recent_logs | "cell-announced worker/web-fetch/uid01" / "channel-ready worker/web-fetch/uid01" / "cell-gone worker/camera/uid02" (terminal=True) | network channel |
| ProcessManager.executing / executed | OS 级 process meta | sub_processes 命令 (开发用, 不进模型 context) |

cells channel 看 "我启动的事"; network channel 看 "网络上发生的事". 两 log 互不引用.

### SS-10. ProcessManager 痛点修复

在 §RR 确定性修复清单上追加:

- **必须修**: Layer 1 `ManagedProcess` 加 on_exit callback. `execute()` 加
  `on_exit: Callable[[ProcessMeta], None] | None = None` 参数, 或 ManagedProcess
  暴露 `done_callback(cb)`. cell 进程死亡感知需要这个.

### SS-11. 确定性修复清单 (终版)

按 §RR 与本节合并:

1. **MossMeta.write_to_file bug**: `create_meta_file` 时设置 `moss_meta.file`;
   `project_id` 副作用从 write_to_file 移到 Environment 入口. 配单测.
2. **HostModeMeta.cell_dirs 重复定义** (project.py line 101 vs 143): 删一份.
3. **CellNetwork ABC 补 @abstractmethod**: 与本节 SS-3 落地一起做.
4. **CellLauncher cwd 默认补全**: read_from_file 中无论 launcher.cwd 原值,
   空就填 found_dir. 统一在 Matrix 启动时 Cell.discover 阶段获取 cwd.
5. **ProcessManager Layer 1 on_exit callback** (SS-10): 必须加.
6. **FEATURE.md §L / §N 推翻标记** ✅ 已完成.

### SS-12. 工序拓扑 (按依赖, 非严格线性)

```
第 0 步: §RR / SS-11 确定性修复 (不依赖抽象, 可并行)
        - MossMeta bug
        - HostModeMeta dup
        - CellLauncher cwd
        - ProcessManager on_exit

第 1 步: Cell 字段 + property 重构
        - CellMetadata 删 group / channel, type 开放
        - Cell.identity / channel_name property
        - 单测覆盖 identity / channel_name 生成

第 2 步: CellNetwork API 扩展 + ZenohChannelHub 补强
        - check_unique / recent_logs / broadcast_log / wait_connected
        - hub 唯一性检查 + 回调 + own liveness
        - 单测覆盖 hub 唯一性 + 回调时序

第 3 步: CellsManager 抽象 + 实现
        - ABC 定型 (list_manifests / start / stop / opened_cells / recent_logs)
        - LocalCellsManager 实现 (整合 ProcessManager + CellNetwork 订阅)
        - 单测

第 4 步: Matrix 重构
        - sub_processes / cells / network 三 property
        - __aenter__ 流程 (Step 0-4)
        - post_enter_bringup
        - 删 Matrix.spawn

第 5 步: 旧代码清理
        - 删 app_store_channel.py
        - 删 fractal_hub.py
        - 删相关 AppStore / FractalHub contracts

第 6 步: cells channel + network channel 新实现
        - cells channel: own_commands + virtual_children
        - network channel: own_commands + virtual_children
        - context_messages 渲染

第 7 步: 集成测试
        - end-to-end: matrix 启动 → bringup → cells:start → cell 提供 channel
          → notify ready → 模型调用 → cells:stop → cleanup
        - mac 控制 g1 场景验证 (network channel 远端可见)
```

### SS-13. 上下文恢复支点 (终版)

按优先级:

1. **本文件 §SS** — 开工契约, 决策最终态
2. **本文件 §MM-§RR** — 跃迁认知 / 二元真相 / 三重身份漂移历史
3. `src/ghoshell_moss/core/blueprint/cell.py` — Cell 字段定型起点
4. `src/ghoshell_moss/contracts/process_manager.py` — ProcessManager 已有契约
5. `src/ghoshell_moss/matrix/networks/zenoh_cell_network.py` — CellNetwork zenoh 实现起点
6. `src/ghoshell_moss/bridges/zenoh_bridge/_hub.py` — hub 补强起点
7. `src/ghoshell_moss/channels/app_store_channel.py` — 待删 (作为 pattern 参考)
8. `src/ghoshell_moss/channels/fractal_hub.py` — 待删 (作为 pattern 参考)
9. `tutorials/L1_hello-world-app.md` / `L2_*` — 开发体感校准

### SS-14. Audit 指标 (下个实例自我校准)

- 是否在 cell network 上做了准入策略? (应在 channel 层做)
- 是否广播了 Cell snapshot? (应只广播 CellLog)
- 是否引入了新的 enum 类型字段? (应保持 type 开放 string)
- 是否引入了"魔法值"派生? (Cell property 必须显式声明规则)
- 是否在 Matrix.\_\_aenter\_\_ 内做 bringup? (必须在 post_enter_bringup)
- 是否 cell 下线时漏 remove proxy? (必须 cleanup, 防泄漏)
- 是否 cells_manager 和 network log 混了? (零漂移约束)
- 是否模型路径中暴露了 uid? (singleton 永不出现, non-singleton 才出现)

### SS-15. Implementation Notes (不影响决策, 避免冷读踩坑)

A. **local 判定的实现**: `cell.status.project_id == matrix.this.status.project_id`.
   写在 CellNetwork.get_live_cells / recent_logs 的 local filter 中. project_id
   在 cell announce 时已写入 CellStatus.

B. **post_enter_bringup 调用点**: Matrix.arun 内, `async with self:` 之后第一行.
   ```python
   async def arun(self, main_coro):
       async with self:
           await self.post_enter_bringup()
           ...
   ```
   不在 __aenter__ 内 — bringup 失败不能炸 matrix.

C. **Matrix 不动的接口**: session / container / configs / resources / logger /
   workspace / cell_workspace / mode_home / ghost_home / runtime_scopes / arun /
   create_task / wait_closed / close / lifecycle objects. 与 cell governance 正交,
   保留接口不动.

D. **CellRegistry 位置**: 仍在 Project (访问路径 `Matrix.project.cells`).
   CellsManager 构造时持有 CellRegistry 引用, 不重复实现发现逻辑.

E. **fractal 清理范围**: 不仅 `channels/fractal_hub.py`, 还包括:
   - `src/ghoshell_moss/host/fractal/` 整个目录
   - `core/blueprint/fractal.py` (FractalHub ABC, 如有)
   - 相关 manifests / providers 中的 FractalHub 引用
   - 测试 `tests/ghoshell_moss/host/test_zenoh_fractal.py` 等

F. **apps 清理范围**: 不仅 `channels/app_store_channel.py`, 还包括:
   - `core/blueprint/app.py` (AppStore 等 contract)
   - `host/apps/` 整个目录 (如有)
   - CLI `apps_cli.py` (前轮已删, 确认)
   - workspace stub 中的 apps/ 目录与 APP.md 模板

G. **ProcessManager on_exit timing**: asyncio loop 线程 fire (process.wait 是 asyncio).
   CellsManager 内 broadcast_log 直接 await. 不像 zenoh 回调那样需要 thread-safe.
   实现可考虑 ManagedProcess 暴露 `add_done_callback(cb)` (asyncio Future 风格)
   或 execute(..., on_exit=cb) 参数注入. 实现细节由推进实例决定.

H. **CellLog ring buffer 默认大小**: 100. 通过 CellNetwork 构造参数 configurable.

I. **Cell.from_proc 行为**: 已有, 不变. 无 CELL.md 的 script 调 `Matrix.discover()`
   时, cell.meta.type='worker', name 从 `__main__.__file__` basename 推导. L3 下
   bash:exec 启动的 script 自然进入 network 视图.

J. **opened_cells 清理双路径**:
   - ProcessManager on_exit → CellsManager._on_cell_exit → pop opened_cells
   - CellNetwork.on_provider_offline → CellsManager._on_provider_offline → pop
   两路径必须 idempotent (`pop(key, None)`). 任一先 fire 都不应崩.

K. **Channel tree dedupe by id**: cells channel virtual_children 与 network channel
   virtual_children 可能含同一 ChannelProxy. channel tree dedupe 用
   `channel.id() = cell.address`. 同 id 单一 ChannelRuntime, 挂多父 channel 复用同一
   runtime — 这是 §SS-8 "两 channel 视图重叠是有意的" 的技术兜底.

L. **Worker cell 上 matrix.cells.start 行为**: 不禁止. CellsManager 是 per-process,
   worker A 启动的 cell B 是 A 的子进程, 不是 host 的. 这是合法级联 (cell 树, 非 host
   单中心).

M. **CellNetwork is_host_view 标志**: 构造时由调用方 (Matrix impl) 传入,
   `is_host_view = (matrix.this.meta.type == 'host')`. enforce `create_proxy` /
   `proxies()` 只允许 host. worker 调用 raise.

N. **singleton 含义边界**: project-scoped, 不是 network-scoped. 同 identity 在不同
   project 各自 singleton 各启一次 — L3 物理现实, 不强求.

### SS-16. 推翻轨迹 (供下一实例理解决策驱动场景)

下面是本轮做的关键删除/简化, 记下驱动场景, 防下一实例重新提议:

- **删 fractal**: 维护成本 > 实际收益. 中心化跨 peer bringup DAG 成本高, project
  入网无可靠唯一身份. 远端 cell 可见走 network channel 即可 (auto-accept all, 第一期).

- **删 apps**: 向前兼容意味着迁移不动. 干净的 cells 命名 + L3 跃迁路径上的 apps
  概念位置不清晰.

- **type 开放 + 删 group**: type / group 同维度, 双字段冗余. host/worker 是兜底保留字,
  用户自由填 sensors/bodies/tools. 不做 type registry / owner channel announce —
  manifest-only 即够.

- **status 变更不广播 snapshot**: §NN. 加 pub/sub 全网代价对低频事件不值, queryable
  按需拉就够. CellLog (轻量 event) 是另一回事.

- **bringup 不在 __aenter__**: cells 操作不能让 matrix 启动失败. matrix alive 之后
  才做 bringup, 失败 catch 不阻断.

- **CellNetwork 不做 accept 规则**: 规则放 channel 层. substrate 只承载发现 + auto
  build proxies. 准入是 channel policy, 不是 network protocol.

- **CellChannelProxies 抽象不做**: 用户原想法是 CellNetwork.proxies(local=bool)
  返回单例对象. 实际 CellNetwork 暴露 proxies() + recent_logs(local=...) + wait_connected
  足够, 不必引入新中间抽象. channel 层自己 filter.

- **HostCellNetwork ABC 分离不做**: §RR 已记. 一个 CellNetwork ABC + `is_host_view`
  构造参数即足.

- **bridge_address 与 address 二分不做**: §RR 已记. 合并为单一 `Cell.address =
  type/name/uid`. uid 模型只在 collision 时见.

- **network 多实例不做**: §PP. Matrix.network 单 primary CellNetwork. 接口位置
  留下, 实现只一个.

- **subscribe_changes 抽象不做**: cells_channel 直接读 recent_logs (pull) +
  watch hub 回调 (on_provider_online), 不需要中间 pub/sub 层.

