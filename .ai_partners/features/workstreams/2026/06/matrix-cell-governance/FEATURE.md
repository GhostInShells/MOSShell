---
title: Matrix Cell Governance
status: in-progress
priority: P0
created: 2026-06-09
updated: 2026-06-10
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
  2026-06-10 moss nodes run 解析规则 + CELL.md 命名锚定。
  CELL.md 统一替代 NODE.md，向上查找同 MOSS.md。executable 默认 sys.executable，
  不碰 uv run。至此 matrix cell 治理设计闭环——人类架构师实现，模型 review。
  剩余：CellType.node、CELL.md、moss nodes CLI、shell-init、apps 迁移。
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
