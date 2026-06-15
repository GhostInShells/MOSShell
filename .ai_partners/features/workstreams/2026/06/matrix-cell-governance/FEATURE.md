---
title: Matrix Cell Governance
status: in-progress
priority: P0
created: 2026-06-09
updated: 2026-06-15
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
  2026-06-15 协议层重新对齐 + 实施纲领落地：address 自由度收敛、type 开放命名空间 + owner
  channel announce、spawn 二分（spawn_cell / spawn_worker + worker/{uuid} 兜底）、跨进程
  异步基底是 process + Matrix 异步回调机制。本轮新增决策：nodes 进 manifests（待拍板）、
  命名分层（codex / manifests / runtime + 域运维）、script 体系整体删除。推进由人类主导，
  拆 branch 治理。下一实例认知重建支点见 FEATURE.md §A-§G。
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


