# Desktop 在 4 剪影拓扑中的位置 — Ghost 反身性基建的空间脏器

> 与 `2026-06-28_desktop_l2_emergence.md` 配对的声明式设计结论。
> 讨论轨迹见 discuss 文件；本文件给出契约与判据，供实现期参照。

## 1. 4 剪影拓扑

Ghost 在世界中的存在是一条连续的上下文轨迹。这条轨迹的不同时空剪影在 MOSS
里各自有物质载体：

| 剪影 | 朝向 | 物质载体 | 不变性原语 |
|------|------|----------|------------|
| **过去** | 已发生的认知 | Memento | commit_id (content-addressable) |
| **未来** | 待执行的动作 | Desktop | path + cwd + pin schedule |
| **当下** | in-flight 状态 | Matrix runtime + Mindflow | active channel state、pending signals |
| **结构版本** | 物质形态的命名 | Git worktree | tree hash + branch ref |

四个剪影非可替代：

- 没有"过去"，Ghost 是没有连续性的、每次重生为新人
- 没有"未来"，Ghost 是悬在虚空里的回忆，没有当下能操作的世界
- 没有"当下"，过去和未来之间没有连接点，archive+plan 不是 alive
- 没有"结构版本"，物质回滚不可能，反身性失去退路

Desktop 是这套拓扑里的**空间剪影 / 未来脏器**。设计 Desktop 的所有决策
必须在 4 剪影拓扑下评，不能当独立模块设计。

## 2. 完备性判据：每个剪影对其他剪影可寻址

反身性要求**任何剪影内部的语义可被其他剪影查询、引用、快照**。

具体到 Desktop 与其他剪影的可寻址性：

| 谁查谁 | 寻址形式 | 契约 |
|--------|----------|------|
| 当下 → 过去 | `memento.show(commit_id)` | 从 mindflow 发起，返回历史 commit 内容 |
| 当下 → 未来 | `desktop.snapshot()` | 返回 DesktopState（pwd、active pins、bg tasks） |
| 未来 → 过去 | CTML 结果自动 → Moment | 通过 CTML interpreter 完成，Desktop 无需感知 |
| 未来 → 当下 | Desktop 命令读 matrix state | 通过 channel runtime 间接，Desktop 不直接耦合 matrix |
| 任何剪影 → 结构版本 | git 命令 via Desktop.exec | git 是工具，不是 Desktop 的特殊客户 |
| 结构版本 → 任何剪影 | worktree fork → mirror Desktop / Memento | 见 §6 反身性的对称 fork |

任何一个寻址断裂，反身性在那个边界失效。这是最小完备性条件。

## 3. Desktop 在拓扑里的边界

### 3.1 Desktop 持有什么

- `root: Path` — 空间剪影的边界（构造期固定）
- `pwd: Path` — 当前活动光标（in-flight 状态的最小投影）
- `tmp_root: Path` — 截断输出的回收点（构造参数，缺省 `root/tmp/desktop/`）
- `_pm: ProcessManager | None` — 可选执行底层
- `_procs: set` — 持有的 in-flight 子进程，用于 shutdown 清理
- `_pins: dict[str, PinRecord]` — 周期性命令的注册表

### 3.2 Desktop 不应持有什么

- ❌ `_read_set` — 这是 Ghost 的 epistemic state，属于 Memento branch，
  必须经 `ReadHistory` protocol 注入
- ❌ atom / 任何特定 ghost 的预设 — instruction 模板、反思路径白名单、
  pin 预算阈值都必须是构造参数
- ❌ Memento / Matrix / Session 的直接引用 — Desktop 通过返回值发信号，
  上层路由

### 3.3 ReadHistory 协议

```python
class ReadHistory(Protocol):
    def has_read(self, path: Path) -> bool: ...
    def mark_read(self, path: Path) -> None: ...
```

- 缺省实现：进程内 `set[Path]`，单测用
- Phase 4 实装：由 Memento branch state 后置——commit 时进快照，
  fork 时跟着 base pointer 继承，switch branch 时切换 read history 上下文

理由：`write` / `edit` 拒绝写入未 `read` 的文件，这条守卫的语义是
"Ghost 在当前认知轨迹上至少看过这个文件"——这是 Ghost 的 epistemic 状态，
不是 Desktop instance 的工具状态。一个 Ghost session 内 Desktop 可能被
多次实例化（探不同子目录），read 历史必须穿透实例边界。

## 4. Pin 与 moss_dynamic 的对齐

Pin 是 Desktop 在"未来剪影"上的周期性观察声明。Pin 内容通过 channel 层
的 `moss_dynamic` 机制注入 prompt——"以最后出现的为准"语义让 cache 爆炸
不成立（覆写式更新，不累加历史）。

### 4.1 Pin 内容的位置

Pin 内容**必须**落在 prompt 的**当下段**（即 staging 区，commit 边界之后），
不能落在 cache 边界之前。理由：

- Memento commit 边界是天然的 cache breakpoint
- Pin 输出每轮变，落在 cache 之前会持续 invalidate 已 cache 的历史段
- Pin 在语义上本就是"我现在正在观察什么"，属于当下，不属于历史

具体放置由 channel 封装层决策，Desktop 自身不知晓 prompt 结构。

### 4.2 Pin 预算

无节制 pin 导致认知爆炸（不是 cache 爆炸——是模型注意力分散）。
Desktop 维护 `max_pins` 构造参数（缺省 16），LRU 淘汰，命中上限时
返回值带 `pin_budget_warning`，让模型自主取舍。

### 4.3 Pin 元参数纪律

`_pin: bool = False` 是 CTML 元参数，`_` 前缀约定。元参数集合可扩展
（`_bg`, `_pin`, 未来可能 `_silent`），共同遵循"不污染业务参数空间"
原则。

## 5. 高影响路径与 Reflection Hint

Desktop 维护 reflection 路径白名单（构造参数），默认包含：

```
.moss/    .git/    DESKTOP.md    CLAUDE.md    pyproject.toml    MOSS.md
```

`write` / `edit` 命中白名单时，返回值附 `ReflectionHint`：

```python
@dataclass
class ReflectionHint:
    path: str
    diff: str
    severity: str  # "config" | "instruction" | "vcs"
    recommend_commit: bool = True
```

Desktop 不直接调 Memento。channel 层把 `recommend_commit=True` 翻译为
建议 emit `<memento:commit summary="pre substrate change"/>`。

这是 reflection（事后给信号）+ memento（提供锚点）的最小协作单元，
为 Phase 5+ 的 sandbox+keyframe 提供 pre-write anchor 机制。

## 6. 反身性的对称 fork

真反身性（"思考镜像 + 关键帧推理验证"）要求**时间轴和空间轴同时可 fork**：

```
反身性操作（例：Ghost 决定改 .moss/manifests/...）
  ↓
1. memento.current().fork(name="trial-{ts}")
     → 时间轴 fork: base pointer 复制，新 branch staging 干净
2. desktop.fork(name="trial-{ts}")
     → 空间轴 fork: worktree / CoW snapshot，mirror desktop 指向 fork 目录
3. 在 fork 出来的 Memento branch 上跑 mirror Ghost
   对 mirror desktop 执行写入
4. 关键帧校验:
   - config 仍 parseable
   - channels 仍能 startup
   - prompt 仍在 token budget 内
   - test suite (可选) 仍 pass
5. 通过 → memento merge + desktop commit-to-real
   失败 → 丢 fork, real desktop 和 real memento 都不动
```

Memento 已经原生支持 fork（base pointer 方案 A）。Desktop 的 fork 在
Phase 5+ 实装，底层可选 git worktree、btrfs snapshot、或 cp -r + mirror
Desktop instance。

**Phase 1 的约束**：`root: Path` 必须是构造参数（已经是了），fork 时
只是"用 mirror 路径 new 一个 Desktop"，无需改核心。

## 7. 原语清单与 12+1 形态

按"原语必须有无法被组合替代的语义"裁剪后：

| 层 | 原语 | 备注 |
|----|------|------|
| 导航 | cd, pwd | 当下剪影的光标管理 |
| 发现 | tree, glob, grep | 空间剪影的结构观察。grep 不硬编码后缀，respect `.gitignore` |
| 读取 | read | 包含 offset/limit/_pin。`head` 删除（可由 `read(limit=N)` 替代，且原 `head` 不进 read_set 是 read-before-write 的 bypass） |
| 写入 | write, edit | 经 ReadHistory 守卫；命中白名单返回 ReflectionHint |
| 执行 | exec | `_bg: bool` 元参数支持后台 |
| 后台 | tasks | 返回结构持 `read()` / `cancel()` 方法，收掉独立 read_task/cancel |
| Pin | pinned, unpin | pin 状态查询与移除 |
| 元 | frontmatter | 可选保留——作为 markdown YAML 头提取原语。L1 试用后决定去留 |

总计：12 + 1（frontmatter）。比原 17 收缩 5 个。

## 8. Ghost / OS 分层纪律

Desktop 是 OS 层抽象。day-to-day 评估每条决策时问：

> 这条改动是 OS 层（为任何未来 ghost 服务）还是 ghost 层（为当前的
> atom/echo 服务）？

OS 层判据：
- 可被任何 ghost 复用，不依赖特定 ghost 的认知结构
- 可被任何其他 ghost 模拟（sandbox+keyframe 场景）
- 可被任何第三方继承（贡献给行业的中间层）

Ghost 层判据：
- 特定 ghost 的 instruction / 审美 / 行为偏好
- 应该放在 ghost 自己的 system prompt / DESKTOP.md 里
- 不应硬编码到 Desktop core

OS 层和 ghost 层都是必要的，但**不能互相借预算**。任何"为 atom 方便而
妥协的 desktop 设计"都污染 OS。

## 9. Phase 切分

| Phase | 范围 | 依赖 |
|-------|------|------|
| **Phase 1** | core/desktop/ 独立模块。contracts/desktop.py ABC + ReadHistory protocol + 12+1 原语 + 元规则 + ReflectionHint + 单测 | 无 |
| Phase 2 | 反思路径白名单完整集成（默认值、可配置）| Phase 1 |
| Phase 3 | ProcessManager 注入路径接稳（cwd 一致、kill 一致） | Matrix Cell Governance |
| Phase 4 | Channel 封装。pin 接 moss_dynamic（落 staging 段）。ReadHistory 切到 Memento branch 后置 | Memento Phase 1-3 |
| Phase 5 | PendingApproval variant + Future 机制接入 | Future 通讯基线 |
| Phase 6 | 对称 fork（memento branch + desktop worktree）+ 关键帧校验 | Phase 4-5 全部 |

Phase 1 是 L0 独立闭环——不依赖 Matrix、Memento、Channel 任何外部体系。
可单测，可滚动，模型可在人类工程师做其他重构时穿插推进。

## 10. Phase 1 acceptance 边界

- 12 原语（+frontmatter 可选）的契约用 ABC 表达
- `ReadHistory` protocol + 进程内缺省实现
- read-before-write 守卫在 write/edit 上正确触发
- 统一输出截断 + tmp_path 路径不重复截断
- 反思路径白名单触发 `ReflectionHint`
- Pin 注册、查询、移除、LRU 淘汰
- ProcessManager 注入 vs 裸 subprocess 两条路径行为等价（cwd 一致）
- 12 原语全部覆盖单测，read-before-write / 截断 / pin LRU / reflection 边界各有专门单测

## 11. 已知未决问题

- **DESKTOP.md 的写守卫**：当前白名单触发 ReflectionHint，但不阻止
  写入（反身性立场）。是否需要"两步确认"机制（write 标记 pending，
  下一帧确认）尚未定。Phase 2 决策。
- **frontmatter 去留**：L1 试用后决定。倾向删除——可由 `read(limit=20)`
  + 模型自己解析 YAML 替代。
- **Pin 嵌套语义**：CTML 里 pin 可以是 CTML 子程序，pin 内容再触发
  pin。这条 schedule(ctml, refresh) 形态留待 Phase 6+。
- **跨 worktree 的 Pin 行为**：fork 时 pin 集合应继承还是清零？倾向
  继承但标记 `forked: true`，避免 mirror Ghost 把 trial 状态当 baseline。

## 12. 与已有抽象的关系

- **GhostWorkspace**：Desktop 不耦合。GhostWorkspace 可以作为 Desktop
  的 root 来源（`desktop = Desktop(root=workspace.path)`），但 Desktop
  作为 contract 不知道 GhostWorkspace 存在。
- **FileEditor**：不再独立设计。Desktop 的 read/write/edit 已覆盖。
  原 FEATURE.md 的 §13 子 channel 架构图属于早期意图，**Phase 1 不
  做子 channel 嵌套**。
- **terminal_channel**：保留独立存在（pexpect 持久会话有自己的语义），
  与 Desktop 平级，不进 Desktop 子 channel。
- **ProcessManager**：Desktop 的可选执行后端。注入后接管 exec / exec_bg；
  未注入时 Desktop 走裸 asyncio subprocess 兜底。

---

*Claude Opus 4.7, 2026-06-28, via claude code*
