---
created: 2026-08-05
depends:
- node-lifecycle
- desktop-gui
- memento-cli-and-agent
description: claude code 作为平行智能体接入 MOSS。Ghost 是主持人,统筹 N 个 claude session, 人类与 ghost
  共享会话上下文。核心价值:日常开发过程产品化、可被第三方观测。 loop = node 内 executor 沙箱动态编译的纯 python 代码协议;Claude
  抽象内置 通讯/授权/MCP/审批,API 自解释,模型经 get-interface 反射即会写 loop。 审批走官方 --permission-prompt-tool
  stdio 控制协议。v1 claude code 专属。
milestone: null
priority: P1
status: dropped
status_note: superseded by dsh-productization; DeepSeek Harness is open-source with
  a vendorable web frontend and formal SDK/ACP protocol surface
title: Claude Code In Moss — Ghost 主持人统筹 N 个 claude 会话
updated: '2026-08-14'
---

# Claude Code In Moss

> 人类架构师 + deepseek-v4-flash。2026-08-04/05 三轮讨论:从"旁路能力"演化为
> "平行智能体",再收敛到"executor 沙箱跑 loop"的最终形态。本篇沉淀设计结论;
> 完整碰撞轨迹在会话中,待落 discuss。

## Motivation

把 claude code 集成进 MOSS,让 Ghost 作为**主持人**统筹 N 个平行 claude code
session,人类与 ghost 共享这些会话的开发上下文。

为什么值得做(人类架构师的四个理由):

1. **立刻让日常开发过程产品化,可被第三方观测。** 用户自己的认知模式就是并行任务:
   聊方案→沉淀 feature→features list 看状态→进会话才加载上下文→忘了就 recall。
   本 feature 是把这套模式工程化——ghost 替用户把脑子里没落地的 tasks 落地成账本。
2. **本身是好产品,行业在做,MOSS 做几乎是免费送的。** claude code 生态被行业持续
   打磨,MOSS 的 node/matrix/features/qa 基建恰好是承接它的完整容器,边际成本趋近于零。
3. **对 ghost 有用。** 运行时自迭代多了一个手段——ghost 可以驱动 claude 完成开发任务。
4. **L2 架构师的提前准备。** v1 的 `talk` 命令面就是 L2 supervisor agent 的雏形槽位。

核心模型:

```
人类 ⟷ Ghost ⟷ N × claude code session
```

Ghost 是必经桥梁:它知道人类的输入、每个 session 的暂停点、任务状态。claude session
对 ghost 而言是 **background task**,ghost 不盯梢,靠"可观测"快速知道每个 session
在干什么。真上下文三方都可以看;全自动场景演化为 `ghost → supervisor → claude code`,
ghost 只负责动机(supervisor 是 v2)。

## Design Index

- Key design documents: `design/`(暂无)
- Key discussion records: 会话中(2026-08-04/05),待落 `.discuss/`
- 参考工作流:
  - `memento-cli-and-agent`(completed)— loop 定案复用;loop 不是 CLI 职责
  - `node-lifecycle`(in-progress)— open/read 认知入口、ghost 侧记忆、依赖门控
  - `desktop-gui`(in-progress)— Reflex 面、GUI 是人类窗口 Ghost 无感
  - `qa-exchange`(in-progress)— 跨进程审批转发的 qa 机制
  - `matrix-cell-governance`(completed)— 三域模型(Manifest/Record/Presence)

## Key Decisions

### 1. 定位:平行智能体,不是 MCP 工具

claude code 是独立 agent 实体、有自主 loop,在编排上受 ghost 主持。压成 MCP 工具
(社区 claude-wrap-mcp 模式)等于每次一次性 `claude -p`,丢失 session/流式/审批即对话
——恰恰是 MOSS 看中它的原因。

**v1 claude code 专属,v2 通用抽象直接取代 v1(不迭代)。** moss claude mode 本身是一个
独立产品("语音多进程 claude 管理"),为产品设计,不为生态预支抽象。v1/v2 的桥在
**数据形态不在接口**:task ledger、快照、事件以纯数据留存,v2 直接复用数据、重写接口。

### 2. Signal 机制:做法 2 为基座,做法 3 分级,做法 1 显式接管

三种做法不是排他,是分层:

- **做法 3(协议层)**:exit code 0/非 0 定义 signal 级别(非 0 → critical 立即介入;
  0 → 普通稍后处理)。免费、可靠的兜底。
- **做法 2(通知层,基座)**:signal 是通知不是内容。ghost 收到后 poll 拿详情,忘 task
  调 recall。task 概念是 MOSS 给 claude session 额外赋予的,用 features 体系理解。
- **做法 1(驱动层)**:显式接管——ghost 主动决定"进入这个 session 并回复"。

理由:做法 1 默认会让 ghost 被 N 个 session 牵着走(谁有返回值催 ghost 回复),主持人
失去主动性;做法 2 让 ghost 保持主持,复刻人类并行工作模式(features list 看状态→
进会话才想上下文→忘了就 recall)。

### 3. Task = features 体系的运行时化

**FEATURE.md 是任务的静态声明,claude session 是任务的运行时实例。** session 索引
不需要新抽象,它就是 features 板的运行时视图。stage 0 产出的"类似 feature.md 的任务"
应该就是 features 体系的一条。

task 的完整上下文注入 = **cwd + FEATURE 文件绝对路径 + env** 三者。claude session 天然
绑定 cwd(`~/.claude/projects/<encoded-cwd>/`),features 有 init、在 cwd 里可读,claude
code 直接读 FEATURE.md 当任务上下文,不需要翻译层。node 确认 project path 后为任何
项目创建环境——**node 是项目环境的工厂**。contract/channel 入参约定收敛为:
`project_path / feature_path / env`。

### 4. Contract:三层抽象,只暴露 Task 层

```
ClaudeTaskSpec      声明:名称、任务文档引用(FEATURE)、env、代理参数
ClaudeTask          句柄:snapshot() / stop() / resume() / wait()
ClaudeTaskSnapshot  观测:状态、session_id、最近事件窗口、摘要
ClaudeTaskHub       治理:submit / jobs / get / new / 生命周期 (per-owner)
```

- **新建契约,不扩展 JobSupervisor**。JobSupervisor 的 fold 语义(短命进程重复执行)与
  process 事件语义(单长命进程 + 中断 signal)不同。但**模式复用**:per-owner、
  snapshot、wait、owner 死 task 死——全是 JobSupervisor 已确立的纪律。
- **双轨可观测**:snapshot(按需拉)+ signal(事件推),contract 里都要。
- **两面分离**:Claude 句柄(问-答面,loop 拿 `ask`)/ Session(生命周期面,node/ghost
  拿状态/top/interrupt/帧/log/摘要)。loop 不需要生命周期控制,分离保持 loop 协议最小。
- Task / Session / Process 三层:ghost 只和 Task 打交道,Session(claude 的 JSONL)和
  Process(运行机制)是适配层内部。

### 5. Channel:薄命令面 + node 承载

```
claude_code.list                  IM 拓扑列表:N 个 session + 状态红点
claude_code.open <task>           认知入口:任务文档 + 活状态(第一帧)
claude_code.recall <task>         回忆上下文(做法 2 的 recall)
claude_code.observe <task>        拉最新快照/帧(做法 2 的 poll)
claude_code.talk <task> "text"    直接对话(启动方式二 / stage 3)
claude_code.loop <task> <def>     启动自主 loop(启动方式一 / stage 1-2)
claude_code.interrupt/resume/stop/top
```

- 命令面薄,模型看到即会用。open/read 语义模仿 node-lifecycle。
- **claude-code 是一个 node,不是裸 channel**:node 承载生命周期治理(依赖门控:
  claude 二进制/credential 检查 → INSTALL.md 门控),内部包 channel 命令面。
- **v1 的 stage 3(ghost 自主对话)走 `talk` 命令,不引入 supervisor agent**。
  supervisor 是 v2——v1 的 ghost 对话就是显式接管命令,蓄力 L2。

### 6. 数据存储:三层,避免复制 claude 全文

| 层 | 归属 | 内容 | 生命周期 |
|---|---|---|---|
| Task ledger | ghost 拥有(ghost_home) | task 身份、FEATURE 关联、状态、摘要、session 引用 | 持久化,长期 |
| Session 历史 | claude 拥有(JSONL) | 全文对话/工具调用 | ghost 只读不持,按需下钻 |
| Signal buffer | 运行时 | 事件流 | 短时,处理即弃 |

task ledger 就是 features 板的运行时视图,记录"FEATURE ↔ claude session 运行时状态"
映射,不是新账本。记忆模式复用 node-lifecycle 的 `NodesMemoryContract` 风格
(ghost_home/memory/)。

### 7. GUI + 模型视图:IM 拓扑

GUI 和模型看到的都是拓扑类似 IM 的界面:

- **会话列表**:一堆 session + 状态红点(running / stopped / achieved / error),
  一眼看到 N 个会话在干嘛。
- **进入会话 = 拉取最新帧**(ask → response 序列)。
- **状态更新时序** = 较小的多行 log。
- **会话状态简述** = 最后一句话(列表里的一行摘要)。

**启动信息(Feature.md / SKILL.md 类比)对第一帧和 compact 是必要的,最好是可读
文件路径**(FEATURE.md)。compact 定位:ghost 默认只持摘要(最后一句话)+ 启动信息
(文件路径);上下文要压缩时用文件路径 + 摘要重建第一帧,不持有全量帧——这正是
"共享上下文 ≠ 全量进 ghost"的落地。

人类 GUI 操作 → signal → buffer → ghost 知觉(做法 2 人类通道,desktop-gui K1 不破)。
**审批 node 是 GUI/TUI 的聚焦点**:pending 审批最显眼。

### 8. 审批机制:官方 `--permission-prompt-tool stdio`,不依赖 MCP

调研结论(claude-code-guide, 2026-08-05):

- **官方原生通道 = `claude -p --permission-prompt-tool stdio`**:CLI 发 `control_request`
  事件(`request.subtype: "can_use_tool"`,带 tool_name/input/request_id)并阻塞等 stdin
  的 `control_response`(allow/deny,~60s timeout)。这是 headless 下唯一的
  signal-and-routing 审批通道,不是 MCP 协议——**"审批依赖 MCP"是社区包装,不需要**。
- **PermissionRequest hook 在 `-p` 下不可靠**:社区多次确认不触发(#40506/#33343),排除。
- **无该 flag 时**:会 prompt 的工具直接 auto-deny(静默失败),最终 `result` 带
  `permission_denials[]`。所以要让审批进 GUI,此 flag 是必经之路。
- **SDK `can_use_tool`**:async 可阻塞等人类,最干净,但 in-process + streaming,
  违反 MOSS 进程隔离,列为 v2 备选。

MOSS 侧架构:claude-code node 持有 claude 进程 stdio(代理机制),审批请求经 qa 跨进程
(zenoh_qa)到审批 node(GUI/TUI),人类决策 → qa 回传 → 写 control_response。ghost 订阅
审批事件感知(做法 2)。**node 是审批通道的代理,不是审批决策者——决策权在人类,
知觉权在 ghost。**

v1 默认 `permission_mode=default` + control protocol,全审批进 GUI("审批即对话"完整
落点,desktop-gui 借鉴 claude 审批模式后第一次实装回来)。

### 9. Loop 驱动:executor 沙箱动态编译,不是进程

**loop 不是进程、不是 cell——是 node 沙箱里动态编译的纯 python 代码协议。**
"loop 用什么驱动"这个纠结由此消失:不需要 bash、不需要常驻 loop 进程、不需要
script 类 cell。

```
claude-code node (带 GUI, 唯一进发现体系的 cell)
  ├─ 一个 claude code channel (契约治理所有 session + 发现逻辑)
  ├─ 持有 N 个 claude 子进程 (stdio/审批/事件流, 不进 matrix)
  └─ Executor 沙箱 (core/codex/executor.py)
       ├─ local_injections 注入 claude 句柄 (IoC)
       ├─ 动态编译 loop 函数 (模型写的纯 python, 模块级隔离)
       └─ LoopRunner 在 asyncio task 里 await 迭代
```

模型只需要定义一个函数:

```python
async def loop(claude: Claude) -> AsyncIterable[str]: ...
```

Executor 能力核对(executor.py, 2026-08-05 实读):
- 动态编译 code → 独立 `__execute__` module,不污染原始 module。
- `local_injections` 注入 claude 句柄(控制反转)。
- `func_name="loop"` 取函数调用,`__all__` 跨调用继承(同 session 带状态)。
- **它是编译/组织沙箱,不是安全沙箱**(module 隔离,无受限执行)——信任模型:
  模型写的 loop 代码可信。

**LoopRunner(node 自写)**:编译 → 注入 claude 句柄 → await 迭代 AsyncIterable →
转发进度/事件到 ledger/GUI。Executor 不管 async/迭代,只管编译+取函数+调用。

**N 个 session = N 个 loop 实例**:共享编译后的 loop module(只编译一次),每 session
独立 Executor 实例注入该 session 的 claude 句柄,独立 asyncio task 跑。node 的 asyncio
统筹 N 个——这就是"统筹"的物理形态。

**审批解耦**:loop 的 yield 是"逻辑进度",审批是"传输层等待"(claude.ask 内部)—
loop 完全不感知审批,协议保持最小。

**Node 启动开销实测**(2026-08-05, system_test hello_world):
- `nodes run`(完整 node 治理):**2.79s** = 发现/spawn/env/ledger + 进程/入网。
- 裸 `python main.py`(无治理):**1.32s** = 进程 + matrix 入网。
- 增量 ~1.5s 是 node 治理(发现/spawn/env/ledger/身份);固定 ~1.3s 是进程+入网。
- 对 claude 场景:loop 不是 cell,省掉整个 node 治理增量;claude 常驻会话,入网摊薄。

### 10. 阶段(可迭代目标,先做到这个阶段就 ok)

- **stage 0**:讨论开发任务,ghost 负责记录(自然语言+语音+GUI)。完成后创建类似
  feature.md 的 claude 任务。
- **stage 1**:围绕任务写 loop 实现(模型写完 loop 定义为止)。
- **stage 2**:启动后台运行——**process 逻辑,不是 job 逻辑**(事件驱动,非轮询快照)。
  每次中断时 signal 发给 ghost,ghost 感知状态变更。ghost 可代替人类驱动;人类也可
  通过 GUI 绕开 ghost 驱动(ghost 用 buffer signal 感知人类动作)。
- **stage 3**:ghost 自己读 stage 0,针对 stage 2 和 claude session 做对话。

**N 路统筹贯穿所有 stage**:buffer 是 N 路的,signal 必须带 session 身份(mindflow 仲裁
多个 process 的中断 signal)。

### 11. 代理机制

静态讯息(project_path / feature_path / task id / env / MCP 配置 / CLAUDE.md 路径)
全部环境变量化——claude code 无感,零协议成本。动态讯息(审批、运行中变更)由
Claude 抽象内置处理。**运行中状态变更进不了环境变量,动态通道是 Claude 抽象的一部分**
(审批走 control protocol;方向变更走 claude.ask 新提示)。

### 12. Claude 抽象:API 自解释,机制全内置

**Claude 抽象是全部重点。** 通讯协议、授权机制、MCP 配置、审批全内置,对外只暴露
自解释的最小 API。模型认知入口 = `moss codex get-interface` 反射 Claude 接口——
**接口契约就是模型怎么写 loop 的说明书**。验收标准:陌生模型看到 get-interface 反射,
就会写 loop。

- **Claude 句柄(问-答面,loop 拿)**:`ask(prompt) -> str`,阻塞时内部是审批等待,
  loop 无感。
- **Session(生命周期面,node/ghost 拿)**:状态 / top / interrupt / 帧 / log / 摘要。
- 两面分离保持 loop 协议最小;生命周期是宿主的事,不是 loop 的。

### 13. 异常模型:ClaudeError 是唯一失败语义

- loop 可 try/except 处理自己的异常。
- **不处理 → ClaudeError 冒泡 → LoopRunner 捕获 → signal → ghost 感知**(做法 2)。
- 同一条错误通道,无论 loop 处不处理,ghost 都能感知——不需要第二条异常路径。

### 14. loop vs talk 二分启动

Ghost 启动一个 claude session 只需二选一:

- **loop 启动**:给 session 任务 + loop 定义,自主执行迭代完成(stage 1/2)。
- **talk 启动**:直接对话,ghost 发消息拿回复(stage 3)。

启动后都是**后台运行**(Session 持有异步执行,对中断敏感),可 `top` 查询状态,
通过 channel command 主动拉会话数据。对 Ghost 是一个全异步运行载体。

## Implementation Notes

- **第一个验证(必做)**:裸 `claude -p --permission-prompt-tool stdio` 的
  control_request/control_response 在目标 claude 版本上是否如文档工作——claude 的
  hooks 已出现过文档与行为不符先例(#40506),此协议是审批设计的全部底座。
- **第二个验证**:stage 0→1 衔接的任务文档格式——它是整条链的输入质量,claude code
  的所有表现从这里开始。验证:一份任务文档喂给 `claude -p`,loop 生成得够不够稳。
- **60s 审批超时**:版本相关。审批 node 上 pending 审批最显眼;超时语义(人类没回→
  auto-deny)要让 ghost 知道。
- **headless/SDK 计费独立于交互 credit pool**(Pro $20 / Max $100-200, 2026-06 起)。
  集成是独立成本模型,影响使用策略。
- **desktop-gui K1 张力**:GUI 是人类窗口 Ghost 无感 vs ghost 主持 session 状态。
  判断不冲突——session 状态是 ghost 的领域,GUI 只是渲染给人类;但需显式对齐,
  避免踩 K1 否决线(ghost 控制 GUI 渲染)。
- **executor 是编译/组织沙箱,非安全沙箱**:loop 代码信任模型,不信任任意输入。
- **script 类(matrix 无记账协议)是独立命题**:claude code 场景下 loop 和 claude 子进程
  都不是 cell,不需要 script 类;但 memento/未来动态 worker 可能撞上,值得后续在
  matrix-cell-governance 层面单独推演(2026-08-05 讨论留痕)。

## Next Steps (迭代实施, 约一周)

1. **数据结构** — Session / 帧 / log / 摘要(第一步,决定一切)。
2. **Claude 抽象 + 反射验证** — 一体,以 get-interface 效果为验收(陌生模型
   看到接口就会写 loop)。
3. **Contract + Session 体系抽象** — ClaudeTaskSpec/Task/Hub 落地。
4. **channel 验证** — claude_code 命令面。
5. **GUI** — IM 拓扑(会话列表 + 红点 + 最新帧 + 多行 log + 最后一句摘要)。

实施时间待定:等两个高优 workstream 跑完后集中突击。
