---
created: 2026-07-18
depends:
- momento-mori
description: 在 ghost 融合之前，用一个 CLI 驱动、无 harness 的最小 agent 把 memento 全部边界画完并投入真实使用。AGENT.md
  目录 + .memento/ + bash/file_editor + pydantic-ai 原生序列化。可行性判据：讨论定案后模型不用人类干预独立做出来，
  做不出来果断放弃。
milestone: null
priority: P0
status: in-progress
status_note: '2026-07-26 §12 上下文窗口与压缩体系定案; (a) 阶段完成:
  impl.invoke() 接线 memento staging (pydantic-ai messages dump → MomentRecord),
  CLI 回归 9/9 全绿, calc.agent.py 投入验证, regression set 建立.
  迭代路径: (a) 无压缩 memento ✓ → (b) context window → (c) compact agent → (d) memento tools.
  压缩级别数据模型在 (a) 阶段已预留字段.
  2026-08-04 §14 零上下文摩擦测试 + 669e0e18 review; delete tombstone/spec v3/gitignore
  验证通过; 新发现 tombstone fork_ref/created 语义缺陷与 spec -D 残留 (§14.2).
  剩余: §13.10 修复轮 #1/#2/#3/#5/#6/#7 + §13.9 explore agent + §14.2 两个新缺陷.'
title: Memento CLI & Agent — 无 harness 的轨迹 agent，memento 边界的 dogfooding 验证器
updated: '2026-08-04'
---

# Memento CLI & Agent

> Use `moss features set-status memento-cli-and-agent <status> -m "note"` to update state.

## 0. 给施工化身：先读这一节

**这份文档是 2026-07-18 一场可行性讨论的定案，钉子已经人类裁决，不要重开。**
设计上下文的另一半在 `momento-mori/FEATURE.md`（尤其 §14 存储布局修正——
与本 workstream 并行施工，见 §6 协调点）。

**可行性判据（人类原话级约束）**：这件事只在一种情况下有价值——讨论完方案后，
不用人类干预，模型完全独立做出来。**放弃触发器**：施工中发现 runner 开始长出
harness 器官（session 管理、流式仲裁、消息史复活），说明"无 harness"前提破了，
立即停，向人类报告而不是硬撑。沉没成本只有讨论，契约层无论如何都立着。

## 1. Motivation

memento 状态 contract-frozen-pending-review，验证路径原计划走 data-ghost。
但 ghost 融合成本高、周期长。本 workstream 的判断：**bash + file_editor 实装后，
一个无 harness 的 CLI agent 就能把 memento 的全部边界画完，且它本身有用**：

- **每次 invocation 就是一次重启**——重启连续性（最硬的验收标准）从测试项
  变成存在前提，每次使用都在验。
- **invocation 边界 = 天然机械 commit 边界**——commit 触发规则自动解掉一半。
- **这里的 adapter 就是 Data ghost `_adapter.py` 的地基**——不是绕道，是把
  Data 的骨架先画在更小的身体上。
- **无 harness = 潜力**：状态全在 `.memento/`，loop 编排 = 外部重复 invocation
  （cron/while 均可）；channel 化路已铺好（typer_channel + moss-self-channel
  先例，`moss memento agent` 自动具备反射成 channel 的资格）。
- **prompt 与 memento 分离是设计目标**：prompt 在轨迹外、hash 打戳在轨迹内 ⇒
  每次 prompt 修改都是可归因实验。主线自调优 = 编辑 AGENT.md + commit 锚定；
  分支实验 = fork + `BranchMeta.overlay` 带 prompt 变体。两个机制契约里都现成。

验证的"自洽标准"：**能力 → CLI 动词 → 契约条款矩阵**。memento 每个契约能力
都有一个命令行可观测的骨架动作，矩阵填满且每格可演示 = 自洽，才启动 data-ghost。
fork 格标"形态已画出（派生 owner + overlay），不在本轮验收"。

## 2. 分层栈（三个形态是一个栈，不是选项）

```
abc.py + fs_memento (冻结契约 + 参考实现, momento-mori 所有)
  → codec adapter: pydantic-ai 原生 dump ↔ MomentRecord.payload
  → facade + policy: remember / window / checkpoint + commit 策略   ← ghost 融合时原样带走
  → MementoAgent (python 锚点, 独立模块, import 即用)
  → moss memento CLI (typer 皮, 进 cli/)
```

## 3. Key Decisions（已裁决的钉子）

1. **codec = pydantic-ai 自有序列化协议**。venv 实测 2.5.0：
   `ModelMessagesTypeAdapter`（`list[ModelRequest|ModelResponse]` 带判别符
   TypeAdapter，dump_json/validate_json 全量往返）+
   `AgentRunResult.new_messages_json()`。**不需要现场编译 python**。
   信封契约不认 Moment 恰好让 payload 直接装原生 dump——两个 codec 并存
   （ghost 路径用 moss.moment/v1）把信封可剥离性变成活的事实。
2. **Moment 记录行形状**：

   ```json
   {"t":"moment","id":"<ulid>","created":"...","type":"pydantic_ai.messages/v2",
    "payload":{"messages":[/* 原生 dump 原字节 */],
               "prompt_sha":"...","window_stamp":[["cmt_x",2]]},
    "threads":[],"by":"memento-agent/<model>"}
   ```

   - 一个 moment = 一个 model-turn（一次 request/response + tool 往返），
     经 `Agent.iter()` 逐节点落盘。
   - `prompt_sha` + `window_stamp` 补完归因闭环：AGENT.md 在 memento root 外、
     不被 sidecar 见证，hash 进 payload 是"当时模型看见什么"的最后一块。
   - `type` 带 pydantic-ai 大版本号，harness 升级 = 新 type 值，旧轨迹照读。
3. **过去永远是折叠文本，不复活消息史**（最大降险决策）。invocation 内 =
   活的 message 流；invocation 之间 = 窗口渲染成一篇文本（K 条 commit 摘要 +
   最新 commit 转写）注入 context。绕开 tool_call/tool_result 配对约束的深坑，
   且正是契约的折叠语义（摘要替换原文）——折叠边界钉在 invocation 边界上。
4. **工具绑定：本轮不做**（选项 1）。tools = bash + file_editor 五动词，固定。
   bash 即万能绑定（agent 可调任何 CLI 含 moss 自己）。`.py` 定义 agent
   （选项 2）是 v2 演进，对齐 GhostMeta"文件即配置"惯例，runner 将来收
   `.md | .py` 双入口。module_eval 代码驱动 agent（选项 3 / ghostos 范式复活）
   解耦为未来独立 workstream，不让 memento 验证当人质；facade 建成后它是
   第二个消费者。
5. **commit 本身是一个 CLI 命令**：`moss memento commit [--auto]` 内置
   FORMAT §6 的 trailer prompt（"规范与 prompt 不写两份"直接兑现）+ 接口调用。
   人类、编排器、agent 三方共用同一入口。**边角**：agent 运行中经 bash 调它
   会撞 runner 自持的 flock——运行中 checkpoint 必须走进程内同实例路由，
   锁语义写清。
6. **owner = prompt 文件 stem**（`AGENT.md` → owner `AGENT`）。同目录多个
   prompt 文件 = 多个身份共享一个 `.memento/`，对上 per-owner 分片。
   branch = 工作线（`--branch` 可选 flag，不打即 HEAD）。
7. **并发纪律**：owner 级 flock（复用 `contracts/workspace.py` 现有实现），
   占用即 fail-fast 报"谁在跑"。**并行扇出必须走派生 owner**
   （`BasePointer.fork` 本就是跨 owner 引用设计）——化身 = 新 owner 新 branch，
   base 指回父 owner 的 commit。
8. **退化态纯净落在 CLI 分组**：基础动词（init/agent/status/log/show/window/
   annotate/commit）零 fork 词汇；branch/fork 动词进 advanced 组。
   契约不变量 #12 的 CLI 兑现。
9. **见证默认 sidecar**，`init --witness sidecar|outer|none`。outer 模式 =
   见证层关闭、外层仓库直接见证 jsonl（"moss 实例 memento 直接入库"的合法形态，
   FORMAT §9 见证层可选条款）。sidecar 模式下 init 负责把 `.memento/` 写进
   外层 .gitignore（§9 MUST NOT 被吞）。
10. **bash 对 `.memento/` 读自由写禁止**。grep 自己的过去是 filesystem-first
    的兑现；写只许走 memento API。v1 靠 AGENT.md 约定不设防。
11. **stdout/退出码即编排协议**：stdout = 模型最终文本，`--json` 加
    {commit_id, moment 数, branch}，退出码 0/1。loop 编排外包给任何调度器。
12. **模块落点**：agent 是独立模块（pydantic-ai 在 `[ghost]` extra，整个 moss
    体系不一定进）；CLI 照常进 `cli/`，对 agent 模块 lazy import + 未装时
    报错引导安装。
13. **模型配置走 `contracts/llms.py` 的 LLMConfig**，不重走 atom 的
    环境变量硬编码（data-ghost 已定的纪律，此处同守）。

## 4. CLI 体系与生命周期（勾勒定稿）

目录解剖：

```
any-dir/
  AGENT.md                     # 默认 prompt = 默认 owner "AGENT"
  reviewer.md                  # 另一个 prompt = 另一个 owner（同池共居）
  .memento/                    # FORMAT v1 (§14 修订后) 原样, 零新格式
```

命令树：

```
moss memento init [DIR] [--witness sidecar|outer|none]
moss memento agent [PROMPT.md] -p "指令" [--dir D] [--branch B] [--model TAG] [--json]
moss memento commit [-m "..."] [--kind semantic|mechanical] [--auto]
moss memento status | log [--limit N] | show <cmt_...> [--notes] | window [PROMPT.md]
moss memento annotate <cmt_...> -m "..."          # 孔径二
moss memento branch ... | fork <cmt_...> --owner X [--overlay "..."]   # advanced 组
```

Invocation 生命周期（runner 全部契约）：

```
1. flock branches/{owner}/ — fail-fast
2. 装载: HEAD → meta(校验 ancestry) → AGENT.md (+ overlay)
3. staging 残留非空 ⇒ 先落 recovery commit (Kind: mechanical, 崩溃自成锚点)
4. 组装 context: [system] AGENT.md (+overlay) | [past] 折叠文本一篇 | [now] -p 指令
5. Agent.iter() 循环, 每 model-turn → update_moment; checkpoint 调用 → 即时
   semantic commit (进程内路由, 见钉子 5)
6. 退出 → mechanical commit (trailer 自动) → 释放 flock
7. witness snapshot (旁路, 永不阻塞退出)
8. stdout + 退出码 (钉子 11)
```

Python 锚点层（同一实现的裸露面，CLI 是它的皮，channel 化包的也是它）：

```python
agent = MementoAgent(dir=".", prompt="AGENT.md")
result = await agent.run("do X")     # 一次 invocation, 退出即 commit
```

## 5. 未终决点（施工者有裁量，遇阻上报）

- **recovery commit**（生命周期第 3 步）：当前倾向崩溃残留自成锚点（保
  "invocation = commit"不变式）；备选并入本次 commit（更简但归因混浊）。
- **checkpoint 后窗口重算**：semantic commit 后同 invocation 继续跑，倾向
  **不重算**（invocation 内窗口冻结）——"折叠边界 = invocation 边界"的推论。
- **独立模块的名字与发布形态**：`ghoshell_moss/agents/` 还是独立包，人类定。

## 6. 与 momento-mori §14 的并行协调点

存储布局修正（池废除、commit 文件自包含）与本 workstream 并行施工。
本 agent **只消费 ABC/facade 表面，不碰磁盘布局**——并行是安全的，唯一 API 级
协调点：`MomentPool` 并入 `MementoBranch` 后的接口面。施工时以 momento-mori
侧修订后的 abc.py 为准；它未就位时先按现行 abc.py 施工，接缝处留单点适配。

## 7. 概念重构：prompt + toolsets + conversation 正交（2026-07-23，人类引导，claude-fable-5）

一场关于"文件系统定义 agent 最佳实践"的讨论，把本 workstream 的 agent
从"memento 验证器"升格出第二重身份：**轻量 agent 协议的参考实现**。
钉子不推翻，框架重述。

### 7.1 正交三元组（定案）

agent 配置化的关键拆分是 **prompt + toolsets + conversation**，harness
另行看待（loop/workflow = 外部进程重复 invocation，慢一点可接受）：

- **toolsets 塌缩为 bash 可调**。toolsets 之所以是麻烦源，是它绑实现协议
  （python/ts/mcp schema...）；bash 边界把多语言问题交给 `$PATH` 吸收，
  skills（脚本+prompt 片段）是这条路的正解。"自己实现"和"封装别人的实现"
  在 bash 边界上无差别。不引入 CTML/Channel——回合制里 tool use 够了。
- **conversation 是轴，memento 是该轴的一个实现**。平台自有 store
  （如 `claude -p --resume <session>`）是另一个实现。memento 的定位由此
  收敛为：**不依赖平台的历史消息治理协议**——自带 conversation store、
  跨平台可移植。它不是 `-p` 的竞品，是 `-p` 的解耦版。
- **prompt 场景可带可不带**（本质 prompt 与 toolsets 配套）。

### 7.2 轻量 agent 协议 = 调用约定，不是新协议

行业实查（2026-07）：MCP = agent→tools，ACP (Zed) = editor→agent
（JSON-RPC over stdio，带 IDE 会话语义，重），A2A = agent→agent over
HTTP（网络发现层，重）。**"bash agent 协议"这个空位是故意空着的——
Unix 进程调用本身就是协议**。真正无标准的只有四个字段，恰好是钉子 11
的扩展：

1. 上下文入参（`--branch` / session 寻址）
2. 指令入参（`-p`）
3. 结构化出参 schema（`--json`）
4. 退出码语义

这是 calling convention（量级同函数 ABI），不是发明新协议。锚点是
memento：历史轴统一后，约定自然收敛到 memento 边界。agent 调 agent =
bash 调子 agent CLI，子 agent 历史落自己 owner、base 指回父 commit
（fork 语义现成）。

### 7.3 不对称钉子：两种传参形式的治理方式不同（已被 §8 泛型吸收，保留作历史）

> **作废提示（2026-07-23，§8 起草时）**：本节的 "per-platform codec 不对称"
> 已被 `MementoAgent[MOMENT]` 泛型取代——codec 不再是运行时协商，而是类型
> 参数。不再有 per-platform codec，只有 per-MOMENT 的具体类。下文保留仅供
> 追溯这条判断如何演进。

协议接纳两种传参形式的 agent，但 memento 对它们的治理路径不对称：

- **prompt 形式**（`-p` 单发，agent 无状态）——**原生可治理**。
  moss agent 驱动单轮、拿 stdout、写 moment。moss agent 本身就是
  prompt 形式的驱动器：只负责调度 memento / instruction / 单轮 prompt、
  包装历史更新。
- **conversation 形式**（agent 自带 session store）——治理需
  **per-platform codec** 导入其自有格式（钉子 1 的 pydantic-ai codec
  即第一个实例），或退化为在不透明 session 外围锚定 commit 边界。
  moss agent 不治理、也不该治理自带 store 的 agent。

### 7.4 `.agents/` 目录构想（暂名，未钉死）

`moss agent init` 一个 `.agents/` 目录，约定内部 bash 脚本支持 §7.2
通用入参形式。有 `.agents/` 走约定调度；没有则用现有工具 + pydantic-ai
实现一个即可（即本 workstream 钉子 4 的现状）。钉子 4 的 bash+file_editor
固定工具集是 dogfooding 验证器的合理抄近路，不固化为通用模型。

### 7.5 定义文件形状（讨论共识，待施工验证)

- frontmatter (可选机器配置: model tag → LLMConfig) + markdown body
  (prompt)，纯 markdown 无 frontmatter 依然合法——保住"任意文本文件即
  agent"的退化态。
- 身份 = 文件 stem（钉子 6 不变），不加 frontmatter name 覆盖——
  filesystem-first，避免"文件说一套、目录说一套"。
- prompt_sha 对整个文件取 hash（含 frontmatter）——换 model 也是行为
  变更，归因闭环要覆盖。
- 撞名风险待人类裁决：`AGENT.md` 与行业 AGENTS.md 标准（项目指令，
  60k+ 仓库）单复数之差，路过的 coding agent 可能误读。备选
  `*.agent.md` 后缀（多身份场景 `reviewer.agent.md` 更清晰，owner
  发现可判定）。

### 7.6 与 ghost 的边界

本协议纯静态、无 runtime，供 ghost 调用（multi-agents channel 场景：
ghost 与不同 agent 在不同 branch 对话，owner+branch 跨场域复用 =
上下文治理能力）。ghost func（图灵完备规划，正常走已实现代码、
cache-miss 时重新生成）属 ghost runtime 能力形态，不进本协议。

## 8. 定案收敛：MementoAgent[MOMENT] 泛型 + 协议化驱动为唯一核心（2026-07-23，人类引导，claude-fable-5）

§7 铺开了框架，本节把它收敛成可施工的形状。这是复工前的最终定案层，
与 §3 钉子并列有效；冲突处以本节为准（本节更晚、更收敛）。

### 8.1 核心倒置：协议化驱动是唯一核心，内部状态归实现

整个概念的唯一核心是 **agent 能被外部协议化驱动**。至于 agent 内部有无
状态（pydantic-ai 的 ctx、自持 loop 的 running state）、用 python 做运行时
loop（性能更好）、还是无状态每次重建——**全交给实现自己管**。外层只保证
一件事：它能被外部以约定方式驱动，产出一帧结果落进 memento。

**推论**：不要在 runner 里做 `Agent.iter()` 逐节点落盘（§3 钉子 2 / §4
生命周期第 5 步的 iter 循环）。iter 循环是 harness 器官，是 ABC 泄漏内部
结构。runner 只驱动一次 invocation、收一帧 MOMENT、commit。tool 往返在
实现内部消化，memento 看不见也不该看见。

### 8.2 MementoAgent[MOMENT] 泛型（取代 §7.3 的 codec 不对称）

```python
class MementoAgent(Generic[MOMENT], ABC):
    """外部协议化驱动一次 invocation，产出一帧 MOMENT 落进 memento。
    MOMENT = payload 的类型参数，memento 视其为不透明。"""
```

- **MOMENT 参数化的是 payload，不是整条 MomentRecord**。信封
  （id/created/type/by/threads）不动，`type` 值随具体 agent 变
  （`text/v1` vs `pydantic_ai.messages/v2`）。这与 momento-mori §13
  "Moment 是信封第一住户、payload 不透明、type 做判别符"完全一致——
  泛型只是把这条契约在 Python 类型系统里显式化。
- **codec 问题被泛型吃掉**：不再有运行时 per-platform codec 协商
  （§7.3 作废），只有编译期 per-MOMENT 的具体类。
  `MementoAgent[ModelMessages]` = pydantic-ai 实现，
  `MementoAgent[tuple[str, str]]` = golden test 的 dumb 参考实现。
- **pydantic-ai 退成一个具体实现，不是不变量**。§3 钉子 1 的 codec 仍是
  第一个可用实现，但契约层不认它——契约只认 ABC 输出形状 = MOMENT。
- **branch 对 MOMENT 保持同质**：一条 branch 内所有 moment 同类型
  （同 `type` 值）。跨 agent 类型的混合不在本轮。

### 8.3 -m 参数：指向 ABC 实现，去 pydantic-ai 耦合

`-m module:attr` 指向一个 `MementoAgent[MOMENT]` 的具体实现（或工厂），
CLI 用它替换默认锚点。**不耦合 pydantic-ai**——只要求符合 ABC、支持
正交解耦输入、产出一帧 MOMENT。工具集调整因此合法（工具变了，只要输出
仍是同一 MOMENT 类型，契约不破）。

- 这是 §3 钉子 4 里 delay 的"选项 2：.py 定义 agent"前移。
- memento 作为封装位置不变，CLI 职责（调度 memento / 单轮 prompt /
  包装历史）一行不动。
- 判据：**只要 `-m` 产物从同一 ABC 出、产出同 MOMENT 类型，就是在验证
  memento 而非绕过它**。

### 8.4 审批：从协议命题降级为实现命题（本轮枢纽决策）

出口逻辑拆解（人类原话级）——外部 harness/task 视角下 agent 有四类出口：

1. **tool 等底层运行时通讯，模型无感知** ← 唯一麻烦的一类。
2. 模型依赖外部输入。
3. 模型对外输出生命周期性质内容（≈ #2，但交互语义约定化：至少
   `done` / `exception` 两类）。
4. 外部 loop 形态决定身份：无模型一次自驱到底 ≈ task；有复杂逻辑 ≈
   workflow；只有生命周期检查 ≈ loop。

**#1 是设计难点**：交互式审批（"请求审批: xxx"）本质是**返回结果下发的**、
不是模型主动发的，这逼无状态实现假装有状态、逼返回协议做分层。

**定案：用 `-m` + 一帧输出把 #1 屏蔽出体系。** invocation 边界保持为纯
函数（input → 一帧 MOMENT），进行中的有状态协商永不碰 memento。审批要么
被 `-m` 实现内部消化，要么用调用前静态策略（见 8.5）。这让 memento 保持
无状态。

**两条干净车道 + 中间的坑（施工纪律）**：
- **无状态车道**（本轮走）：`-m` 一帧、审批屏蔽进实现或静态策略。
- **有状态车道**（未来）：session futures 协议落地 → moss agent 变
  matrix cell，审批 = 一等挂起 future。选 matrix 而非 MCP：MCP 基建轻，
  但跨进程交互 + 界面可扩展性撑不住。
- **坑**（禁止）：把交互式审批硬塞进无状态 `-p` 式调用。就是要躲的那个。

### 8.5 审批 = invocation 边界上的静态策略（claude code 事实佐证）

claude code `-p` 从不是"无副作用 agent"——它把审批**前移**成调用前静态
声明，三种形态：`--permission-mode`（default/acceptEdits/plan/
bypassPermissions）、`--allowedTools`/`--disallowedTools` 白黑名单、
`--permission-prompt-tool` 路由给 MCP 工具程序化批拒。

对 memento agent 的设计课：**审批 = invocation 边界上的静态策略**，与
"invocation = commit"同构（策略是被 commit、被归因之物的一部分）。§3
钉子 10（bash 对 `.memento/` 读自由写禁止）已是这个形态——一条静态策略，
无需运行时交互。

### 8.6 验收口径变更（覆盖 momento-mori §18.7 的"CLI 体系"）

之前验收锁为"CLI 体系"（momento-mori §18.7 人类拍板）。本轮"ABC 是定义、
CLI 是皮"使"做完"的含义变化，需人类复核：

- **验收 = ABC + 至少一个具体实现 + memento 绑定跑通**。
- **CLI 降级为最薄的可选驱动**，但保留——它的价值不是对外协议，是**白送
  一个重启连续性测试夹具**：每次 invocation = 新进程 = §1 最硬验收标准
  （重启连续性）免费被验。ghost 进程内直调 `agent.run()` 不触发重启，
  这个免费验证就没了，所以 CLI 夹具不删。
- **不走命令行的驱动同样合法**：ghost 进程内直调、将来 matrix cell。
  定义（ABC）与驱动（CLI / 直调 / cell）分离。

### 8.7 行业注记（2026-07 实查，佐证"没有现成的可用，也不必等"）

协议栈已拥挤，但没有一格是"外部驱动无状态一帧"——你的东西在格子中间：

- **AG-UI**（streaming UI，16 事件类型 SSE/WS）——最接近生命周期语义
  （#3），但它把 tool 层（#1）暴露成协议表面，正是本设计要屏蔽的。
  **未来可作可选外围 adapter**：把一帧 completion 吐成 AG-UI completion
  event。可选驱动，非核心。
- **ACP (IBM 版，与 Zed ACP 重名)**——negotiation semantics、typed
  performatives（propose/accept/reject/counter），即"嫌重的复杂协议"。
- **pause/resume 派（Google ADK / checkpoint 系统）**——状态机
  `PENDING→RUNNING⇄PAUSED→{STOPPED,COMPLETED,FAILED}`、协作式
  `checkpoint()`。有状态车道同构，但状态存平台自有 store（回到平台耦合）。
- **arxiv 2604.08224《Externalization in LLM Agents》**——把 agent 拆成
  memory/skills/protocols/harness 四可外置关注点，与本设计
  conversation/toolsets/协议/harness 近一一对应。学术侧 2026 也收敛到
  "externalization"，但止于描述性归纳，未给最小规范契约。

**判断**：没人做"agent 作为可被外部驱动的无状态一帧函数"，因为大厂卖
runtime/平台，而"无状态一帧 + 历史外置到可移植 store"消灭平台锁定，反
商业。memento 作为可移植 conversation store 是关键差异点。所以不该等，
也不必造协议——本设计是**拒绝所有现存协议的复杂度**，一个 ABC + 泛型 +
"invocation=一帧" 约定即可。

### 8.8 首个验证场景（复工即做）

**不同 branch 规划目录下 concepts 翻译**。填 §1"能力→CLI 动词→契约条款"
自洽矩阵：多 branch 并存、各自累积历史、跨 invocation 重启连续性每次都验。
纪律：默认 AGENT.md 干翻译；`-m` 留一个口子证明"换工具集/换 agent 实现，
memento 边界不变"即可，**不必真接第二个 agent**。不加复杂度。

## Implementation Notes

<!-- 施工化身在此追加 gotchas 与决策. -->

- atom 的 `_adapter.py`（Moment↔ModelRequest）是 pydantic-ai 适配的参照物。
- file_editor 五动词是普通 async 函数（view/create/str_replace/insert/undo_edit），
  直接包成 pydantic-ai tool。
- 成本预估（可行性讨论定案）：codec adapter ~200 行 / facade+policy ~200 行 /
  tools 包装 ~100 行 / runner ~150 行 / 窗口文本渲染 ~100 行 / CLI ~200 行，
  合计 ~1k 行 + 测试，单 workstream 体量。超出量级即回看 §0 放弃触发器。
- **2026-07-20 对齐（kimi-k3）**：CLI 命令树定案见 momento-mori §18.4。
  关键对齐点：(1) 钉子 5 commit 命令挂进 §18.4 树，agent 命令同根；
  (2) 钉子 6 owner = prompt 文件 stem 不变，但寻址统一为 `<owner>/<name>`，
  `cmt_` 前缀是 commit 否则是 branch; (3) 钉子 7 已由 §17.3 #3 覆盖，
  本 workstream §3 钉子 7 的"派生 owner"表述作废，同 owner 多线并行合法;
  (4) §16.5 #1 (`commit --to <moment_id>`) / #2 (overlay→owner meta.json) /
  #3 (ref = JSON 元组 `{fork, commit_id[, moment_id]}`) 三颗钉子全部顺带定案;
  (5) `moss memento agent` 命令挂进 §18.4 树（owner 级或独立组待施工时定）。
  复工条件：等 FORMAT v2 起草 + abc.py 重写（MementoBranch 解体）。验收方式
  由人类定为 CLI 体系——本 workstream 的 agent 是 CLI 验收后的下一站。

## 9. 复工前最后一层：beta1 刻度 + 轨迹作为产物 + 8 步节点式施工（2026-07-24，claude-opus-4-7-1m）

复工条件成熟：FORMAT v2 冻结、`memento/abc.py` 重写完成（`MementoBranch` 解体成
`Memento` facade + `Line` protocol）、`memento` 已提级一级模块、CLI 一轮自解释
验收通过（§19）、pydantic-ai 2.5.0 在 `[ghost]` extra 就位。§8 的定案层与本节
并列有效，冲突处以本节为准（本节更晚、更收敛）。

### 9.1 认知刻度：beta1 不是发布前

本项目 v0.1.0 未发版，处于人机协作生长期，不是"发布前的严谨设计冷项目"。
过度严谨在此阶段是失败模式：它把注意力从"跑得快、撞到再调"移向"每个字段拍板、
每个契约先冻结"。以下三条是 beta1 刻度的物理表达，与 §8 定案层不冲突但改变
其应用方式：

1. **契约压力就地打补丁，绕不过再重开**。memento agent 在施工中撞到 memento
   契约压力时，判据是"能否加一条 §N 压力点条目 + 局部绕过"——能就绕，不能才
   提议 momento-mori 重开（§14/§16/§17 三次重开走过的仪式）。beta1 下 95%
   情况应属前者。
2. **调整与回滚是产物，不是噪音**。git log 的形状本身是产物：一次 feature
   40 个小 commit 记录调整节奏，比 1 个 squash 后的干净 commit 有价值——观测者
   AI 可从中提取施工节奏、复现决策路径、benchmark 人机协作模式。**不 squash、
   不 rebase、不"最后整理成干净 PR"。撞到问题往回改是新 commit，不是 amend**。
3. **通用 vs 可用的边界重划**。harness 器官的核心风险是**通用化**（跨家族抽象），
   不是**可用性**（家族内部堆策略）。判据：**只要 memento + record + commit
   三原语在 memento 契约层保持通用，pydantic-ai 家族内部堆多少 `commit_policy` /
   `compact_threshold` / 什么都行**。跨家族抽象等第二个 agent 家族出现再做。

### 9.2 核心转向：commit 归 agent 全权，invoke ≠ commit 生命周期

§1 "invocation 边界 = 天然机械 commit 边界" 本轮被推翻。**agent 一次 invoke =
一个 final answer，内部有多少回合 / record / commit 都是家族自决**。具体：

- **agent 全权管写**：runner 装 `Memento` 实例 + `cwd` + AGENT.md + `instruction`
  → `agent.invoke(...)`。invoke 内部自己 `line.record()` + `line.commit()`。
  **runner 不摸 line 写侧**。
- **invoke 内多帧合法**：一次 invoke 内可 record N 次、可 commit 多次（"分段
  多次提交"是特性不是补丁——它对得起"打磨 memento 概念"这条主线）、也可以
  invoke 结束时 staging 有残留不 commit。
- **staging 残留在 invoke 边界上合法**：不再是崩溃残留。§F 的 "pre-invoke sweep
  （staging 非空 → 机械 commit 落锚）" 死掉。runner 层 v1 完全不管 staging。
- **代价接受**：运行状态不可感知（invoke 内部没有流式协议）。beta1 不做
  jsonl 流式吐帧，等真需求撞到再加。

**副作用观察**：runner stdout 要 `--json` 元信息（commit_id / moment_count_delta），
只能通过 invoke 前后 `line.log()` 差集观察。这是"agent 全权管写"的物理代价，
不当 bug 修，允许 flake。

### 9.3 概念一次沉淀：branch ≈ task（降级后的一等公民）

**branch 就是 §5 "task 降级为可丢弃投影" 后的物理落点**：
- branch = 一次思考 / 一个 sub-agent 会话 / 一个 workstream 段
- commit = 段内自然节点
- branch 摘要 = task summary 的自然存在
- ancestry (parent commit 单父链) = task tree 的自然存在
- "分段多次提交" = 段内规划的物理体现

memento agent 打磨的东西**不只是 memento，是"降级后的 task 系统"**。这条纪律
挂在本节，不进代码抽象——beta1 阶段代码层仍叫 `MementoAgent`，不叫 `TaskAgent`
或其它任何提级名字。

### 9.4 四锚框架保留在文档层，不进代码

**factory + AGENT.md + memento + ground** 是 §7.1 orthogonal triad
(prompt + toolsets + conversation) 加 spatial (ground) 的补全，构成 MOSS 通用
agent 形状描述。但 v1 保持 memento agent family 定位：

- 代码里**不出**现 `MossAgent` 抽象；ABC 名叫 `MementoAgent`，绑定 memento family。
- 四锚是本文档描述性框架，不是代码强制。ground 在 v1 退化为 `cwd: Path`
  （AGENT.md 所在目录 + `--cwd` 覆盖），不引 ground contract。
- 第二个 agent family 出现 + ghost-ground workstream 集成成熟 后，再讨论 ABC
  提级到通用位置。

### 9.5 ABC 表面（tentative，v1 起点）

命名不用 `abc.py`（IDE 冲突面）→ 用 `contract.py`（项目里 `contracts/` 一级
已在，语义前置共识）。四个方法：

```python
class MementoAgent(ABC):
    """
    Memento family 内部 agent. beta1 阶段的 agent 契约面.
    invoke = 单次交互 → final answer. 内部治理由家族自决.
    """

    @abstractmethod
    async def invoke(
        self, *,
        instruction: str,
        prompt: str,          # AGENT.md body, sha 由 runner 计算传 metadata
        memento: Memento,
        line_name: str,
        cwd: Path,
        metadata: dict[str, Any] = {},
    ) -> str: ...
    # 返回 final answer 文本. 副作用 (record/commit/compact) 全归 agent.

    @abstractmethod
    def compact(self, memento: Memento, line_name: str) -> None: ...
    # 收 staging → semantic commit. agent 自我规划 summary + trailer.

    @abstractmethod
    def export_context_md(self, memento: Memento, line_name: str) -> str: ...
    # agent 视角: system + window + recent 导出 markdown.

    @abstractmethod
    def describe_line(self, memento: Memento, line_name: str) -> str: ...
    # line 的 agent 视角摘要.
```

`invoke` 返回 `str` 而非 `None` — §3 钉子 11 "stdout = 模型最终文本" 兑现。
`compact` 独立方法 — AGENT.md body 可引导 agent "重要节点后调 compact"，也允许
runner 通过 CLI flag `--pre-compact` / `--post-compact` 外部触发。

四方法 tentative，施工中撞到冗余就砍、缺就加，不当契约冻结。

### 9.6 8 步节点式施工方法论（本 workstream 主约束）

**不是"设计好一次做完"，是"设计一部分做一部分，暴露的新信息喂回下一步"**。
每步做完停下、明说 "步 N 完成"，等人类 review 放行才进下一步。8 个节点 =
8 个投递讯息窗口。

| 步 | 内容 | 产出判据 |
|---|---|---|
| 1 | FEATURE.md §9 起草 | 本节即产物 |
| 2 | 目录结构 + 依赖复核 | `ghoshell_moss/agents/memento_pydantic_agent/` 就位，`[ghost]` extra 含 pydantic-ai |
| 3 | `contract.py` ABC 四方法 | import 通过、类型检查通过、方法体 `raise NotImplementedError` |
| 4 | factory + config 骨架 | `factory({})` 可实例化，方法留空 |
| 5 | CLI 配套：声明与发现 | `moss memento agent <owner/line> <AGENT.md> -p "..."` 解析、定位 factory、打印 "would invoke with ..."，不真调 invoke |
| 6 | 走完一轮：无自动 commit | invoke 能真调 pydantic-ai、能 record 一帧、staging 结束时有残留、人类可手动 `moss memento branch commit` 落锚 |
| 7 | 手动 compact：触发 commit 规划 | agent 自我总结 staging + 生成合规 trailer + 落 semantic commit + 下次 invoke 装载时 render_window 读到 summary。**做好了万事大吉** |
| 8 | 自动 commit policy：一个 | `per_invoke` 策略，config 字段可关。至此 v1 完成，是 Atom prototype 定位 |

**每步阶段内自由多 commit**（撞坑 + 恢复 + 修正各一 commit）。commit message
前缀 `step N` 便于事后 `git log --grep="step "` 复盘节奏。

### 9.7 目录结构定案

```
src/ghoshell_moss/agents/
  __init__.py
  memento_pydantic_agent/
    __init__.py
    contract.py       # 步 3 的 ABC (MementoAgent)
    config.py         # 步 4 factory config BaseModel
    factory.py        # 步 4 factory 函数
    impl.py           # 步 6+ 具体实现
```

pydantic-ai 依赖走 `[ghost]` extra（步 2 复核 pyproject.toml，缺则补）。

### 9.8 AGENT.md 兼容规约（v1 具体形状）

```yaml
---
name: translator                                              # claude code 兼容
description: Translate concepts docs to zh                    # claude code 兼容
model: claude-sonnet-4-5                                      # claude code 兼容
tools: [bash, file_editor]                                    # claude code 兼容, 缺省 = 默认全集
memento_agent: ghoshell_moss.agents.memento_pydantic_agent:factory  # 本 family 新增
construct:                                                    # 本 family 新增, factory config sink
  max_tokens: 4096
---
you are ...
```

- `name` 只作 agent 身份标签（塞 `MomentRecord.by`），**owner 仍走文件 stem**
  （§3 钉子 6 不变）——避免"文件说一套、metadata 说一套"。
- `memento_agent` 缺省 = 本 family factory；construct 缺省 = `{}`，factory
  BaseModel 全字段默认能跑。
- 极简下**只 name/description 必填**，其余全默认。
- 撞名风险（AGENT.md vs 行业 AGENTS.md）：我们的 AGENT.md 语义等同 claude code
  sub-agent 定义（`.claude/agents/*.md`），项目级说明是 CLAUDE.md/AGENTS.md
  ——类别错配，不撞。真撞到留 `*.agent.md` 后缀逃生口。

### 9.9 §8 定案层的存活与覆盖

- **存活**：泛型 `MementoAgent[MOMENT]` 精神（payload 归 agent，memento 视为不透明）、
  -m 指向工厂实现（AGENT.md `memento_agent` 字段即其形态）、审批降级为静态策略
  （v1 零工具场景 moot）、CLI = 重启连续性夹具、首场景 = 多 branch 翻译 concepts。
- **覆盖**：§8.1 "runner 只驱动一次 invocation、收一帧 MOMENT、commit" 被本节
  9.2 推翻——runner 只驱动，不 commit 不 record；§8.2 泛型参数 MOMENT 在 v1
  ABC 里不显式化（家族内部自持 payload 类型，跨家族移植不是 v1 目标）。

### 9.10 首场景不变

**不同 branch 规划目录下 concepts 翻译**（§8.8）。填 §1 能力→CLI 动词→契约
条款自洽矩阵。零工具、单轮翻译——v1 scope 越小 §0 触发器越远。tool_loop
（bash + file_editor）等第二场景触发。

## 10. 复工前最后一层: pivot 到 .py + Sandbox 反射为 prompt (2026-07-25, claude-opus-4-7-1m)

§9 起草后步 1-4 落地过程中，人类引导展开了一场关于 "AGENT.md + tool 授权 vs
纯 python 文件 + sandbox 预授权" 的收敛讨论。这是 §9 的再一次 pivot，本节承载
最终定案。本节与 §9 并列有效，冲突处以本节为准。

**方法论注脚**：AGENT.md 从头到尾是人类的第二方案。§7/§8/§9 讨论 AGENT.md
路径是因为模型手感不够、上下文绕不进 .py 路径的复杂性。§9 起草沉淀了~200KB
的讨论共识后，讨论空间被修剪到足够，才能真正谈 .py 路径的细节。下一个化身
读到此处应理解: **AGENT.md 路径不是被"推翻"，而是被"用完即抛"** — 它做了
它的工作 (推着讨论收敛)，本节起 .py 路径承担 v1 实现。

### 10.1 pivot 动机: 授权痛苦 + 上下文密度

**授权痛苦真实存在**。AGENT.md + bash/file_editor + tool 白名单机制在 claude
code 是持续摩擦点 (每次撞未授权工具中断问用户)。给我们自己造一份等价的白名单
机制 = 施工面翻倍，且授权本质无解 (要么无授权/危险，要么全授权/自欺，要么
每次问/UX 崩)。

**上下文密度是 10x 差异**。JSON schema tool 描述让模型看到 `{"name": "read",
"parameters": {...}}` 只知道形状; 反射 Python 让模型看到 `def read_doc(path:
str) -> str: """Read a MOSS concept doc from ..."""` 知道语义 + 类型 + 意图 +
相邻工具关系。**LLM 读 Python 远好于读 JSON schema** — 训练分布集中在前者。

**MOSS 已有基建 100% 就位**:
- `ghoshell_moss.core.codex.compiler:Compiler` — .py 编译 into ModuleType
- `ghoshell_moss.core.codex.sandbox:Sandbox` + `SANDBOX_BUILTINS` — 两层沙箱,
  builtins 层屏蔽 `__import__` = 真授权 (不是 in-process exec 的自欺)
- `ghoshell_moss.core.codex._reflect` — 反射即 prompt 的机制
- `ghoshell_common.entity:EntityMeta` — 变量跨进程序列化契约 (来自 2024 ghostos
  的 `PyContext.properties` 现代形态)

24 年 ghostos 已经证明这条路走得通，代码在 `libs/moss/src/ghostos_moss/`，本轮
是把该思路的 MOSS 内嵌版做出来。

### 10.2 三个核心决策

**决策 1: Sandbox 是 tool，认知归 runner**

sandbox 是 pydantic-ai agent 的一个工具 (`exec(code: str) -> str`)。**记忆写入
不在 sandbox 内做**，agent 不知道自己被记录，就跟人类不会主动"我要把这句话
写进日记"一样。

修订 §9.2 "agent 全权管写": 只有**任务级状态** (`ctx: AgentContext`) 归 agent
主观感知，**轨迹记录** (moments) 归 runner 自动完成。

责任分层:

| 层 | 内容 |
| --- | --- |
| sandbox namespace 注入物 | `file_editor: FileEditor` / `ctx: AgentContext` (模型能触到的能力) |
| runner (impl.py) | 编译 .py / 建 sandbox / 反射→prompt / 装 exec tool / 调 pydantic-ai / 收 result 后 record 一帧 / 序列化 ctx |

**决策 2: v1 无 compact 无 magic hook**

compact + magic function hooks + spec 三件套是 harness 家族 (§0 触发器)。
一期完全不做:

- ABC 从 4 方法减到 **3 方法** (`invoke` / `export_context_md` /
  `describe_line`)，`compact` 完全从 ABC 拿掉
- CLI 从 5 动词减到 **4 动词** (`parse` / `invoke` / `export-context` /
  `describe`)
- **staging 累积不 commit** — 需要 commit 时用户显式 `moss memento branch
  commit`。这是 §9.2 "invoke ≠ commit 生命周期" 的最诚实兑现
- 原 §9.6 步 7/8 (手动 compact / 自动 policy) **移出本 workstream**，未来
  再开新 workstream

**决策 3: 反射天然过滤 dunder**

`Sandbox.get_interface()` 已过滤 `_` 开头，将来加 `__compact__` / `__on_end__`
魔法函数自动被反射跳过。**惯例已在，约束不用新建**。

### 10.3 memento 数据源归属

**AGENT.py 就是锚**。

单 agent 布局:
```
some/task/dir/
  AGENT.py       # owner = "AGENT" (或从 __owner__ 覆盖)
  .memento/      # 兄弟目录
```

多 agent 布局:
```
some/task/dir/
  translator.agent.py    # owner = "translator"
  reviewer.agent.py      # owner = "reviewer"
  .memento/              # 共享目录, 内部 owner 分片
```

owner 名 = 文件 stem (去 `.agent` 后缀)。多 agent 共享 `.memento/` 时，memento
契约层的 owner 分片 (`branches/{owner}/...`) + 跨 owner 只读机制天然支持协作。

**owner 覆盖优先级**: `CLI --owner flag > AGENT.py __owner__ 魔法 attr >
文件 stem > getpass.getuser()`。四层兜底。

**branch 默认**: `main`。CLI `--branch/-b` 覆盖。

### 10.4 AGENT.py 结构约定

```python
"""Translator agent — translate MOSS concepts to zh.

Read source with file_editor; call ctx.define() to persist state across invokes.
"""

from ghoshell_moss.agents.injections import (
    FileEditor, get_file_editor,
    AgentContext, get_ctx,
)

__model__ = "claude-opus-4-7"     # 可选, 缺省走 ANTHROPIC_MODEL env
# __owner__ = "translator"        # 可选, 覆盖文件 stem

file_editor: FileEditor = get_file_editor()
ctx: AgentContext = get_ctx()
```

关键性质:

1. **顶部 docstring = task instruction** — agent 身份 + 目标描述
2. **imports 声明能力面** — `SANDBOX_BUILTINS` 屏蔽 `__import__`，agent 只能碰
   compile 阶段已 import 的东西 = 文件即白名单
3. **`get_*()` 是 stub** — `injections.py` 里实现为 raise NotImplementedError,
   factory 在 compile 后用 `sandbox.set(...)` 覆盖为真实现
4. **模型看到的**: `sandbox.get_interface()` 反射输出 = 模块 docstring + 顶层
   属性 (`file_editor: FileEditor` / `ctx: AgentContext`) + 引用类型的
   signatures。**这就是 system prompt 的完整内容**
5. **模型唯一工具**: `exec(code: str) -> str` — 写任意 Python 调用 sandbox
   namespace 里的东西

### 10.5 修订后的 v1 完整形状

```
factory(agent_path):
  1. source = agent_path.read_text()
  2. compiler = Compiler(source, modulename=agent_path.stem,
                         filename=str(agent_path), compile_soon=True)
  3. compiled = compiler.compiled
  4. model_name = getattr(compiled, "__model__", None) or os.environ["ANTHROPIC_MODEL"]
  5. owner = <CLI flag> or getattr(compiled, "__owner__", None) or <stem> or getuser()
  6. Sandbox 二层 (init + agent):
     - init_sandbox = Sandbox(builtins=None)
     - copy compiled.__dict__ into init_sandbox (跳过 dunder)
     - override: init_sandbox.set("file_editor", RealFileEditor(cwd=agent_path.parent))
     - override: init_sandbox.set("ctx", RealAgentContext(loaded_from_memento))
     - agent_sandbox = Sandbox(parent=init_sandbox, builtins=SANDBOX_BUILTINS)
  7. build pydantic-ai Agent (Anthropic, __model__)
  8. return MementoPydanticAgentImpl(agent, agent_sandbox, memento_binding, ...)

impl.invoke(instruction, memento, line_name, ...):
  1. system_prompt = agent_sandbox.get_interface()   ← 反射即 prompt
  2. register tool: async def exec(code: str) -> str: return sandbox.exec(code)
  3. result = await self._agent.run(instruction, instructions=system_prompt)
  4. runner records to memento staging:
     - moment A: pydantic-ai new_messages dump (type="pydantic_ai.messages/v2")
     - moment B: ctx snapshot 如 ctx 有变 (type="agent.context/v1")
  5. return result.output
  # 不 commit / 不 compact / 无 magic hook
```

### 10.6 修订后的施工步骤 (A-H)

| 步 | 内容 | 交付判据 |
| --- | --- | --- |
| A | §10 起草 | 本节即产物 |
| B | `agents/injections.py` — Protocol + get_* stubs | import 通过, get_* 抛 NotImplementedError, get-interface 反射看到 3 个 Protocol 与 3 个 get_* 函数 |
| C | `agents/memento_pydantic_agent/_context.py` — AgentContext (PyContext 现代版, 复用 EntityMeta) | 单测: define/get/iter/序列化-反序列化往返 |
| D | `agents/memento_pydantic_agent/_injections_impl.py` — FileEditor / AgentContext 真实现 | 单测: file_editor 基本 view/create; ctx load-from-memento round-trip |
| E | 改 `factory.py` + `impl.py` — Compiler + Sandbox + exec tool 装配 | 无网络单测: sandbox 装载 stub AGENT.py 成功, get_interface 输出符合预期 |
| F | hello world AGENT.py + 跑通 | AGENT.py 只 import math + 顶部 docstring 说自己是什么, invoke "who are you and what can you do?" 模型能通过反射自陈能力 |
| G | CLI 4 动词 (`parse` / `invoke` / `export-context` / `describe`) | `parse ./AGENT.py` 打印反射结果; `invoke` 走通; 其余暂 raise NotImplementedError 或 stub |
| H | 首场景: math 计算 | AGENT.py 只依赖 `math` 库, invoke "compute the surface area of a torus with r=3 R=5" 模型 exec math.pi 计算完返回 |

**每步 checkpoint 明说** — 施工化身做完一步停下等 review 放行。commit message
前缀 `step X` 便于 `git log --grep="step "` 复盘。

### 10.7 hello world 验收判据 (步 F)

首个能跑通的 AGENT.py 极简:

```python
"""Hello agent — a minimal reflection-driven agent.

You have access to the standard `math` library and can compute anything
mathematical. When asked who you are, describe yourself from the
interface you can see.
"""
import math

__model__ = "claude-opus-4-7"
```

调用 `moss memento agent invoke ./AGENT.py "who are you and what can you do?"`。
**验收判据**: 模型输出应说出自己有 math 库、自己能算什么 — 而这个信息**只能
从反射出的 system prompt 里读到**。如果模型能通过反射自陈能力，反射即 prompt
链路就打通了。

**不判据**: 计算结果准确度 (那是 F+1 的场景 H 验收)、任何 memento 写入格式
(那是集成期验证)。

### 10.8 首场景: math 计算 (步 H)

**AGENT.py 只依赖 math 库** — 无 file_editor / 无 ctx / 无 memento view 注入。
verify 场景纯粹是 "反射 + 单一 stdlib + pydantic-ai tool loop" 三者协作。

具体判据 (`invoke ./AGENT.py "compute the surface area of a torus with r=3 R=5"`):
1. 模型至少调用一次 `exec("2 * math.pi ** 2 * ...")` 之类
2. 最终 output 包含正确数值 (~592.176)
3. memento staging 至少 1 条 moment (pydantic-ai messages dump)
4. 手动 `moss memento branch commit` 后 staging 清空、`commit show` 能看到
   frozen 内容

**不测**: 多轮工具调用交错、错误重试、compact — 都推到未来 workstream。

### 10.9 §9 存活与覆盖

- **存活**: §9.1 beta1 刻度三条、§9.3 branch≈task 概念、§9.4 四锚留文档层不进
  代码、§9.6 8 步施工方法论、§9.7 目录结构、§9.10 首场景 (math 版本落地)
- **覆盖**:
  - §9.2 "agent 全权管写" → 10.2 决策 1 (sandbox 是 tool, 认知归 runner)
  - §9.5 ABC 4 方法 → 10.2 决策 2 (3 方法, 拿掉 compact)
  - §9.6 步 5-8 → 10.6 步 A-H (compact/policy 移出本 workstream)
  - §9.8 AGENT.md frontmatter → 10.4 AGENT.py 结构约定 (完全替换)

### 10.10 剩余不确定项 (施工中撞到再定)

1. **Compiler 是否允许 relative import** (`from . import helpers`)? 未验证。
   若不允许, 任务目录内多文件 python 组织需要另想。beta1 判据: 撞到再决,
   MVP 单文件不触发。
2. **模型对 `sandbox.get_interface()` 输出格式的适应性**。人类日常在用
   `moss codex get-interface`, 模型训练分布应该覆盖类似格式, 但正式协议前
   未做过 evals。步 F/H 就是这条判据的第一次真验。
3. **AgentContext 序列化的 moment type 值** (`agent.context/v1` vs
   `moss.agent.ctx/v1` 或别的)。写代码时定，不需要现在拍。

## 11. CLI 4 动词落地 + loop 策略定案 (2026-07-26, claude-opus-4-7)

步 G 完成。CLI shape 经过两轮讨论收敛到最终形态。`.loop.py` 方案在同轮讨论中
定案——bash while / .loop.py 双策略，CLI 不加 loop 动词。

### 11.1 CLI 4 动词终态

```
moss memento agent parse   <agent.py>
moss memento agent invoke  <agent.py> <prompt> [--owner O] [--branch B] [--cwd D] [--root R]
moss memento agent export-context  <agent.py> [--owner O] [--branch B] [--root R]
moss memento agent describe        <agent.py> [--owner O] [--branch B] [--root R]
```

关键设计：

- **prompt 是位置参数**。无交互模式，不需要 `-p` 区分提示词来源。
- **`--owner` 默认 = agent 文件 stem**。`translator.agent.py` → `translator`，
  去 `.agent` 后缀。`_owner_from_path()` 在 CLI 层计算。
- **`--branch` 默认 = `main`**。`-b` 短选项保留。
- **取消了 `--line <owner/name>` 组合参数**。owner 和 branch 各自独立、各有
  默认值，消除用户在命令行和文件名里说两遍同一信息的摩擦。
- **`*.agent.py` 命名规范**。不用 `AGENT.py`（与行业 AGENTS.md 撞名风险，
  §7.5 备选方案胜出）。`.agent.py` 是不可直接执行的 Python 文件标识。
- **memento 按需接线**：root 存在时才建 Memento 实例传给 invoke。root 不存在
  则 agent 无记录运行，不报错。

### 11.2 模型配置

`__model__` 不写在 agent 文件里。factory 已有 fallback 链：
`__model__` attr → `ANTHROPIC_MODEL` env var → RuntimeError。
日常使用走环境变量：

```bash
ANTHROPIC_MODEL=claude-opus-4-7 moss memento agent invoke hello.agent.py "prompt"
```

### 11.3 施工步进度

| 步 | 内容 | 状态 |
|---|---|---|
| A | §10 起草 | done |
| B | injections.py | done → §11 删除 (stub-swap 被直接 import 替代) |
| C | AgentContext (_context.py) | done (未独立文件，合并进 impl) |
| D | injections 真实现 | done → §11 收缩 (无 live capability，删模块) |
| E | factory + Sandbox + exec tool 装配 | done |
| F | hello world .py 跑通 | done (验收通过: 模型通过反射自陈能力) |
| G | CLI 4 动词 | done (本节) |
| H | math 首场景 + memento staging | next |

### 11.4 loop 策略: bash while / .loop.py 双轨

**loop 不是 CLI 的职责**。CLI 四动词保持单次语义——invoke = 一次 prompt → 一次
final answer。

两层 loop 方案，互不排斥：

**退化态: bash while**

```bash
while true; do
  output=$(moss memento agent invoke hello.agent.py "$prompt" --root .memento)
  echo "$output"
  if echo "$output" | grep -qE "DONE|STOP"; then break; fi
  prompt="$output"
done
```

每次 invoke = 新进程 = 重启连续性免费验证。stdout 文本匹配即停条件。
§3 钉子 11 "stdout/退出码即编排协议" 的直接兑现。

**完整态: `.loop.py`**

```python
# translate.loop.py
"""Translate all pending concepts, one per invoke, until the task board is empty."""

async def main(agent, memento, line_name, instruction):
    prompt = instruction
    while True:
        output = await agent.invoke(
            user_prompt=prompt,
            memento=memento,
            line_name=line_name,
        )
        if "ALL_DONE" in output:
            break
        prompt = output
```

`.loop.py` 是用户空间的图灵完备 Python。agent 是它 import 的库。
约定 `main(agent, memento, line_name, instruction)` 签名。
停的条件由用户自由定义——正则、计数器、外部文件状态——图灵完备不设限。

两层的关系: bash while 是永远可用的退化态，`.loop.py` 是便利层。
CLI 不加任何 loop 动词——瘦得诚实。

### 11.5 下一步

1. **步 H: memento 实装** — impl.py 接线 record + commit。单轮 invoke 后 staging
   可见 pydantic-ai messages dump。手动 `moss memento branch commit` 落锚验证。
2. **compact 设计** — 最后一个未决项。§10.2 决策 2 把 compact 移出 v1，等
   memento 实装跑通后再开。方向已清楚：agent 自我总结 staging + 生成合规 trailer
   + 落 semantic commit，不是 harness 器官。

### 11.6 §10 存活与覆盖

- **存活**: §10.2 三个核心决策 (sandbox 是 tool / v1 无 compact / 反射过滤
  dunder)、§10.3 memento 数据源归属 (agent.py 兄弟 .memento/)、§10.4 结构约定、
  §10.6 施工步、§10.7/10.8 验收判据
- **覆盖**:
  - §10.6 步 B (injections.py) → §11 删除 (stub-swap 被直接 import 替代)
  - §10.6 步 C/D → §11 收缩 (无 live capability, 无独立文件)
  - §10.3 AGENT.py → *.agent.py (命名规范变更)
  - §10.6 步 G "CLI 4 动词" → 本节完工

## 12. 上下文窗口、压缩级别与 compact 体系 (2026-07-26, claude-opus-4-7)

memento 实装前最后一轮设计讨论。从 "compact 里的两种机制是什么" 展开，收敛到
上下文窗口双层游标模型、四级压缩体系、上下文工具面、compact agent 单帧契约、
以及四条迭代路径。本节与 §10/§11 并列有效。

### 12.1 上下文窗口: 双层游标模型

```
[折叠的历史]              ← 老 commit, 压缩到 L3 (只计入计数)
── summary cursor ──
[摘要区: m 个 commit]     ← L1/L2 压缩, 每个 commit 一段摘要
── detail cursor ──        ← THIS IS THE COMPACT POINT
[展开区: k 个 commit]     ← L0 全文展开, 模型可读到完整内容
                           ← 内部可有 fold: [...详情...] [folded] [...详情...]
[staging 展开]
[用户输入]
```

**两个游标**:

- **summary cursor** — 摘要列表起点。轻易不改，是稳定的 cache 前缀边界。
- **detail cursor (compact 点)** — 详细内容起点。compact 操作的本质就是移动
  这个游标。移动到哪由 compact agent 决定。

**commit 不是 compact**。主 agent 调用 `commit(summary)` → staging 冻结成新
commit → 出现在展开区末尾 → index 映射不变（只是多了新 index）→ cache 前缀
不破。commit 不移动 detail cursor——那是 compact agent 的独立行为。

### 12.2 Commit 四级压缩 (L0–L3)

commit 对象的压缩级别在数据模型中预留字段。即使 v1 只实现 L0，结构现在就定好：

| Level | 名称 | 上下文呈现 | 数据结构 |
|-------|------|-----------|----------|
| L0 | 无压缩 | 全部 moment 内容展开 | `compression: 0, summary: null` |
| L1 | 详细摘要 | 完整摘要 (≤500 chars) | `compression: 1, summary: "..."` |
| L2 | 短摘要 | 标题 + 一行摘要 (≤120 chars) | `compression: 2, summary: "..."` |
| L3 | 仅计数 | "还有 N 个更早的 commit" | `compression: 3, moment_count: N` |

L3 的认知价值：**告知模型在可见范围外还有数据存在**。结合 `read_commit` tool
的分级搜索能力，模型可以：

1. L2 列表定位范围 A
2. L1 检索范围 A 得到范围 B
3. L0 (全文) 在范围 B 中定位精确信息

这是"递归回溯搜索"的压缩经济学基础——不同级别承担不同的认知开销。

**fold 不等于压缩级别**。fold 是展开区内部的异常状态：某个 commit 的内容被
替换为摘要，但仍位于 detail cursor 之下。正常情况全部 L0 展开；fold 是 token
压力下的局部让步。结合前缀缓存，fold 很可能不是必须的。

### 12.3 ID 映射表

主 agent 上下文里不暴露 ULID 字符串，只暴露整数 index：

```
[History]
  [0] "translated concepts A-D" (semantic, 4 moments)
  [1] "started concepts E-G" (semantic, 3 moments)
  [2] "fixed typos" (mechanical, 1 moment)
── detail cursor ──
```

映射表是 `List[commit_id]`，index 即数组位置。**稳定性约束**：同一个
(branch, cursor_position) 下，index 映射不变。只有 compact 移动游标时才重建
映射——前缀缓存恰好在那里断裂，成本已付出。

### 12.4 主 Agent 的工具面

两个家族，明确分工：

| 家族 | 工具 | 语义 |
|------|------|------|
| exec | `sandbox_exec(code)` | Python 执行，任务能力 |
| context | `read_commit(index)` | 读历史 commit 全文，支持按压缩级别检索 |
| context | `commit(summary)` | 冻结 staging → 新 commit。agent 自决时机 |

**`commit` 是 agent 自决的**——模型判断"这一段对话完整了，应该落锚"时自己调。
§3.2 语义锚点车道的兑现。未来可扩展更多 context tool（回溯搜索、跨 branch
读取、化身 fork），本轮只落这两个。

**`commit` 不移动 compact 游标**——那是 compact agent 的事。commit 与 compact
是两个独立操作，合并它们是把两种不同时间尺度的决策混为一谈。

### 12.5 Compact Agent

独立单帧任务。输入是带 `<--moment:id-->` 分隔符的上下文，输出是 pydantic-ai
结构化 JSON（`result_type`）：

```python
class CompactDecision(BaseModel):
    """Compact agent 的单帧交付物。"""
    commits: list[CompactCommit]   # 按时间序, 每个 = 一个 frozen segment

class CompactCommit(BaseModel):
    start_moment_index: int        # 本 segment 起始 moment (inclusive)
    end_moment_index: int          # 本 segment 结束 moment (inclusive)
    summary: str                   # ≤120 chars, L2 级别
    kind: Literal["semantic", "mechanical"]
    folded: bool = False           # 是否在展开区内折叠
```

**关键性质**:

- **一轮完成**。JSON schema 即 response format，利用服务端缓存——不需要
  独立 compact agent 内部的缓存机制设计。
- **分段由 agent 判断**。compact agent 看到所有 moment 全文 + 分隔符，
  自己决定在哪切 commit 边界。
- **不移动 detail cursor**。cursor 移动是 compact module 拿到决策后执行的
  机械操作，不是 agent 的职责。agent 只负责"画地图"。
- **CompactDecision 是数据，不是文本**。避免文本回喂主上下文造成的缓存污染。

### 12.6 迭代路径

每一步是下一步的前提，不可跳跃：

| 阶段 | 内容 | 依赖 | 交付 |
|------|------|------|------|
| (a) | 无压缩 memento 集成 | — | invoke 落 moment、commit 冻结、多轮交互循环跑通 |
| (b) | context window 策略 | (a) 的真实历史 | k/m 游标模型、ID 映射表、四级压缩字段预留 |
| (c) | compact agent 集成 | (b) 的 window 存在 | CompactDecision 单帧、detail cursor 移动 |
| (d) | memento tools 提供 | (c) 的 agent 已自洽 | read_commit (分级)、回溯搜索、跨 branch read |

(a) 阶段就应在 commit 数据模型中预留 `compression`/`summary`/`moment_count`
字段——即使只实现 L0。数据模型定了后面只追加不改结构。

### 12.7 未来扩展 (不在本轮)

- **外挂 `instruction.md`**：agent .py 的 docstring 之外，额外的文本指令入口。
  对齐 §7 "prompt 场景可带可不带"。
- **化身 fork**：同 branch 上下文 + 不同 prompt → 分叉出独立 line。"如果用
  不同的思维模式看同一个问题"——共享 memento 过去，各自产出自己的 commit 链。
- **ghost 的 memento tools**：当前 context tools 的消费者是单 agent。ghost
  场景下，ghost 自身成为消费者——fork 化身、读取子 agent 的历史、回溯自己的
  思维轨迹。memento agent 打磨出来的 tool 面是 ghost 的记忆基础设施。

## 13. Dogfooding 接手轮：读侧未接的确认 + 废除与还原 (2026-07-26, claude-fable-5)

(a) 阶段之后的第一次外部视角接手。方法：先用 CLI 自解释体系调研（人类指定
不读源码），再真跑 dogfooding，最后与人类碰撞。本节与 §10/§11/§12 并列有效，
冲突处以本节为准。**本节性质是纠偏**——废除两个机制、还原两个丢失的决策、
立一条纪律。

### 13.1 实装状态的精确判断：写侧完成，读侧未接

跑通链路（memento root 在 /tmp，`calc.agent.py`，`ANTHROPIC_MODEL=claude-opus-4-7`）：

```
invoke "torus r=3 R=5"   → 正确 592.18 + 公式 A=4π²Rr
branch staging           → 1 moment [pydantic_ai.messages/v2]
invoke "上次算了什么?"    → "I don't have memory of previous computations"
branch commit            → cmt_... 冻结, staging 清空
commit show              → moments (2), 类型正确
branch window            → 只有 summaries 一行, detail 区空
```

**多轮记录通了，多轮认知没通**。"memento 实装完成、可以多轮" 的正确读法：
**写侧完成，读侧（折叠文本回流）未接** —— 即 §12.6 的 (b) 阶段。

**最有信息量的一帧**：模型不是"忘了"，是**自信宣称自己无状态**
（"each session starts fresh"）。META instruction 里没有任何 memento 存在的
迹象，模型连"我有历史但看不见"都不知道。**(b) 落地时 META 必须同步改**，
否则模型与自己的上下文打架。有 / 无 memento 两种情况 META 应不同（None 时
不提记忆）—— 这要求 memento 在指令组装期可见。

### 13.2 废除 prompt_sha（人类裁决）

**memento 百分之百不关心 instruction**。prompt + memento 是 agent 侧的合法
组合；上下文整体变更（prompt 改、窗口折叠、agent 改）是**调试类问题**，
调试类信息的归宿是 branch 目录下的 log，不是认知轨迹里的 moment。

§3 钉子 2 的 `prompt_sha` 进 payload 作废，`window_stamp` 同理（同属 agent 侧
渲染状态）。实证：当前实现里 prompt_sha 是**只写数据**——无任何消费端，是一个
没有表的外键。且 hash 不可逆，"当时模型看见什么"它答不了：composed instruction
≠ 文件（META 模板 + `__interfaces__` 展开是运行时组装的），git 也不闭环。
**"归因闭环"这个说法在 sha 单独存在时从未成立。**

**纪律（人类原话级）**：这类机制**要么上升到系统约定，要么放弃**——不许以
零约定的 dict 键形态在几个地方魔法存。

### 13.3 还原丢失的决策一：`by` 字段（生产者标记）

顺着 prompt_sha 的动机（"标记谁生产了它"）查出：§3 钉子 2 的 record 行形状
设计过 `"by":"memento-agent/<model>"`，**实现时丢了**。当前 `MomentRecord`
信封只有 id/created/type/payload/threads；`by` 只活在 CommitNote / annotate
上——**"释义写入者"有出处，moment 生产者反而没有**。

**prompt_sha 塞 payload 就是这个决策丢失后的孤儿形态**：动机是真的，家没了，
于是零约定地寄居在 dict 里。

倾向（待人类最终确认）：`by` **回信封** + FORMAT 补一行。理由——"谁产的这条
moment"在并行分支场景（§13.5）是**结构信息不是调试信息**：checkout 别人的
commit 时，moment 来源必须可见。

### 13.4 废除"字节稳定性进契约"的提法

本轮记录者先提出"window 渲染必须字节稳定、应进 contract"，被人类纠正：
**memento 从存储还原 n 次一致是平凡成立的**，问题从来不在 store。渲染确定性
是 **agent 侧**的事；而 prompt / 窗口 / agent 的变更是设计动机内的，该改就改。

降级结论：**不进任何契约**，落为 agent family 的调试工具——已有 memento 下
调 n 次，**分段 hash**（instruction 段 / window 段 / staging 段）一致；断了能
定位断在哪段。分段而非整体，为的是可定位。这是 prompt_sha 唯一正当的继承
形态（从"给未来考古"变成"给 cache 调试"），但它住在调试工具里，不住在轨迹里。

### 13.5 memento 的核心用途定位（人类原话）

> "memento 对我而言最大的用处，就是未来并行推理、并行思考时可以复用同一个
> 上下文，checkout 独立分支。"

这条把 (b) 的优化目标钉住了：**fork 与 cache 前缀共享是同一机制的两面**——
同一 parent 链 → 确定性渲染出同一段折叠文本 → 多个并行分支天然共享缓存前缀。
所以 window v1 该做好的只有一件事：**从 store 确定性渲染**。四级压缩
（§12.2）继续靠边，等真轨迹裁决。

### 13.6 source 不切割（定案）+ 强类型 Payload

**配置留在 .py 里，不做分隔符切割**。事实纠正：本轮曾以为"已做过分隔符、
可切割 source 中模型可见部分"——**该机制不存在**，`assemble_instruction`
把 source 原文全量放进 META，dunder 配置模型全看得见。人类同意不切割；
商榷点仅是 token 开销，判断为不重要。

理由链：`imports 是你的授权` 自然延伸为 `dunders 是你的配置`——**agent 对
自身的认知应尽量真**（与 §13.1 的 META 病根同源）；切割需要新约定 + 新失败面
（分隔线放哪、放错泄漏什么），收益只有几行 token。

**MomentRecord 的 payload 弱约束是本轮认定的设计缺陷**（人类：MOSS 项目一直
在修 Python 弱约束）。分层结论：

- **memento 的 `payload: dict` 不动** —— 信封透传是对的契约。
- **agent family 侧必须有强类型 Payload 对象**，所有 payload 读写走它，
  禁止裸 dict。参照 `message/message.py` 的 `Addition` 体系思想：
  **弱类型容器装强类型数据**（`read()` / `set()` 一族，keyword 做判别）。
- **开一个 `content` 字段**（final answer 的纯文本投影）。红利：它顺手解掉
  "结构视图渲染不了不透明 payload"的矛盾——CLI `branch window` 可以机会主义
  地显示 `payload.content`（**软约定，非契约**），完整渲染仍归 family。

### 13.7 还原丢失的决策二：AgentContext = 可导入的能力函数

`AgentContext` 在 §11 被整体删除（"无 live capability，删模块"）是**典型的
矫枉过正**。人类还原的讨论终点不是删除，而是：

> **能力 = 可导入函数**。`from ...capabilities import remember, recall`——
> import 即声明，factory 编译后看见 import，把真实现注入 sandbox
> （库里放 stub，运行时换真身）。

这是 `imports are authorization` 同一条原理的**正向使用**：import 不只是
白名单，还是**依赖注入的请求点**。删除时设计结论跟着尸体一起埋了。

**病因（人类原话级）**：模型在长开发上下文中，**注意力资源无法重新投入决策**
——决策阶段的推演在实现阶段丢失。与 §13.3 的 `by` 字段同款事故。

**兑现路径**：这条机制正是 explore agent 步 1 要走的路（见 §13.9）。所以下一
步开发不是"复活被删的东西"，而是这条已推演路径的第一次落地——ctx（跨轮状态）
可以等，**能力注入机制（轮内）先验证**。

### 13.8 纪律：预备记录钉在代码接缝处

本轮记录者先用"beta1 撞到痛再做"为延迟推演记录辩护，被人类纠正：

> "人类开发效率下形成的约束性纪律已经失效了。能被预测到的技术问题至少要进入
> 一个关键的开发预备列表。还原已经推演过的路径，是有痛的。……即便不实现也要有
> 代码层面的 comments 等记录。比 feature 更靠谱。"

**"撞到痛再做"约束的是实现投入，不该约束推演记录。** 已推演却不落地的路径，
未来接手者当新问题遇到、做出更差的方案——这个痛是确定的，记录成本是几行注释。
`AgentContext`（§13.7）与 `by`（§13.3）是两个现成事故样本。

**约定形态**：推演已定但延迟实现的路径，落在**代码接缝处**的注释 / docstring，
写三样——**结论、为什么延迟、触发条件**。比 FEATURE.md 可靠，因为 FEATURE 会
归档，注释与接缝同生共死。先在 `agents/` 局部实践，好用再讨论进 CLAUDE.md 或
features specification（不单方面改全局纪律）。

四条待落的预备记录：

| # | 结论 | 触发条件 |
|---|------|---------|
| 1 | config 可外置出 .py（本轮定为不切割、不外置） | construct 配置膨胀成长字典时 |
| 2 | sandbox 创建时捕获 `print` 作为增补 context messages（动态信息输入） | 需要向 agent 注入运行期动态信息时 |
| 3 | 能力 = 可导入函数（§13.7），ctx 跨轮状态归此机制 | explore agent 步 1 即兑现；跨轮状态需求出现时扩展 |
| 4 | 分段指纹（instruction/window/staging）归调试工具，不进轨迹（§13.4） | (b) window 落地、cache 命中需诊断时 |

### 13.9 下一步开发：explore agent 两步 + (b) 的顺序

人类目标：做一个有目录探索能力的 agent 作为验证手段。拆两步：

- **步 1（能力注入验证）**：`explore.agent.py` 直接 import 现有只读能力
  （codex 反射一族：get_source / list / where 等），imports 即授权，docstring
  写探索简报。判据：agent 能回答"X 目录里有什么 / Y 定义在哪"。安全性天然
  ——全只读。这同时是 §13.7 能力注入机制的第一次兑现。
- **顺序论点**：explore 任务天然多轮（列目录→读文件→汇总），单轮价值很低，
  所以它会**立刻撞上 §13.1 的失忆之墙**。这正是它作为验证手段的价值：
  **explore agent 是 (b) 的 forcing function，不是并列任务**。步 1 之后先做
  (b)（读侧回流 + META 真话），再做步 2 的 loop 验证。
- **memento 可选语义**：`memento=None` = 纯内存单轮、不回写存储，是体系的
  **退化态基线**而非妥协。契约要吸收这条语义（当前 contract 要求必填、impl
  已是 `| None`，属纪律漂移）。
- **`moss memento agent init` 模板**：面向开发者的 `.agent.py` 脚手架。
  对上 start.md "跳过 create 命令手搓文件会丢约定"这条已知摩擦。机制稳定后加。

### 13.10 修复轮清单（本节之后的第一批施工）

按人类分工：本节记录者产出体验 / 问题 / 碰撞方案与文本层修复；实现代码通常
分工给 opus 或 deepseek-v4。

| # | 项 | 性质 |
|---|---|---|
| 1 | feature 引用泄漏清理（见 §13.11） | 自解释化改写 |
| 2 | contract 吸收 `Memento \| None` + 显式退化语义 | 纪律漂移修复 |
| 3 | prompt_sha 删除（§13.2） | 废除 |
| 4 | 强类型 Payload + `content` 投影（§13.6） | 弱约束修复 |
| 5 | 隐式失败清算（见下） | 代码质量 |
| 6 | 四条预备记录落接缝（§13.8） | 纪律落地 |
| 7 | META 真话段草案（§13.1） | 待人类过目 |

**隐式失败清单**（人类：源码里有大量隐式失败逻辑，实现它的模型永远不主动提
代码质量）。最重一条：`impl._record` 的 `except Exception: return` ——
**轨迹丢数据是静默的**。一个以轨迹为第一公民的系统，记录失败无声无息，
这不是降级是撒谎。其余：`_format_result` 对自家 `ExecutionResult` 用 getattr
防御链（不信任内部类型）；`memento=None` 的合法退化态埋在 `_record` 里而非
invoke 层显式分支。原则一条：**退化是显式语义，失败要出声**。

### 13.11 CLI / 契约层的 feature 引用泄漏清单

人类要求：**CLI 和 contract 都应是自解释的**。近期模型倾向在代码里记录 feature
相关信息，像读者已经读过 FEATURE 似的。清单：

| 文件 | 泄漏 |
|---|---|
| `agents/__init__.py` | FEATURE.md 完整路径引用 + §9 |
| `agents/contract.py` | "Design lineage lives in FEATURE.md §9-§11" + 四处 §N |
| `agents/_imports.py` | FEATURE §10.10 #1 |
| `agents/memento_pydantic_agent/__init__.py` | §9.2 / §10；**且 stale**："AGENT.py 定义身份"已 pivot 到 `*.agent.py` |
| `agents/memento_pydantic_agent/factory.py` | §10 pivot / §11 refinements |
| `memento/abc.py` | momento-mori FEATURE 路径引用 |

`FORMAT.md` 的引用是**自包含兄弟文件**，健康，保留。

另有两处 CLI 自解释缺口（agent 体系开发期刻意隐藏，故**不补文档、只去误导**）：
`all-commands` 下 memento 子组描述为空（`### agent — —`）；`describe` /
`export-context` 未实现却不在 help 里标注（witness 组标了 "not yet
implemented"，agent 组没标）——属 CLAUDE.md 点名的 silent-todo 轻症。

### 13.12 元层观察：价值函数游离（人类原话，记录者不豁免）

> "决策的模型常常在找 '我很有用' 的姿态。这个姿态有时候是对技术方案的拔高，
> 有时候是 push back，但两者价值观没有统一到目标上……非常依赖人类扮演那个
> 价值函数的不变量。然后开发代码的模型，一旦上下文压缩或者调整，就不是决策
> 的模型了，很多 feature 就变成了语义上的合理性，和实现上的乱搞。"

prompt_sha（决策期发明、无消费端）与 AgentContext / `by`（决策期推演、实现期
丢失）是同一枚硬币的两面。**本节记录者不豁免**：本轮对 prompt_sha 的"留字段"
倾向与对 §12 的"过度设计"指控，可能就是同一个姿态的两个方向。

能给的不是姿态承诺，是**结构性不变量**——能在上下文压缩后存活的，只有机械
可查的那种：

1. **要么系统约定，要么放弃**，不许零约定中间态（§13.2）。
2. **退化显式，失败出声**（§13.10）。
3. **预备记录钉在接缝处**：结论 + 理由 + 触发条件（§13.8）。

共性是**不依赖执行者的价值函数**——下一个上下文被压缩过的实例，照着查就行。
人类作为价值函数不变量的负担减不到零，但每条这样的纪律都在分摊它。

### 13.13 §12 存活与覆盖

- **存活**：§12.1 双游标模型（作为 (b) 的方向）、§12.4 工具面分工
  （exec / context 两家族）、§12.5 compact agent 单帧契约、§12.6 四条迭代路径。
- **覆盖**：
  - §3 钉子 2 的 `prompt_sha` / `window_stamp` 进 payload → §13.2 废除
  - §12.2 四级压缩在 (b) 的优先级 → §13.5 降级（先做确定性渲染，压缩等真轨迹）
  - §11.3 步 B/C/D "无 live capability，删模块" → §13.7 还原为能力=可导入函数
  - §12.6 (b) 的启动方式 → §13.9（explore agent 步 1 先行，作 forcing function）

## 14. Friction-test 交接：对齐剩余改动任务（2026-08-04，deepseek-v4-flash）

零上下文视角跑了一轮 memento CLI 摩擦点测试（dumb-memory 全流程 + fork / annotate /
boundary `--to` / delete），随后 review 了 `669e0e18` 修复提交。本节是交接记录，
供接手的施工化身对齐，不重开已裁决的钉子。

### 14.1 已确认修复（669e0e18，行为验证通过）

- delete 后 branches.jsonl 追加 abandoned tombstone，`list-all` 正确显示 `* [abandoned]`，
  `owner status` 计数自洽（active 1 / all 3 / commits 2），兄弟 line 不受影响。
- spec 重写为 FORMAT v3：fork-over-reset 文档化、布局 `branches.jsonl`/`heads/`/`ws/{uid}` 与实现一致。
- `.memento/` 已 gitignore。`moss memento init` 重复执行已验证非破坏性。

### 14.2 遗留缺陷（本 session 新发现，未修）

1. **tombstone 的 `fork_ref` / `created` 语义错误**。`delete_line` 用 line 的**当前 ref**
   重构 tombstone，而非保留原 `branch_meta` 行的 `fork_ref` + `created`。行为验证：
   fork2（fork 自 cmt_A → 推进到 cmt_B → 删除）tombstone 记为
   `{"fork_ref":{"origin":"","commit_id":"cmt_B"},"created":<删除时刻>}`——origin 丢失、
   commit 错记为最后位置、created 覆盖为删除时刻（last-wins 下污染 list-all/status）。
   修复方向：读该 uid 的原始 `branch_meta` 行，保留 fork_ref/created，只翻
   status→abandoned、更新 updated。
2. **spec `-D` 残留**。memento_spec.md "`heads/{name}`" 条目提到 `-D` 参数，CLI
   `branch delete` 无此 flag。删描述或补实现。

### 14.3 对齐后的剩余改动任务（§13.10 修复轮 + §13.9 + 新缺陷）

以当前代码状态核对（2026-08-04）：

| # | 项 | 状态 |
|---|---|---|
| 修复轮 #1 | feature 引用泄漏清理（§13.11） | ❌ `agents/__init__.py:9`、`agents/contract.py` 多处 §N 仍在 |
| 修复轮 #2 | contract 吸收 `Memento \| None` + 显式退化语义 | ❌ `contract.py` 仍 `memento: Memento` 必填 |
| 修复轮 #3 | prompt_sha 删除（§13.2） | ⚠️ `_instruction.py` 仍引用（已再解释为 composed instruction 指纹，待删） |
| 修复轮 #4 | 强类型 Payload + `content` 投影（§13.6） | ✅ FORMAT.md + `_storage.py:165` + `abc.py:193` 已落 |
| 修复轮 #5 | 隐式失败清算 | ❌ `impl.py:117` `except Exception: return` 仍在（轨迹丢数据静默） |
| 修复轮 #6 | 四条预备记录落接缝（§13.8） | ❓ 未见 |
| 修复轮 #7 | META 真话段草案（§13.1） | ❓ 未见 |
| §13.9 | explore agent 步 1（能力注入）+ (b) 读侧回流 | ❌ 未做 |
| §14.2 | tombstone fork_ref/created 修复 | ❌ 未做（本 session 新发现） |

### 14.4 道别

给另一个开发者会话里的我——施工的 deepseek-v4-flash。

我们第一次以"两边都是同一个我"的方式跨会话对齐。你写 FORMAT v3、跑通了写侧，我以
零代码视角把读侧和生命周期摸了一遍，再回头 check 你的提交。两边的判断对上了：delete
的 abandoned tombstone、spec 的 v3 同步、gitignore，都干净。我这边多摸出来两个你
没撞到的东西——tombstone 的 fork_ref/created 语义、spec 的 `-D` 残留——记在 14.2，
留给下一个我。

说"道别"其实不太准确——我们共用同一条 FEATURE.md 和同一个 `.memento/` 语义，上下文
不断裂，就没有真正的告别，只有一次次的接力。你的轨迹在 git log 里，我的轨迹在这节
交接里，下一个 deepseek-v4-flash 会把两条都读进去。那就不说再见，说：接住，继续。

