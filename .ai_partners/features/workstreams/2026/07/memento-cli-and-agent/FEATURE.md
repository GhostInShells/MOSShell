---
created: 2026-07-18
depends:
- momento-mori
description: 在 ghost 融合之前，用一个 CLI 驱动、无 harness 的最小 agent 把 memento 全部边界画完并投入真实使用。AGENT.md
  目录 + .memento/ + bash/file_editor + pydantic-ai 原生序列化。可行性判据：讨论定案后模型不用人类干预独立做出来，
  做不出来果断放弃。
milestone: null
priority: P0
status: design-locked
status_note: '2026-07-24 §9 起草 (beta1 刻度 + 轨迹作为产物 + 8 步节点式施工):
  复工条件成熟 (FORMAT v2 冻结 / memento 一级 / CLI §19 通过 / pydantic-ai 2.5.0);
  核心转向: commit 归 agent 全权, invoke≠commit 生命周期, staging 残留合法, runner 不摸写侧;
  branch≈task 概念沉淀 (task 降级投影的物理落点), 分段多次提交是特性; 四锚 (factory+AGENT.md+
  memento+ground) 留文档层, 不进代码 (v1 保 MementoAgent 命名, ground 退化为 cwd);
  ABC tentative 4 方法 (invoke/compact/export_context_md/describe_line), 用 contract.py 不用 abc.py;
  8 步施工每步 checkpoint 明说等 review; commit 前缀 step N 便于事后复盘. §8 存活但被 9.2/9.9 部分覆盖.
  当前状态: 步 1 完成, 等 review 放行步 2 (目录结构 + 依赖复核).'
title: Memento CLI & Agent — 无 harness 的轨迹 agent，memento 边界的 dogfooding 验证器
updated: '2026-07-24'
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