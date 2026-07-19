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
status_note: 13 钉子已裁决; 施工暂停等 momento-mori §16 (branch=ref 重构) 冻结 — MementoBranch ABC
  将解体, 消费面变
title: Memento CLI & Agent — 无 harness 的轨迹 agent，memento 边界的 dogfooding 验证器
updated: '2026-07-19'
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

## Implementation Notes

<!-- 施工化身在此追加 gotchas 与决策. -->

- atom 的 `_adapter.py`（Moment↔ModelRequest）是 pydantic-ai 适配的参照物。
- file_editor 五动词是普通 async 函数（view/create/str_replace/insert/undo_edit），
  直接包成 pydantic-ai tool。
- 成本预估（可行性讨论定案）：codec adapter ~200 行 / facade+policy ~200 行 /
  tools 包装 ~100 行 / runner ~150 行 / 窗口文本渲染 ~100 行 / CLI ~200 行，
  合计 ~1k 行 + 测试，单 workstream 体量。超出量级即回看 §0 放弃触发器。