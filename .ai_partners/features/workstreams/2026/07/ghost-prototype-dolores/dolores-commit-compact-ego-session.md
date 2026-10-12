# Commit / Compact Ego Session

> dolores context 管理的第二阶段前置调研。2026-09-09 与人类架构师讨论收敛。
> 结论：**dsh session 机制原生支持 commit / compact 两个动作，无需在 dsh 侧发明新东西。**
> 但两者都不走官方 `session.fork`——统一走 Dolores Ego plugin 层，底层 `ctx.agents.create`。
> 关联：`ghost-prototype-dolores` FEATURE.md；此机制接 memento 的 commit/segment 之后。

## 意图

回合制 agent 可以花两分钟现场做一次 compact（模型读全历史 → 生成摘要 → replace）；实时交互 ghost 的
compact 必须**瞬间发生**，不能占用主循环的关键路径。

解法是把"慢的摘要生成"从关键路径上分期前置：**前序阶段性 commit 预先固化摘要，compact 时只做拼装。**
于是 dolores 需要两个 ego 动作：

- **commit** — 在 completed turn 边界 fork 一份，让后台 agent 生成该点之前的摘要，存为可拼装的 commit。
- **compact** — 上下文到限时 create 新 session，seed = 已就绪的 commit 摘要（作第一条 user message，xml-like
  容器）+ last-commit 之后的增量原文。动作本身零模型调用，毫秒~百毫秒级。

两个动作共享前提：**agentPreset 一致，工具构建继承（不产生毒 session）。**

## 两个动作 → dsh 机制映射

| 动作 | dsh 机制 | 原生支持 |
|---|---|---|
| commit = fork last completed turn | `session.fork`（省略 `atSeq` 即 last completed turn） | ✅ |
| compact = create 新 session + 摘要作第一条 | `ctx.agents.create`（seed 第一条 = 摘要 `user/message`） | ✅ |

摘要的 xml-like 容器 dsh **完全不管语义**：`deriveEventMessage` 对 `user/message` 是 verbatim 透传
（`surface.ts:96-98`，`return event.data`），不做 framing/parse。塞进 `<container>` 或任何 xml 形状，
dsh 一字不改喂给模型。

## 关键事实（源码锚点）

- **fork 只认 completed turn。** `atSeq` 校正到所在 turn 的 `turn/end`（`api-proxy.ts:2283-2289`）；
  open turn 则 `fork-unavailable`（`api-proxy.ts:2290-2297`）。→ commit 只能在静止点（turn 结束/idle）做。
- **fork 继承 agentPreset + 工具构建，不是毒 session。** fork handler 走
  `composeAgent(resolveSessionPreset(source))` 重建工具 + `meta.agentPreset`
  （`api-proxy.ts:2321-2336`）。注释明说：seeded history 是在这些工具下产出的，换组合会
  strand 已有的 tool call。
- **compact 的 create 是既有先例。** ego/create 已把 `messages` 数组逐条
  `session.append('user/message', ..., {surfaceOp:'append'})` 注入（`plugin.ts:381-390`）。摘要只是把
  "多条"换成"一条 xml-like 容器"，机制相同。create 的 `seed` 接受任意 seq 0 连续事件数组。
- **agentPreset 从 log 解析，不是只看 header。** `resolveSessionPreset({header, events})`
  （`api-proxy.ts:480-485`）：preset 存在 log 事件流里，fork 后 `resolveSessionPreset(source)` 能还原。
- **compact 的 seed 是重组，不受 fork 的"纯前缀"限制。** 只要满足通用约束：seq 0 连续、turn 平衡、
  无 dangling tool call。

## 分层判定：为什么官方 fork 不可用

官方 `session.fork` 是 `ctx.agents.create` 之上的 **UI 特化封装**，它锁死三个自由度，与需求逐条冲突：

| 官方 fork 锁死的 | 需求 | 冲突 |
|---|---|---|
| `seed = events.slice(0, cut)` 纯前缀 | compact 要"摘要作第一条 + 增量"，是重组 seed | ❌ 无摘要注入面 |
| `setup = composeAgent(...)` 固定 | 额外挂摘要工具 | ❌ setup 不是调用方写的 |
| `workspace.attachSession(childId)` 强制进 workspace | 后台 agent 不进 workspace | ❌ 无参数关闭 |

走 plugin 不是另起炉灶——官方 fork 与 plugin 要写的，底层是**同一个 `ctx.agents.create`**
（agent-loop 的 `createAgent`）。走 plugin 只是绕过官方 fork 强加的三条约束，拿回 seed 内容、
setup、workspace 三个自由度。

**分层结论：**

```
DshSession（通用抽象）       ← 保持官方动词，含 fork() 但仅作"继承式快照"语义
    ↓ 不承载 commit/compact
Dolores Ego plugin 接口      ← 新增 commit / compact 两个 plugin_call 路由
    ↓ 内部
ctx.agents.create({          ← 三者统一走这里
  seed: <摘要作第一条 | fork 前缀>,
  meta: { agentPreset, cwd },
  setup: <挂摘要工具 / 继承工具>,
  // 不调 attachSession → 自然不进 workspace
})
```

## Open Seams

1. **commit 的静止点**：fork 只认 completed turn。实时 ghost 中 turn 边界难界定，commit 要挑静止点或
   等 idle，否则 fork-unavailable。摘要生成由 fork 出的后台 agent（一次性、不进 workspace、挂摘要工具）
   消化，不阻塞主 ego。
2. **摘要重编码成 seed 事件**：commit 摘要是 MOSS 侧产物，compact 时须重编码成带 `surfaceOp:'append'`
   的 `user/message` 事件插入 seed 首条，再拼增量原文。整个 seed 须 seq 0 连续、turn 平衡。
3. **增量原文切片 + seq 重排**：last-commit 之后的原文仍在源 log，须按 commit 切点精确切片并重编号
   为 0 连续，切在 open turn 里 seed 校验直接拒。
4. **fork 后台 agent 的 model/provider 一致性**：历史里的 assistant 消息标注原 provider/model。后台
   agent 若换 provider 重放会冲突——同类型 agent 约束须钉死（child 与 parent 同 provider/model）。

## 装线计划 — 13 步（2026-09-13 与人类架构师对齐）

> 边界锁定：严格按本 13 条接线，不引入本清单外的自创设计。10–13 各是一刀，分开做。
> note 生产的执行基元（session ref fork 跑一轮）见 [dolores-reentrant-ego-session.md](dolores-reentrant-ego-session.md)。

1. **memento 根路径在 runtime 定义**，默认 `ghost_home/.memento/ego/`。
2. **runtime 按约定路径实例化并持有 memento**（`new_local_memento`）；启动时取 `main` 分支，无则建。
3. **基于 memento 构建 xml-like view 区块**：`<branch name=".." index="..">` 包若干
   `<commit seq="27-1027" created="..">note</commit>`；分支无 commit 时显示 `no commit`。
4. **view 边界进 meta 配置**（`.dolores.yml`，对应 `Branch.view(n=..)`）。
5. **memories 拓扑 = `groundset root → memento → epoch`**；ground 留在 memories 首位，不上移 instruction。
6. **ego 正常退出必 commit**；异常退出不管。
7. **commit 数据对象 = `{ref, prev_turn}`**：`ref` 是 `DshSessionRef`；`prev_turn` 是同 session 上一个
   commit 生成的 turn，作本 commit 区间下界。memento 只存 metadata、不解析。
8. **ego 注入 memento**（typed injection，不 back-ref ghost）。
9. **ghost 把 observer 传给 ego**，ego 在 memory 里独立组装 epoch start point。
   （`mindflow.moments` 的命名/生产时机由人类架构师在 mindflow 侧手动调整。）
10. **ego 定义 `commit(message="")`**：无 message → 立刻落锚点并返回；有 message → 锚点 + note 同落。
    **note 生产不绑定 ego 生命周期。**
11. **note 生产 = session ref fork 跑一轮**（旁路单轮基元）。下一步做。
12. **aexit 时 last turn 若无 commit 则补 commit**（去重/补漏）。这一步不做。
13. **ego `compact()`**：~~能换掉 runtime 持有的 ego 实例；create ego session 传入**可为空的 session ref**
    ——取自最后一个 **note 非空**的 commit，从源 session 拆出 ref 之后的 message，建新 session 运行。~~
    **→ 已被 2026-10-12「压缩机制 — 收敛设计」取代（见文末）：改走同 session 平面上的 surface replace，
    不新建 session。**

### 依赖：memento 坐标改造（→ memento-mori 实现）

- **`BranchView` 持有 `CommitView`；`Branch` 暴露 `CommitView`**（`CommitView` = `CommitRef` + `Note`
  成对 + 坐标，替掉现在只有 `id/message/seq` 的 `CommitSummary` 投影）。
- **坐标 = `{branch_index}-{commit_seq}`**（形如 `27-1027`）。选它是因为 ULID 26 位、20 条 commit 的
  view 就有 ~500 token 纯开销；坐标把纯开销压到个位数 token。
  - `branch_index` = owner 的 `branches.jsonl` 行序（append-only → 永久稳定；改名不动、删除不回退）。
  - `commit_seq` = 该支 `commits.jsonl` 行序，**commit 时定死**（生产即有精确 seq）。
  - 一个 commit 只住一条 branch 的 `commits.jsonl`，故坐标在 owner 内唯一，天然解掉「子支 seq 从 1
    重数、与父支撞号」。
  - `id`（ULID）留在数据模型做全局身份（`ForkRef`/文件系统/跨 owner），**不进 view**。
- **memento 须提供配套 index 查找**：坐标 → commit（不是渲染侧临时拼）。这是契约面的新增，待 memento-mori
  定形（`Memento.resolve_commit(coord)` / `branch.get_commit(seq)` + `branch.index` 一类）。

## 压缩机制 — 收敛设计（2026-10-12 与人类架构师对齐）

> 取代第 13 步的「create 新 session」版本。[dolores-memento-plan.md](dolores-memento-plan.md) §6 当初列了
> 两条路（**新建 session** / **同 session 平面继续**），此处选定**后者：同 session 平面上的 surface replace**。
> 理由：session id / agent / tools / 既有 commit ref 全部稳定，无重接线 race，不丢 subagent 等工作态；
> `collapseTurn` 已证明活 session 上 replace 可行。

### 一个原语

**compact = 原地把 surface 前缀 `[head, cut)` 替换成一条 `user/message`，内容 = ghost 的 memory
（ground + 重新渲染的 memento branch view）；`cut` 之后的原文 tail 原样保留。**

- 替换内容必须是 **memory（视图）**，不是单条 note —— note 是增量语义（`_note_prompt`：「只记自上个
  commit 以来」），单独拿一条替换整段会丢掉它之前的全部；只有 branch view 是整条轨迹的汇总，且它本就是
  session 创建时的 seed 材料（plugin `memorySeed` / `_runtime.memories()`）。于是**启动压缩与运行中压缩
  是同一件事**：模型的「头部」永远是 `memory(视图) + 自 cut 以来的原文 tail`。
- 旧 memory（head 处那次注入）与旧轮次一起被 replace 遮蔽，新 memory 落位。

### 切点 = 安全 compact 位点（SCS）

- **SCS = 倒数第二个 commit 的 `end_turn`**（即 `cut(n-1)`，`n` = tip）。永远把**最新一个 commit 的 span
  留在压缩区外**，作原文 tail。→「不丢最近几轮」是**结构保证**，不靠算。
- 保留 tail 的大小由 **commit 节奏**决定：`force_tokens`(T) 越密，tail 越小。T 与 `compact_tokens` 的关系
  就是「保留多少原文」的旋钮。
- `cut` 落在**已完成 turn 的边界**上 → tool-call/result 配对平衡，replace 合法。
- SCS 随 commit 前推；compact 时直接切，无重算。

### 触发与时机

- **阈值是绝对值** `compact_tokens`（模型窗口的某比例；架构师给 400k–800k 量级），不是 K/T 那种增量。
  取值来源 = `assistant/message` 的 usage（最近一次调用的真实 prompt 大小）。
- **判定**放在**上一个有效 run**（无异常 / abort）结束时；**执行**放在**下一个 run 启动时**（pre-step、
  请求之前）。→ 请求之前就压好，**溢出不会发生**（不是「先失败再补救」）。代价：下一个 run 启动有轻微延时。

### ego 启动：不再回放最近上下文

- `create_session` 的 seed 退化为 **memory 一段（ground + branch view）**，**不再**带源 session 的原文 tail。
- 退役启动路径上的 `resume_ref` / `buildEgoSeed` / `resolveCut` / `surfaceTailSeed` / `loadSourceEvents`。
  **dsh 历史消息协议再变，启动都不碰**（这是防「升级后 ghost 永远起不来」的关键）。
- 跨重启的连续性完全由 memento 承担：视图 + `read`/`chat`。丢的是最近原文轮次（重启 = 新 epoch）。

### python → plugin 协议

```
POST /moss-api/ghost/dolores/compact
  { memory: <content blocks>, cut_turn: <int> }   # 压到该 turn 边界, 保留其后原文
```

- 协议**语义化、不带版本形状**：`cut_turn` 由 python 从 SCS 给；turn→seq 解析、边界校正、replace 形状
  全在 plugin 那一侧。python 侧跨 dsh 版本一行不改。
- 版本耦合收进 plugin 内部封装（探测 + 归一，**不散落**）：
  - `appendReplace(session, range, msg, sourceSeqs)` —— 归一 `{op,start,end}` vs `{op,startSeq,endSeq}`
  - `onRequestError(ctx, handler)` —— 归一 retry 返回形状
  - `classifyOverflow(err)` —— 先认 code，认不到文本兜底
- 其余路由不变（ego/create、thinking/enter|exit、tool-result、bypass/run、read）。

### 溢出兜底（已实现）

- 主动压缩覆盖常态；`agent/request-error` recovery 作后手：认 `CONTEXT_WINDOW_EXCEEDED` → 有损压掉
  「当前 turn 之前」的全部历史 → `{kind:'retry'}`。防单步（一个超大 moment）直接冲过上限。
- 有损压缩**模型无关**（零 LLM 调用，溢出路径上不再发请求）：替换内容 = 缓存的 memory 文本块（plugin
  在 ego/create 与 compact 时记下）+ 一条「上下文被压缩」说明。self-bounding：无节点可压时退回原错误，
  不无限重试。
- 替换内容含一条说明 notice，把「发生了什么」直接告诉模型 —— 主动 compact 的 memory 视图前同样前置一条。

### 未决 / 注意

- branch view 受 `view_limit`（默认 10 条 commit）截断 → **压到视图时，超过 limit 的旧 commit 淡出模型
  可见记忆**（仍在磁盘，`read`/`chat` 可达）。这是「渐进披露」，但 retain 值待确认。
- `surfaceOp` 字段名跨版本分歧（0.1.5-rc.1 要 `{op,startSeq,endSeq}`）已在 plugin 封装层解决，python 侧无感。
- 旁路「多档（角色）」是后续方向（subagent / commit / clone 并行化身），本设计不涉及；见
  [dolores-reentrant-ego-session.md](dolores-reentrant-ego-session.md)。
