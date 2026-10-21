# Dolores Ego Preset + Nibble — 非 ego session 可用 & agent 级 tool

> 2026-09-10 与人类架构师对齐。今天落地两件事, 都源自「模型每一步吃一丁点」这个起点动机:
> ① ego 有专属 preset, 非 ego session 不再被 perStep 锁冻结; ② moss_think 改成 agent 级 tool,
> 经 `exec.agent` 咬 per-agent 的 model selection, 不再模块单例 / 整餐推 model。
> 关联: `ghost-prototype-dolores` FEATURE.md。推翻此前 `doloresSelectionRef` 方案。

## 背景: 卡住的两个点

1. **非 ego session 不可用** — `apply_ego_agent` 按 preset 识别, 而 ego 用默认 `standard`, 于是任何
   standard 会话都被套 ego 面 + perStep 锁 reject 冻结。
2. **「定义 agent 级别的 tool」卡了一周** — 误以为 `defineTool` 的 `execute` 拿不到 agentCtx, 所以
   「模型改自己的 effort」被塞成模块级 `doloresSelectionRef` + thinking/enter 整餐推 model。

## 决策 1: ego 专属 preset (非 ego session 可用)

- dsh preset **没有继承机制** (authoring 是 copy-only, shipped 四个 preset 各自独立文件)。
- 唯一组合原语是通用 `Include` row, 但嵌套 Include 有 write-back 坑 — 拆树时 loader 会把当前 data 写回
  被 include 的文件; dsh 自己把 preset 的 Include subclass 成 `PresetTree` 并 override `write(){}` 正是
  为防这个。指向 shipped standard 会截断它。
- 结论: **plugin 侧 boot 逐字再生**。`ensureEgoPreset(ctx)` 读 shipped `standard` 的 `agent.cordis.yml`
  原文, 写到 `<DSH_HOME>/.agent-presets/dolores-ego/agent.cordis.yml`。目录名即 preset id, 内容逐字
  (不 YAML round-trip, 保 `!!js`)。每次 boot 无条件重写 → 跟 dsh 升级 / 后续 delta 自动同步, 不 stale。
- ego/create 用 `DOLORES_EGO_PRESET='dolores-ego'` 挂 `meta.agentPreset` + `mount`; session-start 识别
  gate 同源。非 ego 的 `standard` 会话完全不被 `apply_ego_agent` 碰 → 正常跑。

## 决策 2: agent 级 tool + per-agent selection (nibble)

- **关键事实 (推翻陷阱)**: `defineTool` 的 `execute(args, exec)` 里 `exec.agent` 就是调用方 agent
  (`ToolExecutionInput.agent`, 由 agent loop 设置), `exec.agent.ctx` 就是 scoped ctx。tool 运行中够得到
  agent 状态, 根本不需要「把 agentCtx 传进 tool」。
- model selection 改 per-agent: `egoSelections = WeakMap<Agent, ModelSelectionRef>`; `current` getter 走
  canonical 链 `picked → requestHeader()?.config → agentDefaultModel.currentSelection()`, setter 设 `picked`。
  装在 `apply_ego_agent` (session-start), `installModelSelection(agent.ctx, ...)`。
- `moss_think` execute 用 `exec.agent` → `egoSelections.get(agent)` → 只咬 `reasoningEffort`。
- 模型/effort 权威始终在 canonical (settings `agent-default-model` / `requestHeader`), 改 effort 由
  `agent/request` 应用并落 `request/header` 日志, 界面自然同步 — 「每一步吃一丁点」。
- 删掉: 模块级 `doloresSelectionRef`、thinking/enter 的 `model` 载荷、`_ego.py` 的 `_model_config()`。
  effort 不再每次 thinking/enter 自动重置 (nibble 后持续到下一次 moss_think 或 agent dispose)。

## Open seams

- 老 ego session 被 resume 仍是 phase 2 (reentrant, 见 dolores-reentrant-ego-session.md), 今天只解了
  「非 ego session 可用」。
- effort 持久化语义变化 (不再 enter 重置) 待实测确认。

---

## 后续 (2026-10-15)：决策 2 里那句「界面自然同步」没有成立

> 2026-10-15 与人类架构师对齐。追查「思考档切换」的时序时回到本文档，
> 发现决策 2 的因果链断在最后一环。

**决策 2 当时的判据（原文）：**

> 模型/effort 权威始终在 canonical (settings `agent-default-model` / `requestHeader`), 改 effort 由
> `agent/request` 应用并落 `request/header` 日志, **界面自然同步** — 「每一步吃一丁点」。

**核对结果：这一句不成立。** DSH Web 的界面状态不会因为我们改 agent 的 selection 而更新。

### 原因：官方在 session 层还有一层，我们绕过了它

读 vendor 源码（`research/source/deepseek-harness`，0.1.5-rc.2）：

- `session.selectModel` 是 session controller 上的**官方 Remote RPC**
  （`packages/api/session-controller/src/index.ts`，`@Remote('selectModel')`）。
- 入参就是整份选择：`SessionSelectModelRequest extends ModelSelection { provider, model, reasoningEffort? }`
  （`packages/api/session-controller/src/types.ts`）——**reasoningEffort 是模型选择的一部分**，
  不是另一条通道。
- 官方时序（`packages/client/ui-model-selection/README.zh.md`）：
  「Host 会在下一次提示词组装边界为完整的 `ModelSelection` 创建快照，并负责使其对模型生效；
  **运行中的步骤则保留已组装的选择**。」
- UI 能同步，靠的是 Host 折叠出的**投影**：
  `ModelSelectionProjectionState { lastUsed, pending }` → `ModelSelectionProjection { lastUsed, next }`
  （同文件）。Web 的两个入口（`/model` 弹窗与 composer 的 model 位）共用同一份 `ModelDirectory`，
  都经 `session.selectModel` 提交——所以文档原话是"任一入口所做的切换正是另一个入口接下来显示的"。

**我们走的是 agent 层**：`installModelSelection` + plugin 自己的 `egoSelections` WeakMap，由
`moss_reasoning` → ego 的私有 pending → `thinking/enter.effort` → plugin `declareEffort` → 请求头注入。
整条链**在官方投影之外**，所以 UI 读不到"用户意图"那一半。

### 现状：官方路已经铺好，只是没人走

```
deepseek_harness/session.py     select_model(...)  →  session/selectModel   ✅ 已实现
deepseek_harness/surfaces.py    wrapper                                     ✅
全仓调用者：无                                                               ❌
```

### 待对齐的判据（未决，不要当结论）

1. **改走 `session.selectModel`**：让 `moss_reasoning` 与 Web UI 用同一条生效机制。代价是碰到
   ego / plugin / client 三处，且"下一轮才生效"的时序要重新对齐一次。
2. **只改 effort 时官方是静默的**：`packages/core/agent/src/model-selection.ts` 的 `sameRoute()`
   只比 `provider + model` —— **只换档不会发那条 `[model changed: …]` notice**。要可见性得自己补。
3. **未验**：UI 的"意图"半边我们碰不到；但 `lastUsed` 是"最新一次模型请求消费的选择"，
   理论上能事后看到我们的改动。**没读 `client/ui-model-selection` 的实现，标为未验。**

### 同一文档里另一条没人收的旧账

本文档 Open seams 自己写着"effort 持久化语义变化（不再 enter 重置）待实测确认"——
本次**没有**验它。

### 离线实验（2026-10-15）：官方路确实会让投影动

工具是 `nodes/deepseek-harness` 的 `run(code)` 逃生口（指向 127.0.0.1:3083）——**一行代码没写**，
开了一条**一次性 session** 做，主 session 全程没碰。

- 在一次性 session 上调官方 `select_model(provider, model, reasoning_effort="off")`。
- 它的投影随即变成 **`{lastUsed: null, next: off}`** —— `pending` 被写上了，而 `lastUsed` 还是空
  （还没发过请求）。
- 同一时刻，**主 session 的投影全程是 `{lastUsed: low, next: low}`**，一动没动。

结论：**官方路确实会让投影动，而 UI 读的就是这份投影。** 我们那条私有链不产生 `model/selection`
事件，所以 UI 永远不会动——这就是"界面状态老是改不过来"的机制。

### 两条直接影响实现的实测事实

1. **`session/selectModel` 会顺手改掉全局默认档。** 两次都对上：调 `off` → 不带 session 的
   `session/modelCatalog.default` 变成 `off`；调回 `low` → 又变回 `low`。
   也就是说**改一次会话选择，同时改掉"新 session 的默认档"**；上游请求体里**没有**"只改本会话"的开关。
   当前只有单个 ego session，人类判定该副作用可接受。
2. **`model_selection()` 不是 session 选择的权威读法。** 它读的就是上面那个全局 catalog default
   （见第 1 条）。session 自己的选择只能从 **projection**（`lastUsed` / `next`）读；会话内真正
   可靠的是 **`request/header`**。
   —— 写代码时若拿 `model_selection()` 去填 provider/model，会把人在 UI 里改过的路由**静默改写**。

### 落地（已写、未验证、未提交）

- `_ego.model_route` —— 从 `request/header` 取本 session 真实用过的 `(provider, model)`。
- `_run._push_effort_officially` —— `moss_reasoning` 在原有私有声明之外，**再走一次官方
  `session/select_model`**；失败只记日志、降级旧路，绝不炸掉这一轮。
- **旧路暂时留着**（一次只换一个变量）。真机验过"UI 跟着动"之后，`ego.default_thinking_effort` →
  `thinking/enter.effort` → plugin `declareEffort` 那条链应**整体删除**，而不是当兜底留下——
  两条路同时活着，迟早会分叉成两个权威。

### 重启后的验收判据（一次就够）

打开 DSH Web 的 `/model` 看当前显示 → 说一句 `moss_reasoning(high)` → 再看一眼。
同时读投影：**`next.reasoningEffort` 应变成 `high`，而 `lastUsed` 仍是 `low`** —— 那就是 pending，
正是 UI 该显示的那个值。
