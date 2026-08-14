---
title: DSH Productization
status: draft
# priority: importance within the current stage (iteration cycle) — not development urgency
priority: P1
created: 2026-08-14
updated: 2026-08-14
depends: []
milestone: 0.1.0
description: >-
  将 DeepSeek Harness (dsh) 作为 MOSS 可驱动的外部 agent 面做产品化集成，
  候选取代 claude-code-in-moss。开箱暴露其 mode / 工具 / session / 权限面，
  不做应用实现。
---

# DSH Productization

> Use `moss features set-status dsh-productization <status> -m "note"` to update state.
> 本 feature 是 8 月命题：**候选取代 `claude-code-in-moss`**。尚未实际运行，以下判断均为预判方向，留待施工时验证。

## Motivation

DeepSeek Harness（`deepseek-ai/deepseek-harness`，MIT，2026-08-13 开发者预览）是一个
「Model + Harness = Agent」的 coding harness，对标 Codex / Claude Code。与
`claude-code-in-moss` 要桥接的 claude code 相比，dsh 有**更工程化的协议面**：

- 正式 Python SDK（`deepseek-harness`）+ TypeScript SDK + ACP 三套驱动协议；
- `session_id` + `session_root`（JSONL）做跨进程 resume，不靠 daemon；
- 「一切皆插件」（Cordis 内核），能力面可外部组合。

因此 dsh 是 MOSS 做「外部 agent 面产品化」的更强候选。本 workstream 的定位是**开箱不实现**：
把 dsh 的 mode / 工具 / session 轨迹 / 权限预设原样搬到 MOSS，暴露给 Ghost 驱动，
不替它写应用逻辑。

## Design Index

- 参考 feature：`claude-code-in-moss`（同类的「桥 Node harness 进 Python MOSS」先例）
- 集成关键面：`python/sdk`（Python SDK 源码）、`packages/acp`（权限机器裁决协议）

## Key Decisions

<!-- 以下为预判方向，非最终裁决。实际运行后由施工化身修正。 -->

### 集成路径（预判）

- **走 Python SDK，不走 CLI headless**。`dsh --profile headless "task"` 是一次性 print
  模式（无 resume、每次新建 agent）；loop + resume 只在 SDK 层
  （`DeepSeekHarness.run(input, { session_id })` → `RunResult`）。
- **Cell 作宿主**：Cell 生命周期 = dsh runtime 生命周期（start = spawn，stop = reap），
  把 dsh 的「spawn-per-instance」映射到 MOSS 的「cell 作为进程单元」。
- **session_root 落 MOSS 存储**：MOSS 管 `session_root` + `session_id`，跨进程 resume 由 MOSS 持有。
- **工程点**：SDK 是同步的（threading reader），MOSS 是 async，需 executor / 线程桥。

### 融合点（为后续施工实例提供的预判方向）

1. **权限**：dsh 权限 = 两个独立 knob（沙箱 `workspace-write`/`danger-full-access` +
   审批 `ask`/`never`），审批是可插拔 waterfall 应答者 seam。预判：MOSS 可把自己的 Ghost
   插成那个应答者（走 ACP `session/request_permission`，逐条 `allow_once`/`reject_once`），
   或走 SDK 预配置 preset。对应 MOSS「限制权限是为了授权」——沙箱 = 划边界，审批 = 敢放行。

2. **provider 协议化**：dsh 的 seam+inject ≈ MOSS 的 blueprint+IoC provider（都是「装线前
   协议化」）。但协议化对象不同——dsh 协议化**外围**（fs/process/sandbox/shell/subagent/llm
   可替换），MOSS 协议化**核心**（memento 是 contract，session 归属装线前由外部定）。预判：
   这是「能否让 Ghost 拥有 dsh 连续性」的分界，也是 feature scope 的上限。

3. **历史轨迹类交互方式**：dsh 的 `session_event_read/search/trace`（模型主动查自己历史，
   工具形态）≈ memento 的 `log/window/show/confluences`（历史组装进上下文，底物形态）。
   同一命题、不同交付。预判：这是两者最近的收敛点，值得持续对照——尤其 dsh 只有 fork
   无 confluent，memento 有 reference-confluent（图结构）。

4. **开箱能力面**：4 mode（standard/minimal/PTC·Code Mode/creator）+ 30+ 模型可见工具 +
   session 轨迹 + 权限预设。预判：GUI 集成直接暴露这些「开箱」面（会话栏 / 模式选择 /
   工具活动流 / 权限预设 / 自我查询入口），不重实现。

5. **`-p` 交互方式**：dsh 的 loop 原语是 SDK 的 `run(input, { session_id })`，语义是
   quiescence-based（`finalResponse` = 区间内最后一条 committed assistant 文本，非因果
   绑定 prompt），不是 turn-causal。预判：GUI 集成要按「activity interval」而非「一问一答」
   来建模多轮交互。

### 定位边界（预判）

- **可「驱动」，不可「拥有连续性」**：SDK 让 MOSS 能驱动 dsh 的 session，但 session 焊在
  dsh runtime 内，不能 memento 化（不能外部 fork / confluent）。集成分界 = 执行反转了、
  连续性未反转。
- **开箱不实现**：只暴露 dsh 的开箱能力面，不做应用实现。

## Survey Findings（源码级确认，2026-08-14）

本轮做了源码级调研（未运行），把多数「预判」钉成事实或修正。核心结论：

### 三套进程外协议，别混（本轮最大澄清）

dsh 对外的「协议」不是一套，是三套独立的：

| 协议 | 载体 | 角色 |
|---|---|---|
| **SDK**（`dsh-sdk-protocol`） | dsh 私有 JSON-RPC stdio | 驱动 agent：3 req（initialize/session/prompt/shutdown）+ 4 notif（session.event/session.status/subagent.started/subagent.finished）。raw lossless 44 型 session.event 流 + session_id 跨进程 resume |
| **ACP**（`dsh-acp`） | 标准协议（agentclientprotocol.com）automation 子集，JSON-RPC stdio | 审批/自动化：7 方法，committed answers only + `session/request_permission` 权限仲裁 + fresh sessions only（无 resume） |
| **Web/host** | Typert RPC over HTTP（:3080） | 浏览器连 host，不是 SDK 线 |

- SDK = **agent 面**（resume + raw 轨迹）；ACP = **func/自动化面**（fresh + committed + 权限）。
- 权限仲裁走 ACP `session/request_permission`，不走 SDK（SDK 的 responder 面是死能力，预留给未来）。这回应「单 turn 叫 func、多 turn 叫 agent」的区分。

### Python SDK 是「驱动整个 dsh」，不是「连 dsh」

- spawn 常驻子进程（`subprocess.Popen`），stdio JSON-RPC，**一次 spawn 跨多次 run 复用**（非 spawn-per-call），无自动重 spawn。
- 同步阻塞 + threading reader，无 asyncio（MOSS 需 executor/线程桥——原预判确认）。
- 控制面 = 常驻 reader 线程 + 订阅队列（pull）+ 按请求 on_notification 回调（push），**不是全局 on_event 注册**。传输层全双工，API 层同步 pull。
- Python 类型面极薄：models.py 只有 `JsonObject`（裸 dict）+ `Notification(payload: dict)` + `InitializeResponse`。44 型 SessionEvent 全不声明、靠字符串 key 访问，只「看懂」3 个 event type（assistant/message、turn/end、agent/inbox/spliced）。

### provider 注册 = build-time cordis.yml 声明，非环境发现

- `DSH_CORDIS_CONFIG` 只是「指向哪个 cordis.yml」的文件指针，不是运行时发现。
- 进程内 26 个 seam（swappable capability），core services 不可替换——`ctx.sessions` 确认为 core，session 焊死 runtime 内，「不可 memento 化」坐实。
- 依赖是纯 IoC：`inject` 声明 → 等 service 出现 → `apply(ctx)` 注册。星形依赖（provider → abstract Definition），非 provider 网状。
- 唯一运行时挂载例外：`initialize` 握手的 llm-deepseek fallback（DeepSeek-specific）。

### 工具注入三条路 + SDK 默认配置极简

- 工具注入：① TS Cordis 插件（`ctx.tools.register(defineTool(...))`，build-time）；② **MCP client**（MOSS 跑 MCP server，dsh `mcp-client` 连，工具以 `mcp__<server>__<tool>` 进模型——跨语言现成桥梁，MOSS 已有 `moss-shell mcp`）；③ skills（skill-filesystem 发现 + tool-skill 暴露）。
- **MCP 只做 client 不做 server**：dsh 连别人的 MCP server，不把自己暴露成 MCP server（对外是 SDK/ACP/Typert）。且 MCP 只桥 tools，不桥 resources/prompts。
- **SDK 默认 cordis.yml 只有 8 条目**（纯 chat spine：jsonrpc server + agent spine + llm + JSONL 持久化 + bash/fs 执行器），**无 model-facing 工具**。完整 dsh-base ~100 条（tool-bash/tool-fs/tool-web/tool-skill/tool-subagent 等 + skill 四件套）。
- skills 四件套：skill（Definition）+ skill-filesystem/skill-badge（Provider）+ tool-skill（Consumer）。

### workspace / main / profile / 安装

- 无 `dsh init`。只需 `cwd`（DSH_CWD）+ `session_root`（DSH_SESSION_ROOT）两个路径指针。
- 「main」= wheel 里 bundled 单文件 Node 可执行，无需系统安装；注册文件 = cordis.yml。
- `dsh --profile <name>` = 组装（composition），非 UI 类型。单进程单端口（web :3080）内部多 session；多开 = 多进程多端口。TUI = 外部插件 `turtle-ui`（`github:deepseek-harness/turtle-ui`），当前装不了（仓库未公开/私有）。
- **无 brew 公式**，唯一分发渠道 npm（`npx @deepseek-ai/dsh web` / `npm i -g @deepseek-ai/dsh`，需 Node ^22.19 或 >=24）。

### agent-surface 重定位（本轮最重要合成）

dsh = agent-surface 骨架下的**第三个 concrete agent**（memento/claude/dsh），不是新协议命题：

- **fusion point #4（开箱暴露 30+ 工具）是三重错帧**：① 工具不在 SDK wire 上（无 tool RPC）；② SDK 默认 cordis.yml 根本不挂工具；③ agent-surface §2.3「表面是最小驱动契约，不是 concrete 全能力面」——工具/mode 是 dsh 内部模型可见面，不是给 MOSS 的。
- **应重定位**为「给 dsh 写一个 Agent 表面 adapter」（create/__call__/context/4 控制函数），不是「把 dsh 的 30 工具搬成 MOSS channel」。
- **agent-surface §2.3 已回答「能否拥有 dsh 连续性」**：不该拥有——surface 是兼容契约，concrete agent 自持连续性。「可驱动不可拥有」不是 limitation，是正确落点。
- 建议挂 `depends: agent-surface`，与 claude-code-in-moss 同骨架。

### 架构差别（MOSS vs dsh）

MOSS = 单 Ghost 多界面（matrix 承载 N 界面 cell）；dsh = 单进程多 session agent，界面（web/tui/headless）是 profile 的 patch 层附属，session 才是核心。

## Open Problems

- 仍未实际运行 dsh，以上为源码级判断，需跑通 SDK 后验证（最小闭环见 Implementation Notes）。
- `finalResponse` 的 quiescence 语义（activity interval）对 GUI 多轮交互的实际体验影响未知——注意：这是 agent 的原生语义，不是要 reduce 成 turn-based 的东西。
- MCP 只桥 tools 不桥 resources/prompts——`moss-shell mcp` 实际暴露的是 tools 还是 resources/prompts，决定「注入工具」这条路的真实通量（待核）。
- ACP `session/request_permission` 的触发源与超时/fail-closed 语义（MOSS 不答会怎样）待确认——决定融合点 #1 能否落地。
- 集成层需容忍开发者预览的协议漂移（`serverInfo.version` 恒 0.0.1，无版本协商）。

## Implementation Notes

<!-- 施工化身在此追加 gotchas 与决策。 -->

- 先跑通最小闭环：Python SDK spawn dsh → `run(input, session_id)` → 读 `RunResult.events`
  渲染轨迹，验证 loop + resume + 权限预设三件事。
- dsh 处于开发者预览，官方明示破坏兼容性变更——集成层需容忍协议漂移。
- 参考 `claude-code-in-moss` 的桥接骨架，但用 Python SDK 替换 CLI 解析。
