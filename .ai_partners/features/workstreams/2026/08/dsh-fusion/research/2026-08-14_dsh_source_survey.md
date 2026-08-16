# DSH 源码级调研 — 2026-08-14

> 从 `dsh-productization` FEATURE.md 提炼的源码级调研(未运行)。后被
> [2026-08-15_dsh_deep_dive.md](2026-08-15_dsh_deep_dive.md) 与
> [2026-08-16_dsh_kernel_privilege.md](2026-08-16_dsh_kernel_privilege.md)
> 的实验验证修正——凡冲突处以 08-15/16 为准。
>
> 预判方向(集成路径/融合点/定位边界)不在此保留,见 FEATURE.md 的 Legacy 指针。

## 三套进程外协议,别混(本轮最大澄清)

dsh 对外的「协议」不是一套,是三套独立的:

| 协议 | 载体 | 角色 |
|---|---|---|
| **SDK**(`dsh-sdk-protocol`) | dsh 私有 JSON-RPC stdio | 驱动 agent:3 req(initialize/session/prompt/shutdown)+ 4 notif(session.event/session.status/subagent.started/subagent.finished)。raw lossless 44 型 session.event 流 + session_id 跨进程 resume |
| **ACP**(`dsh-acp`) | 标准协议(agentclientprotocol.com)automation 子集,JSON-RPC stdio | 审批/自动化:7 方法,committed answers only + `session/request_permission` 权限仲裁 + fresh sessions only(无 resume) |
| **Web/host** | Typert RPC over HTTP(:3080) | 浏览器连 host,不是 SDK 线 |

- SDK = **agent 面**(resume + raw 轨迹);ACP = **func/自动化面**(fresh + committed + 权限)。
- 权限仲裁走 ACP `session/request_permission`,不走 SDK(SDK 的 responder 面是死能力,预留给未来)。

## Python SDK 是「驱动整个 dsh」,不是「连 dsh」

- spawn 常驻子进程(`subprocess.Popen`),stdio JSON-RPC,**一次 spawn 跨多次 run 复用**,无自动重 spawn。
- 同步阻塞 + threading reader,无 asyncio(MOSS 需 executor/线程桥)。
- 控制面 = 常驻 reader 线程 + 订阅队列(pull)+ 按请求 on_notification 回调(push),**不是全局 on_event 注册**。
- Python 类型面极薄:models.py 只有 `JsonObject`(裸 dict)+ `Notification(payload: dict)` + `InitializeResponse`。44 型 SessionEvent 全不声明、靠字符串 key 访问,只「看懂」3 个 event type。

> 08-15 修正:官方 SDK 太简陋,`DshChannel` 直接 speak stdio JSON-RPC(自定义协议客户端)是验证过的路。

## provider 注册 = build-time cordis.yml 声明,非环境发现

- `DSH_CORDIS_CONFIG` 只是「指向哪个 cordis.yml」的文件指针,不是运行时发现。
- 进程内 26 个 seam(swappable capability),core services 不可替换——`ctx.sessions` 确认为 core,session 焊死 runtime 内。
- 依赖是纯 IoC:`inject` 声明 → 等 service 出现 → `apply(ctx)` 注册。星形依赖,非 provider 网状。
- 唯一运行时挂载例外:`initialize` 握手的 llm-deepseek fallback。

## 工具注入三条路 + SDK 默认配置极简

- 工具注入:① TS Cordis 插件(build-time);② **MCP client**(dsh 连别人的 MCP server,工具以 `mcp__<server>__<tool>` 进模型);③ skills。
- **MCP 只做 client 不做 server**,且只桥 tools,不桥 resources/prompts。
- **SDK 默认 cordis.yml 只有 8 条目**(纯 chat spine),**无 model-facing 工具**。完整 dsh-base ~100 条。
- skills 四件套:skill(Definition)+ skill-filesystem/skill-badge(Provider)+ tool-skill(Consumer)。

## workspace / main / profile / 安装

- 无 `dsh init`。只需 `cwd`(DSH_CWD)+ `session_root`(DSH_SESSION_ROOT)两个路径指针。
- `dsh --profile <name>` = 组装(composition),非 UI 类型。单进程单端口(web :3080)内部多 session。
- 无 brew 公式,唯一分发渠道 npm(`npx @deepseek-ai/dsh web`,需 Node ^22.19 或 >=24)。
- **无 per-project 配置发现**:dsh 唯一从 cwd 读的配置是 `.env`。不同 project 配不同 agent 能力无原生机制,`DSH_HOME` 重定向是当前可用兜底。

## web 组件化 & vendor 可行性(源码确认)

- **store 分离**:状态引擎 = zustand vanilla + immer(`packages/client/runtime`),React-free;`web-react` 只是 glue。
- **client 侧自跑一个 Cordis ctx**(浏览器端),与 backend ctx 分离,唯一桥是 `connection`(RPC over WebSocket/HTTP)。
- **UI 由 slots 组装**(`ctx.slots`):`root` → sidebar/conversation/details/shell.overlay。chat = `conversation` 槽,可裁剪。
- **contract 面**:`ISession`(8 verb:prompt/cancel/rename/loadOlder/command/updateQueue/readAttachment + projections + 读快照)、`SessionsPort`(list/create/open)、`IWorkspaces`(CRUD)。

**关键结论:chat 界面是 SessionEvent 流的纯投影**,自己不持有会话状态。proxy 只需转发 event 流 + 8 verb 即可正确渲染,无需复刻状态机。

chat 的 backend 依赖全集(协议面):
- 读面:SessionEvent 流 → ConversationSnapshot;projections(todo/plan/goal 派生状态)。
- 写面:prompt(queue|steer)/ cancel / rename / command / updateQueue / readAttachment / feedback。
- 系统指令:插件注册的 prompt section + persona 配置拼装;compact 是斜杠命令(`command` verb)。
- fork:`ctx.sessions.fork(...)`。

**控制反转**:dsh web = Node backend(`ctx.agents` 持有 live session)+ 浏览器 client(RPC);owner 是 backend。要 ghost 可控,须把 session owner 从 dsh backend 反转到 MOSS。

**收敛形态**:matrix node 持有 session(owner 归 MOSS),vendor 的 chat 界面作 GUI 子进程,父进程 proxy 转发 event 流 + 8 verb。

**两个额外结论**:
- dsh 前端是近乎通用的 agent session 表面——只认 `ISession` + `SessionEvent`,任何满足 contract 的 backend(dsh / claude code / MOSS)都能挂它当界面。
- 存储侧(对 memento 有参考):session 日志 = zstd 压的 append-only JSONL(真值)+ SQLite(二级索引 + 投影缓存)。
