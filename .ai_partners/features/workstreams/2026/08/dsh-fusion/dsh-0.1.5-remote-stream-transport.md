# dsh 0.1.5 传输重接 — Remote Stream + Web 鉴权

> dsh-fusion 的传输层重接决策。0.1.1-rc.2 → 0.1.5-rc.1 破坏了 `deepseek_harness/`
> 的三条基石：mux 端点、帧形状、鉴权门。本 workstream 只重接传输，不改融合语义。
> 结论：**走 web 面（`/api/remote.mux` 逻辑流 + token→cookie 鉴权），弃 sdk/acp stdio 面**——
> 后者拿不到 assistant 实时流，会杀死 Dolores 的 CTML-first。
> 关联：`ghost-prototype-dolores` workstream 的 O11；落点 `deepseek_harness/launcher.py`。

## 破坏清单

### 传输层

| 维度 | 旧 (0.1.1-rc.2) | 新 (0.1.5-rc.1) |
|---|---|---|
| WS 端点 | `/api/events.mux` (+`/api/events.host`) | `/api/remote.mux`（单条，多路复用逻辑流） |
| 握手 | 裸 WS 连上即收帧 | 先 `{type:'open', streamId, endpoint, payload}` 开逻辑流 |
| 下行帧 | `{type:'server-request', method, payload}` | `{type:'item', streamId, value}`；value = `ready\|emit\|waterfall\|cancel` |
| 上行回话 | plugin HTTP 路由 | waterfall 回 `POST /api/$events/result`（client-request 信封） |
| 鉴权 | 无 | `requestRejection`：Host/Origin fence + 签名 cookie |

### 两条逻辑流

- **`$events`** — 应用级转发事件（payload `{args:{}}`，须为空对象）。帧：`ready`(带 clientId)
  / `emit`(单向，`api-session/status|added|removed|activity|error` 等) / `waterfall`(需回话，
  `approval/request`、`user-questions/request`) / `cancel`。
- **`session/follow`** — session-controller 的 stream 型 Remote 方法。帧 `SessionFollowFrame` =
  `snapshot` + 有序 durable 事件 + **`assistant-stream`（live 的 start/chunk/end）**。
  **open 时须 `assistantStream:true` 才有逐 token 实时流。**

### 事件层小改

- `assistant/chunk` → `assistant/attempt`（chunk → `stream` 数组）。
- durable 的 assistant 事件是 **per-step 一次**（`assistant/message`/`assistant/attempt` 带整段
  `stream`），逐 token 的实时 chunk 只在 process-local `assistant-stream` 帧里。
- `todo/write` 移出核心；新增 `system/message`。

### 未破

- `client-request` 信封 `{type, rpcId, method, payload}` **schema 未变** → `DshClient.call` /
  `launcher.call` 逻辑原样可用，唯一新增是带 Cookie。
- plugin `ctx.webServer.register` 的上行路由**不经 `requestRejection`**（只有 `/api` 通道和 mux
  upgrade 被拦）→ 上行（thinking/enter|exit、tool-result 等）不受影响。

## Surface 取舍：为什么否决 sdk / acp

dsh 另有两个官方程序入口：`--profile sdk`（stdio JSON-RPC）与 `--profile acp`（Agent Client
Protocol），无鉴权，看似更干净。但：

- SDK 只转发 **durable session-log 事件**（`session.event`）；逐 token 的 assistant 流是
  process-local presentation frame，**只有 web 面的 `session/follow` 提供**。
- **Dolores 的 CTML-first 依赖 token 流出即执行**；走 SDK = 模型整段输出完才到 MOSS，实时性死。
- web + sdk 技术上可并存（都只是 dsh-base 上的 plugin row），但 stdout 冲突（web 打 URL 行污染
  JSON-RPC 帧），且无必要。

→ 结论：sdk/acp 对 Dolores **干净但残废**，否决。这条写死，防止下个实例再兜一圈。

## 鉴权方案

### token 来源

- dsh web 启动时把 `dsh web: <url>?token=<launchToken>` **打到 stdout**（`console.log`,
  `dsh-web-app/lib/index.js`）。launcher 本就消费 stdout → 正则抓取。
- `DSH_WEB_URL`（注入 shell 的环境变量）是 **clean url、不含 token**，不是来源。
- launch token **每进程随机、无 env 入口、不落盘**。
- 兜底来源：自定义 **`DSH_WEB_TOKEN`** 环境变量，`DshConnection` 启动时读一次 → 使基类无子进程
  也能独立起。

### token → cookie

- `GET /?token=…` → 303 + `Set-Cookie`（`dsh-auth-<hash(authority)>`）→ 会话 cookie。
- 之后 WS upgrade 与所有 `/api` 调用都带 `Cookie`。
- cookie 是 **opaque bearer**，存/重放**零协议耦合**。

### ticket 缓存（落 dsh home）

- 交换一次后把 cookie 存成 ticket，复用免重复授权。价值：①**连接可独立启动**（连不是我们 spawn
  的 dsh）；②免于 stdout 行格式漂移。
- key = **(dsh_home, authority)**；校验 authority 一致 + 未过期；401/过期 → 弃、回落 token、
  重铸回写。
- 落点：**dsh home** 路径下（`~/.dsh` 非 git 仓库，不需 gitignore）；权限 0600。
- 风险：ticket = 本地 dsh 全权 bearer，而 dsh 有仓库 shell 权限 → 泄漏 ≈ 工作区任意代码执行。
  故 0600 + 不进 VCS。
- **否决自签 secret**：直接读写 credentials secret 自签 cookie 会耦合 dsh 内部存储格式（HMAC
  载荷结构/版本），版本漂移即炸。存 ticket 不触这条线。

## 落点

| 文件 | 改动 |
|---|---|
| `launcher.py` | `DshConnectionConfig` 加 `token`；`DshConnection.token()` 访问器（config → `DSH_WEB_TOKEN` 兜底）；`DshLauncher` 覆盖 `token()`，stdout 发现优先；`_wait_started` 先等 token 就绪（拿不到即故障）；WS 改 `/api/remote.mux` + 逻辑流 open/dispatch；Cookie 注入；4 处 print 换 logger |
| `client.py` | 信封未变，仅需带 Cookie（经 http client 注入） |
| `session.py` | 帧来源改 `session/follow`；事件名小改（`assistant/attempt` 等） |
| `types/` | 新增 `SessionFollowFrame` / `$events` 帧；`SessionEvent` 对齐 |
| `__init__.py` | `DSH_VERSION` 0.1.1-rc.2 → 0.1.5-rc.1（传输重接完成后） |

## 分步

1. **token 落线**（✅）：config 入参 + `token()` + `DSH_WEB_TOKEN` 兜底 + launcher stdout 发现 +
   故障语义 + print 清理。
2. **cookie 交换 + 鉴权接线**（✅）：`_authorize` token→cookie、WS/HTTP 带 Cookie。ticket 缓存
   **延迟**（launcher 每次都有 stdout, 边际收益低; 独立连接由 `DSH_WEB_TOKEN` 覆盖）。
3. **下游重写**（🟡 半完成）：remote.mux 逻辑流 + `$events`（emit/waterfall/cancel/ready）+
   waterfall 回话已落地、经活 dsh 验证启动成功（token → cookie → remote.mux → $events ready）；
   `session/follow`（per-session 事件流）未接。
4. **事件层对齐**（⏳）：`session/follow` 帧 + `SessionEvent` 投喂 + `DshSession.run()` 重接 +
   `DSH_VERSION` bump。

## Open Seams

1. **authority 拼写** — cookie 名由 `Host` 头派生，`127.0.0.1` 与 `localhost` 不同名，须全链路
   钉死一种。
2. **`$events/result` 载荷形状** — `dispatchRpc` 收 `payload`，browser 侧发 `{args: result}`；
   信封内层待实测确认。
3. **follow 帧排序契约** — durable 事件与 `assistant-stream` 帧的交错顺序、`snapshot` 与 live
   衔接，待实测。
4. **ticket 与 dsh home 绑定** — dsh home 可经 `DSH_HOME` 配置，ticket 须随 home 走。
