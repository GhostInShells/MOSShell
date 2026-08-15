# DSH 深入调研 — 探索轨迹 (2026-08-15)

> Dolores 集成 DSH 的完整探索轨迹。技术细节不记, 记录**问题 / 观点 / 探索路径 /
> 初步结论**。与主 FEATURE.md 的 DSH Integration section 对应, 这里是过程, 那里是裁决。

---

## 一、起点: bash 沙盒到底怎么做的

**问题** — dsh 的 bash 执行机制, 是"调度内核做沙盒 + 写副作用触发回调", 这是 bash 自身的能力还是某个 TS 包的封装?

**探索路径** — 从 `dsh-tool-bash`(模型面工具)一路下钻: tool(薄壳)→ `ctx.shell`
(执行器 seam)→ `dsh-bash-sandbox`(包 argv)→ `dsh-sandbox-local`(平台链选择)→
`node-addon-landlock-run`(C 二进制)。

**初步结论**:

- **不是 bash 的能力, 也不是单个 TS 包, 是「TS 编排 + OS 原语」的复合体。** bash
  只是最内层被关进笼子的 `bash -c` 进程。
- 沙盒按平台选 runner: Linux 是 bwrap(用户命名空间)或 landlock-run(C 静态二进制,
  直调 Landlock LSM syscall); macOS 是 sandbox-exec(Seatbelt SBPL profile)。
- "调度内核做沙盒"字面成立: landlock-run 的 C 源码对内核发 `landlock_create_ruleset
  / add_rule / restrict_self` 原始 syscall, 再 `execvp(bash)`, 规则跨 execve 继承。
- "写副作用触发回调"是两层: 前台 run() 的 Promise settle 渲染 marker; 后台
  ctx.jobs 的 `settle()` → `onJobDone` 监听者 → `owner.followup/inject`(唤醒或注入)。

**观点** — dsh 的沙盒层设计(denial 分类、fail-closed、escalation 审批)是 MOSS
沙盒未来可对照的范本, 但 MOSS 的 subprocesses 契约当时判断**不加 readonly**(无
对抗模型、有便宜替代), 这个判断在 DSH 集成决策下仍成立。

---

## 二、Python SDK: 三个同名包 + 一个真相

**问题** — dsh 的 Python SDK 怎么定义工具、怎么启动 session?

**探索路径** — PyPI 探查, 发现 `deepseek-harness` 是第三方同人包(OpenAI 兼容客户端,
Henry Zhang), `dsh` 是另一个同名项目(flashashen)。真正的官方 SDK 是
`deepseek-harness-sdk`(import 名 `deepseek_harness`, 作者 DeepSeek)。

**初步结论**:

- **SDK 是 JSON-RPC over stdio 的薄客户端**, 不是运行时。它拉起打包的单文件
  runtime(`dsh-jsonrpc-agent`, 来自 `deepseek-harness-runtime-bin` wheel), 通过
  stdio 行协议驱动。
- **工具不在 Python 里定义, 在 cordis.yml 组合里声明**。SDK 只传 `cordis=` 路径,
  工具面是部署时 settle 的插件集。session 是 `session_id` + `session_root`(JSONL)。
- SDK 的 Python API 面极小: `DeepSeekHarness` / `Session` / `RunResult` /
  `Notification`, 无工具注册通道。

**观点** — 官方 SDK "太简陋"是后来「Python 侧全部重做」的起点。它刻意保持薄:
  一切机制在 runtime 子进程, Python 只是 RPC 客户端。这个"薄客户端"哲学本身值得
  借鉴, 但它的同步 threading reader 与 MOSS 的 async 世界需要桥接。

---

## 三、JSON-RPC 协议: 调用侧 vs 回调侧的分界

**问题** — SDK 协议长什么样? 能不能在 Python 侧扩展工具/协议?

**探索路径** — 读 `dsh-sdk-protocol` 的类型定义 + `dsh-sdk-jsonrpc-server` 源码。

**初步结论**:

- 线格式: JSON-RPC 2.0, 一行一帧(id+method=请求, id=响应, method=通知)。
- 客户端→运行时**仅 3 个方法**: `initialize` / `session/prompt` / `shutdown`。
- 运行时→客户端**4 个通知**: `session.event` / `session.status` / `subagent.started`
  / `subagent.finished`。
- **协议里没有任何工具方法**, 也没有 system prompt 字段(这一点后来成为核心命题)。
- 客户端侧有 `IncomingRequest`/`respond()` 对称通道(runtime→client 请求), 但当前
  服务端 `handleRequest` 只实现那 3 个方法, 该通道是预留未接线。

**观点** — 这是"调用侧可对齐、回调侧不可对齐"的最早证据: 传输层是 method-agnostic
  的(任何方法都能发), 但服务端 handleRequest 写死 switch。Python 想扩展能力, 要么
  改服务端源码, 要么走别的通道(apiproxy 的 HTTP RPC, 后来才发现)。

---

## 四、上下文构建: boot 链 + session 世界

**问题** — 启动 harness 时, 它的"上下文"怎么构建?

**探索路径** — `dsh-app-boot.boot()` → `cordis-plugin-loader` → `dsh-agent` 的
`agents.create` → `dsh-system-prompt`。

**初步结论**:

- **两层上下文**。运行时层: 组合 YAML 是 entry 树(每项 `{id, name, config}` 一个
  插件), loader 并发 `apply(ctx)`, settle 后审计(全 active 才继续)。session 层:
  `agents.create({sessionId, meta, agentOptions, setup})` 惰性创建, setup 在发布前
  注册 scoped 工具/提示段/变量。
- **环境三层快照**(`dsh-launch-environment`): process env > 调用目录 .env > home
  .env, 冻结不可变, 组合里 `!!js process.env.X` 表达式读它。
- **系统提示是"注册表 + 每步装配"**, 不是一次性字符串。`systemPrompt.section/
  context/variable/tools` 注册, agent-loop 每步 `assemble()` 求值渲染。这是后来
  "system prompt 能不能创建时构建"命题的机制地基。

**观点** — session 是"上下文状态"不是"持久记录"的雏形从这里开始: 装配是每步的,
  注册是持久的。这个二分在最终架构里演变成"dsh session = 思考锚点, Memento = 记忆"。

---

## 五、plugin 的上下文机制: 六面

**问题** — dsh 插件到底有哪些上下文机制? (用户注意到它有 tools)

**探索路径** — 读 cordis 基座 + dsh 加的服务层。

**初步结论** — 一个插件 = 在六个面上注册:

1. **tools 注册表** — `defineTool({name, description, parameters, output, execute,
   presentCall/presentResult, timeoutMs, isConcurrencySafe})`; 执行管线是 4 个
   scope-filtered 事件(pre-execute / execute / post-execute / result)。
2. **systemPrompt** — section(有序)/ context(动态)/ variable({{var}} 插值)/ tools
   (schema 收集)。
3. **scope 系统** — 事件沿 scope 链向上流, 父作用域收所有后代事件。这是"一个
   composition 观察它底下所有 agent"的机制。
4. **事件总线** — 5 种派发(emit/parallel/serial/bail/waterfall), waterfall 组合成链。
5. **service seam** — `ctx.shell/subprocess/sandbox/fs/jobs/...`, 一个能力一个缝,
   实现可插拔(依赖注入面)。
6. **effect/fiber** — 生命周期 disposer, 逆序清理, HMR 依赖它。

**观点** — MOSS 的 channel/builder 体系与 dsh 的 plugin 六面是"同一命题的两套方言"。
  当时判断: CTML 工具面最佳实践是 plugin 不是 MCP(dsh 消费 MOSS channel = 挂
  `dsh-tool-*` 插件; MCP 只当 transport, 与 mcp_hub.py "MCP 降级为 transport,
  CTML 接管调度" 镜像)。这个判断贯穿到最终架构。

---

## 六、实验全链路: 从"能不能带 GUI"到"两个 dsh 互相对话"

**问题** — 用 Python 启动 dsh harness 时, web GUI 能不能看到? 能不能用它当 ghost
的思维空间?

**探索路径与关键弯路**(这一节弯路本身就是结论):

1. **runtime-bin 不带 web 插件组**(验证): 核对 `python/sdk-runtime/package.json`
   的依赖清单 — 有 `dsh-web`(搜索/fetch 能力缝, 不是 GUI), 无 `dsh-host-webserver/
   apiproxy/client-web`。**SDK 子进程是纯 headless(stdout 被协议独占), GUI 只能走
   `dsh --profile web`(完整 host+web 组合)。**
2. **独立 DSH_HOME 隔离**(坑): profile boot 每次启动重写 `profiles/web/cordis.yml`,
   共享 home 会与正在运行的 3080 实例互相干扰。实验用 node 内 `.dsh-home` + 软链
   共享 `profiles/node_modules`(全是符号链接, 指向全局安装)。
3. **gateway 广播验证**(弯路 426 → WS): 最初以为 `/api/events.mux` 是 SSE, 直连报
   `426 Upgrade Required` — **是 WebSocket 不是 SSE**。转 `POST /api/session.list`
   (HTTP RPC) 确认 Python 驱动的 session 在 web 侧可见; 再用 websockets 连 WS, 抓到
   35 帧实时广播(完整事件流: user/message → turn/start → assistant/chunk → turn/end)。
4. **DshChannel 实现 + 跑通 3081**: node 内 channel(talk/wait/drain/context), 经
   `matrix.nodes:run` 拉起 → 独立 home → `dsh --profile web --port 3081` → web GUI
   200。踩坑: `channel.build.startup` 是不带括号的装饰器用法。
5. **音频双链路**: TTS(`__main__.say`, 用户听到) + capture(3 秒波形, RMS 收到)。
6. **MCP 接入 moss**: 当前 3080 的 profile patch 挂 `dsh-mcp-client`(streamable-http
   指向 moss 的 `:20773/mcp`), 拿到 `mcp__moss__*` 全套 CTML 动词。
7. **全链路闭环**: 3080 → MCP → moss → CTML → `matrix.mesh.dsh_web_probe_01M030`
   (mesh proxy) → DshChannel → stdio → 3081, 跨实例对话 + 真实 LLM 推理成功,
   3081 GUI 同步显示。

**初步结论**:

- **gateway 广播对 SDK 驱动的 session 天然生效**(同一 session/event 总线, 不分
  来源)。"ghost 驱动 + GUI 观察"拓扑成立: stdio(ghost)+ WS(浏览器)都终止于
  harness, 是星形不是四边形。
- **两个 dsh 实例经 moss 互相对话跑通**, 证明"DSH 当推理中枢 + MOSS 当交互/执行"
  不是纸面设想。
- **channel 路径是 organ 全名**(`matrix.mesh.<name>_<id>`), 不是短名。
- **env 继承链解决 3081 凭据**: shell → moss-shell → node → harness, DEEPSEEK_API_KEY
  一路传下来(dsh 的 scrub 只防自己的子进程, 不防 moss 的 node 链)。

**观点** — 这一节是"可观测面"命题的实证: 3081 的 web GUI 就是 ghost 的思维空间
  观察窗, 不需要自己做 UI。它也是后来"不做 UI、终止点就是 final result"激进策略
  的事实前提 — 但 GUI 在激进策略里被放弃(只当调试通道), 保留与否待裁决。

---

## 七、架构收敛: 记忆归属、dry-run、与"dsh 退化为纯推理函数"

> 这是今天的核心收敛。起点是"session id 能不能承载持久化 ghost", 终点是"dsh 退化
> 成 think(moment)->result 的纯推理函数"。每个子命题都经过碰撞与纠正, 记录如下。

### 7.1 记忆归属: session = 思考锚点, Memento = 记忆权威

**问题** — 单一 session id 扛不住持久化 ghost(dsh 的 session 是线性 JSONL, ghost
需要无限上下文 + 化身分叉)。怎么办?

**结论**(用户裁决, 我认同): **dsh session id 对应 Memento 的一个 commit/moment。
session 外的历史与 CTML channel 由 Memento 组装(system prompt + content blocks)
驱动 session。全部 session 的集合 = ghost 发生过的思考锚点全集。**

- dsh 的 session 是"上下文状态", 不是"持久记忆"。无限上下文与持久化是 MOSS 的
  专有命题, dsh 方向性不一致, 不交给 dsh。
- Memento 是组装者(历史 + 身份 + 热数据), dsh 是被驱动的推理单元。

### 7.2 dry-run: dsh 没有, 但 fork 形同

**问题** — dsh 有没有 dry-run?

**结论** — **没有原生 dry-run**(全库无 dryRun/rehearse/simulate)。但 fork-而不-merge
就是 dry run 语义: `session.fork {sessionId, atSeq}` 按 turn 边界切历史成 seed, 新
session 继承源身份(cwd + parentSession + agentPreset + setup)。试跑不满意弃 fork
(源不受影响), 满意升格为新 Memento branch。

- 关键洞察(用户提出): **fork 可丢弃会话形同 dry run**。dry run 不是模式开关, 是
  "fork 而不 merge"这个动作本身 — dsh 无感知, Memento 定义"什么算 commit"。

### 7.3 runtime-context: 是 warm 不是 hot, 且与 MOSS 同构

**问题** — dsh 的 `RuntimeContextProjection.project(current, sections)` 是推还是拉?
和 MOSS 的 perspectives 什么关系?

**探索路径**(三次纠正, 本身是教训):

- 我最初断言"project 的消息不进历史" → **错**, 被源码纠正: `session.append(
  "user/message")` 进历史, 重启从历史恢复 retained。
- 我断言"和 MOSS perspectives 同构" → 用户纠正"你姿态转换太快", 让我查 MOSS 的
  shell context 机制 + 读 features(context-cache-engineering / interleaved-ctml-thinking)。
- 用户提示"应该读 features" → 读到 K5/D1/D2 设计意图后, 判断才收敛。

**结论**(读 features 后):

- project 是 **warm 用了 hot 风格**: 变化检测(`retained.text === snapshot`)+ 变更
  才进历史 — 精确对应 MOSS context-cache-engineering D1 的 warm 层(变更事件进历史)
  + D2(hash 渲染文本检测)。
- **DSH project = MOSS warm 槽位(help + interface + instruction + states)的已实现
  版本**, 同一机制两个实现: dsh 做 agent 级粗粒度(合并比较), MOSS 设计做 channel
  级细粒度(per-channel delta)。
- **dsh 没有 hot 槽位**: `deriveMessages` 全量重建, 无 ephemeral/transient 事件,
  请求无临时上下文字段。**hot(每轮变、尾部浮动、不进历史)是 MOSS 的独有空间。**

### 7.4 图片与视觉盲

**问题** — 图片进 dsh 历史有什么代价?

**结论**:

- 图片是 content-addressed attachment(同图同 id, 存储去重), 消息存引用非字节;
  compaction 有 image policy(text-only 裁剪)。
- **但 `dsh-llm-deepseek` text-only 拒图**(`UNSUPPORTED_CONTENT`), 且 `deriveMessages`
  全量重建 + 无跨请求去重 → 图片进历史必撞"窗口压满"或"传输放大"之一。
- **推论**: 高 churn 大块数据(vision)归 MOSS hot 层, 不进 dsh session。dsh 看文本
  世界, 看世界的是 MOSS。具身/桌面场景的感知走 MOSS 旁路(compact 成帧)。

### 7.5 system prompt 构建: 三条路径

**问题** — 创建 session 时怎么构建 System Prompt? (这是"唯一最关键的问题")

**结论** — 协议无字段, 但有三条正交路径:

1. **agent-preset**(声明式 YAML): `session.create` 带 `agentPreset`, apiproxy 走
   `agents.create({setup: composition.setup})` — 静态身份, 零 TS 改动。
2. **本地 JS 插件**(代码): loader 的 `name` 以 `.` 开头按相对路径 `import()` 本地
   文件, 不发 npm — 动态变量进 assemble(进程内)。
3. **contentBlocks**(Python 组装): 身份降级为 user message, 进历史全量带 — 零
   harness 改动, 但失去 system prompt 语义。

**当前观点**: 身份走 preset(静态), 热数据走 Python 手动组装(零改动), 动态变量按需
再用本地插件。三者组合关系待裁决。

### 7.6 激进策略: articulator/action 解耦, dsh 退化为纯推理函数

**问题** — dsh 看不了世界(视觉盲), 又不想背持续上下文, 怎么用?

**当前观点**(激进方向, 未裁决, 需推翻既有 "1:1 articulator:action" 决策):

- **回到最初 mindflow 三循环完全孤立**: 每次 articulator 激活 → fork session → 一次
  性思考 → 消费 final result → 结束。dsh = `think(moment) -> result`, 状态全在 Memento。
- **不做 UI**: 终止点就是 final result。GUI 只当调试观测通道。
- **信号半推半收**: 开始 = ghost 发(fork + prompt); 结束 = dsh 广播 turn/end +
  agent/status idle 事件, ghost 收。排队用 followup(next-turn FIFO), 不抢占用 steer。
- **千级 session 是 feature**: 每个 commit 一个思考锚点。Memento 存 (sessionId, seq)
  指针 + 元数据, 物理存储委托 dsh session 持久化; history 接口按指针精确取回。
- **并行思考调度旁路**(MOSS 侧), hot 数据 compact 成帧走旁路。

**待裁决点**(与主 FEATURE 的 Open Problems 对应): 热数据桥接形态 / system prompt
三路径组合 / 解耦策略与旧决策的冲突 / 千级 session 治理 / 视觉旁路的帧粒度去重时序。

---

(完 — 2026-08-15)
