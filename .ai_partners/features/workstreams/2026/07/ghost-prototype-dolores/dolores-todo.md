# Dolores TODO

> dolores 的问题清单 —— **单一事实源**。dogfood 只负责发现与记录，状态在此维护。
> 状态: `open`(待修) / `uncertain`(不确定) / `fixed`(已修, 带 commit) / `verified`(下轮 dogfood 验证) / `invalid`(判定非 bug)。
> 由 `ghost-prototype-dolores` FEATURE.md 关联索引。dogfood 发现新问题在此登记，修复/验证在此改状态。

> **状态快照 (2026-09-07 更新)**：D1/D5/D7/D21/D23/D26 → `fixed` 待回归（见归口 commit）；D24 → `open`(检查未启动)；D9/D17/D18 → `invalid`。下一轮 dogfood 优先跑 D1/D5/D7/D21/D23/D26 回归。

> **2026-09-09 收尾判定**：D2 → `fixed`(human-confirmed)。D3 → 降 `P3`（除非用 channel 否则无法试开启）。D27 → `invalid`（方案已换，零件事）。D20 对齐修改已落地；D19 协议纪律补充已落地（markdown 与 CTML 互补不复述）。

> **2026-09-08 落地**：mindflow interleaved — incomplete impulse(首包)也送入思考单元, effort='none' 只观测不行动, complete 尾包折进响应帧再行动（「首包抢占注意力但不行动」）；attention 拆分 `draw_from`(冻结创建者)/`impulse`(活量) 并修吸收/衰减/挑战四处错位。

> **2026-09-11 拆分**：D22 拆两半——图片协议半边 → `fixed`（字节头嗅探，3080 实机验证）；moment dynamic context 丢失半边 → 新 D28 `open`（未定位，含与 D22 同源的假说）。

> **2026-09-11 新增**：D29 `open` — `session/frozen` 是 plugin 自造类型，不在 dsh `KNOWN_SESSION_EVENT_TYPES` 中，持久化读取门（`dsh-session-persistence/lib/index.js:1119`，未知类型须 `ignorable:true`）会**拒绝整条 log 加载**；`Session.append` 无 `ignorable` 写入口（`dsh-session/lib/index.js:1444-1475`）。→ 带此事件的 session resume 失败。

> **2026-09-12**：D29 → `invalid`。旁路机制取代「冻结」，`notifySessionFrozen` + `session/frozen` 已删除，零件事。

> **2026-09-12 新增**：D30 `fixed`（`e7924674`）— 打断时 tool 结果迟到打穿整轮。触发链 = 语音/输入抢占正在跑的 thinking → exit 非 yield → plugin cancel → abort 删 pending 号 → MOSS 回的 `/tool-result` 撞空号 400 → articulate 报错、该帧 moment 丢失。修法见 D30 行；**待回归**（需重启 dsh 生效，plugin 是内核插件）。

> **2026-09-12 收尾**：D25 → `fixed`。need_observe 亮着不思考的真正原因不是 observe 回路断，而是续帧被 plugin 的「inputs 为空不起 turn」规则挡成缓冲帧。`needsObserve` 标记 + steer 开轮已落地（`_ego.py` / `plugin.ts`）。同一条根因串起三个现象：续帧滞留、下一轮帧顺序错乱、fetch 的 moment 迟到一轮。**待重启回归**。D30/D25 均未做全量回归（用户明确叫停），只跑了 `test_dolores.py`。

> **2026-09-12 dsh 0.1.5 升级 + prompt 架构对齐**：环境升 Node 24.21（`import.meta.main` 需 ≥24.2；TUNA nodejs-release 已冻结，改 npmmirror 精确版本）；`~/.dsh/profiles/node_modules` 两个 pnpm 实体包清掉（heal 重建 symlink）。对齐结论：standard 的 host 平面三段（web-surface / deliverable-file-references / file-reference）由 host bundle 注入，改用选择性压制（plugin shadow web-surface + patch disable ui-deliverables/file-reference-local，保留 harness:source 与 TOOL_* 工具散文）；dsh `renderContextSnapshot` 是 append+文字 supersede 而非 replace-op，不可借作 MOSS dynamic context。决策详见文末「dsh 0.1.5 升级决策」。

## 缺陷

| # | 状态 | Pri | 问题 | 发现 | 归口 |
|---|------|-----|------|------|------|
| D1 | fixed | P0 | 反馈回路缺失 — speech 返回可听时长(played Ns / STOPPED 301)+ 命令结算入 InterpreterStoppedEvent，行动有后果 → 本能训练回路打通 | dogfood-2 | `7cbdc3ce`+`48447180`+`d228b0c1`+`f940d983`+`df3a14d6` |
| D2 | fixed | P0 | 帧纪律 — enter moment 未入 session / 奇数帧丢、回复后 flush。debug 发现帧在历史轨迹、界面未渲染，疑似展示层而非 tracer 丢帧 | dogfood-2 | human-confirmed 2026-09-09 |
| D3 | open | P3 | 模式默认 — 按会话种类设默认（实时→CTML / 阅读→文本）+ 双通道原语 + 不对称成本。除非用 channel 否则无法试开启 | dogfood-2 | — |
| D4 | fixed | P2 | effort 机制 — 自救工具(think) + effort 映射 + 文档化降级(Reasoning Effort 段)已落地 | dogfood-2 | `4fda96a0` |
| D5 | fixed | P0 | TUI 生命周期 — perStep 锁改 global ctx(agentPreset+sessionId gate)+ ego tools 改 agent scope，纠正首轮后二轮发不了的作用域 | dogfood-2 | `4fc8b0c5` |
| D6 | fixed | P0 | 沙箱 cwd 错位 — DSH cwd = ghost home 而非 project 根，ghost 无法读写仓库、无自迭代能力 | dogfood-2 | project_home → project root |
| D7 | fixed | P0 | dsh 提示词打架 — CTML-first 输出协议反转(`699984f2`+`4fc8b0c5`)+提示词重排完成；three-homes 工作区分层与身份宣言在 inception `_prompts.py` 明确，五处冲突消解(待回归) | dogfood-2 | `699984f2`+`4fc8b0c5` |
| D8 | fixed | — | 回声全量重渲染（facade-delta 未生效） | ego-wiring | `2e57a8f8` |
| D9 | invalid | — | baseline `<key>value</key>` 渲染污染 — 记录错误，key 作 tag 判定正确，无需改 | ego-wiring | — |
| D10 | fixed | — | yield 返回 "ok" 哑载荷 | ego-wiring | `59f13736`+`ab6aaac1` |
| D11 | fixed | — | fetch/wait 词汇（when vs who） | ego-wiring | `ab6aaac1` |
| D12 | fixed | — | "thinking" 占位符泄漏 | ego-wiring | — |
| D13 | fixed | — | dsh UI 生命周期（消息被吞） | ego-wiring | `ea90993a` |
| D14 | fixed | — | exit 失败闸门残留 | ego-wiring | `ea90993a` |
| D15 | fixed | — | inputs_messages 不一致（executing 归 context） | ego-wiring | — |
| D16 | fixed | — | observe 镜像风险（moment index 帧带序号） | ego-wiring | `59f13736` |
| D17 | invalid | P1 | 语言不匹配 — 中文输入，ghost 全文英文回答，markdown 内反而中文（未复现，待重观；暂判非 bug） | dogfood-3 | — |
| D18 | invalid | P1 | markdown 内自指重新发声 — 判定非 bug(模型输出问题)；parser 正确处理 `<|Markdown|>…</|Markdown|>` 成对 escape(已有单测 `test_dolores.py`)，可补边界单测 | dogfood-3 | — |
| D19 | open | P2 | 长篇大论 — 缺「简洁/少即是多」规则（旧 persona/behaviors 有，重写丢失）。待补进交互礼仪；(已把 `__content__` 从语音拿掉，只有 `say` 发声) | dogfood-3 | — |
| D20 | fixed | P1 | fetch wait_actions_done 三处张力 — 已随工具改名 `moss_fetch_next_moment`→`moss_wait_action_done` 且参数拿掉(无参, 默认 always wait+refresh+observe) 消解 | dogfood-3 | 2026-09-13 |
| D21 | fixed | P2 | dsh 侧先停 + 界面无中断 — teardown 时序已修(interpreter __aexit__ 清 clear_after_exit + mindflow 改 wait_compiled，待回归)；**界面无中断能力/双向同步未启动调研** | dogfood-3 | `4fc8b0c5` |
| D22 | fixed | P1 | 图片协议传输 — MOSS 图片消息 → dsh 图片消息: media_type 按扩展名猜(JPEG 存成 .png → 声明 image/png)，dsh attachment admission 拒 `Declared image type does not match its bytes`。已改字节头嗅探；3080 实机验证 accepted 且模型真读到图 | dogfood-3 | `message/contents/images.py` from_file 字节嗅探 (待 commit) |
| D23 | fixed | P2 | shell trajectory 验证方式 — help(notice)+interface 各自独立判断 delta 已落地，不再每次一起传 | dogfood-3 | `36dcaefd` |
| D24 | open | P0 | interpreter error 被 wrap 成 command error — is_notifiable(≥300) 语义已铺垫(`ceb9eef7`)；区别于 command error + 关闭方式未定。**检查未启动** | dogfood-3 | `ceb9eef7`(铺垫) |
| D25 | fixed | P0 | **need_observe 亮着却不思考** — 命令执行完 `InterpreterStoppedEvent(need_observe=True)` → `when_need_observe` 置位 observer → 帧循环 `while need_observe()` 生成**回声续帧**并 `thinking/enter`。续帧常无 percepts（它是 ghost 自己要求回看，不是外部输入），而 plugin 的规则是「inputs 为空不起 turn」→ 续帧只能滞留 `pendingMoments` 等下一次真实输入捎带：现象就是 need_observe 亮着却停住、下一轮开头莫名多出帧且顺序错乱、fetch 的 moment 也迟到一轮。修法：`_ego.needs_observe()` 从 `moment.previous.need_observe` 判定续帧身份，`thinking/enter` payload 增 `needsObserve`；plugin 在 inputs 为空且 `needsObserve` 时用 **steer**（而非 inject，inject 只投递不唤醒）开一轮，载荷带 moment_id。**待回归**：重启 dsh 后，命令执行完应自动续一段思考而不是等输入；`plugin.ts:796` 那一支无命中即可止损回滚 | 实机 dogfood（13:33 现场） | `a2613f08` 后的续提交（见 git log） |
| D26 | fixed | P0 | tui 遇 interpreter error 崩溃 — exeception 处理加固(print+continue)+runtime stop 非零退出，不再静默崩溃(待回归) | dogfood-3 | `261adbbd` |
| D27 | invalid | P1 | perStep reject 界面提示 — 调研路径搞错，可在 reject 处发 stream/error 类事件给界面提示。方案已换，零件事 | dogfood-3 | — |
| D28 | open | P1 | moment dynamic context 丢失 — 看 moment 疑似彻底丢了 dynamic context。未定位；**假说**：与 D22 同源——moment 带图且媒体类型错时 `durableMomentContent` 在 `thinking/enter` 中抛错 → 整个 enter 返 400 → context/inputs/epoch 全未注入。D22 修复可能一并解决；若 dynamic context 不含图则属另一机制，待下轮 dogfood 复现 | dogfood-3（D22 拆分） | — |
| D29 | invalid | P1 | `session/frozen` 使 log 不可 resume — plugin 自造类型不在 `KNOWN_SESSION_EVENT_TYPES`，读取门拒整条 log；`Session.append` 无 `ignorable` 写入口。已随旁路机制删除 `notifySessionFrozen`（冻结被旁路取代） | dsh 调研（旁路机制） | — |
| D30 | fixed | P0 | **打断即静默失效 — tool 结果迟到打穿整轮**：thinking 结束时 tool 仍在等 MOSS 回话（fetch/interleaved），exit 先 `agent.cancel` → abort 监听器 `pendingCalls.delete(callId)` 注销号 → MOSS 侧随后到达的 `/tool-result` 撞空号报 400 → 异常穿 `_dispatch_tool_result` → `logos()` → `_articulate` 整轮 articulate error。一次打断 = 一句 aborted + 一轮报废，且**该帧 moment 丢失**。实测同会话复现两次，均在「新输入抢占正在跑的 thinking」之后。修法（两层）：plugin 在非 yield 的 thinking/exit **先结算** pending tool（回普通结果 `{interrupted}`）再 cancel，并登记 settled-call 墓碑使迟到回话被安静吞掉（200 dropped）而非 400；abort 监听器改为 reject 但不删条目。MOSS 侧 `_dispatch_tool_result` 吸收 RPC 失败（warn + drop），迟到结果永不烧轮。**待回归**：重启 dsh 后复现「打断中含 fetch」场景，日志应见 `settled N pending tool call(s)` 且零 `no pending tool call` | 实机 dogfood（现场打断复现） | `e7924674` |

> dogfood-3 追加验证通过：perStep 锁上移全局生效；prompt 顺序调整后 CTML 默认输出立现。

## 未接能力

| # | 状态 | 能力 | 依赖 |
|---|------|------|------|
| W1 | open | Memento 持久化轨迹 — 纯内存历史换 commit 轨迹持久化（重启不丢、化身分叉） | momento-mori 契约 |
| W2 | open | Ghost 反身 channel — 以 `ghost` 名注册 channel，感知/操纵自身唯一入口 | — |
| W3 | open | 独立思维模块 — 并行化身（fork）+ 关键帧自测（checkpoint self-eval） | — |
| W4 | open | 模型自感知切换 — `ghost.model` channel 暴露 current/list/switch-model/window-status | — |

## 设计问题

| # | 状态 | 问题 |
|---|------|------|
| O1 | open | 时序对齐 — thinking 期内存状态如何按 moment commit 切分，未验证 |
| O2 | open | Memento 上下文映射 — 持久化轨迹如何组装为 articulator 可消费的上下文 |
| O3 | open | 千级 session 治理 — fork/commit 累积的 session 生命周期（GC/归档/索引），`(sessionId, seq)` 指针方案未实测 |
| O4 | open | protocol notice 重心 — 关键不是围绕 CTML，而是强调「输出=行动」；控制单轮输出内容 + 赋予连续输出能力 |
| O5 | open | 中断能力 — 未 wrap 叙述会发声是机制（要强调）；thinking 期可中断：replan ctml='' 不执行 or moss_shell_interrupt |
| O6 | open | Matrix 能力声明 — 通过 matrix 可见/可管理自身能力，默认只提供一小部分（修正「行动」修辞过度） |
| O7 | open | HARNESS_IDENTITY_TEXT 调整 — 考虑尊重 dsh，不再过度强调 GIS/MOSS 身份 |
| O8 | open | GhostRuntime 内核不允许崩溃 — ego 坏了要有感知，运行时异常经 tui error output 打印 |
| O9 | fixed | dolores-ego preset 工具面 keep/drop — 定稿 2026-09-13：keep agent-instructions/shell/fs/jobs/plan-mode/delegation/todo/web；drop persona(plugin shadow)/skill(MOSS 自有)/goal(loop 建立)/compaction(旁路 commit 取代)/ask-user/present。理由见 agent.cordis.yml 头注释。delegation 长期要换 per-directory 授权 + dolores clone + 代码驱动 loop(#8) |
| O10 | open | dynamic context 落点 — dsh renderContextSnapshot 是 append 非 replace；MOSS dynamic context 用自身 log replace op（仅 thinking/enter、工具调用不携带、turn/start 替换上一轮），具体落点待定 |
| O11 | open | dsh 0.1.5 传输协议重接 — mux `/api/events.mux`→`/api/remote.mux` + `server-request`→`emit/waterfall/cancel` 帧 + `$events/result` RPC + token 鉴权；事件层小修(assistant/chunk→attempt、todo/write 移出、+system/message)。launcher.py/client.py/session.py/types 重写，生命周期已解耦故边界 bounded |

## dsh 0.1.5 升级决策 (2026-09-12)

> 决策记录，非缺陷。dsh 0.1.5 升级引发的 prompt / 工具架构对齐结论。

### 环境升级（已完成）

- dsh 0.1.5 CLI 入口 `if (import.meta.main)` 需 Node ≥ 24.2.0；已升 24.21.0。nvm 全局包按 Node 版本隔离，需在新版本下重装 `@deepseek-ai/dsh@0.1.5-rc.1`。
- 镜像：TUNA `nodejs-release` 冻结在 v24.1.0（已废）；npmmirror CDN 文件真、`index.json` 过期 → 必须精确版本 `nvm install v24.21.0`。
- `~/.dsh/profiles/node_modules/@deepseek-ai/{dsh-sdk-protocol,dsh-sdk-jsonrpc-server}` 是旧 pnpm 实体包，heal 拒收 → 移走，heal 用 symlink 重建。

### 0.1.5 的 prompt 表面破坏

- `PERSONA_SECTION` / `PERSONA_ORDER` 删除 → `PERSONA_PREFIX_SECTION`(order 0) / `PERSONA_SUFFIX_SECTION`(order 10200)；order 用 `getSectionOrder()` 取值，勿硬编码。
- `@deepseek-ai/dsh-persona` config：`text` → `prefix`(必填)/`suffix`，新增 `complete` / `includeRuntimeContext`。
- standard preset 其它漂移（`command-goal`/`present`/`tool-web.fetch:true`/`modelSelectionSettings`）不再逐字跟随，见下。

### 传输协议破坏（未修, 独立大活）

- mux WS 端点 `/api/events.mux` + `/api/events.host` 整体删除 → 换成 Gateway Remote Stream：`/api/remote.mux`（`packages/api/gateway/src/stream-protocol.ts`）。
- 帧格式从 `{type:'server-request', method, payload}` 换成 `{type:'ready'|'emit'|'waterfall'|'cancel'}`；waterfall 回话走 `$events/result` HTTP RPC（`$events` 是逻辑流端点）。
- 鉴权：mux upgrade 过 `connection.requestRejection`，token 经 `authenticatedUrl`（部署后防攻击那层）。
- **事件结构是小改**：13 种事件大体稳。`assistant/chunk`→`assistant/attempt`(chunk→stream 数组)、`todo/write` 移出核心 SessionEventMap、新增 `system/message`。`tool/call`/`tool/result` 载荷稳定（工具桥不动）。
- 生命周期逻辑与通讯已解耦（防御设计），修复边界 = launcher 传输层重写 + session_events 小修，两件事解耦。落点 → O11。

### prompt 架构决策

1. **独立 dolores-ego preset**（取代「逐字复刻 standard + plugin shadow」）。文件为 repo 自持，launcher 启动前复制进 `.agent-presets/dolores-ego/agent.cordis.yml`，每次覆盖；plugin 的 `ensureEgoPreset` 删除。dsh 升级时手动 rebase 这一文件（跟随版本、非 delta、非运行时 transform）。
2. **选择性压制（放弃 `complete:true`）**：shadow persona prefix(instruction) + suffix(空) + harness:identity(GIS/MOSS) + app:web-surface(空)；patch 层 disable `ui-deliverables` + `file-reference-local`；保留 `surfaceContext:true` 让 `harness:source` 与 TOOL_* 工具散文由 dsh 自己注册、自动跟版本。只压掉 web-surface / deliverable-file-references / file-reference 三个毒段，其余官方散文（工具使用纪律）保留。
3. **分段结构在 dsh 侧，文本 python 侧 create 时改写**（非 enter；create 写 = system prompt session 内静态 = cache 安全）。
4. **preset 工具面**（O9）：shell/fs/jobs/plan-mode/delegation/todo/web 留，skill/goal/compaction/ask-user/present 砍。ghost 的手 = MOSS channel（CTML 实时交互）+ dsh coding 工具（自迭代）。
5. **plugin 改名 `moss_dolores_plugin.ts`**（注册路径变）；复制责任交 launcher 后 plugin 只做协议桥 + moss_* + shadow。

### runtime-context 边界

- dsh `renderContextSnapshot` = append + 按文本 dedupe + 文字 "supersede earlier snapshots"，**非 replace-op**；且 `preStep` 每步注入、会带进工具调用。**不可借作 MOSS dynamic context**。
- MOSS dynamic context 用 **MOSS 自己的 log replace op**：仅 thinking/enter、工具调用不携带、每次 turn/start 替换上一轮（原设计如此）。落点 → O10。
