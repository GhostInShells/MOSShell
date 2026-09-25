# Dolores TODO

> dolores 的问题清单 —— **单一事实源**。dogfood 只负责发现与记录，状态在此维护。
> 状态: `open`(待修) / `uncertain`(不确定) / `fixed`(已修) / `verified`(下轮 dogfood 验证) / `invalid`(判定非 bug)。
> 由 `ghost-prototype-dolores` FEATURE.md 关联索引。dogfood 发现新问题在此登记，修复/验证在此改状态。

> **2026-09-25 上下文回顾（补漏恢复 + 失败落盘）**：见 D35。关键现场——`fa97362d`（2026-09-15「给旁路任务一个生命周期」）在引入"关停取消在飞旁路"的同时，把启动补漏 `resume()` 和它的配置 `resume_tail` 一起删了。删除给出的论据是"非阻塞 teardown"，但那只正当化**取消**，推不出删补漏；且当时冷读能力（`09bb8b94`）已就位，"源 session 已死也能补"的前置本就成立。`dolores-memento-plan.md:21` 的计划条目至今仍在，代码没了、docstring 反而写"不做重启补漏"——**属误删，不是记录在案的设计反转**。另立 O18（零有效工作的区间要不要 commit）留待数据。

> **2026-09-23 全链路实机运行复盘**（见 [dolores-full-chain-live-run-retro.md](dolores-full-chain-live-run-retro.md)）：新增 D31–D34、方向 O12–O17、礼仪 W7。核心 = prompt 反转释放焦虑（ctml 显式围栏 + interleaved 默认 wait_action_done + moss_wait_action_done 拿掉）。最终回归在 dolores ghost 侧语音逐条试。

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
| D31 | fixed | P0 | 中断回合后音频有时未中断 — clear player 不重启设备 + 终止流 + 统一 seed key | 全链路实机 | 待回归 |
| D32 | open | P1 | AEC 回声 — 已找到原因并大规模重构（miniaudio factory 统一 player/capture、AEC 挂 emit clock），待实机测试；耳机+公放无问题，问题在特定接法 | 全链路实机 | 待实机 |
| D33 | fixed | P1 | facade 反转不刷新 — shell trajectory frame index 与 tracer event index 解耦 | 全链路实机 | 待回归 |
| D34 | open | P1 | harness waterfall 提示词丢失 — 模型没授权不会发申请 | 全链路实机 | — |
| D35 | fixed | P1 | **空 note 无补漏 + 失败不落盘** — 两条互补的空洞：① `__aexit__` 关停取消在飞旁路 → commit 的 note 留空（非阻塞设计使然，本身正确），但没有启动补漏把空 note 捡回来（`fa97362d` 误删 `resume()` + `resume_tail`，见文首 2026-09-25 注）。② 旁路失败（传输异常 / 空文本）只置内存态 `BypassState.FAILED`、**不写 Note** —— 重启即失忆，且留空会被补漏当成"还没生产"每次开机重跑。修法：恢复 `backfill()`（**非阻塞派发**，与运行期同一条 `schedule_note` 路径，不是旧的顺序 `await`；幂等依据是 memento 里那条 Note **本身**，ready 与 terminal error 都算"有"；无 ref 的外来 commit 静默跳过）；`_fail()` 在真失败时写 **terminal error note**（memento 早备好 `Note.error`，读侧标 broken、view 折叠），**取消不写** —— 留空好让下次开机补漏捡起来。`.dolores.yml` 暴露 `resume_tail`（0 = 关） | 2026-09-25 上下文回顾 | 本轮 |
| D36 | fixed | P0 | **思考档被永久钉死 / 界面改档不生效** — plugin 的 `ensureEgoSelection` 只有 `current/assembled`, **没有官方 `dsh-api-session-controller` 的 `consume` 释放**: 一旦写进 `picked` 就永久优先于 `request/header` 与 `agentDefaultModel`. 而 `_ego` 的初值硬编码 `"off"`, 于是**开机第一帧就把 `reasoningEffort: 'off'` 钉进之后每一个请求**; `moss_reasoning` 之后又把 picked 钉成新值, 界面再也改不动. 实机证据 (本次 session `session.v3.jsonl.zstd`): 首条 `request/header` = `reasoningEffort: off`, 而 `settings.yaml` 的 `agent-default-model.reasoningEffort` = `high` —— 人类设的档被 ghost 静默压掉; 全程只有一次 `change`（moss_reasoning 下发 low）, 页面改动从未进入任何请求配置. 另有两处同源缺陷: ① enter 把 `'none'` 帧的 effort 当"档位"下发, plugin 白名单不含 `'none'` → 静默丢弃**且**声明已被消费, `moss_reasoning` 会随机失效; ② `effort` 与 `reasoning_effort` 两个字段下同一个值, turn 驱动与档位混在一条判定里. 修法: **单字段 `effort`**（`'none'` = 不驱动 turn, `''` = 不表态, 其余 = 合法档位, 档位解析与 turn 驱动分开判定）; plugin 侧补 **apply-then-release**（pending 被 `request/header` 采纳即释放, 权威交还 canonical 链; 档位不丢, 它已落 header）; 声明**只在被真正采用的那一帧消费**; startup 增 `default_thinking_effort`（空 = 不表态, 缺省不覆盖 dsh/UI） | 第五轮回归（23:37–00:15 现场） | 本轮 |
| D37 | fixed | P1 | **落锚不可执行 —— 提醒与手段不匹配** — `memento_notice kind="warn"`（K=50k）催 "commit an anchor now", 但 ghost **没有任何动词能落锚**: `ghost.memento` 只有 `read/chat/view/history`, 内观 tool 面亦无. 模型收到一条自己执行不了的指令, 只能假装履行（与"挂出来却只会答做不了"同源）. 定案 (人类 09-26 定): 落锚**做成 channel 命令** `ghost.memento:commit(text__)`, **不进内观 tool 面** —— 记忆是器官, 写面与读面同处一处; 内观 tool 只留决策必需的（`moss_reasoning` 一类), `branches` 留给未来应用. 约定: **自己写 note 就不触发旁路**, 留空则旁路代笔 —— 判据是"这条 commit 有没有 Note"(与 `backfill` 同一判据, 落在 `schedule_note` 内部而非调用方自律); 区间上界 = **最后一个已完成 turn**（落锚落不住脚下这一轮）; 没有落锚后端就不挂这条命令. 未做 (同族, 人类点名的下一步): `compact` 的主动暴露 —— 顺序是先落锚再压。同轮补齐 `node(coord)` (打开某条锚点自己的节点空间: `ensure_memento` get-or-create + 一层 ls, 格式抄 `core/file_editor`; 磁盘活走 `to_thread`; 一碰就建 ⇒ "节点存在"≠"有内容", 契约里写明了) | 09-26 现场 | 本轮 |
| D38 | fixed | P0 | **感知延迟 —— 模型看到的永远是动作前的世界** — `moss_interpret` 等完 observed 就直接 `observe()` 签 moment, 中间**不刷暖层**; 而 metas 只在**下一次 interpreter 派发前**刷 (`ctml_shell.interpreter` L442-444 `await refresh_metas(timeout=prepare_timeout)`, 注释"阻塞等待刷新结果")。于是**任何改状态的命令, 其效果都要等下一道命令走一趟派发才进入模型视野**。实机证据 (09-26 现场): `ghost.frame:resolve` 01:21:59 生效 (返回 `1/5`), 而**同一帧**的 facade-delta 仍是 5 条问题; 直到下一条命令 (`resolved`, 01:22:19) 走完派发, 才刷出 4 条。对照: `moss_observe` 早就做了这一步 (`_run.py` 的 `_handle_observe`: `await self._facade.shell.refresh_metas(timeout=5.0, stale_time=1.0)`), interpret 漏了 —— 同一件事两处实现, 一处漏。修法: interpret handler 签 moment 前做同一步; 单测断言顺序 (refresh 必须早于 observe); 测试助手 `_run()` 补一个默认 noop facade (原来传 `facade=None`, 三个 interpret 用例因此裸奔)。**预测**: 1-8/1-9 那次 memento 逐字重发是同一条的症状 (迟到的刷新让相邻两帧的 `previous_metas` 错位), 不一定是独立的 cache bug; 修完若仍重发, 才是第二处 | 09-26 现场 | 本轮 |
| D39 | fixed (待重启实测) | P0 | **moss_reasoning 不终结本轮 —— 替身发明了生产没有的 API, 异常被静默吞掉** — 现象: 调用 `moss_reasoning` 后本轮照跑到 `completed`, 模型只收到 `null`. 实机证据 (session 日志): `moss_wait_next` 01:49:04 tool/result 同秒 `turn/end aborted/hook/moss cancel_turn`; `moss_reasoning` 01:53:23 与 01:59:31 两次都是 tool/result 文本 `null` + 本轮继续. 根因 (运行时 introspect 取得, 非推断): `_run.py::_handle_reasoning` 调 `self._thinking.add_echoes(observe=True)`, 而 `add_echoes`/`moments` 定义在 **`Action` 侧** (`core/blueprint/mindflow.py:1034`, 类 `Action`), `Thinking` 上**没有**该方法 → AttributeError → 被 `ToolCallParameter.run_tool` 的裸 `except Exception` 吞成 `ToolCallResult(result=None, error=...)`, 而 `_dispatch_tool_result` 只发 `result` 不发 `error` → **cancel 随之丢失**, 本轮永不结束. 替身 `FakeRunThinking.add_echoes(*messages, observe=False)` 正是生产不存在的签名, 因此单测 139 全绿而生产必挂 —— 今晚第三次遇到同一形状: **表面承诺了一个不存在的机制**. 修法: (1) 改调 `self._thinking.observer.add_echoes([], need_observe=True)` (`Observer(Moments)` 上确实有, 与 `_mindflow.py:707/835` 同一惯用法); (2) 人类裁定并采纳: `wait_actions_done()` 放在 cancel **之前** —— cancel 会掐掉在飞动作, 记账工具不许比它记账的世界更快; (3) `run_tool` 的 except 补 `_logger.exception` —— 静默吞异常是这条 bug 藏了整晚的唯一原因; (4) 新增 `test_tool_surface_matches_production_thinking`: 断言 handler 用到的 API 在生产 `Thinking` 上真实存在、且 `add_echoes` **不在**其上. 同轮结论**修正**: 早先"档位应用点应从 turn start 挪到 turn/end"的论证**依赖了"切断没发生"这个 bug** (没切断时, 本轮的下一个请求会先用上新档位) —— 切断修好后, "下一个请求"就真属于下一轮, enter 时写 pending 即正确. 因此**只修切断, 先实测**; 只有实测仍晚一拍, 才动 plugin 的应用点 | 09-26 现场 | 本轮 |
| D40 | fixed | P0 | **静默失败的制度面 —— 日志写了, 但不在当前 node 上** — 人类提问: "接口定义错了、被 test mock 错了, 是不是一点错误日志都没有? system error log channel 存在吗?" 查证分三层: ① **通道存在但没挂在我身上** —— `runtime_error` channel (只 `pull`) + `RuntimeErrorLogImpl` (挂 `moss` logger 的 ERROR 级 Handler, 早于 provider bootstrap, 64 FIFO + CRITICAL 永不丢) 在 `stubs/workspace/modes/default/src/HOST/channels.py` 里有, 而活的 `.moss/modes/default/src/HOST/channels.py` 里没有 → facade 里看不到, 无法 pull. ② **那次 bug 就算挂了也一条都没有** —— 静音发生在**生成 record 之前**: 旧的 `run_tool` 裸 `except` 不记日志, 异常被译成 `result=None` 就蒸发. 实证: `grep -E "reasoning\|Thinking" moss.log.2026-09-25` → 0 命中 (那晚 8MB 日志零痕迹); 对照 CTML 层的错误是响的 (01:40 `<ghost.memento:commit>` not found 留了 9 条 ERROR + traceback). ③ **mock 那半是契约缺口, 任何日志体系都看不见** —— `FakeRunThinking.add_echoes` 不是写错, 是**比生产更宽松**, 测试里那条路径不抛异常, 所以永远没有 except 会响; 只有反射式契约测试能抓 (D39 的第 4 条修法). **人类同时纠正了 logger 的取得方式**: `get_moss_logger()` 是兜底, dolores ghost 往下**都应持有 matrix/shell 传下来的 logger, 层层传递而不是重建** —— 每层再调一次会把整棵 ghost 的日志从当前 node (`Matrix.logger` = "Logger belonging to the current node") 的语境里摘出去. 落实: `_runtime` 构 `DoloresEgo` 时**漏传 `logger=`** (旁边的 `EgoMementoManager` 传了) → 整个 ego 子树掉回兜底; 补 `logger=self.logger`; `_ego.run_thinking` → `DoloresRun(logger=)`; `_run`/`_tools` 的**module 级 `_logger` 全删**, 改 `self._logger` / `run_tool(..., logger=)`; 保留 `logger or get_moss_logger()` 作最后兜底. 同轮补齐**静默失败面**: `_handle_tool_use_event` 无人认领的 tool 名 (调用永远等不到回话, 原先零日志)、参数解析失败 (`ValidationError` 只回模型不记日志)、CTML 参数流损坏 (解释器没见到, shell 的 ERROR 不会出现)、`logos()` 退出时残留未关闭的 articulator 流 (开了边界没人关 → 不 commit 不 settle 不记日志)、`thinking/enter` 失败 (一路在抛但没一处写日志, 与 exit 不对称)、memento 切点重建失败 (只 warning 不带 traceback = 连续性丢失的原因永久丢掉)、`_commit()` 的三条 no-op 路径 (阈值说要落锚而什么都没落)、`assistant/message` 缺 usage (窗口恒 0 → K/T 阈值永不触发, 自动落锚整体静默失效, 报一次)、self-wake 无广播 (自醒整体哑掉, 报一次)、`ego/create` 未回 `thinkingToken` (症状在下一步, 原因看不见). 新增 9 条**可观测性**测试 (`TestSilentFailureSurfaces` / `TestEgoSilentFailureSurfaces`), 断言的都是"出错时日志里有没有东西", 其中一条专门断言日志落在**调用方传进来的那个 logger** 上 | 09-26 现场 | 本轮 |


> dogfood-3 追加验证通过：perStep 锁上移全局生效；prompt 顺序调整后 CTML 默认输出立现。

## 未接能力

| # | 状态 | 能力 | 依赖 |
|---|------|------|------|
| W1 | open | Memento 持久化轨迹 — 纯内存历史换 commit 轨迹持久化（重启不丢、化身分叉） | momento-mori 契约 |
| W2 | open | Ghost 反身 channel — 以 `ghost` 名注册 channel，感知/操纵自身唯一入口 | — |
| W3 | open | 独立思维模块 — 并行化身（fork）+ 关键帧自测（checkpoint self-eval） | — |
| W4 | open | 模型自感知切换 — `ghost.model` channel 暴露 current/list/switch-model/window-status | — |
| W5 | fixed | 读自身 channel facade 的两个 tool — 让 ghost 能拉取某个 channel 的当前开放面 (操作表面), 看清自己此刻能做什么. 实现走 `moss_*` tool 方案 (经 `MShellContextFacade` 读, 与既有工具同构); 不做绑定 shell 的 channel module 方案. 与 O6 (Matrix 能力声明) 同源 | `660bc9a0` (moss_channels + moss_channel_facade) |
| W6 | open | features 场脚手架 — stubs 内 `.ai_partners/features/` 未 `moss features init` 铺开, 当前只留 signpost. 治理 K6 的"未来再做", 本期不做 | — |
| W7 | open | 长程说话全程 buffer + 尾包发 signal 提示拉（礼仪，以后再说） | — |
| W8 | open | 快速响应 — ghost channel define (char, desc, template) 三元组，出现在 channel 的 named notice（变化整体重发不增量）；tool `moss_react`(char, kwargs: dict[str,str]\|None, wait_next_moment=True)：单字符执行 template.format(**kwargs)，command 不 observe，wait_next_moment=True 立刻 yielded 否则继续；模型可自定义，startup 机制默认加载 | — |

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
| O8 | open | GhostRuntime 内核不允许崩溃 — ego 坏了要有感知，运行时异常经 tui error output 打印。已部分落地: **异常有感知** (articulate error → `session.output('error')`)，但**僵死无感知** — `logos()` 无 turn 超时、`wait_until_done` 无时长上限，模型挂住不出 turn/end 时 articulate task 永久 `await anext(events)`、attention 卡死。turn 级看门狗待做 (每 run 首 event 等待套超时，超时 `abort_thinking()` + 打 error 面) |
| O9 | fixed | dolores-ego preset 工具面 keep/drop — 定稿 2026-09-13：keep agent-instructions/shell/fs/jobs/plan-mode/delegation/todo/web；drop persona(plugin shadow)/skill(MOSS 自有)/goal(loop 建立)/compaction(旁路 commit 取代)/ask-user/present。理由见 agent.cordis.yml 头注释。delegation 长期要换 per-directory 授权 + dolores clone + 代码驱动 loop(#8) |
| O10 | open | dynamic context 落点 — dsh renderContextSnapshot 是 append 非 replace；MOSS dynamic context 用自身 log replace op（仅 thinking/enter、工具调用不携带、turn/start 替换上一轮），具体落点待定 |
| O11 | open | dsh 0.1.5 传输协议重接 — mux `/api/events.mux`→`/api/remote.mux` + `server-request`→`emit/waterfall/cancel` 帧 + `$events/result` RPC + token 鉴权；事件层小修(assistant/chunk→attempt、todo/write 移出、+system/message)。launcher.py/client.py/session.py/types 重写，生命周期已解耦故边界 bounded |
| O12 | open | final answer 必须支持 ctml — ctml 反转成 `<|CTML|>` 显式围栏 + `__content__` 语音反转 + 区分 say(高级)/普通，减少 token；对抗预训练「发出请求→等返回」本能 |
| O13 | open | interleaved 反转 — 默认 wait_action_done，思维奔逸改特例，结果返回 observe，replan 拿掉 |
| O14 | open | moss_wait_action_done 拿掉 — 提示用 ctml interrupt 在飞的行动 |
| O15 | open | channel node 赋名 + 通知机制旁路大改（问题最大）；模型 ctml 出错转 bash 起 node 造成多管理面 |
| O16 | open | 重启不从 last session 还原最后帧，与 compact 分开；「总痛苦守恒」反面，清空/折叠必要；考虑 clear 函数回 memento 态 |
| O17 | open | 是否允许模型在 memento channel 自定义 commit instruction（未定论） |
| O18 | fixed | **"零有效工作的区间不 commit"** — 定论 2026-09-25：**先正常（照常 commit），未来再改，不做特殊机制**。理由：按"有没有意义"筛 span 会把判断调用塞进写入路径，与 memento 的设计前提（诚实 append-only 日志 + 读侧折叠）冲突；噪声在读面已被 `view_limit`/broken 折叠管住。若将来真要治，只能用客观量（如 `end_turn - start_turn > 2`）且只落在 exit 封尾路径 —— 稳态 commit 由 `force_tokens` 驱动，本以"真实 token 累积"为前提，再叠 turn 数守卫会与它打架 |

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
