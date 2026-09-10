# Dolores TODO

> dolores 的问题清单 —— **单一事实源**。dogfood 只负责发现与记录，状态在此维护。
> 状态: `open`(待修) / `uncertain`(不确定) / `fixed`(已修, 带 commit) / `verified`(下轮 dogfood 验证) / `invalid`(判定非 bug)。
> 由 `ghost-prototype-dolores` FEATURE.md 关联索引。dogfood 发现新问题在此登记，修复/验证在此改状态。

> **状态快照 (2026-09-07 更新)**：D1/D5/D7/D21/D23/D26 → `fixed` 待回归（见归口 commit）；D24 → `open`(检查未启动)；D9/D17/D18 → `invalid`。下一轮 dogfood 优先跑 D1/D5/D7/D21/D23/D26 回归。

> **2026-09-09 收尾判定**：D2 → `fixed`(human-confirmed)。D3 → 降 `P3`（除非用 channel 否则无法试开启）。D27 → `invalid`（方案已换，零件事）。D20 对齐修改已落地；D19 协议纪律补充已落地（markdown 与 CTML 互补不复述）。

> **2026-09-08 落地**：mindflow interleaved — incomplete impulse(首包)也送入思考单元, effort='none' 只观测不行动, complete 尾包折进响应帧再行动（「首包抢占注意力但不行动」）；attention 拆分 `draw_from`(冻结创建者)/`impulse`(活量) 并修吸收/衰减/挑战四处错位。

> **2026-09-11 拆分**：D22 拆两半——图片协议半边 → `fixed`（字节头嗅探，3080 实机验证）；moment dynamic context 丢失半边 → 新 D28 `open`（未定位，含与 D22 同源的假说）。

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
| D20 | open | P1 | fetch wait_actions_done 三处 — 默认 True / 工具描述已改「Wait for already-emitted...」(与默认一致, 不再"Fetch now") / prompt 仍「optionally waiting」，与 default=True 轻微张力待统一 | dogfood-3 | — |
| D21 | fixed | P2 | dsh 侧先停 + 界面无中断 — teardown 时序已修(interpreter __aexit__ 清 clear_after_exit + mindflow 改 wait_compiled，待回归)；**界面无中断能力/双向同步未启动调研** | dogfood-3 | `4fc8b0c5` |
| D22 | fixed | P1 | 图片协议传输 — MOSS 图片消息 → dsh 图片消息: media_type 按扩展名猜(JPEG 存成 .png → 声明 image/png)，dsh attachment admission 拒 `Declared image type does not match its bytes`。已改字节头嗅探；3080 实机验证 accepted 且模型真读到图 | dogfood-3 | `message/contents/images.py` from_file 字节嗅探 (待 commit) |
| D23 | fixed | P2 | shell trajectory 验证方式 — help(notice)+interface 各自独立判断 delta 已落地，不再每次一起传 | dogfood-3 | `36dcaefd` |
| D24 | open | P0 | interpreter error 被 wrap 成 command error — is_notifiable(≥300) 语义已铺垫(`ceb9eef7`)；区别于 command error + 关闭方式未定。**检查未启动** | dogfood-3 | `ceb9eef7`(铺垫) |
| D25 | open | P0 | observe=True 未生成下一帧 thinking — 反而要界面驱动，这是 bug | dogfood-3 | — |
| D26 | fixed | P0 | tui 遇 interpreter error 崩溃 — exeception 处理加固(print+continue)+runtime stop 非零退出，不再静默崩溃(待回归) | dogfood-3 | `261adbbd` |
| D27 | invalid | P1 | perStep reject 界面提示 — 调研路径搞错，可在 reject 处发 stream/error 类事件给界面提示。方案已换，零件事 | dogfood-3 | — |
| D28 | open | P1 | moment dynamic context 丢失 — 看 moment 疑似彻底丢了 dynamic context。未定位；**假说**：与 D22 同源——moment 带图且媒体类型错时 `durableMomentContent` 在 `thinking/enter` 中抛错 → 整个 enter 返 400 → context/inputs/epoch 全未注入。D22 修复可能一并解决；若 dynamic context 不含图则属另一机制，待下轮 dogfood 复现 | dogfood-3（D22 拆分） | — |

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
