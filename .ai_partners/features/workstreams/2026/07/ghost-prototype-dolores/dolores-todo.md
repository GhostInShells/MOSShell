# Dolores TODO

> dolores 的问题清单 —— **单一事实源**。dogfood 只负责发现与记录，状态在此维护。
> 状态: `open`(待修) / `uncertain`(不确定) / `fixed`(已修, 带 commit) / `verified`(下轮 dogfood 验证) / `invalid`(判定非 bug)。
> 由 `ghost-prototype-dolores` FEATURE.md 关联索引。dogfood 发现新问题在此登记，修复/验证在此改状态。

## 缺陷

| # | 状态 | Pri | 问题 | 发现 | 归口 |
|---|------|-----|------|------|------|
| D1 | open | P0 | 反馈回路缺失 — speech 无返回提示 + 命令无 `<result>` 回执，行动无后果 → 本能训练被杀死 | dogfood-2 | — |
| D2 | uncertain | P0 | 帧纪律 — enter moment 未入 session / 奇数帧丢、回复后 flush。debug 发现帧在历史轨迹、界面未渲染，疑似展示层而非 tracer 丢帧 | dogfood-2 | — |
| D3 | open | P1 | 模式默认 — 按会话种类设默认（实时→CTML / 阅读→文本）+ 双通道原语 + 不对称成本 | dogfood-2 | — |
| D4 | fixed | P2 | effort 机制 — 自救工具(think) + effort 映射 + 文档化降级(Reasoning Effort 段)已落地 | dogfood-2 | `4fda96a0` |
| D5 | open | P0 | TUI 生命周期 — 首轮能发消息、第二轮起不能（round1 后新回归） | dogfood-2 | — |
| D6 | fixed | P0 | 沙箱 cwd 错位 — DSH cwd = ghost home 而非 project 根，ghost 无法读写仓库、无自迭代能力 | dogfood-2 | project_home → project root |
| D7 | open | P0 | dsh 提示词打架 — DSH 系统提示与 MOSS 元指令五处冲突（工作区/身份/输出机制/perStep/输入来源），元指令缺「裁决级」取舍 | dogfood-2 | — |
| D8 | fixed | — | 回声全量重渲染（facade-delta 未生效） | ego-wiring | `2e57a8f8` |
| D9 | invalid | — | baseline `<key>value</key>` 渲染污染 — 记录错误，key 作 tag 判定正确，无需改 | ego-wiring | — |
| D10 | fixed | — | yield 返回 "ok" 哑载荷 | ego-wiring | `59f13736`+`ab6aaac1` |
| D11 | fixed | — | fetch/wait 词汇（when vs who） | ego-wiring | `ab6aaac1` |
| D12 | fixed | — | "thinking" 占位符泄漏 | ego-wiring | — |
| D13 | fixed | — | dsh UI 生命周期（消息被吞） | ego-wiring | `ea90993a` |
| D14 | fixed | — | exit 失败闸门残留 | ego-wiring | `ea90993a` |
| D15 | fixed | — | inputs_messages 不一致（executing 归 context） | ego-wiring | — |
| D16 | fixed | — | observe 镜像风险（moment index 帧带序号） | ego-wiring | `59f13736` |
| D17 | uncertain | P1 | 语言不匹配 — 中文输入，ghost 全文英文回答，markdown 内反而中文（机制不明，待问 ghost） | dogfood-3 | — |
| D18 | open | P1 | markdown 内自指重新发声 — `<|Markdown|>` 内编号项重新触发语音，疑似自指 | dogfood-3 | — |
| D19 | open | P2 | 长篇大论 — 缺「简洁/少即是多」规则（旧 persona/behaviors 有，重写丢失） | dogfood-3 | — |
| D20 | open | P1 | fetch wait_actions_done 三处不齐 — 默认 True / 工具描述「Fetch now」/ prompt「optionally waiting」 | dogfood-3 | — |
| D21 | open | P2 | dsh 侧先停 + 界面无中断 — final result 后 dsh 比 moss 先停，dsh UI 无中断能力（双向同步有鬼主意） | dogfood-3 | — |

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
