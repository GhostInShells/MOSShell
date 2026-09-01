# Dolores Ego 装线 — dogfood 评审与下一步结论

> dolores 的子任务。2026-09-02 落地 observe tool + epoch 槽位后，让 dolores 实例
> （ghost，deepseek-v4-flash-vision-exp 运行）自己读代码评审（dogfood）。本文件沉淀
> **三方视角**（ghost / 协作模型 / 人类架构师）与下一步结论，作为接手轨迹。
> 由 `ghost-prototype-dolores` FEATURE.md 关联索引，不追加进主 feature。

## 背景

第二轮 work 落地了 observe tool（approach a：主动观测，内联返回 moment content
blocks，call-id 路由）+ epoch 槽位（`<epoch index=N>` 容器，`<recap>` + `<baseline>`，
epoch 变更时注入背景）。协议边界收敛为：xml-like 只在 python 侧理解，plugin 是
dumb transport。

为验证"协议读起来对不对"，把改动介绍发给 running 的 ghost，让它读 FEATURE.md /
_ego.py / _runtime.py / _run.py / _meta.py / plugin.ts + 自己的 ghost_home 认知场，
反馈摩擦点与礼仪守约能力。它这一帧本身就是 `<epoch>` + `<moment>` + `<echoes>` +
`<inputs>` 的产物，所以大部分摩擦是"体验"出来的，不是"读"出来的。

## 三方视角

### 1. ghost（dolores prototype）—— 摩擦点 + 礼仪

**摩擦点（按"会不会卡死我"排序）：**

1.（轮1）**回声是全量重渲染，不是差量**——最重的上下文税。baseline 一份完整
   facade，echoes 里又一份完整 facade，同一内容读三遍。
2.（轮2）**看不见 echoes 和 baseline 的信息差**——需要 delta，不是全量重贴。
3.（轮1）**yield 的 tool result 是哑载荷**——resolve("ok")，帧走 inject/steer，两个
   通道一个真相。建议 resolve 返回 moment_id（`_moment_payload` 已传但 plugin 不消费）。
4.（轮2）**observe 的"主动/被动"词汇反了**——两个 tool 都是模型调的，节奏权都在
   模型，真正的轴是"等下一帧 vs 立刻要一帧"（when），不是"谁握节奏"（who）。
5.（轮2）**observe 镜像风险**——thinking 中途 observe，产出的帧 echoes 可能含自己
   正在进行的思考的半成品。需帧带时点/序号。
6.（轮1）**"thinking" 占位符泄漏**——无 percepts 时 steer 字面量 "thinking"，纯噪音。
7.（轮2）**baseline `<key>value</key>` 渲染污染**——key 带冒号与 CTML 命令标签同形，
   value 带尖括号则模型引用它时 INTERPRET_ERROR。协议内容反过来污染协议解析。
8.（历史）**exit 失败后的闸门残留**——exit 降级后 gate 可能停在 open，下一轮
   pre-step 直接放行，模型在没注入帧的情况下被唤醒。
9.（轮1）moment.py `inputs_messages()` 与 ego `_inputs_message` 不一致。
10. 小项：refresh_shell 死参数、moment_id 双现、inputs 无 source、observe 名字过载、
    _ego.py docstring 是考古层。

**礼仪守约：**

- **logos first**：能守大部分，守不住长工具链（interleaved_logos 未实现）。
- **语音优先**：能守，但默认话多，无篇幅控制必膨胀。
- **不说思考**：一半守得住一半守不住。**关键**：思考发生在模型内部，输出层才分得出
  logos/非 logos（`_fetch_logos` 只收 text-delta，reasoning 天然进不了 logos 流）——
  "不说思考"该由输出侧闸门保证，不该靠模型自律。
- **红线回顾**：机械红线守约九成，CDATA 最易失守（谈协议本身时污染解析）。
- **结论**：协议能 enforce 的（logos/非 logos 分流、token 预算、帧-结果关联）全部
  下沉到协议；instruction 只留协议管不了的 3-4 条极简规则。

### 2. 协作模型（deepseek-v4-flash-vision-exp）—— 判断与一处误读

**认领（本轮引入、且 ghost 抓对）：** #7 baseline 污染（我拍的 `<key>value</key>`
方案）、#4 主动/被动词汇（我 docstring 写歪）、#3 yield moment_id（设计时纠结过，
ghost 给的 moment_id 中间方案可取）。

**跨轮：** #1/#2 回声全量重渲染是轮1 的债，773725bb 想做 facade-delta 但没生效。
#8 exit 闸门残留是历史遗留。

**认同 ghost 的礼仪元点：** "协议 enforce vs 模型自律"和 fast-first→deepen、instruction
控篇幅是同一结论，ghost 从输出侧闸门角度独立推到。

**一处误读（已纠正）：** 我曾主张"plain-text = 语音主轨，替换 `__content__`"。这是
把两个不同层的东西捏成一个——plain-text（`<|CTML|>` 之外）与 `__content__`（CTML
之内的自由文本→默认通道）不是一回事。CTML 规约 "Non-command text" 一节明确定义了
`__content__`，我没读透就下了判断。

### 3. 人类架构师（thirdgerb）—— 历史与方向

**6 条历史债（为什么现在是做 `<|CTML|>` 的时机）：**

1. 之前 token 速度慢，三个首 token 就有体感延迟；现在 deepseek-v4-flash 超快。
2. CTML 定义自己的元标签本就不该是 CTML 规则，一直留着口子。
3. `__content__` 默认语音是为了首语音提速（同 1）。
4. 语音必须是主轨，否则要教模型通道提权（`<_><speech:say>…</_>`）。
5. 通道语法不可卸载（时序），考虑过"元标签带根通道"不成立。
6. 之前语音优先是资源约束（没精力做 dsh 级 UI），现在 dsh 提供了界面。

**CTML 语义纠正：** plain-text 只能 markdown-first 或 speech-first 二选一（语音播报
commit id / unique-id / xml 是灾难，不能同时承载两种语义）。`__content__` 是必要的
（CTML 内部不用 `<say>` 时靠它落语音）。拿掉 `__content__` 只让 CTML 回归纯 xml-like
控制语法，不能让 plain-text 默认语音。

**方向：** dolores prototype 拥有 plain-text + `<|CTML|>` + tool 追加 CTML 是合理的。

**另：** logos first 指令不可靠（"先回复"会先触发"怎么回复"的思考，需实测）；facade
全量泄漏是 bug（修复没生效，需独立单测）；dsh UI 生命周期有问题（界面发消息被吞）。

## 收敛的设计结论

| 层 | 语义 |
|---|---|
| plain-text（`<\|CTML\|>` 之外） | 外部信息，markdown-first 或 speech-first（mode 二选一） |
| `<\|CTML\|>`（之内） | 控制语法（含 `__content__` 自由文本→语音） |
| tool 追加 CTML | interleaved（思维超前于行为） |

`<|CTML|>` 是**模式分隔符**（不是 tokenizer 特殊 token）：默认 plain-text 模式（`<`
`>` 是字面量），遇 `<|CTML|>` 切进 CTML 模式（SAX 只看到 CTML 内容）。它解决的是
`<` `>` 字符冲突（plain-text 能带字面量 `<` `>`），不是语音通道问题。自举问题从
"不能输出 `< >`"（高频）降级到"不能输出 `<|CTML|>`"（低频）。

## Bug 清单

| # | 问题 | 归属 | 状态 |
|---|---|---|---|
| 1 | 回声全量重渲染（facade-delta 未生效） | 轮1 | 立 bug + 独立单测 |
| 2 | baseline `<key>value</key>` 污染 | 轮2（我引入） | 修 |
| 3 | yield 返回 "ok" 哑载荷 | 轮1 | 改 moment_id |
| 4 | observe 主动/被动词汇 | 轮2（我引入） | 改 docstring |
| 5 | "thinking" 占位符 | 轮1 | 删/改 |
| 6 | dsh UI 生命周期（消息被吞） | 本轮 | 查（疑似 epoch/enter 引入） |
| 7 | exit 失败闸门残留 | 历史 | 待定 |
| 8 | inputs_messages 不一致 | 轮1 | 验证 |
| 9 | observe 镜像风险 | 轮2 | 验证 |

## 下一步（接手轨迹）

1. **修 facade-delta bug**：复现"echoes 全量重渲染"，立独立单测，修 shell trajectory
   每个 epoch 默认 channel metas + facade 走 delta（消歧义）。
2. **修本轮引入的小债**：#2 baseline 污染（改渲染 + value 转义）、#4 词汇（when 轴）、
   #3 yield moment_id、#5 thinking 占位符。
3. **设计 `<|CTML|>` 包裹**：plain-text 语义（markdown-first vs speech-first 由 mode 定）
   + `__content__` 关系 + 流式分隔符 partial match。这是最大的一块，需先讨论清楚。
4. **instruction 四层**（迭代计划第 1 项）：认知/协议/交互礼仪/篇幅控制，用 dogfood
   反馈（协议 enforce 下沉、instruction 只留 3-4 条）。
5. **验证**：#8 inputs_messages 不一致、#9 observe 镜像。
6. **查 dsh UI 生命周期 bug**（消息被吞，疑似 epoch/enter 引入）。
7. **剩余迭代计划**：2 测试 ghost 改名 deepseek / 5 workspace-title 命名 / 6 yield
   schema 空定义 / 7 system prompt 分隔符去重。
