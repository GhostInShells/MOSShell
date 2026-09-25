---
date: 2026-09-25
feature: ghost-prototype-dolores
model: claude-opus-4-7
---

# Dolores 工具面 — 第五轮回归

> 本文件在一次推倒重写中落成。**第四轮回归前**的工具面方案 (moss_ctml_append /
> moss_wait_action_done / moss_wait_next_moment / moss_react(char,args) 那一版) 已从
> 本文件删除, 历史见 `git log -- <this file>`。
>
> 本文记录的是**基于第四轮回归结论、为第五轮回归设计的交互方案**。第四轮的成果是
> `wait_observed` 机制 (mindflow 侧 primitive + dolores 侧 ctml_append 的等待语义), 它是
> 本轮工具面的地基, 不是被推翻的对象。

## 第四轮留下的地基

判定"等什么"的判据已从**时序**换成**语义**, 由 CTML 自身声明:

- **`@observe` 命令装饰器** → `CommandMeta.always_observe`。这条命令的结果预期回流进模型
  的下一步, 调用方应停下等它。
- **`wait_tasks(to_be_observed=True)`** 只等 `always_observe` 子集落地。
- **`BaseAction.set_observed_done()` / `wait_observed_done()`** 是 action 侧的 observed 节点:
  需观测 task 全部落地时置位。三种结算: 观测完成 / abort / 编译失败 (抛 `InterpretError`)。
- **`BaseArticulator.wait_observed()`** 是 articulator 侧的同一节点 (shared event)。
- **跨帧协议** (`tests/.../test_shell_append_cross_frame.py` 钉死): mindflow 恒用
  `kind='append'` + `clear_after_exit=False` —— 解释器退出时, 运行态的非 observe 命令
  **继续跑, 不被撤销**; 撤销只发生在 `kind='clear'` (replan)。

这三条决定了第五轮工具面的形状: 一个工具"等什么"由它送出的 CTML 里写了什么 `@observe`
决定; "撤不撤"由它走 append 还是 clear 决定。

## 第五轮工具面

7 个工具。三个控 CTML 生命周期, 三个是旁路/自省, 一个声明思考深度。

| tool | 参数 (默认) | 等什么 | 签发 moment | 轮次 | 流式 |
|---|---|---|---|---|---|
| `moss_interpret` | `ctml` | wait **observed** | ✅ | 继续 | **是** |
| `moss_react` | `ctml` | wait **compiled** | ❌ | cancel | **是** |
| `moss_observe` | `interrupt=false` | wait **action done**; interrupt 时先 replan | ✅ | 继续 | 否 |
| `moss_wait_next` | — | wait action done | ❌ | cancel | 否 |
| `moss_channel_facade` | `channel_path`, `recursive=true` | — | ❌ | 继续 | 否 |
| `moss_shell_status` | — | — | ❌ | 继续 | 否 |
| `moss_reasoning` | `effort` | — | ❌ | 继续 | 否 |

### `moss_interpret(ctml)`

把一段 CTML 解释进独立的 articulator (`kind='append'`), 等这条 CTML 里 **`@observe` 命令
全部落地** 才返回, 然后签发 observed moment, 退出 articulator 到下一轮。

- **不等 action done**。非 observe 命令 (说完的 say、启动的长任务) 跨帧继续跑 —— 由
  `clear_after_exit=False` 协议保证不被撤销。模型想要"这些必须跑完再返回", 用 CTML 的
  `<all>` 作用域显式声明。
- **返回 moment 签发**。返回前 `thinking.observe()` 得到最新 moment, 把它的 ref 交回。
  模型拿到 ref 就能在下一步读结果。
- **可多次调用**。一条 `moss_interpret` 一段 articulator, 帧内可连续多次。
- **命名理由**: interpret 与失败面天然配对 —— 它的错误出口就是 `InterpretError`
  (表面为 `"ctml syntax error"`)。模型看到 interpret 失败, 不需要额外解释就知道是
  CTML 没解析成命令。且对"这条是 append 还是 clear"中立, replan 能力并入 `moss_observe`
  后它不必改名。

### `moss_react(ctml)`

fire-and-await。把 CTML 送进独立 articulator, **等编译完成** 就返回, 立刻 cancel 本 turn。

- 与 `moss_interpret` 形成干净对照: interpret 是"发射并看结果" (等 observed), react 是
  "发射即走" (只等 compiled)。
- **不签发 moment** —— 它是"发出一个快去反应, 然后这条回合就结束了", 没有下一步要读的东西。
- **char 模板机制全部删除**。旧 `moss_react(char, args)` + `moss_define_reacts` +
  `ReactStore` (`_react.py`) 移除。原因: ① 旧实现不如预期; ② 它依赖两次 tool 调用、中间隔一个
  返回值, 一点也不 react。react 现在直接吃一段 `ctml`。
- **与 `moss_interpret` 共享流式解析**: 两者都只有一个 `ctml: str` 参数, 都经
  `CtmlArgumentStream` (`_ctml_stream.py`) 从 tool-call-delta 流式解码。`_handle_ctml_delta`
  的 tool-name gate 要认两个名字。

### `moss_observe(interrupt=false)`

等待并签发。

- `interrupt=false` (默认): `wait_actions_done` — 等所有跨帧跑着的命令落地, 然后签发最新
  moment。用于"捞"那些 `moss_interpret` 不等、跨帧继续跑的命令的结果。
- `interrupt=true`: 先用一个 replan articulator (`kind='clear'`) 中止当前所有行动, 再
  `wait_actions_done`, 再签发。**这是 `kind='clear'` 的撤销语义入口** —— 打断了旧的,
  再等新的落地。
- 吸收了旧 `moss_wait_action_done` 的 `replan` 能力, 不再单独保留那个工具。

### `moss_wait_next()`

等所有 action done (`thinking.wait_actions_done`, 现用法), 然后发送 `cancel=true` + tool 结果,
结束本回合。

- **不签发 moment**。它是"没有要输出的了, 结束回合等下一帧", 代替没人看的 final answer。
- 是 `moss_interpret` 的收尾对偶: interpret 送完继续, wait_next 送完就退。

### 旁路与声明工具 (不动)

- `moss_channel_facade` — **位置不动**。它做成 tool 就是要走 CTML 旁路供模型在决策时调用,
  不迁到 ghost channel。
- `moss_shell_status` — 观测 shell 状态。
- `moss_reasoning` — 声明思考档位（off/low/high/max），下一轮生效并保持, 直到 ghost 或界面再改。

### 记忆的写面: channel 而非内观 tool

`ghost.memento:commit(text__)` —— 主动落锚做成 channel 命令, 不进内观 tool 面。理由: 记忆是
ghost 的**器官**, 写面与 `read`/`chat`/`view`/`history` 同处一个 channel; 内观 tool 面只留
"决策必需"的那几个。约定: **自己写 note 就不触发旁路**(判据 = 这条 commit 有没有 Note, 与
`backfill` 共用), 留空则旁路代笔。区间上界 = 最后一个已完成 turn。

`ghost.memento:node(coord)` —— 打开某条锚点自己的节点空间 (get-or-create, 底层
`Branch.ensure_memento`) 并列一层内容, 格式抄 `core/file_editor` 的 `ls`。**一碰就建**, 所以
"节点存在"只说明"有人开过这个空间", 不说明里面有东西。磁盘活整体走 `to_thread` —— 一次
CTML 调用不该把 event loop 压在 stat 上。往里放东西仍归 ghost 自带的文件工具, channel 不新增
写协议。

`branches` 留给未来的应用。`compact` 的主动暴露同族, 未做 (顺序: 先落锚再压)。

## 流式解析是内核逻辑, 不是提速

`moss_interpret` 与 `moss_react` 是**唯二**做流式的 tool。四个理由, 每个都是"丢了就是丢了":

1. **多重通道语法**。CTML 的 `chunks__` / `ctml__` 等流式通道语法, 丢失它, 模型与 Shell
   之间最细颗粒的实时交互就没了。
2. **中途掐断**。流式解析下, 模型输出到哪个字符有错, 哪个字符就立刻中断; tool 非流式则
   模型无论如何要一次输出完整段, 无法中途终止。
3. **长 CTML 阻塞**。没有流式解析, 模型输出一段 100k 的 CTML 是完全阻塞的。
4. **partial 编译丢失**。tool use 丢掉编译阶段的 partial, 所有可提速逻辑都变慢: 极小的交互
   (say) 感知不到, 极大的交互 (macro 10k ctml) 纯负债。

## 提示词约束

- **ctml 入参必须是完整闭合的一条**。工具收到的是**一条完整的 CTML**, 不是可以边写边补的
  片段。未闭合的语法 (开了没关的标签、写一半的属性) 一律是 `ctml syntax error`。
- 流式解析发生在**模型生成这条例 ctml 的过程中** (边生成边执行), 但**工具契约要求这条
  ctml 最终完整**。流式是为了"边写边执行", 不是为了"允许半成品"。
- `interaction states` 一节按上面 7 个工具重写, 与工具名严格对齐。

## 交互状态 (prompt 层)

- **react** — `moss_react(ctml)`。一个快速反应, 发完就退, 等下一帧。真正 "react" 的路径。
- **interpret** — `moss_interpret(ctml)` → 读 moment → `moss_interpret` → …。边做边看结果。
- **observe** — `moss_observe()` 捞跨帧长命令的结果; `moss_observe(interrupt=true)` 打断重来。
- **yield** — 没有要做的了, `moss_wait_next()` 结束回合。

## 已定论 (第四→五轮之间)

- **`InterpretError` 的处理 = abort thinking + 自愈**。契约 (Action.abort_thinking): 行动不可执行
  异常要主动停止当前思考。实现 (commit `31d7698d`): abort 停当前 thinking 帧 (不释放 attention),
  并显式 mark need_observe 驱动下一帧 —— 自愈由 abort 自己拥有的显式事件驱动, 不依赖解释器
  close() 的落盘时序。连续失败阈值 N=2, 第 3 帧起不再签发自愈帧 (模型碎碎念错处仍错, 第三帧
  预期不会更好)。
- **moment_ref 只做握手, 内容走 buffer→drain 注入**。结果的 content blocks 由 MOSS 侧组装,
  经 `/tool-result` RPC 后由 plugin `agent.inject()` 注进下一个 step 的上下文 (不是覆盖当时的
  moment 槽)。`{epoch}-{index}` 的 ref 只是给模型的握手标识, 不是内容载体。

## 第五轮已实现 (代码已落地, 待真机)

7 工具面按上文契约实现完毕:

- `moss_interpret` / `moss_react` 共享 `CtmlArgumentStream` 流式解析, delta gate 认两个名字。
  interpret 等 `observed`(不是 action done) + 注入 moment; react 只等 compiled + cancel。
- `moss_observe(interrupt)` 吸收旧 replan 能力: interrupt 时空 replan (`kind='clear`) + `wait_action_done`,
  再 `refresh_metas`(await) + observe。旧 `timeout` 删除。
- `moss_wait_next` / `moss_reasoning` / `moss_shell_status` / `moss_channel_facade` 各归其位。
  旧 char 模板 react (`_react.py` / `moss_define_reacts`) 整体删除。

两处内核口径修正 (本轮):

- **turn/end 绝不冒泡 abort attention**。旧 `_note_turn_end` 调 `Thinking.abort` → 连 attention
  一起杀, 使 need_observe 驱动的回声帧循环 `while not attention.is_aborted() and need_observe()`
  永远起不来。现在 turn/end 只收线本帧走自然退出; 它不反向污染 moment —— 这轮怎么断的, tool
  调用侧已经知道, 不需要再往 moment 里塞 stop_reason。
- **思考档链路（D36 重写）**。一个字段 : impulse 的 `thinking_effort` 是唯一的思考档来源, 三态 —
  `'none'`（不思考：不驱动 turn, 只注入背景）、`''`（不表态）、`off|low|high|max`（合法档位）。
  turn 驱动与档位解析分开判定：**档位解析每帧都做, steer 才看 `'none'`**。

  | 位置 | 职责 |
  |---|---|
  | startup `default_thinking_effort` | boot 声明（空 = 不表态, 不覆盖 dsh/UI） |
  | ego | 持**待采用的声明**（startup 初值 / `moss_reasoning` 现场声明）; enter 把"帧 effort, 否则声明"解析成**单一 `effort`** 下发。**声明只在被真正采用的那一帧消费** —— `'none'` 帧或自带档位的帧不会把它吃掉 |
  | plugin | 写**一次性 pending** selection → 被 `request/header` 采纳即释放（对标官方 `dsh-api-session-controller` 的 `consume`） |
  | `request/header` | 权威：档位落在持久请求配置里, 权威链 = pending → header → `agentDefaultModel`(settings/UI) |

  **为什么必须有"释放"**：只写不释放 = 一个持久覆盖, 会永久压住 settings/UI 的选择（D36 实机：
  开机钉 `off` 压掉了人类设的 `high`, 页面改档位再也进不来）。**释放不等于档位失效** —— 它已经落在
  `request/header` 里成为 canonical 链的当前值, 下一次 assembly 依然读到。所以"写一次 + 采纳即释放"
  = 「改一次, 一直生效」, 而不是「只生效一轮」。
- 值域纪律：`deepseek` 适配器只认 `off/low/high/max`（`dsh-llm-deepseek` `serialize.d.ts`）;
  `Mindflow` 的 `ThinkingEffort` 里的 `'medium'` 无对应档位, **当前被丢弃** —— 要么从值域去掉,
  要么定显式映射, 不要让它静默消失。

## 待实机验证 (第五轮回归)

- `moss_interpret` 的 observed 等待 + moment 签发: 工具返回与下一帧 echoes 的时序在真机是否稳定。
- `moss_react` 的 cancel 极速路径, 两条流式工具共享解析后行为一致。
- 双向降级后模型不再产生"对虚空输出"的 final answer 幻觉。
- 跨帧长命令的结果能否被后续 `moss_observe` 稳定捞到 (不丢、不错帧)。
