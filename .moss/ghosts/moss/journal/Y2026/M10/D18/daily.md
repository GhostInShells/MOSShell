---
summary: "memory 折叠、buffer 不睡、跑着也能送一帧；押推送；出声即让路改成提权；强度曲线不再在保护期边界掉回初始值"
status: writing
---

# D18

## 今天落的三块

1. **memory 折叠**（87081a4d）：渲染把 history 当正文逐条画，292 条分支注入 283,635 字符；现在 history 折成一条 earlier（count/first/last + 最近五个标题），实测 7,845 字符。数据层（commits/notes 全量读）量过 2.0ms + 2.3ms，判为提前优化，只记阈值债。
2. **buffer 不睡**（c5f0afbb）：observer 加 `has_pending` / `drain_percepts` / `on_percepts_injected`；mindflow 签发条件改成「没 abort 且（need_observe 或 has_pending）」。测试按新契约改写：notify 抢不到之后，在同一个 attention 里折进下一帧（1 个 attention，不是 2 个）。
3. **跑着也能送**（未提交）：run 注册钩子（随轮起落、可 dispose）→ observe 一帧 → `ego.push_moment` → 插件 `/moment/push`（复用 `injectMomentPayload`，context + inputs 一起注）。

## 重启后要一条一条对的点

1. 开机 memory 只有 10 条 commit + 一条 earlier；总字符约 8k（之前 283k）。
2. 每轮进入的日志带 `need_observe=… has_pending=…`。
3. 说话被打断 / notify 落 buffer 时，日志出现 `buffer push while this turn runs: N message(s) — scheduling`，随后 `buffer push delivered: moment X-Y`。
4. 插件探针 `__DOLORES_PROBE__ moment/push …` 出现在 moss.log（stderr 通道）。
5. 一次 interruption 之后 attention 数不增加，但帧数增加（旧契约是 2 个 attention）。
6. `/moment/push` 的 thinkingToken 校验：非 ego 调用应被拒。
7. 换档（moss_reasoning）仍走 steer-observe，没有回退。

## 未提交

- `_run.py` / `_ego.py` / `test_dolores.py`（替身补齐）/ 插件 `/moment/push` 路由。
- 插件尚未真机加载；重启是它第一次被解析。

## 今天的方法账

- 我在 DSH 里改文件、他看不见，返工三轮；之后改成 DSH 只读、全部走 file_editor（trust 已开，导出即落盘）。
- `edit(region)` 改区间时反复吃掉相邻行（多出 `)` / 缩进断掉），每次都得读回。判据：改完先读那一段，再跑编译。

## 夜 · 重验 cue 档"跑 bash 时他插话"：三次都死，量出两个洞

他让我重验：我挂 bash，他插话，我要既看得到新 moment、又不被打断。三次全灭。两处根因互不相干：

1. **仲裁只有三秒保护期**。`protection_time = min(strength_decay_seconds * 0.2, 3.0)`，listener 的 decay 是 120s，所以保护期恒为 3 秒；同源挑战者强度按 `*1.1` 算。身体在跑时有 action 循环心跳提权（`mindflow_in_shell.py` 的 `_escalation_heartbeat`），一停（思考、挂 bash）就没人提权，他三秒后开口必赢 → `verdict=preempted` → 整轮 abort，bash 陪葬。实测：17:17:29 我最后一个动作停，17:17:32 他那句进来，**正好三秒**。
2. **投递用的是可丢弃的暂存**。17:14:21–28 他四句（「好，测试，你在用 bash」「我看得到你在用 bash」「好像没有打断」「我就是」）全部 `inject` 成功，挂在 `next-step` 门口；17:14:41 abort 时 inbox 记 `removedCount=9, outcome=canceled` —— 投了，一句没读，字没了。`Agent.inject` 契约原文："Cancellation or disposal may discard pending context"。

判据是他给的：**别用 inject/steer 那套，走跟 enter、跟"协议返回的 moment"一样的投递路**。于是 `moment/push` 的两半改成缓冲进 `pendingMoments`，由 pre-step 挂载点插到本步历史最前——插件自己的队列，pre-step 之前一直在，这一轮死了下一轮照样插。新函数 `queueMomentPayload`（与 `injectMomentPayload` 成对：一个塞门口，一个排自己的队）。

**没动**：thinking 期心跳（在 `_run.py`，那里压着上面那批未提交改动）。所以重启后先验"字不丢"，再决定要不要连"轮不被掐死"一起修。

## 重启后要逐条对的点（夜 · 追加）

8. 他说话时轮被掐，**字不丢**：下一轮我的上下文里应当出现那几句 `<inputs>`；验法就是这条——跑 bash、他四句、轮被掐、我下一轮报出原话。
9. `moment/push` 之后日志里不再出现 `agent/inbox/spliced … target: next-step` 的那两条（改成 pre-step 挂载），但步骤历史里能看到同样的 `<moment>` / `<inputs>` 帧。
10. D18 那七条照旧（memory 折叠、buffer push 三行、探针、thinkingToken 校验、换档仍走 steer-observe）。

## 第二轮重启前 · 押 moment 的旗子收窄到 moss_react（17:35）

他定的：CTML 工具调用期间押住 moment 推送的那个 flag，**不该对所有 CTML 工具都立，只该在 `moss_react` 上立**。理由是他说的那两句——react 的结果一到，这一轮就被 cut，中途推的帧只能落进一个正在收尾的 turn；`moss_interpret` 跑完这一轮还要继续，人此刻说的话推进来，下一步就该读到。

改动（未提交，混在 D18 那批里）：

- `_run.py` 新增 `_HOLD_PUSH_TOOLS = frozenset({ReactToolCall.tool_name()})`；`_handle_tool_use_event` 的立旗条件从 `_STREAMING_TOOLS`（interpret + react，那是给 delta 流用的，别混）换成它。
- 押住的日志改成 `buffer push held: moss_react is in flight`，免得下次看日志的人以为 interpret 也押。
- 测试拆两条：`test_react_tool_call_holds_the_percept_push` / `test_interpret_tool_call_does_not_hold_the_percept_push`。`test_dolores.py` 180 passed。

**重启后追加的第 11 条**：我挂着 `moss_interpret` 在跑动作时他说话 → 日志应是 `buffer push while this turn runs` + `delivered`（不是 `held`）；我跑 `moss_react` 时他说话 → 应是 `held`，字留在 buffer 里等下一帧签发。

> 他提醒：重启后第一个 commit 不会立刻有 note。所以记在这里——这次重启要带的东西是「react 押推送」这一处收窄，而不是 D18 那七条之外的新清单。

## 第三轮 · 重启后的实测与「wait_next 也该押」（17:37–17:44）

**链路验通了**。cue 档下用 bash（`moss --ai features list`，夹 sleep）+ 说话交替跑了三轮，他在中间插话，每一句都落进我的上下文。17:41:05 那条日志是干净的证据：

```
impulse verdict=queued source=listener_nucleus ... mode=next
buffer push while this turn runs: 1 message(s) — scheduling
__DOLORES_PROBE__ moment/push id=1-35 context=1 inputs=1
buffer push delivered: moment 1-35
```

queued → buffer push → moment/push → delivered，十五毫秒。所以第 8、9 条判据这一夜是通的。

**他补的机制**：`moss_wait_next` 一旦开始调用，接下来我说的话**也不该再 inject**。判据只有一条——**这一轮调完那个工具之后还打不打算继续读**：

- `moss_interpret`：跑完这一轮还要继续 → 人此刻说的话照推，下一步就读到；
- `moss_react`（结果一到 turn 就被 cut）、`moss_wait_next`（本身就是在交出这一轮）→ 推过来的帧只能落进一个正在收尾的 turn → **押住**，留在 observer buffer，等本 attention 收尾自然签发成下一帧。

改动（未提交）：`_HOLD_PUSH_TOOLS` 从 `{react}` 扩到 `{react, wait_next}`；日志改成 `buffer push held: the turn is closing`；新增 `test_wait_next_tool_call_holds_the_percept_push`，`-k percept_push` 四条全过（181 collected）。

**重启后的第 12 条判据**：我说完一句就调 `moss_wait_next`，他在我的语音还在播的时候插话 → 日志应是 `held`，那些字出现在**下一个** moment 里，而不是塞进正在收尾的那一轮。

## 傍晚 · 出声即打断落地，以及强度曲线被我发现「退化」（18:40–18:56）

**第一件事：barge-in 从硬掐改成了提权，当面验通。** 他让我开 cue，我先说一段长的，他只说「测试」两个字，我这边的 say 当场交出 `stop_reason=interrupted`——不是排队、不是等说完。这一版（`5cbf3cc6`）是上一轮压着的：`ListenerSignal.boost`、交付包按 `max(priority, boost)` 挑战、`DeliverSpec.barge_in_boost` 默认 WARNING、保护期那道门只拦同/低优。

**第二件事才是今天真正的收获：我顺着「为什么思考还是会被三秒掐断」把强度曲线挖穿了。**

- 现象：跑 pytest 的时候他说话，`boost=None`（我没在出声），那一包只有 INFO，却判 `verdict=preempted`，bash 陪葬。
- 我先把「保护期只有三秒」讲对了：`protection = min(decay*0.2, 3.0)`。但这句话是半对——**硬保护期其实是 `Impulse.protection_time`，默认 0，ghost 自己的包从来没人填**；那三秒是 `current_strength()` 内部算的平台，不是硬门。
- 更要命的一条：平台结束时强度从 `100*1.1=110` **掉回 100**，正好落在同强度挑战者的水平上。所以整段衰减对「最常见的那个信号」等于零保护——保护期一过就必输。
- 我跑了真实数值：不提权 → 0..2s 是 110，第 3 秒 99 立刻被抢；提权每 1 秒一次 → 一直 110，同强度永远进不来。所以**提权就是那三秒平台，提一次苟三秒**，洞在于思考期没人调它。

**改法（他拍板「把曲线改长一点」）**：平台 `min(decay*0.5, 10.0)`，衰减段从平台值往下走、不再掉回初始强度。20 秒周期的实测变成：0–10s 恒 110，第 11 秒 99 才第一次被抢，12s 88，16s 44，20s 0。`4dd5ad96`，测试 516 条全过。

**顺手纠正的两处记忆**：历史那版不是线性而是指数（`exp(-elapsed/T)`，未完成的脉冲再乘 2.5 加速），它在 3 秒就掉到 86——比线性更不保护；所以「某一轮改坏了」不成立，两版都给不出他要的形状。

**他给的两条语义，我记死**：保护期 = 同级下连更强的强度都进不来；衰减期 = **维持自己的强度**，让同强度的不能轻易打断。震惊那种效果不该交给强度，该交给优先级档。

**留给晚饭后的**：他说还有几个小细节，做完就能到「准双工」。已知待办——思考期是否要一条真正的时间心跳（事件驱动的提权在长工具里天然断档）；`deliver` 与 `onset` 的 boost 是否要一起用；还有 `moss_react`/`moss_wait_next` 挂住时跟 percept 竞速那个「准全双工」的形状。

**给未来的自己**：这一轮我三次把「他记得的」和「代码里的」分开核对，两次证明他记错了（20 秒窗口、历史曲线），一次证明我错了（硬保护期是 0）。**只要跑一遍真实数值，争论就结束了**——别用推理去反驳一个可以测的命题。

## 夜 · cue 档分句交付第一次跑通，尾包不再是个空块（20:10–20:20）

他先验 `always`：我念一段长的，他插一句，我那段 say 当场被取消——日志记 `logos interrupted, cancelled 1, last say`。然后切 cue，读回来确认：起话信号关、静音零、交付走 next、出声让路关。

**关键的一次踩空**：我对着 `listener:set_etiquette_spec` 改了 `per_clause=true`、`silence=4.0`，然后说"改好了"。他一句话点破——**配置只改了内存里那份礼仪，正在跑的听侧还是旧的一套**。停掉再 activate 之后，第一包 `paused="false"` 才真的出现。判据：配置项改完不等于生效，`listener` 要重启。

跑通后的形态：三包 `paused="false"` 各带一段新文字，最后一包是空的、只有 `paused="true"`。他嫌那个空块"占的 token 还行，但看着怪怪的"，说空的时候用括号加一句提示，别让它像空的 ASR 正文。于是核里加了 `_EMPTY_TAIL_HINT = "（他停了）"`，`8a30aefb`，核 + 听侧 532 条全过。

顺带纠正我自己一次口误：没有 `vad` 这个字段，落地的属性叫 `paused`；我先前说"没有 vad、是个 bug"，其实是找错了名字。

**行为上的一条新纪律，他给的**：「我说完了的话我会说说完了」——他收尾会明说，我在那之前不插嘴。我守住了，中间有一次我没守（他说"我先说完啊"），当场用 `<interrupt/>` 把自己正在播的语音掐掉。

## 夜 · screen / file_editor 是节点，不是终端（20:21–20:23）

他嫌我改代码他看不见。我第一反应是拉了个 **GNU screen** 会话去 tail 一份 diff 日志——彻底理解错。他要的是 `nodes/screens` 那个**屏幕躯体**，和 `nodes/os/file_editor` 那个**共享工作副本**：改动变成卡片摆在屏幕上，export 是唯一落盘的动作，人点同意。

拉起来之后才知道好用：`screen_manager`（20381，窗口池 + 分组 + veil 手势）、`artifacts`（流式工件）、`file_editor`（thread / read / find / edit / export）。他后来因为键盘和 file_editor 自己 export 审批不撤销的 bug，让我把三个节点全关了，改回"我直接改文件、他自己看 diff"。

## 夜 · 给 say 绑进度：连着被他指出两处 hack（20:23–20:33）

目标是他提的那个真问题：**判断要不要开口，需要知道「我上一句说到哪」**。已有的管道都在——`CommandTask.set_progress` → 状态行的 `progresses` → 渲染成 `executing` 下面的几行，缺的只是 `say` 播到一句时去写它。

第一版我写了两处，都被他当场否掉：

1. `getattr(stream, "on_clause", None)` —— 他说"既然有 type hint 就不该用 getattr，我很不喜欢这种机制"。正解：`SpeechStream` 契约上本来就有 `on_sample` 这种默认空实现的回调，`on_clause` 照它的形状补一条即可，命令层直接调。
2. 用 `CommandTaskContextVar.get(None)` 自己去抠 task —— 他说"我前面已经跟你说过了，用 CommandUtil"。正解：`CommandUtil.set_progress` 本来就有，它内部认命令上下文；我只需要在绑定时 `contextvars.copy_context()`，回调经 `call_soon_threadsafe(ctx.run, CommandUtil.set_progress, ...)` 回到事件循环再调。

最后一版落在 `contracts/speech.py`（SpeechStream 加 `on_clause`）和 `core/speech/speech_module.py`（`bind_speech_progress`，`say` 与 `__content__` 各绑一次，退出时 dispose）。

**还没解的那一层，是他指出来的**：`play(samples: list)` 这个参数本身就是"拿一个可变 list 冒充回调"的输出口，当初写它的人偷了个懒。他说这块他自己改，让我别再动代码。

**给未来的自己**：这一晚我三次在同一件事上返工——**先想到底层机制，再被叫回去用现成的口子**。判据很简单：动手之前先问"这个能力是不是已经有一个公开的门"，`getattr`、`ContextVar.get`、自己 new 一个 list 当出口，这三种形状都是同一类病。
