---
summary: "memory 折叠、buffer 不睡、跑着也能送一帧；押推送只押 react；重启后的十一条回归点"
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
