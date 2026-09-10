# DshSession 单轮对话接口 (run) 与标准 cancel

> 2026-09-11。承接 2026-09-10 ref span/seed/run 的收尾: dsh-fusion 的「最后一步」——
> 让 `DshSession` 具备一个**官方 API 标准的单轮对话接口** + 一个**标准 cancel 接口**,
> 这是「可以 loop 一个 session」的关键。落点 `src/ghoshell_moss/deepseek_harness/session.py`。

## 「官方 api 标准」的落点: 官方 Python SDK 的 `Session.run`

原先"官方 api 标准"有歧义 (ACP `session/prompt`? apiproxy `session.prompt`?)。源码锚定后
确定 = **官方 dsh Python SDK 的 `Session.run(input) -> RunResult`**
(`research/source/deepseek-harness/python/sdk/src/deepseek_harness/api.py`)。它的契约:

```
session_prompt(sessionId, content) -> messageId          # fire-and-return
  → 等 agent/inbox/spliced 回执 (含 messageId, 确认入队)
  → 收集该 session 的 session.event, 直到 session.status == idle
  → RunResult(session_id, final_response, finish_reason, events, notifications, session_root)
```

- `final_response` = 区间内**最后一条 `assistant/message`** 的 text 块拼接 (`final_response()`)。
- `finish_reason` = 区间内**最后一条 `turn/end`** 的 `data.reason.kind` (`finish_reason()`)。

**SDK 没有 cancel** (client 无 `session_cancel`)。"标准 cancel" = dsh 的 `session.cancel`
(apiproxy, keepInbox), DshSession 早已薄封装 (`cancel()`),此处保持,只补文档。

## 分层 (对齐 SDK 的 client.session_prompt / Session.run 两层)

| DshSession 方法 | 对应 SDK | 语义 |
|---|---|---|
| `prompt()` / `cancel()` | `client.session_prompt` / `session.cancel` | 原生动词薄透传, fire-and-return (返回 accepted), 不阻塞 turn |
| `run()` | `Session.run` | **组合出的阻塞单轮**: 发 prompt → 等本轮 turn/end → 返回 `DshRunResult` |

`DshRunResult` 镜像 SDK `RunResult`, 去掉传输专属的 `notifications` / `session_root`,
保留 `(session_id, final_response, finish_reason, events)`。`str` 入参按 SDK `normalize_input`
包成单个 text 块。

## MOSS web 侧的三处适配 (与 SDK 的可见契约一致, 传输不同)

1. **回执门控**: SDK 靠 `agent/inbox/spliced` 里的 messageId 确认入队; apiproxy `session.prompt`
   **不回 messageId**。改以**本轮第一个 `turn/start`** 为起点, 直接锚定 turn 号 (比回执更强)。
2. **结算边界**: SDK 停在 whole-agent **idle**; MOSS 停在**本轮 `turn/end`**。理由: 需要"只跑一轮",
   不被队尾排队工作拖住; 且 live 实测 `host/session-status` 的 running 镜像在 turn 在飞时会读成
   False (见下), idle 不可作单轮边界。
3. **并发门控**: 同一 session 同时只允许一个 `run()` 在飞 (对齐 ACP "one in-flight request per
   session"), 重入抛 `RuntimeError`。

## 真实验证 (2026-09-11, dsh web @ http://127.0.0.1:3080)

`DshConnection(host=127.0.0.1, port=3080)` + `create_session` 直连活 dsh:

- **正路**: `run("Reply with exactly the two words: pong test")` →
  `finish_reason="completed"`, `final_response="pong test"`, events 从 `turn/start` 到 `turn/end`
  共 20 条 (含 `agent/inbox/spliced` / `step/*` / `request/*` / `assistant/chunk`)。
- **cancel**: 多步 turn (list→read→summarize) 跑到一半 `cancel()` (accepted=True) → `run()`
  以 `turn/end{reason.kind: "aborted"}` 结算。dsh 原生 reason 直接透传 (ACP codec 另把它映射成
  end_turn, `cancelled` 由 ACP 桥 out-of-band 标记 — MOSS 层面保 dsh 原生即可)。
- **观察 (非本步 scope)**: cancel 前 `session.running` 读到 **False**, 而 turn 实际仍在飞
  (task 未 done, events 还在 `step/start`)。说明 web mux 的 `host/session-status` running 镜像
  在本场景下不可靠 —— 这佐证了 run() 用 turn/end 而非 idle 作边界是对的。若将来 `when_running/
  when_idle` 要被外部依赖, 需单独排查该镜像。

## 落点

- `src/ghoshell_moss/deepseek_harness/session.py`: `DshRunResult` + `_normalize_content` /
  `_final_response` + `DshSession.run()`; `cancel()` 补对称文档。
- `tests/ghoshell_moss/deepseek_harness/test_session.py`: run 单轮收集 / 第二轮隔离 /
  cancel 结算 / 并发拒绝 / cancel 返回值。
