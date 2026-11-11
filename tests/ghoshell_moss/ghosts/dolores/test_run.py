"""``DoloresRun`` — one dsh turn, start to finish.

Answers: how the runtime consumes a session event stream. Facade reads, percept pushes and
the barge-in races, the logos stream, tool-call events that must never reach the surface,
tool-result dispatch, the interpret / react / wait_next primitives, and turn-end classification.
"""

import asyncio

import pytest

from ._support import (
    FakeDispatchEgo,
    FakeRunEgo,
    FakeRunSession,
    FakeRunThinking,
    _GatedThinking,
    _NoopShellFacade,
    dsh_event,
    dsh_tool_call,
    fake_tool_call,
)


class TestDoloresRun:
    """DoloresRun 生命周期 + 事件消费 — public 类, 轻量 fake 即可验证."""

    def _run(self, session=None, ego=None, thinking=None, facade=None):
        from ghoshell_moss.ghosts.dolores._run import DoloresRun

        session = session or FakeRunSession()
        return DoloresRun(
            ego=ego or FakeRunEgo(session),
            thinking=thinking or FakeRunThinking(),
            thinking_event=asyncio.Event(),
            facade=facade or _NoopShellFacade(),
        )

    @classmethod
    def _text_chunk(cls, text: str, seq: int = 1):
        return dsh_event(
            "assistant/chunk",
            {"turn": 1, "step": 1, "chunk": {"type": "text-delta", "text": text}},
            seq=seq,
        )

    @pytest.mark.asyncio
    async def test_channel_facade_recursive_lists_and_single_reads(self):
        """moss_channel_facade: recursive 列前缀下 channel, 非 recursive 读单个完整面."""
        from ghoshell_moss.ghosts.dolores._run import DoloresRun

        class _Meta:
            def __init__(self, description):
                self.description = description

        class _Facade:
            def channel_metas(self, available_only=True):
                return {"ghost.frame": _Meta("the frame"), "ghost.voice": _Meta("the voice")}

            def get_channel_full_facade(self, path):
                return f"<{path}>full surface</{path}>" if path == "ghost.frame" else ""

        session = FakeRunSession()
        ego = FakeDispatchEgo(session)
        run = DoloresRun(
            ego=ego, thinking=FakeRunThinking(), thinking_event=asyncio.Event(), facade=_Facade(),
        )
        await run._handle_tool_use_event(
            dsh_tool_call("moss_channel_facade", '{"channel_path": "ghost"}', call_id="call_rec"), {},
        )
        await run._handle_tool_use_event(
            dsh_tool_call("moss_channel_facade", '{"channel_path": "ghost.frame", "recursive": false}', call_id="call_one"), {},
        )

        results = {call_id: result for call_id, result, *_ in ego.rpc_calls}
        assert "ghost.frame" in results["call_rec"]
        assert "the voice" in results["call_rec"]
        assert "full surface" in results["call_one"]

    @pytest.mark.asyncio
    async def test_closing_tool_races_the_percept_push(self):
        """收轮工具在飞时人插一句: 立旗 + 照推 —— 不再死押, 这一轮要接着走."""
        run = self._run()
        run._thinking_event.set()
        seen: dict = {}

        async def _fake_dispatch(event, streams):
            seen["in_flight"] = run._ctml_in_flight
            run._on_percepts_injected(["he said something mid-action"])
            seen["barge"] = run._barge_in_event is not None and run._barge_in_event.is_set()
            seen["push"] = run._buffered_push
            run._buffered_push.cancel()

        run._dispatch_tool_use_event = _fake_dispatch
        await run._handle_tool_use_event(
            dsh_tool_call("moss_react", '{"ctml": "<noop/>"}', call_id="call_react"), {},
        )

        assert seen["in_flight"] is True, "react 必须在立旗状态下执行"
        assert seen["barge"] is True, "人插话必须叫醒那个等待, 否则竞速永远没人赢"
        assert seen["push"] is not None, "竞速赢了就该照推 moment"
        assert run._ctml_in_flight is False, "工具返回后旗子必须放下"
        assert run._barge_in_event is None, "竞速信号只归这一次调用"

    @pytest.mark.asyncio
    async def test_wait_next_tool_call_arms_the_barge_in_race(self):
        """moss_wait_next 同样进竞速 (它也是在等身体), 只是赢下之后是"这轮接着走"而不是"收轮"."""
        run = self._run()
        run._thinking_event.set()
        seen: dict = {}

        async def _fake_dispatch(event, streams):
            run._on_percepts_injected(["he said something while I yield"])
            seen["barge"] = run._barge_in_event is not None and run._barge_in_event.is_set()
            seen["push"] = run._buffered_push
            run._buffered_push.cancel()

        run._dispatch_tool_use_event = _fake_dispatch
        await run._handle_tool_use_event(dsh_tool_call("moss_wait_next", "{}"), {})

        assert seen["barge"] is True
        assert seen["push"] is not None

    @pytest.mark.asyncio
    async def test_percepts_stay_in_the_buffer_once_the_turn_is_closing(self):
        """竞速已经判给身体 (旗子撤下) 之后落地的 percepts 只留 buffer —— 这一轮正在收尾."""
        run = self._run()
        run._thinking_event.set()
        run._ctml_in_flight = True
        run._barge_in_event = None
        seen: dict = {}

        run._on_percepts_injected(["late words"])
        seen["push"] = run._buffered_push

        assert seen["push"] is None, "定了收轮之后不许再推给一个正在收尾的 turn"

    @pytest.mark.asyncio
    async def test_percepts_during_settle_only_arm_the_signal(self):
        """settle 期间 (不安全) signal 只置位 signal_event, 不 drain —— 留给 settle 竞速决定.

        这是 safe flag 的核心: turn/end 后到 settle 结论之间, 若急着 drain, 竞速判 action_done
        退出时 signal 已经被抽干 (has_pending 假), mindflow 不签下一帧 → 两头落空. 所以不安全窗口
        只通知不 drain, 竞速判退出时 signal 仍留在 observer 兜底.
        """
        run = self._run()
        run._thinking_event.set()
        run._settling = True
        run._signal_event = asyncio.Event()

        run._on_percepts_injected(["他说了一句"])

        assert run._signal_event.is_set(), "settle 期间 signal 必须叫醒竞速"
        assert run._buffered_push is None, "settle 期间不 drain、不 push"
        assert run._pushed_this_run is False, "settle 期间不标记 push"

    @pytest.mark.asyncio
    async def test_settle_exits_when_actions_done_and_no_signal(self):
        """turn/end(normal) 后 settle: 无 push、无新 signal → action_done 赢 → 退出本 run."""
        run = self._run()
        run._thinking_event.set()
        assert await run._settle_after_turn_end() is False, "无 signal 时 settle 判退出"

    @pytest.mark.asyncio
    async def test_interpret_tool_call_does_not_hold_the_percept_push(self):
        """moss_interpret 不立旗: 它跑完这一轮还要继续, 人此刻说的话下一步就该读到."""
        run = self._run()
        run._thinking_event.set()
        seen: dict = {}

        async def _fake_dispatch(event, streams):
            seen["in_flight"] = run._ctml_in_flight
            run._on_percepts_injected(["he said something mid-action"])
            seen["push"] = run._buffered_push
            assert run._buffered_push is not None, "interpret 期间必须照常开推"
            run._buffered_push.cancel()

        run._dispatch_tool_use_event = _fake_dispatch
        await run._handle_tool_use_event(
            dsh_tool_call("moss_interpret", '{"ctml": "<noop/>"}'), {},
        )

        assert seen["in_flight"] is False, "interpret 不押推送"
        assert run._ctml_in_flight is False

    @pytest.mark.asyncio
    async def test_non_ctml_tool_does_not_hold_the_percept_push(self):
        """非 CTML 工具 (bash / 工具面读) 不立旗 — 跑 bash 时人说的话照旧推进来."""
        run = self._run()
        seen: dict = {}

        async def _fake_dispatch(event, streams):
            seen["in_flight"] = run._ctml_in_flight

        run._dispatch_tool_use_event = _fake_dispatch
        await run._handle_tool_use_event(dsh_tool_call("moss_channel_facade", '{}'), {})

        assert seen["in_flight"] is False

    @pytest.mark.asyncio
    async def test_channel_facade_reports_unknown_path(self):
        """未知 path (非 recursive) 报错不抛 (工具结果回给模型, 不烧整轮)."""
        from ghoshell_moss.ghosts.dolores._run import DoloresRun

        class _Facade:
            def channel_metas(self, available_only=True):
                return {}

            def get_channel_full_facade(self, path):
                return ""

        session = FakeRunSession()
        ego = FakeDispatchEgo(session)
        run = DoloresRun(
            ego=ego, thinking=FakeRunThinking(), thinking_event=asyncio.Event(), facade=_Facade(),
        )
        await run._handle_tool_use_event(
            dsh_tool_call("moss_channel_facade", '{"channel_path": "nope", "recursive": false}'), {},
        )
        assert "no such channel" in ego.rpc_calls[0][1]

    @pytest.mark.asyncio
    async def test_ghost_event_escalates_the_attention_only_while_the_turn_runs(self):
        """DSH 每签发一个事件就提权一次 —— thinking 期间人开口才抢不动、只能排队折进来.

        反例 (D18 实测): 没有提权时, 只跑 tool call 的那几秒里强度按衰减曲线掉下去, 人一句
        INFO 就赢下仲裁 → preempt → 整轮 abort, 正在跑的 tool call 一起陪葬.
        """
        log: list = []
        run = self._run(thinking=FakeRunThinking(log))
        event = dsh_event("tool/start", {"turn": 1})

        run._thinking_event.set()
        await run._on_event(event)
        assert "escalate" in log

        # 收线之后不再提权 —— 否则一个已经结束的 attention 永远不衰减.
        log.clear()
        run._thinking_event.clear()
        await run._on_event(event)
        assert "escalate" not in log

    @pytest.mark.asyncio
    async def test_shell_status_reads_facade(self):
        """moss_shell_status 读 shell 状态描述, 不产生 moment."""
        from ghoshell_moss.ghosts.dolores._run import DoloresRun

        class _Status:
            def description(self):
                return "idle, 2 actions running"

        class _Facade:
            def status(self):
                return _Status()

        session = FakeRunSession()
        ego = FakeDispatchEgo(session)
        run = DoloresRun(
            ego=ego, thinking=FakeRunThinking(), thinking_event=asyncio.Event(), facade=_Facade(),
        )
        await run._handle_tool_use_event(dsh_tool_call("moss_shell_status", call_id="call_s"), {})
        assert ego.rpc_calls == [("call_s", "idle, 2 actions running", None, False)]

    @pytest.mark.asyncio
    async def test_shell_clear_reaches_shell_directly(self):
        """moss_shell_clear 直接调 shell.clear(), 不经过 CTML 的 __main__ 队列.

        这是排队堵死时的逃生口: CTML 的 <clear/> 自己也是一条命令, 会和被堵的命令排在一起.
        它不该签发 moment, 也不该掐回合 —— 下一步要能看见清完之后的现场.
        """
        from ghoshell_moss.ghosts.dolores._run import DoloresRun

        calls = []

        class _Shell:
            async def clear(self):
                calls.append("clear")

        class _Facade:
            def __init__(self):
                self.shell = _Shell()

        session = FakeRunSession()
        ego = FakeDispatchEgo(session)
        run = DoloresRun(
            ego=ego, thinking=FakeRunThinking(), thinking_event=asyncio.Event(), facade=_Facade(),
        )
        await run._handle_tool_use_event(dsh_tool_call("moss_shell_clear", call_id="call_c"), {})

        assert calls == ["clear"], "shell.clear() 没被调用 —— 逃生口等于不存在"
        call_id, result, error, cancel = ego.rpc_calls[0]
        assert call_id == "call_c"
        assert error is None
        assert cancel is False, "清壳不该掐掉回合, 否则下一步看不到结果"
        assert "cleared" in result

    @pytest.mark.asyncio
    async def test_observe_observes_moment(self):
        """moss_observe 等动作完 + observe 最新 moment (返回 moment_ref 并注入)."""
        from ghoshell_moss.core.blueprint.moment import Moment

        from ghoshell_moss.ghosts.dolores._run import DoloresRun

        class _Epoch:
            index = 3

        class _Observer:
            epoch = _Epoch()

        class _Shell:
            async def refresh_metas(self, timeout=None, stale_time=None):
                pass

        class _Facade:
            def __init__(self):
                self.shell = _Shell()

        class _Thinking(FakeRunThinking):
            async def wait_actions_done(self):
                pass

            def observe(self):
                return Moment(id="m", index=7)

            @property
            def observer(self):
                return _Observer()

        session = FakeRunSession()
        ego = FakeDispatchEgo(session)
        thinking = _Thinking()
        run = DoloresRun(
            ego=ego, thinking=thinking, thinking_event=asyncio.Event(), facade=_Facade(),
        )
        await run._handle_tool_use_event(dsh_tool_call("moss_observe", call_id="call_w"), {})
        assert ego.rpc_calls == [
            ("call_w", {"moment_ref": "3-7"}, {"context": [{"type": "text", "text": "moment:3-7"}], "inputs": [], "moment_id": "3-7"}, False),
        ]

    @pytest.mark.asyncio
    async def test_observe_interrupt_replans_then_observes(self):
        """interrupt=true → 先空 replan (kind='clear') 中止当前行动, 等它落地, 再 wait + observe."""
        from ghoshell_moss.core.blueprint.moment import Moment

        from ghoshell_moss.ghosts.dolores._run import DoloresRun

        class _Epoch:
            index = 3

        class _Observer:
            epoch = _Epoch()

        class _Shell:
            async def refresh_metas(self, timeout=None, stale_time=None):
                pass

        class _Facade:
            def __init__(self):
                self.shell = _Shell()

        class _Thinking(FakeRunThinking):
            def __init__(self):
                super().__init__()
                self.replans: list[bool] = []

            def articulator(self, replan=False, wait_action_done=False):
                self.replans.append(replan)
                return super().articulator(replan=replan, wait_action_done=wait_action_done)

            async def wait_actions_done(self):
                pass

            def observe(self):
                return Moment(id="m", index=9)

            @property
            def observer(self):
                return _Observer()

        session = FakeRunSession()
        ego = FakeDispatchEgo(session)
        thinking = _Thinking()
        run = DoloresRun(
            ego=ego, thinking=thinking, thinking_event=asyncio.Event(), facade=_Facade(),
        )
        await run._handle_tool_use_event(
            dsh_tool_call("moss_observe", '{"interrupt": true}', call_id="call_i"), {},
        )

        assert thinking.replans == [True]
        assert thinking.articulators[0].sent == []  # 空 replan, 没发内容
        assert ego.rpc_calls[0][1] == {"moment_ref": "3-9"}

    @pytest.mark.asyncio
    async def test_aenter_binds_listener_and_aexit_cleans_up(self):
        session = FakeRunSession()
        ego = FakeRunEgo(session)
        run = self._run(session=session, ego=ego)
        async with run:
            assert run._thinking_event.is_set()  # 交易中 (run aenter 置位)
            assert len(session.handlers) == 1  # catch-all 监听已绑
            await asyncio.sleep(0)  # 让出 loop, enter task 跑
            assert ego.enter_calls == 1
        assert not run._thinking_event.is_set()  # 交易结束 (run aexit 复位)
        assert ego.exit_calls == 1
        assert len(session.handlers) == 0  # 解绑

    @pytest.mark.asyncio
    async def test_logos_skips_text_and_stops_on_turn_end(self):
        """plain text 跳过 (不解析为 CTML), turn/end 被转发成 GhostEvent 后 logos() 自止."""
        session = FakeRunSession()
        ego = FakeRunEgo(session)
        thinking = FakeRunThinking()
        run = self._run(session=session, ego=ego, thinking=thinking)
        async with run:
            await session.emit(self._text_chunk("plain text nobody reads", seq=1))
            await session.emit(dsh_event("turn/end", {"turn": 1}, seq=2))
            collected = []
            async for delta in run.logos():
                collected.append(delta)
        # plain text 不产出 logos; 流以 thinking 开场, turn/end 只收线、不再转发.
        assert [d.event for d in collected] == ["ghost/lifecycle/thinking"]
        assert thinking.articulators == []  # 无 articulator 被创建
        assert thinking.abort_reasons == []  # completed → 不打断

    @pytest.mark.asyncio
    async def test_logos_forwards_complete_dsh_events_only(self):
        """只转发完整事件 (turn/tool/assistant-message); assistant/chunk 逐 token 流被排除."""
        session = FakeRunSession()
        ego = FakeRunEgo(session)
        thinking = FakeRunThinking()
        run = self._run(session=session, ego=ego, thinking=thinking)
        async with run:
            await session.emit(dsh_event("turn/start", {"turn": 1}, seq=1))
            await session.emit(dsh_event("assistant/message", {"turn": 1, "step": 1, "message": {}}, seq=2))
            await session.emit(dsh_event("tool/call", {"callId": "c1", "name": "mystery_tool", "arguments": "{}"}, seq=3))
            await session.emit(dsh_event("tool/result", {"message": {}, "error": None}, seq=4))
            await session.emit(self._text_chunk("reasoning nobody sees", seq=5))  # 排除: 流
            await session.emit(dsh_event("turn/end", {"turn": 1}, seq=6))
            collected = []
            async for delta in run.logos():
                collected.append(delta)
        # lifecycle/thinking 开场; 之后只剩行动脊椎 —— tool/call. turn/*、assistant/message
        # 和 tool/result 还在 session log 里, 但不再往观测面复制一遍.
        assert [e.event for e in collected] == [
            "ghost/lifecycle/thinking",
            "dsh/tool/call",
        ]
        # 转发的 payload 是原始信封 (to_dict), data 原样在内.
        assert collected[1].payload["seq"] == 3
        # content 是给表面的那一行: 摘要由我们自己拼, 不指望 dsh 给.
        assert collected[0].content == "thinking"
        assert collected[1].content == "mystery_tool"

    @pytest.mark.asyncio
    async def test_tool_event_summary_prefers_the_callers_own_description(self):
        """观测面上那一行取调用方自己写的 description, 不是参数原文."""
        session = FakeRunSession()
        ego = FakeRunEgo(session)
        thinking = FakeRunThinking()
        run = self._run(session=session, ego=ego, thinking=thinking)
        async with run:
            await session.emit(dsh_event(
                "tool/call",
                {"callId": "c1", "name": "bash",
                 "arguments": '{"command": "git status", "description": "Show working tree status"}'},
                seq=1,
            ))
            await session.emit(dsh_event("turn/end", {"turn": 1}, seq=2))
            collected = []
            async for delta in run.logos():
                collected.append(delta)
        assert collected[1].content == "bash: Show working tree status"

    @pytest.mark.asyncio
    async def test_tool_event_summary_never_leaks_argument_text(self):
        """没有 description 就只报工具名 —— 参数里可能有文件正文/路径/口令, 不许上总线."""
        session = FakeRunSession()
        ego = FakeRunEgo(session)
        thinking = FakeRunThinking()
        run = self._run(session=session, ego=ego, thinking=thinking)
        async with run:
            await session.emit(dsh_event(
                "tool/call",
                {"callId": "c1", "name": "write",
                 "arguments": '{"file_path": "/tmp/secret.md", "content": "private text"}'},
                seq=1,
            ))
            await session.emit(dsh_event("turn/end", {"turn": 1}, seq=2))
            collected = []
            async for delta in run.logos():
                collected.append(delta)
        assert collected[1].content == "write"
        assert "private" not in collected[1].payload.get("content", "")
        assert "secret" not in str(collected[1].content)

    @pytest.mark.asyncio
    async def test_own_tool_calls_never_reach_the_surface(self):
        """moss_interpret / moss_react 是管道不是行动: 不上观测面."""
        session = FakeRunSession()
        ego = FakeRunEgo(session)
        thinking = FakeRunThinking()
        run = self._run(session=session, ego=ego, thinking=thinking)
        async with run:
            await session.emit(dsh_event(
                "tool/call",
                {"callId": "c1", "name": "moss_surface_probe",
                 "arguments": '{"ctml": "<say>hi</say>"}'},
                seq=1,
            ))
            await session.emit(dsh_event("turn/end", {"turn": 1}, seq=2))
            collected = []
            async for delta in run.logos():
                collected.append(delta)
        assert [getattr(d, "event", None) for d in collected] == ["ghost/lifecycle/thinking"]

    def test_is_own_tool_call_discriminates_by_namespace(self):
        """判据挂在 moss 命名空间上 —— 与 tool 分发用的是同一条前缀."""
        from ghoshell_moss.ghosts.dolores._run import _is_own_tool_call

        def call(name: str):
            return dsh_event(
                "tool/call",
                {"callId": "c1", "name": name, "arguments": "{}"},
            )

        assert _is_own_tool_call(call("moss_interpret")) is True
        assert _is_own_tool_call(call("moss_react")) is True
        assert _is_own_tool_call(call("bash")) is False
        assert _is_own_tool_call(call("read")) is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "reason, expected",
        [
            ({"kind": "completed"}, None),
            ({"kind": "max-tokens"}, None),
            ({"kind": "brand-new-kind"}, None),
            ({"kind": "aborted", "reason": {"kind": "hook", "reason": "moss cancel_turn"}}, None),
            ({"kind": "aborted", "reason": {"kind": "user"}}, "turn/end aborted (user)"),
            ({"kind": "interrupted"}, "turn/end interrupted"),
            ({"kind": "blocked", "reason": {"kind": "user"}}, "turn/end blocked (user)"),
            ({"kind": "error"}, "turn/end error"),
        ],
    )
    async def test_turn_end_classifies_before_ending_the_frame(self, reason, expected):
        """turn/end 分两类: 自然收线不动 attention, 中断类 abort 并带 reason.

        旧版这条叫 `test_turn_end_never_aborts_thinking`, 断言"无论 kind 都绝不 abort" —— 那是把一个
        方向的判据当成了通性: 它只对"我们自己让 plugin 切的 turn"成立 (cause=hook, 身体在返回前已
        drain). UI 点停与运行时故障若也走自然退出, 身体会继续跑 (语音照旧播完), 故障还会自续成环.
        现在判据挂在 cause 上 (不挂 turn 号), 见 ``_run._turn_end_abort_reason``.

        端到端判据 (语音是否真停) 不在单测里: 见 regressions/dolores-ghost-prototype TC-003.
        """
        session = FakeRunSession()
        ego = FakeRunEgo(session)
        thinking = FakeRunThinking()
        run = self._run(session=session, ego=ego, thinking=thinking)
        async with run:
            await session.emit(self._text_chunk("<say>hi</say>", seq=1))
            await session.emit(dsh_event("turn/end", {"turn": 1, "reason": reason}, seq=2))
            async for _ in run.logos():
                pass
        if expected is None:
            assert thinking.abort_reasons == []
        else:
            assert thinking.abort_reasons == [expected]

    @pytest.mark.asyncio
    async def test_turn_end_interrupt_lands_before_the_consumer_can_run(self):
        """中断在**入队前**落地: 消费被 tool handler 占住时, 人点停也要立刻生效.

        2026-10-04 D04 实测的根: ``_handle_react`` 在 logos() 里 await wait_actions_done() 等身体跑完,
        期间的 turn/end 只能排队 —— 人点的停 (aborted/user) 在队列里等了 41 秒, 语音自己播完才被消费.
        判据是入队时就 abort (Attention.abort 同步立即生效), 不等消费.
        """
        session = FakeRunSession()
        ego = FakeRunEgo(session)
        thinking = FakeRunThinking()
        run = self._run(session=session, ego=ego, thinking=thinking)
        async with run:
            await session.emit(self._text_chunk("<say>hi</say>", seq=1))
            await session.emit(
                dsh_event(
                    "turn/end",
                    {"turn": 1, "reason": {"kind": "aborted", "reason": {"kind": "user"}}},
                    seq=2,
                )
            )
            # 关键: 这里**不消费** logos() —— 消费正被占住, 模拟 D04 的真实时序.
        assert thinking.abort_reasons == ["turn/end aborted (user)"]

    @pytest.mark.asyncio
    async def test_our_own_hook_cancel_does_not_abort_on_enqueue(self):
        """正常收轮 (cause=hook) 不得因为入了队就 abort —— 否则每次 react 一开口就被自己清掉."""
        session = FakeRunSession()
        ego = FakeRunEgo(session)
        thinking = FakeRunThinking()
        run = self._run(session=session, ego=ego, thinking=thinking)
        async with run:
            await session.emit(self._text_chunk("<say>hi</say>", seq=1))
            await session.emit(
                dsh_event(
                    "turn/end",
                    {"turn": 1, "reason": {"kind": "aborted",
                                           "reason": {"kind": "hook", "reason": "moss cancel_turn"}}},
                    seq=2,
                )
            )
        assert thinking.abort_reasons == []

    @pytest.mark.asyncio
    async def test_enter_error_propagates_and_aborts(self):
        session = FakeRunSession()
        ego = FakeRunEgo(session)
        thinking = FakeRunThinking()
        ego.enter_error = RuntimeError("enter boom")
        run = self._run(session=session, ego=ego, thinking=thinking)
        with pytest.raises(RuntimeError, match="enter boom"):
            async with run:
                async for _ in run.logos():
                    pass
        assert thinking.abort_reasons  # enter 异常 → thinking.abort

    @pytest.mark.asyncio
    async def test_consumer_exception_aborts_thinking(self):
        session = FakeRunSession()
        ego = FakeRunEgo(session)
        thinking = FakeRunThinking()
        run = self._run(session=session, ego=ego, thinking=thinking)
        with pytest.raises(RuntimeError, match="consumer boom"):
            async with run:
                raise RuntimeError("consumer boom")
        assert thinking.abort_reasons

    @pytest.mark.asyncio
    async def test_dispatch_tool_result_passes_call_id_result_and_moment_parts(self):
        """正常路径: callId + result + moment parts 原样交给 RPC."""
        from ghoshell_moss.core.blueprint.moment import Moment

        from ghoshell_moss.ghosts.dolores._tools import ToolCallResult

        ego = FakeDispatchEgo()
        run = self._run(ego=ego)
        result = ToolCallResult(
            call=fake_tool_call("call_ok"),
            result={"moment_ref": "1-7"},
            moment=Moment(id="1-7"),
        )
        await run._dispatch_tool_result(result)
        assert ego.rpc_calls == [("call_ok", {"moment_ref": "1-7"}, {"context": [{"type": "text", "text": "moment:1-7"}], "inputs": [], "moment_id": "1-7"}, False)]

    @pytest.mark.asyncio
    async def test_dispatch_tool_result_absorbs_rpc_failure(self):
        """迟到回话不再致命: RPC 失败被吸收, 不冒泡 (旧行为会打穿整轮 logos 流)."""
        from ghoshell_moss.ghosts.dolores._tools import ToolCallResult

        ego = FakeDispatchEgo(raise_on_rpc=RuntimeError("no pending tool call for call_late"))
        run = self._run(ego=ego)
        result = ToolCallResult(call=fake_tool_call("call_late"), result={"moment_ref": "1-8"})
        await run._dispatch_tool_result(result)  # 不抛
        assert ego.rpc_calls == []

    @pytest.mark.asyncio
    async def test_dispatch_tool_result_without_moment_sends_empty_parts(self):
        """interleaved_ctml 这类不携带 moment 的 tool: moment parts 为 None."""
        from ghoshell_moss.ghosts.dolores._tools import ToolCallResult

        ego = FakeDispatchEgo()
        run = self._run(ego=ego)
        result = ToolCallResult(call=fake_tool_call("call_ctml"), result="ok")
        await run._dispatch_tool_result(result)
        assert ego.rpc_calls == [("call_ctml", "ok", None, False)]

    @pytest.mark.asyncio
    async def test_dispatch_tool_result_carries_the_call_turn(self):
        """结果协议带上 tool 被调用时的 turn —— 插件据此把"早到结果"绑到正确的轮次, 并按轮清理.

        插件侧 /tool-result 可能先于 dsh 派发该 tool 的 execute 到达 (MOSS 走 mux 事件流更快);
        结果里的 turn 让插件在 execute 尚未注册时也能按轮归属, 而不是悬空.
        """
        from ghoshell_moss.deepseek_harness.v0_2_0 import ToolCallEvent

        from ghoshell_moss.ghosts.dolores._tools import ToolCallResult

        ego = FakeDispatchEgo()
        run = self._run(ego=ego)
        call = ToolCallEvent(callId="call_turn", turn=7)
        result = ToolCallResult(call=call, result="ok")
        await run._dispatch_tool_result(result)
        assert ego.rpc_calls == [("call_turn", "ok", None, False)]
        assert ego.rpc_turns == [7]

    @pytest.mark.asyncio
    async def test_wait_next_result_carries_cancel(self):
        """moss_wait_next 用结果协议里的 cancel flag 收线 — 不是 final answer, 也不用 abort."""
        from ghoshell_moss.ghosts.dolores._run import DoloresRun

        class _Thinking:
            def __init__(self):
                self.actions_done = 0

            async def wait_actions_done(self):
                self.actions_done += 1

        thinking = _Thinking()
        session = FakeRunSession()
        ego = FakeDispatchEgo(session)
        run = DoloresRun(
            ego=ego, thinking=thinking, thinking_event=asyncio.Event(), facade=None,
        )
        await run._handle_tool_use_event(dsh_tool_call("moss_wait_next", call_id="call_w"), {})

        assert thinking.actions_done == 1
        assert ego.rpc_calls == [("call_w", "yielded", None, True)]

    # ── ctml 流式接线 ───────────────────────────────────────────────

    @staticmethod
    def _ctml_delta(call_id, *, name=None, arguments_delta=None, seq=1):
        """assistant/chunk 的 tool-call-delta — 名字只挂在首个 delta 上."""
        chunk = {"type": "tool-call-delta", "id": call_id}
        if name is not None:
            chunk["name"] = name
        if arguments_delta is not None:
            chunk["argumentsDelta"] = arguments_delta
        return dsh_event("assistant/chunk", {"turn": 1, "step": 1, "chunk": chunk}, seq=seq)

    @staticmethod
    def _tool_call_event(name, arguments, call_id, seq=1):
        return dsh_event(
            "tool/call",
            {"turn": 1, "step": 1, "callId": call_id, "name": name, "arguments": arguments},
            seq=seq,
        )

    @staticmethod
    def _turn_end(seq=1):
        return dsh_event("turn/end", {"turn": 1, "reason": {"kind": "completed"}}, seq=seq)

    @pytest.mark.asyncio
    async def test_interpret_delta_streams_into_articulator_then_returns_moment(self):
        """流式 ctml: delta 逐片进 articulator, tool/call 收线 → 等 observed → 签发并注入 moment."""
        session = FakeRunSession()
        ego = FakeDispatchEgo(session)
        thinking = FakeRunThinking()
        run = self._run(session=session, ego=ego, thinking=thinking)
        async with run:
            await session.emit(self._ctml_delta("c1", name="moss_interpret", arguments_delta='{"ctml":"<say>', seq=1))
            await session.emit(self._ctml_delta("c1", arguments_delta='hi</say>"}', seq=2))
            await session.emit(self._tool_call_event("moss_interpret", '{"ctml":"<say>hi</say>"}', "c1", seq=3))
            await session.emit(self._turn_end(seq=4))
            async for _ in run.logos():
                pass

        assert len(thinking.articulators) == 1
        assert "".join(thinking.articulators[0].sent) == "<say>hi</say>"
        assert ego.rpc_calls == [
            ("c1", {"moment_ref": "3-7"}, {"context": [{"type": "text", "text": "moment:3-7"}], "inputs": [], "moment_id": "3-7"}, False),
        ]

    @pytest.mark.asyncio
    async def test_reply_first_runs_the_interpret_path_and_does_not_close_the_turn(self):
        """reply_first 是同一机制换个名字: 流式进 articulator, 签 moment, cancel 为假."""
        session = FakeRunSession()
        ego = FakeDispatchEgo(session)
        thinking = FakeRunThinking()
        run = self._run(session=session, ego=ego, thinking=thinking)
        async with run:
            await session.emit(self._ctml_delta("c1", name="moss_reply_first", arguments_delta='{"ctml":"<say>', seq=1))
            await session.emit(self._ctml_delta("c1", arguments_delta='先看一眼</say>"}', seq=2))
            await session.emit(self._tool_call_event("moss_reply_first", '{"ctml":"<say>先看一眼</say>"}', "c1", seq=3))
            await session.emit(self._turn_end(seq=4))
            async for _ in run.logos():
                pass

        assert "".join(thinking.articulators[0].sent) == "<say>先看一眼</say>"
        assert ego.rpc_calls == [
            ("c1", {"moment_ref": "3-7"}, {"context": [{"type": "text", "text": "moment:3-7"}], "inputs": [], "moment_id": "3-7"}, False),
        ], "与 interpret 同一条路: 带 moment 回来, cancel 为假 —— 这一轮还要继续"

    @pytest.mark.asyncio
    async def test_interpret_refreshes_metas_before_signing_the_moment(self):
        """签 moment 之前必须把暖层刷到"动作之后" —— 否则模型看到的还是动作前的世界 (晚一拍)."""
        from ghoshell_moss.ghosts.dolores._run import DoloresRun

        log: list[str] = []

        class _Shell:
            async def refresh_metas(self, timeout=None, stale_time=None):
                log.append("refresh_metas")

        class _Facade:
            def __init__(self):
                self.shell = _Shell()

        class _Thinking(FakeRunThinking):
            def observe(self):
                log.append("observe")
                return super().observe()

        session = FakeRunSession()
        ego = FakeDispatchEgo(session)
        run = DoloresRun(
            ego=ego,
            thinking=_Thinking(log=log),
            thinking_event=asyncio.Event(),
            facade=_Facade(),
        )
        async with run:
            await session.emit(self._ctml_delta("c1", name="moss_interpret", arguments_delta='{"ctml":"<say>', seq=1))
            await session.emit(self._ctml_delta("c1", arguments_delta='hi</say>"}', seq=2))
            await session.emit(self._tool_call_event("moss_interpret", '{"ctml":"<say>hi</say>"}', "c1", seq=3))
            await session.emit(self._turn_end(seq=4))
            async for _ in run.logos():
                pass

        assert "refresh_metas" in log and log[log.index("refresh_metas") + 1] == "observe"
        assert ego.rpc_calls == [
            ("c1", {"moment_ref": "3-7"}, {"context": [{"type": "text", "text": "moment:3-7"}], "inputs": [], "moment_id": "3-7"}, False),
        ]

    @pytest.mark.asyncio
    async def test_interpret_error_returns_cancel(self):
        """interpret error → 一行 ctml syntax error + cancel (细节留给下一轮 echoes)."""
        from ghoshell_moss.core.concepts.errors import InterpretError

        class _ErrorThinking(FakeRunThinking):
            def articulator(self, replan=False, wait_action_done=False):
                art = super().articulator(replan=replan, wait_action_done=wait_action_done)
                art.interpret_error = InterpretError("bad ctml")
                return art

        session = FakeRunSession()
        ego = FakeDispatchEgo(session)
        thinking = _ErrorThinking()
        run = self._run(session=session, ego=ego, thinking=thinking)
        async with run:
            await session.emit(self._ctml_delta("c1", name="moss_interpret", arguments_delta='{"ctml":"<bad>"}', seq=1))
            await session.emit(self._tool_call_event("moss_interpret", '{"ctml":"<bad>"}', "c1", seq=2))
            await session.emit(self._turn_end(seq=3))
            async for _ in run.logos():
                pass

        assert ego.rpc_calls == [("c1", "ctml syntax error", None, True)]

    @pytest.mark.asyncio
    async def test_interpret_without_deltas_sends_whole_argument(self):
        """无 delta (形状不符/整段回退): tool/call 一次性发完整 ctml."""
        session = FakeRunSession()
        ego = FakeDispatchEgo(session)
        thinking = FakeRunThinking()
        run = self._run(session=session, ego=ego, thinking=thinking)
        async with run:
            await session.emit(self._tool_call_event("moss_interpret", '{"ctml":"<say>hi</say>"}', "c1", seq=1))
            await session.emit(self._turn_end(seq=2))
            async for _ in run.logos():
                pass

        assert "".join(thinking.articulators[0].sent) == "<say>hi</say>"
        assert ego.rpc_calls == [
            ("c1", {"moment_ref": "3-7"}, {"context": [{"type": "text", "text": "moment:3-7"}], "inputs": [], "moment_id": "3-7"}, False),
        ]

    @pytest.mark.asyncio
    async def test_interpret_opens_and_closes_the_articulator_lifecycle(self):
        """回归: interpret 必须展开 articulator 的 async-with 生命周期 (__aenter__/__aexit__ 各一次).

        旧实现在 send + wait_compiled 后直接 return ToolCallResult, 从不退出边界 —— articulator 停在
        _commit 之后, 收尾永不发生, 插件侧于是把 pending tool 判成 settled, 结果回执被丢弃
        ("tool-result RPC dropped ... already settled plugin-side")。本用例锁住"边界被展开"。
        """
        session = FakeRunSession()
        ego = FakeDispatchEgo(session)
        thinking = FakeRunThinking()
        run = self._run(session=session, ego=ego, thinking=thinking)
        async with run:
            await session.emit(self._ctml_delta("c1", name="moss_interpret", arguments_delta='{"ctml":"<say>', seq=1))
            await session.emit(self._ctml_delta("c1", arguments_delta='hi</say>"}', seq=2))
            await session.emit(self._tool_call_event("moss_interpret", '{"ctml":"<say>hi</say>"}', "c1", seq=3))
            await session.emit(self._turn_end(seq=4))
            async for _ in run.logos():
                pass

        art = thinking.articulators[0]
        assert art.entered == 1, "articulator 必须被展开一次"
        assert art.exited == 1, "articulator 必须被收尾一次"
        assert "".join(art.sent) == "<say>hi</say>"

    @pytest.mark.asyncio
    async def test_interpret_waits_for_observed_not_actions_done(self):
        """interpret 只等 observed 命令落地, 不等动作全部跑完 —— 非 observe 命令跨帧继续跑.

        这是第五轮的关键语义转向: 等什么由 CTML 里的 @observe 声明, 而非"所有动作跑完".
        """
        session = FakeRunSession()
        ego = FakeDispatchEgo(session)
        log: list = []
        thinking = FakeRunThinking(log)
        run = self._run(session=session, ego=ego, thinking=thinking)
        async with run:
            await session.emit(self._ctml_delta("c1", name="moss_interpret", arguments_delta='{"ctml":"<say>', seq=1))
            await session.emit(self._ctml_delta("c1", arguments_delta='hi</say>"}', seq=2))
            await session.emit(self._tool_call_event("moss_interpret", '{"ctml":"<say>hi</say>"}', "c1", seq=3))
            await session.emit(self._turn_end(seq=4))
            async for _ in run.logos():
                pass

        assert log.count("wait_observed") == 1, "interpret 等 observed 命令落地"
        assert log.count("wait_action_done") == 0, "interpret 不再等动作全部跑完"
        # interpret 的 handler 不等 action 停; 这 1 次 wait_actions_done 来自 run 收尾的 settle 竞速
        # (主权在 body action: turn/end 后 run 仍要等 body 静默才退出).
        assert log.count("wait_actions_done") == 1, "interpret 的 handler 不等 action 停 (跨帧继续)"
        assert ego.rpc_calls == [
            ("c1", {"moment_ref": "3-7"}, {"context": [{"type": "text", "text": "moment:3-7"}], "inputs": [], "moment_id": "3-7"}, False),
        ]

    @pytest.mark.asyncio
    async def test_react_streams_and_cuts_turn(self):
        """moss_react: 共享流式解析, 等 compiled + action 停, 再 cancel 本回合, 不签发 moment."""
        session = FakeRunSession()
        ego = FakeDispatchEgo(session)
        log: list = []
        thinking = FakeRunThinking(log)
        run = self._run(session=session, ego=ego, thinking=thinking)
        async with run:
            await session.emit(self._ctml_delta("c2", name="moss_react", arguments_delta='{"ctml":"<say>', seq=1))
            await session.emit(self._ctml_delta("c2", arguments_delta='hi</say>"}', seq=2))
            await session.emit(self._tool_call_event("moss_react", '{"ctml":"<say>hi</say>"}', "c2", seq=3))
            await session.emit(self._turn_end(seq=4))
            async for _ in run.logos():
                pass

        art = thinking.articulators[0]
        assert art.entered == 1, "articulator 必须被展开一次"
        assert art.exited == 1, "articulator 必须被收尾一次"
        assert "".join(art.sent) == "<say>hi</say>"
        # react 的竞速等 action 停 1 次; turn/end 后的 settle 竞速再等 1 次 (run 收口).
        assert log.count("wait_actions_done") == 2, "react 先等 action 停再 cancel, settle 再等一次收口"
        assert ego.rpc_calls == [("c2", "reacted", None, True)]

    @staticmethod
    async def _drain(run) -> None:
        async for _ in run.logos():
            pass

    @pytest.mark.asyncio
    async def test_react_yields_to_a_barge_in_and_keeps_the_turn(self):
        """准全双工: react 正等身体跑完时人插一句 —— 工具立刻返回, 不 cancel, 帧照推.

        判据 (他给的): 竞速赢了不是"结束这一轮", 是"这一轮接着往下走". 所以结果是 cancel=False,
        而且那一帧必须已经落地 (下一 step 才读得到). 身体的动作不取消, 跨帧照旧跑.
        """
        session = FakeRunSession()
        ego = FakeDispatchEgo(session)
        thinking = _GatedThinking()
        run = self._run(session=session, ego=ego, thinking=thinking)
        try:
            async with run:
                await session.emit(self._ctml_delta("c3", name="moss_react", arguments_delta='{"ctml":"<noop/>"}', seq=1))
                await session.emit(self._tool_call_event("moss_react", '{"ctml":"<noop/>"}', "c3", seq=2))
                drain = asyncio.create_task(self._drain(run))
                await asyncio.wait_for(thinking.reached.wait(), timeout=2)
                run._on_percepts_injected(["他说了一句"])
                await session.emit(self._turn_end(seq=3))
                thinking.gate.set()  # 放行 settle 竞速的 wait_actions_done, 让 run 收口
                await asyncio.wait_for(drain, timeout=2)
        finally:
            thinking.gate.set()

        call_id, result, moment, cancel = ego.rpc_calls[0]
        assert call_id == "c3"
        assert cancel is False, "竞速赢了就不许收轮 —— 这一轮要接着读那句新的话"
        assert "continues" in result
        assert ego.pushed_moments, "赢了竞速的那一帧必须落地, 否则下一 step 读不到它"

    @pytest.mark.asyncio
    async def test_wait_next_continues_the_turn_when_someone_speaks(self):
        """moss_wait_next: 交出这一轮之前人说话了 —— 交出的前提没了, 这一轮接着走."""
        session = FakeRunSession()
        ego = FakeDispatchEgo(session)
        thinking = _GatedThinking()
        run = self._run(session=session, ego=ego, thinking=thinking)
        try:
            async with run:
                await session.emit(self._tool_call_event("moss_wait_next", "{}", "c4", seq=1))
                drain = asyncio.create_task(self._drain(run))
                await asyncio.wait_for(thinking.reached.wait(), timeout=2)
                run._on_percepts_injected(["他抢先说了一句"])
                await session.emit(self._turn_end(seq=2))
                thinking.gate.set()  # 放行 settle 竞速的 wait_actions_done, 让 run 收口
                await asyncio.wait_for(drain, timeout=2)
        finally:
            thinking.gate.set()

        call_id, result, moment, cancel = ego.rpc_calls[0]
        assert call_id == "c4"
        assert cancel is False
        assert "continues" in result
        assert ego.pushed_moments

    @pytest.mark.asyncio
    async def test_reasoning_records_depth_and_cancels(self):
        """moss_reasoning: 入口只标记 + cancel 收轮; 等身体/推档位/补回声都在 run 收尾."""
        from ghoshell_moss.ghosts.dolores._run import DoloresRun

        session = FakeRunSession()
        ego = FakeDispatchEgo(session)
        log: list = []
        thinking = FakeRunThinking(log)
        run = DoloresRun(ego=ego, thinking=thinking, thinking_event=asyncio.Event(), facade=None)
        await run._handle_tool_use_event(dsh_tool_call("moss_reasoning", '{"effort": "high"}', call_id="c_r"), {})

        assert ego.default_thinking_effort == "high"
        assert run._need_switch_thinking_effort == "high", "入口只把要换的档位记成状态"
        assert log == [], "工具入口不执行副作用: 不等动作、不推档位、不提交回声"
        assert ego.rpc_calls == [("c_r", {"effort": "high"}, None, True)]

        # 副作用全部在 run 收尾落地, 顺序: 等身体 → 推档位 → 补回声 (都在 exit_thinking 之前).
        # 替身上没有 select_model, 于是走 _push_effort_officially 的"永不致命"退路.
        await run.__aexit__(None, None, None)
        assert log == ["wait_actions_done", "observer.add_echoes:True"], (
            "收尾必须先把身体等完, 再补 need_observe —— 少哪一步, 档位换了下一帧也没人签发"
        )
        assert run._need_switch_thinking_effort is None, "pending 收尾后必须清掉, 不留悬声明"

    def test_tool_surface_matches_production_thinking(self):
        """handlers 在 thinking 上用的每个 API, 生产类必须真的有.

        回归防线. 这块曾经栽在哪: FakeRunThinking 发明了一个生产 ``Thinking`` 上不存在的
        ``add_echoes``, 单测全绿, 而生产里 moss_reasoning 抛 AttributeError → 被
        ``run_tool`` 吞成 result=None (cancel 随之丢失) → 本轮永不结束, 模型只看到 null.
        教训不是"补一个断言", 而是: **替身不许发明生产没有的 API**.
        """
        from ghoshell_moss.core.blueprint.mindflow import Thinking

        for name in ("articulator", "wait_actions_done", "observe", "observer", "abort"):
            assert hasattr(Thinking, name), f"Thinking 上没有 {name} —— handler 会在运行期静默失败"
        assert not hasattr(Thinking, "add_echoes"), (
            "add_echoes 属于 observer (Moments), 不属于 Thinking —— 用错对象就是那个被吞掉的 AttributeError"
        )

    @pytest.mark.asyncio
    async def test_interpret_stream_parse_error_returns_cancel(self):
        """流式解析出错 (非法转义) → 回 "ctml parse error" + cancel (不 fallback)."""
        session = FakeRunSession()
        ego = FakeDispatchEgo(session)
        thinking = FakeRunThinking()
        run = self._run(session=session, ego=ego, thinking=thinking)
        async with run:
            await session.emit(self._ctml_delta("c1", name="moss_interpret", arguments_delta='{"ctml":"<say>\\x', seq=1))
            await session.emit(self._tool_call_event("moss_interpret", '{"ctml":"<say>"}', "c1", seq=2))
            await session.emit(self._turn_end(seq=3))
            async for _ in run.logos():
                pass

        assert ego.rpc_calls == [("c1", "ctml parse error", None, True)]
