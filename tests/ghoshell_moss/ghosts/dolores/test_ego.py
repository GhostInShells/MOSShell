"""``DoloresEgo`` — the thinking half.

Answers: how a frame is entered and left. Depth declaration, the observe continuation,
compaction, epoch boundaries, the ego nucleus' arbitration shape, self-wake, the anchor it
commits after a turn, and the ``turn/end`` table that decides whether thinking must exit.
"""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghoshell_container import Container
from ghoshell_moss.core.blueprint.moment import Echoes

class TestDoloresEgoObserveContinuation:
    """DoloresEgo.needs_observe — observe 续帧标记, plugin 据此在 inputs 为空时也开一轮."""

    @staticmethod
    def _thinking(previous=None):
        return SimpleNamespace(moment=SimpleNamespace(previous=previous))

    def test_previous_requires_observe(self):
        from ghoshell_moss.ghosts.dolores._ego import DoloresEgo

        assert DoloresEgo.needs_observe(None, self._thinking(Echoes(need_observe=True))) is True

    def test_previous_without_observe(self):
        from ghoshell_moss.ghosts.dolores._ego import DoloresEgo

        assert DoloresEgo.needs_observe(None, self._thinking(Echoes(need_observe=False))) is False

    def test_no_previous_is_not_a_continuation(self):
        from ghoshell_moss.ghosts.dolores._ego import DoloresEgo

        assert DoloresEgo.needs_observe(None, self._thinking(None)) is False


class TestDoloresEgoReasoningEffort:
    """enter_thinking 的思考档解析 — 单一 effort 字段; 声明只在被真正采用的那一帧消费."""

    @staticmethod
    def _ego(launcher, *, default_thinking_effort=""):
        from ghoshell_moss.ghosts.dolores._ego import DoloresEgo, DoloresEgoContext

        return DoloresEgo(
            launcher=launcher,
            ctx=DoloresEgoContext(
                project_home=Path("."),
                project_name="pytest",
                name="dolores",
                mode="pytest",
                instruction="i",
                facade=None,
            ),
            default_thinking_effort=default_thinking_effort,
        )

    @staticmethod
    def _thinking(effort):
        from ghoshell_moss.core.blueprint.moment import Moment

        return SimpleNamespace(
            moment=Moment(index=0),
            observer=SimpleNamespace(
                epoch=SimpleNamespace(id="0", index=0, recap=[], baseline={}),
            ),
            effort=lambda: effort,
        )

    class _Launcher:
        def __init__(self):
            self.calls = []

        async def call(self, path, payload=None, *, timeout=None):
            self.calls.append((path, payload))
            return {}

    @pytest.mark.asyncio
    async def test_frame_effort_wins_and_declaration_survives(self):
        """帧自带明确 effort → 直通; 未被采用的声明不消费 (留给下一个真正能带它的帧)."""
        launcher = self._Launcher()
        ego = self._ego(launcher)
        ego.default_thinking_effort = "high"

        await ego.enter_thinking(self._thinking("low"), 1)

        payload = launcher.calls[0][1]
        assert payload["effort"] == "low"
        assert "reasoning_effort" not in payload, "档位只有一个字段, 不再两路下发"
        assert ego.default_thinking_effort == "high"

    @pytest.mark.asyncio
    async def test_none_frame_does_not_eat_the_declaration(self):
        """'none' 帧不驱动 turn, 也不该把待采用的声明吃掉 —— 否则 moss_reasoning 会静默失效."""
        launcher = self._Launcher()
        ego = self._ego(launcher)
        ego.default_thinking_effort = "high"

        await ego.enter_thinking(self._thinking("none"), 1)

        assert launcher.calls[0][1]["effort"] == "none"
        assert ego.default_thinking_effort == "high"

    @pytest.mark.asyncio
    async def test_default_declaration_is_consumed_when_used(self):
        """帧 effort == '' → 采用 ego 的待定声明, 采用即消费 (只采用一次)."""
        launcher = self._Launcher()
        ego = self._ego(launcher)
        ego.default_thinking_effort = "high"

        await ego.enter_thinking(self._thinking(""), 1)

        payload = launcher.calls[0][1]
        assert payload["effort"] == "high"
        assert ego.default_thinking_effort is None

    @pytest.mark.asyncio
    async def test_no_declaration_means_empty_effort(self):
        """无声明 + 帧不表态 → effort 为空串: 不驱动档位变更, 档位归 dsh/UI."""
        launcher = self._Launcher()
        ego = self._ego(launcher)

        await ego.enter_thinking(self._thinking(""), 1)

        assert launcher.calls[0][1]["effort"] == ""
        assert ego.default_thinking_effort is None

    @pytest.mark.asyncio
    async def test_startup_declaration_seeds_the_ego(self):
        """startup 的 default_thinking_effort 落到 ego 的待定声明上 (boot 声明)."""
        launcher = self._Launcher()
        ego = self._ego(launcher, default_thinking_effort="low")

        assert ego.default_thinking_effort == "low"

        await ego.enter_thinking(self._thinking(""), 1)

        assert launcher.calls[0][1]["effort"] == "low"
        assert ego.default_thinking_effort is None

    @pytest.mark.asyncio
    async def test_empty_startup_declares_nothing(self):
        """startup 空值 = 不表态: ego 不带初值, 不覆盖 dsh/UI 持有的档位."""
        launcher = self._Launcher()
        ego = self._ego(launcher)

        assert ego.default_thinking_effort is None

    @pytest.mark.asyncio
    async def test_model_declaration_rides_the_next_frame_and_is_consumed_once(self):
        """模型声明与档位同一条路: 下一帧带出 provider/model, 采用即消费."""
        launcher = self._Launcher()
        ego = self._ego(launcher)

        ego.declare_model("deepseek-official", "deepseek-v4-pro")
        await ego.enter_thinking(self._thinking(""), 1)

        payload = launcher.calls[0][1]
        assert payload["provider"] == "deepseek-official"
        assert payload["model"] == "deepseek-v4-pro"
        assert ego.default_model is None

    @pytest.mark.asyncio
    async def test_no_model_declaration_sends_empty_strings(self):
        """无声明 = 不表态: 空串让插件保持当前模型 (不是把 provider 清零)."""
        launcher = self._Launcher()
        ego = self._ego(launcher)

        await ego.enter_thinking(self._thinking(""), 1)

        payload = launcher.calls[0][1]
        assert payload["provider"] == ""
        assert payload["model"] == ""

    @pytest.mark.asyncio
    async def test_declare_model_carries_the_effort_and_refuses_half_a_route(self):
        """换模型可以顺手带档位; provider/model 缺一不可 —— 半个 route 会写坏下一次请求."""
        launcher = self._Launcher()
        ego = self._ego(launcher)

        receipt = ego.declare_model("deepseek-official", "deepseek-v4-pro", "max")
        assert "deepseek-v4-pro" in receipt and "max" in receipt
        assert ego.default_model == ("deepseek-official", "deepseek-v4-pro")
        assert ego.default_thinking_effort == "max"

        with pytest.raises(ValueError):
            ego.declare_model("", "deepseek-v4-pro")

    def test_model_identity_is_none_until_a_request_is_observed(self):
        """温数据的读口: 还没跑过请求 = None, 不拿目录默认值冒充"我在跑的模型"."""
        launcher = self._Launcher()
        ego = self._ego(launcher)

        assert ego.model_identity is None
        assert ego.model_route is None

    def test_declare_model_updates_identity_immediately(self):
        """声明模型那一刻身份就变 —— notice 不等下一帧 header 才反映改动."""
        launcher = self._Launcher()
        ego = self._ego(launcher)

        ego.declare_model("deepseek-official", "deepseek-v4-pro", "high")

        assert ego.model_identity == ("deepseek-official", "deepseek-v4-pro", "high")

    def test_declare_effort_updates_identity_immediately(self):
        """只改档位那一刻身份也变 —— 模型保持, 档位立即切到新值."""
        launcher = self._Launcher()
        ego = self._ego(launcher)
        ego.declare_model("deepseek-official", "deepseek-flash", "low")

        ego.declare_effort("high")

        assert ego.model_identity == ("deepseek-official", "deepseek-flash", "high")

    def test_declare_model_queues_identity_notice_immediately(self):
        """声明即排 notice —— 下一帧 enter 就带上新身份, 不等 header 回传."""
        launcher = self._Launcher()
        ego = self._ego(launcher)

        ego.declare_model("deepseek-official", "deepseek-v4-pro", "high")

        notices = [n for n in ego._notices if n.meta.tag == "model_notice"]
        assert len(notices) == 1
        assert "deepseek-v4-pro" in notices[0].to_content_string()

    @pytest.mark.asyncio
    async def test_lost_enter_does_not_burn_the_epoch(self):
        """enter RPC 丢了不能打戳 —— 否则 recap (压缩后 memory 的唯一通路) 永不重发."""
        from ghoshell_moss.message import Message

        class _FlakyLauncher(self._Launcher):
            fail = False

            async def call(self, path, payload=None, *, timeout=None):
                if self.fail:
                    raise RuntimeError("enter lost")
                return await super().call(path, payload, timeout=timeout)

        launcher = _FlakyLauncher()
        ego = self._ego(launcher)
        thinking = self._thinking("")
        thinking.observer.epoch = SimpleNamespace(
            id="e1", index=1,
            recap=[Message.new(tag="memory").with_content("ground")],
            baseline={"facade": "<facade/>"},
        )

        launcher.fail = True
        with pytest.raises(RuntimeError):
            await ego.enter_thinking(thinking, 1)
        assert ego._moment_epoch is None

        launcher.fail = False
        await ego.enter_thinking(thinking, 2)
        assert launcher.calls[0][1]["epoch"] is not None
        assert ego._moment_epoch == "e1"

        # 送达后同 epoch 不再重复.
        await ego.enter_thinking(thinking, 3)
        assert launcher.calls[1][1]["epoch"] is None


class _FakeCompactManager:
    """Duck-typed EgoMementoManager — compact 只读阈值配置与 SCS."""

    def __init__(self, cut_turn=7, compact_tokens=0):
        self.config = SimpleNamespace(compact_tokens=compact_tokens)
        self._cut_turn = cut_turn

    def compact_cut_turn(self):
        return self._cut_turn


class _FakeCompactLauncher:
    def __init__(self, replaced: int = 1):
        self._replaced = replaced
        self.calls: list[tuple[str, dict]] = []

    async def call(self, path, payload):
        self.calls.append((path, payload))
        return {"replaced": self._replaced}


class _CancellingLauncher:
    """取消落在 compact RPC 中间 — 插件可能已经换了 surface."""

    async def call(self, path, payload):
        raise asyncio.CancelledError()


def _compact_ego(*, memories, memento_manager=None, launcher=None):
    from ghoshell_moss.ghosts.dolores._ego import DoloresEgo, DoloresEgoContext

    ego = DoloresEgo(
        launcher=launcher,
        ctx=DoloresEgoContext(
            project_home=Path("."),
            project_name="pytest",
            name="dolores",
            mode="pytest",
            instruction="i",
            facade=None,
        ),
        memories=memories,
        memento_manager=memento_manager,
    )
    ego._ego_session_id = "s1"
    return ego


class TestDoloresEgoCompact:
    """compact 的 payload 拆分 — memory 走 epoch recap, surface 只留说明."""

    @staticmethod
    def _memory_message():
        from ghoshell_moss.message import Message

        return Message.new(tag="memory").with_content("ground")

    @pytest.mark.asyncio
    async def test_payload_splits_memory_from_notice(self):
        ego = _compact_ego(memories=lambda: [self._memory_message()])

        memory, notice = ego._compact_memory_payload()

        assert "<memory>" in memory[0].to_content_string()
        assert "compacted" in notice
        assert "<memory>" not in notice

    @pytest.mark.asyncio
    async def test_compact_if_pending_returns_memory_and_sends_notice(self):
        launcher = _FakeCompactLauncher()
        ego = _compact_ego(
            memories=lambda: [self._memory_message()],
            memento_manager=_FakeCompactManager(cut_turn=7),
            launcher=launcher,
        )
        ego._pending_compact = True

        memory = await ego.compact_if_pending()

        assert memory is not None and "<memory>" in memory[0].to_content_string()
        path, payload = launcher.calls[0]
        assert path.endswith("/compact")
        assert payload["cut_turn"] == 7
        assert "<memory>" in payload["memory"][0]["text"]
        assert "compacted" in payload["notice"]
        assert ego._pending_compact is False
        # 成功返回即交付, 不留下跨帧暂存.
        assert ego.take_compact_memory() is None

    @pytest.mark.asyncio
    async def test_nothing_shadowed_is_not_a_compact(self):
        """plugin 返回 replaced=0 → 实际没压, 不开新 epoch."""
        ego = _compact_ego(
            memories=lambda: [self._memory_message()],
            memento_manager=_FakeCompactManager(cut_turn=7),
            launcher=_FakeCompactLauncher(replaced=0),
        )
        ego._pending_compact = True

        assert await ego.compact_if_pending() is None
        assert ego.take_compact_memory() is None

    @pytest.mark.asyncio
    async def test_compact_if_pending_noop_without_pending(self):
        ego = _compact_ego(
            memories=lambda: [],
            memento_manager=_FakeCompactManager(),
            launcher=_FakeCompactLauncher(),
        )

        assert await ego.compact_if_pending() is None

    @pytest.mark.asyncio
    async def test_cancelled_compact_leaves_memory_for_the_next_frame(self):
        """取消落在 RPC 窗口: 插件可能已换 surface —— memory 必须留给下一帧补开 epoch."""
        ego = _compact_ego(
            memories=lambda: [self._memory_message()],
            memento_manager=_FakeCompactManager(cut_turn=7),
            launcher=_CancellingLauncher(),
        )
        ego._pending_compact = True

        with pytest.raises(asyncio.CancelledError):
            await ego.compact_if_pending()

        memory = ego.take_compact_memory()
        assert memory is not None and "<memory>" in memory[0].to_content_string()
        assert ego.take_compact_memory() is None  # 取走即清


class TestDoloresEgoNucleus:
    """DoloresEgoNucleus — BACKGROUND 挑战包 (发完丢), attended 加工成 INFO 运行包."""

    @pytest.mark.asyncio
    async def test_signal_produces_background_empty_impulse(self):
        from ghoshell_moss.core.blueprint.mindflow import Priority

        from ghoshell_moss.ghosts.dolores.nucleus import DoloresEgoNucleus, new_dolores_ego_signal

        nucleus = DoloresEgoNucleus()
        impulses = []
        async with nucleus:
            nucleus.with_bus(lambda signal: None, impulses.append)
            nucleus.add_signal(new_dolores_ego_signal())

        assert len(impulses) == 1
        imp = impulses[0]
        assert imp.priority == Priority.BACKGROUND
        assert imp.messages == []
        # 默认 mode (空) = 正常仲裁, 非 silent buffer.
        assert imp.mode == ""

    @pytest.mark.asyncio
    async def test_startup_kind_carries_command_and_instruction(self):
        from ghoshell_moss.ghosts.dolores.nucleus import DoloresEgoNucleus, new_dolores_ego_signal

        nucleus = DoloresEgoNucleus()
        impulses = []
        async with nucleus:
            nucleus.with_bus(lambda signal: None, impulses.append)
            nucleus.add_signal(new_dolores_ego_signal(
                kind="startup", command="<say>hi</say>", instruction="预热"
            ))

        assert len(impulses) == 1
        imp = impulses[0]
        assert imp.logos == "<say>hi</say>"
        assert len(imp.messages) == 1
        body = imp.messages[0].to_content_string()
        assert "<startup>" in body
        assert "预热" in body

    @pytest.mark.asyncio
    async def test_body_makes_it_a_delivery_on_the_buffer_path(self):
        """有正文的 ego signal 是投递, 不是唤醒: 不抢注意力, 输了仲裁也必须留下."""
        from ghoshell_moss.core.blueprint.mindflow import ChallengeMode, Priority

        from ghoshell_moss.ghosts.dolores.nucleus import DoloresEgoNucleus, new_dolores_ego_signal

        nucleus = DoloresEgoNucleus()
        impulses = []
        async with nucleus:
            nucleus.with_bus(lambda signal: None, impulses.append)
            nucleus.add_signal(new_dolores_ego_signal(
                "the answer", role="mind", description="answer from 1-1",
            ))

        assert len(impulses) == 1
        imp = impulses[0]
        # BACKGROUND: 不加挑战强度, 空闲时才作为初始 attention 赢.
        assert imp.priority == Priority.BACKGROUND
        # notify: 抢占失败时由 mindflow 整个 buffer, 而不是 suppress 掉.
        assert imp.mode == ChallengeMode.notify.value
        assert imp.role == "mind"
        assert imp.description == "answer from 1-1"
        assert [m.to_content_string() for m in imp.messages] == ["the answer"]

    @pytest.mark.asyncio
    async def test_second_delivery_does_not_overwrite_the_first(self):
        """排队, 不是单槽: 两条投递都还没被 raise 时, 后来的不能盖掉前一条."""
        from ghoshell_moss.ghosts.dolores.nucleus import DoloresEgoNucleus, new_dolores_ego_signal

        nucleus = DoloresEgoNucleus()
        async with nucleus:
            nucleus.with_bus(lambda signal: None, lambda impulse: None)
            nucleus.add_signal(new_dolores_ego_signal("first", role="mind"))
            nucleus.add_signal(new_dolores_ego_signal("second", role="mind"))
            first = nucleus.peek()
            second = nucleus.peek()
            assert nucleus.peek() is None

        assert first is not None and second is not None
        assert [m.to_content_string() for m in first.messages] == ["first"]
        assert [m.to_content_string() for m in second.messages] == ["second"]

    @pytest.mark.asyncio
    async def test_attended_rewrites_to_info(self):
        from ghoshell_moss.core.blueprint.mindflow import Impulse, Priority

        from ghoshell_moss.ghosts.dolores.nucleus import DoloresEgoNucleus

        nucleus = DoloresEgoNucleus()
        async with nucleus:
            challenge = Impulse(source="dolores_ego_nucleus", priority=Priority.BACKGROUND)
            rewritten = nucleus.attended(challenge)

        assert rewritten is not None
        assert rewritten.priority == Priority.INFO
        assert rewritten.messages == []

    @pytest.mark.asyncio
    async def test_ignores_foreign_signal(self):
        from ghoshell_moss.core.blueprint.mindflow import Signal

        from ghoshell_moss.ghosts.dolores.nucleus import DoloresEgoNucleus

        nucleus = DoloresEgoNucleus()
        impulses = []
        async with nucleus:
            nucleus.with_bus(lambda signal: None, impulses.append)
            nucleus.add_signal(Signal(name="some/other"))

        assert impulses == []

    @pytest.mark.asyncio
    async def test_meta_factory_builds_nucleus(self):
        from ghoshell_container import Container

        from ghoshell_moss.ghosts.dolores.nucleus import DoloresEgoNucleus, DoloresEgoNucleusMeta

        nucleus = DoloresEgoNucleusMeta().factory(Container())
        assert isinstance(nucleus, DoloresEgoNucleus)
        assert nucleus.name() == "dolores_ego_nucleus"


class TestDoloresEgoSelfWake:
    """DoloresEgo 的 self-wake gate — articulate flag 决定 turn/start 是否自醒."""

    def _ego(self):
        from ghoshell_moss.ghosts.dolores._ego import DoloresEgo, DoloresEgoContext

        # gate 路径不触 dsh / 不读 ghost 反参 — ctx/launcher 传最小 dummy 即可.
        return DoloresEgo(
            launcher=None,
            ctx=DoloresEgoContext(
                project_home=Path("."),
                project_name="pytest",
                name="dolores",
                mode="pytest",
                instruction="i",
                facade=None,
            ),
        )

    @pytest.mark.asyncio
    async def test_turn_start_self_wakes_when_idle(self):
        ego = self._ego()
        emitted = []
        ego.bind_signal_broadcast(emitted.append)
        await ego._on_session_activity(None)  # type: ignore[arg-type]  # gate 不读 event

        assert len(emitted) == 1
        assert emitted[0].name == "dolores/ego"

    @pytest.mark.asyncio
    async def test_turn_start_suppressed_when_articulating(self):
        ego = self._ego()
        emitted = []
        ego.bind_signal_broadcast(emitted.append)
        ego._thinking_event.set()  # 模拟 run 交易进行中 (is_thinking 只读, 直接置 event)
        await ego._on_session_activity(None)  # type: ignore[arg-type]

        assert emitted == []


class TestDoloresEgoCommit:
    """ego 的锚点 commit — 验证期 force_tokens=0 (每 turn commit) 下检查 memento 锚点数据."""

    @staticmethod
    def _turn_end(turn: int):
        from ghoshell_moss.deepseek_harness.v0_2_0 import event_from_dict

        return event_from_dict({"type": "turn/end", "seq": turn, "data": {"turn": turn}})

    @staticmethod
    def _set_up(tmp_path: Path, *, force_tokens: int = 0, warn_tokens: int = 50_000):
        from ghoshell_moss.ghosts.dolores._ego import DoloresEgo, DoloresEgoContext
        from ghoshell_moss.ghosts.dolores._ego_memento import EgoMementoConfig, EgoMementoManager
        from ghoshell_moss.memento import new_local_memento

        memento = new_local_memento(tmp_path / "owner")
        memento.create_branch("main")
        manager = EgoMementoManager(
            connection=None, memento=memento,
            config=EgoMementoConfig(force_tokens=force_tokens, warn_tokens=warn_tokens),
        )
        ego = DoloresEgo(
            launcher=None,  # commit 路径不触 dsh
            ctx=DoloresEgoContext(
                project_home=Path("."),
                project_name="pytest",
                name="dolores",
                mode="pytest",
                instruction="i",
                facade=None,
            ),
            memento_manager=manager,
        )
        ego._ego_session_id = "s1"
        return ego, memento

    @pytest.mark.asyncio
    async def test_each_turn_commit_span_is_chained(self, tmp_path: Path):
        ego, memento = self._set_up(tmp_path)
        for turn in (1, 2, 3):
            await ego._on_turn_end(self._turn_end(turn))

        metas = [c.metadata for c in memento.get_branch("main").commits()]

        # 每 turn 一锚点, 区间半开平铺: (0,1] (1,2] (2,3] —— 下界逐字抄上一个的 end_turn.
        assert [m["ref"]["start_turn"] for m in metas] == [0, 1, 2]
        assert [m["ref"]["end_turn"] for m in metas] == [1, 2, 3]
        assert [m["prev_turn"] for m in metas] == [0, 1, 2]
        assert {m["ref"]["session_id"] for m in metas} == {"s1"}

    @pytest.mark.asyncio
    async def test_exit_seals_turns_left_unratified(self, tmp_path: Path):
        """封尾的本职: 阈值没到、有已完成却没被追认的 turn → 退出时补一个锚点."""
        ego, memento = self._set_up(tmp_path, force_tokens=10 ** 9)
        await ego._on_turn_end(self._turn_end(1))
        assert memento.get_branch("main").commits() == []  # 阈值远未到 → 不落锚点

        await ego.__aexit__(None, None, None)

        commits = memento.get_branch("main").commits()
        assert len(commits) == 1
        ref = commits[0].metadata["ref"]
        assert (ref["start_turn"], ref["end_turn"]) == (0, 1)
        # 封尾是系统记录: 没有缓存的 final answer 时, 至少 title 说明这是 exit.
        assert memento.get_branch("main").notes()[commits[0].id].message == "exit: graceful shutdown"

    @pytest.mark.asyncio
    async def test_exit_adds_nothing_when_already_ratified(self, tmp_path: Path):
        """区间为空 (上一锚点就落在同一个 turn) → 不落空锚点, 也不白跑一次旁路."""
        ego, memento = self._set_up(tmp_path)
        await ego._on_turn_end(self._turn_end(1))
        await ego._on_turn_end(self._turn_end(2))
        before = len(memento.get_branch("main").commits())
        notices = len(ego._notices)

        await ego.__aexit__(None, None, None)

        assert len(memento.get_branch("main").commits()) == before
        assert len(ego._notices) == notices  # 也没有多出"已生成 commit"提醒

    @pytest.mark.asyncio
    async def test_abnormal_exit_does_not_commit(self, tmp_path: Path):
        ego, memento = self._set_up(tmp_path)
        await ego._on_turn_end(self._turn_end(1))
        committed = len(memento.get_branch("main").commits())

        await ego.__aexit__(RuntimeError, RuntimeError("boom"), None)

        # 异常退出不管: 锚点数不变.
        assert len(memento.get_branch("main").commits()) == committed

    @staticmethod
    def _assistant_message(turn: int, *, input_tokens: int, cache_read: int = 0, text: str = ""):
        from ghoshell_moss.deepseek_harness.v0_2_0 import event_from_dict

        message: dict = {"role": "assistant"}
        if text:
            message["content"] = [{"type": "text", "text": text}]
        return event_from_dict({
            "type": "assistant/message", "seq": turn * 10,
            "data": {
                "turn": turn, "step": 1, "message": message,
                "usage": {"inputTokens": input_tokens, "cacheReadTokens": cache_read},
            },
        })

    @pytest.mark.asyncio
    async def test_seed_is_a_baseline_not_uncommitted_conversation(self, tmp_path: Path):
        """首条 usage 是 seed 基线: 重启后的第一轮不该因 instruction+memory 触发锚点."""
        ego, memento = self._set_up(tmp_path, force_tokens=100, warn_tokens=50)
        await ego._on_assistant_message(self._assistant_message(1, input_tokens=10_000))
        await ego._on_turn_end(self._turn_end(1))

        assert ego._growth == 0
        assert memento.get_branch("main").commits() == []
        assert ego._notices == []

    @pytest.mark.asyncio
    async def test_usage_drives_warn_notice_once_per_window(self, tmp_path: Path):
        """增量达 K 排一次 warn notice (每窗口一次); 首条只建立基线, 不计入增长."""
        ego, memento = self._set_up(tmp_path, force_tokens=10 ** 9, warn_tokens=100)
        await ego._on_assistant_message(self._assistant_message(1, input_tokens=80, cache_read=70))
        assert ego._growth == 0  # seed 基线
        await ego._on_turn_end(self._turn_end(1))
        assert memento.get_branch("main").commits() == []  # 未到 T, 不落锚点
        warns = [n for n in ego._notices if n.meta.attributes.get("kind") == "warn"]
        assert warns == []

        # 窗口 200 → 增量 50, 未到 K.
        await ego._on_assistant_message(self._assistant_message(2, input_tokens=200))
        await ego._on_turn_end(self._turn_end(2))
        assert ego._growth == 50
        warns = [n for n in ego._notices if n.meta.attributes.get("kind") == "warn"]
        assert warns == []

        # 窗口 300 → 增量 150 ≥ K → 排一次; 同窗口再涨也只提醒一次.
        await ego._on_assistant_message(self._assistant_message(3, input_tokens=300))
        await ego._on_turn_end(self._turn_end(3))
        warns = [n for n in ego._notices if n.meta.attributes.get("kind") == "warn"]
        assert len(warns) == 1

        await ego._on_assistant_message(self._assistant_message(4, input_tokens=400))
        await ego._on_turn_end(self._turn_end(4))
        warns = [n for n in ego._notices if n.meta.attributes.get("kind") == "warn"]
        assert len(warns) == 1

    @pytest.mark.asyncio
    async def test_usage_drives_force_commit_and_resets_window(self, tmp_path: Path):
        """增量达 T 强制落锚; 提交后增量归零, 下一笔从零重新预算 (不是按绝对窗口)."""
        ego, memento = self._set_up(tmp_path, force_tokens=200, warn_tokens=100)
        await ego._on_assistant_message(self._assistant_message(1, input_tokens=250))
        assert ego._growth == 0  # 首条 = seed 基线

        # 窗口 500 → 增量 250 ≥ T → turn/end 强制.
        await ego._on_assistant_message(self._assistant_message(2, input_tokens=500))
        await ego._on_turn_end(self._turn_end(2))
        commits = memento.get_branch("main").commits()
        assert len(commits) == 1
        assert commits[0].metadata["ref"]["end_turn"] == 2
        assert ego._growth == 0  # 提交后归零

        # 窗口 600 → 增量 100, 不足 T; 预算看增量, 不看"绝对窗口还剩多少".
        await ego._on_assistant_message(self._assistant_message(3, input_tokens=600))
        await ego._on_turn_end(self._turn_end(3))
        assert len(memento.get_branch("main").commits()) == 1
        assert ego._growth == 100

        # 窗口 800 → 增量再 200 ≥ T, 区间接续 (2,4].
        await ego._on_assistant_message(self._assistant_message(4, input_tokens=800))
        await ego._on_turn_end(self._turn_end(4))
        commits = memento.get_branch("main").commits()
        assert len(commits) == 2
        assert (commits[1].metadata["ref"]["start_turn"], commits[1].metadata["ref"]["end_turn"]) == (2, 4)

    @pytest.mark.asyncio
    async def test_manual_commit_waits_for_the_turn_boundary(self, tmp_path: Path):
        """轮中登记的手动 commit 不立即落地 —— 等 turn/end, 边界才切得对."""
        ego, memento = self._set_up(tmp_path, force_tokens=10 ** 9)
        receipt = ego.commit_anchor("the note")
        assert "queued" in receipt
        assert memento.get_branch("main").commits() == []

        await ego._on_turn_end(self._turn_end(1))
        branch = memento.get_branch("main")
        commits = branch.commits()
        assert len(commits) == 1
        assert (commits[0].metadata["ref"]["start_turn"], commits[0].metadata["ref"]["end_turn"]) == (0, 1)
        assert branch.notes()[commits[0].id].message == "the note"

    @pytest.mark.asyncio
    async def test_second_manual_note_updates_the_pending_one(self, tmp_path: Path):
        ego, memento = self._set_up(tmp_path, force_tokens=10 ** 9)
        ego.commit_anchor("first")
        receipt = ego.commit_anchor("second")
        assert "updated" in receipt

        await ego._on_turn_end(self._turn_end(1))
        branch = memento.get_branch("main")
        assert branch.notes()[branch.commits()[0].id].message == "second"

    @pytest.mark.asyncio
    async def test_exit_note_is_a_system_record(self, tmp_path: Path):
        """优雅退出: title 是 exit op, body 是运行时缓存的 last final answer."""
        from ghoshell_moss.memento import CommitMessage

        ego, memento = self._set_up(tmp_path, force_tokens=10 ** 9)
        await ego._on_assistant_message(
            self._assistant_message(1, input_tokens=100, text="the last answer")
        )
        await ego._on_turn_end(self._turn_end(1))
        await ego.__aexit__(None, None, None)

        branch = memento.get_branch("main")
        commits = branch.commits()
        assert len(commits) == 1
        message = branch.notes()[commits[0].id].message
        assert message == "exit: graceful shutdown\n\nthe last answer"
        parsed = CommitMessage.parse(message)
        assert (parsed.op, parsed.content, parsed.body) == ("exit", "graceful shutdown", "the last answer")

    @pytest.mark.asyncio
    async def test_pending_manual_note_lands_at_exit(self, tmp_path: Path):
        ego, memento = self._set_up(tmp_path, force_tokens=10 ** 9)
        await ego._on_turn_end(self._turn_end(1))  # no threshold commit; the span stays open
        ego.commit_anchor("manual note")
        await ego.__aexit__(None, None, None)

        branch = memento.get_branch("main")
        commits = branch.commits()
        assert len(commits) == 1
        assert branch.notes()[commits[0].id].message == "manual note"

    @staticmethod
    def _request_header(turn: int, *, provider: str, model: str, effort: str = ""):
        from ghoshell_moss.deepseek_harness.v0_2_0 import event_from_dict

        config = {"provider": provider, "model": model}
        if effort:
            config["reasoningEffort"] = effort
        return event_from_dict({
            "type": "request/header", "seq": turn * 10,
            "data": {
                "header": {"config": config},
                "reason": "change",
            },
        })

    @pytest.mark.asyncio
    async def test_first_request_header_queues_identity_notice(self, tmp_path: Path):
        """首次观测必发: 思考档决定交互礼仪, 模型开局就得知道自己在哪一档."""
        ego, _ = self._set_up(tmp_path)
        assert ego._notices == []

        await ego._on_request_header(self._request_header(1, provider="deepseek", model="m", effort="off"))

        notices = [n for n in ego._notices if n.meta.tag == "model_notice"]
        assert len(notices) == 1
        text = notices[0].to_content_string()
        assert "deepseek" in text and "m" in text
        assert "off" in text
        assert "directly" in text  # off 的礼仪提示

    @pytest.mark.asyncio
    async def test_unchanged_header_queues_nothing(self, tmp_path: Path):
        """同一身份重复上报不重复排队."""
        ego, _ = self._set_up(tmp_path)
        await ego._on_request_header(self._request_header(1, provider="deepseek", model="m", effort="high"))
        await ego._on_request_header(self._request_header(2, provider="deepseek", model="m", effort="high"))

        assert len([n for n in ego._notices if n.meta.tag == "model_notice"]) == 1

    @pytest.mark.asyncio
    async def test_effort_change_queues_notice_naming_previous(self, tmp_path: Path):
        """思考档切换 → 再排一条, 并点明从哪一档变来 (礼仪随之切换)."""
        ego, _ = self._set_up(tmp_path)
        await ego._on_request_header(self._request_header(1, provider="deepseek", model="m", effort="high"))
        await ego._on_request_header(self._request_header(2, provider="deepseek", model="m", effort="max"))

        notices = [n for n in ego._notices if n.meta.tag == "model_notice"]
        assert len(notices) == 2
        text = notices[1].to_content_string()
        assert "high" in text and "max" in text
        assert "changed" in text

    @pytest.mark.asyncio
    async def test_identity_notice_drains_into_enter_payload(self, tmp_path: Path):
        """notice 经每帧 enter 排空 — 模型在下一轮思考里看到自己的身份."""
        ego, _ = self._set_up(tmp_path)
        await ego._on_request_header(self._request_header(1, provider="deepseek", model="m", effort="low"))

        drained = ego._drain_notices()

        assert len(drained) == 1
        assert "deepseek" in drained[0]
        assert ego._notices == []  # 排空后不残留


class TestTurnEndClassification:
    """turn/end → 要不要显式退出 thinking 的判定表 (纯函数, 不依赖替身).

    这是 cancel 治理的判据面: 自然收线绝不动 attention, 中断类才 abort —— 唯一例外是 cause 属于我们自己
    的 hook cancel (moss_react / moss_wait_next / moss_reasoning 自己收的轮), 身体在返回结果前已 drain.
    2026-10-04 D04 的教训: 判据曾经挂在 turn 号上, 于是每一个 react 轮都被算成"自己人", 人点的
    aborted/user 被一起豁免, 41.3 秒语音全量播完. 端到端行为 (语音是否真停) 归
    features/regressions/dolores-ghost-prototype TC-003, 不在单测里.
    """

    @staticmethod
    def _end(kind: str, turn: int = 7, cause=None):
        from ghoshell_moss.deepseek_harness.v0_2_0 import TurnEnd, TurnEndReason
        return TurnEnd(turn=turn, reason=TurnEndReason(kind=kind, reason=cause))

    @staticmethod
    def _cause(kind: str, reason: str = ""):
        from ghoshell_moss.deepseek_harness.v0_2_0 import AgentCancelCause
        return AgentCancelCause(kind=kind, reason=reason or None)

    def test_natural_ends_do_not_touch_the_body(self):
        from ghoshell_moss.ghosts.dolores._run import _turn_end_abort_reason
        assert _turn_end_abort_reason(self._end("completed")) is None
        assert _turn_end_abort_reason(self._end("max-tokens")) is None

    def test_unknown_kind_is_treated_as_natural(self):
        from ghoshell_moss.ghosts.dolores._run import _turn_end_abort_reason
        assert _turn_end_abort_reason(self._end("something-new-from-dsh")) is None

    def test_interruption_kinds_abort_with_a_reason(self):
        from ghoshell_moss.ghosts.dolores._run import _turn_end_abort_reason
        assert _turn_end_abort_reason(self._end("aborted")) == "turn/end aborted"
        assert _turn_end_abort_reason(self._end("interrupted")) == "turn/end interrupted"
        assert _turn_end_abort_reason(self._end("blocked")) == "turn/end blocked"
        assert _turn_end_abort_reason(self._end("error")) == "turn/end error"

    def test_cause_rides_along_when_dsh_provides_one(self):
        from ghoshell_moss.ghosts.dolores._run import _turn_end_abort_reason
        assert _turn_end_abort_reason(
            self._end("aborted", cause=self._cause("user"))
        ) == "turn/end aborted (user)"
        assert _turn_end_abort_reason(
            self._end("blocked", cause=self._cause("user"))
        ) == "turn/end blocked (user)"

    def test_our_own_hook_cancel_is_not_an_interruption(self):
        """cause=hook: moss_react / moss_wait_next / moss_reasoning 自己收的轮 — 不得 abort."""
        from ghoshell_moss.ghosts.dolores._run import _turn_end_abort_reason
        assert _turn_end_abort_reason(
            self._end("aborted", cause=self._cause("hook", "moss cancel_turn"))
        ) is None
        assert _turn_end_abort_reason(self._end("blocked", cause=self._cause("hook"))) is None

    def test_error_wins_over_a_hook_cause(self):
        """故障优先: 即便这轮也被我们标过收轮, 运行时错误仍然要 abort (否则自续成环)."""
        from ghoshell_moss.ghosts.dolores._run import _turn_end_abort_reason
        assert _turn_end_abort_reason(
            self._end("error", cause=self._cause("hook"))
        ) == "turn/end error (hook)"
