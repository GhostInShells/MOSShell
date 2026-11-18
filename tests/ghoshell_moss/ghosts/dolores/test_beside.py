"""``ghost.beside`` — parallel thought issued now, started at the step edge, delivered later.

Answers: what a ticket promises (a receipt at once, no model call before the edge), what the
group's delivery looks like (one message, one child per direction, failures as `<error>`),
and what the factory refuses (a disabled organ mounts nothing at all).
"""

from __future__ import annotations

import asyncio

import pytest

from ghoshell_moss.ghosts.dolores.mindset.beside import (
    BesideRuntime,
    EgoBesideMindConfig,
)
from ghoshell_moss.ghosts.dolores.mindset.ego_lifecycle import StepEndEvent
from ghoshell_moss.ghosts.dolores.mindset.ego_lifecycle_bus import EgoLifecycleBus


class _Session:
    """The narrow face beside uses: a session id and one off-record round per direction."""

    def __init__(self, replies: list[str] | None = None, *, error: Exception | None = None) -> None:
        self.session_id = "s1"
        self.replies = list(replies or [])
        self.error = error
        self.calls: list[tuple[list, object, dict]] = []

    async def chat_off_record(self, messages, *, anchor=None, **kwargs):
        from ghoshell_dsh.blueprint.agent import ChatReply

        self.calls.append((messages, anchor, kwargs))
        if self.error is not None:
            raise self.error
        item = self.replies.pop(0) if self.replies else "answer"
        if isinstance(item, dict):
            return ChatReply.model_validate(item)
        return ChatReply.model_validate({"text": item})


def _runtime(session, signals: list, *, config: EgoBesideMindConfig | None = None, lifecycle: EgoLifecycleBus):
    return BesideRuntime(
        lifecycle=lifecycle,
        session_source=lambda: session,
        send_signal=signals.append,
        config_source=lambda: config or EgoBesideMindConfig(),
    )


async def _drain(lifecycle: EgoLifecycleBus, runtime: BesideRuntime) -> None:
    """The bus task only spawns; the groups are the runtime's own — drain both, in that order."""
    await lifecycle.wait_all_done(timeout=1.0)
    await runtime.wait_all_done(timeout=1.0)


@pytest.mark.asyncio
async def test_think_returns_a_receipt_and_waits_for_the_step_edge():
    lifecycle = EgoLifecycleBus()
    session = _Session(["first", "second"])
    signals: list = []

    async with lifecycle:
        async with _runtime(session, signals, lifecycle=lifecycle) as runtime:
            receipt = runtime.think("one\n\ntwo")
            # The receipt is immediate; nothing is asked before the edge. It is plain text and
            # it carries the parsed count — the only place a mis-split becomes visible.
            assert receipt == "beside-think #1: 2 direction(s)"
            assert session.calls == []

            lifecycle.fire_step_end(StepEndEvent(turn=2, step=3, seq=30))
            await _drain(lifecycle, runtime)

    assert len(session.calls) == 2
    anchors = {call[1].seq for call in session.calls}
    assert anchors == {30}
    delivery = signals[0]
    assert delivery.name == "mind/delivery"
    text = delivery.messages[0].to_content_string()
    assert '<beside-think seq="1"' in text
    assert '<direction i="1">' in text and "first" in text
    assert '<direction i="2">' in text and "second" in text


@pytest.mark.asyncio
async def test_a_failed_direction_rides_the_same_delivery_as_an_error():
    lifecycle = EgoLifecycleBus()
    session = _Session(error=RuntimeError("model unavailable"))
    signals: list = []

    async with lifecycle:
        async with _runtime(session, signals, lifecycle=lifecycle) as runtime:
            runtime.think("only\n\ntwo")
            lifecycle.fire_step_end(StepEndEvent(turn=1, step=1, seq=7))
            await _drain(lifecycle, runtime)

    text = signals[0].messages[0].to_content_string()
    assert '<error i="1">' in text and "model unavailable" in text
    assert '<error i="2">' in text


@pytest.mark.asyncio
async def test_directions_are_capped_by_config():
    lifecycle = EgoLifecycleBus()
    session = _Session()
    signals: list = []

    async with lifecycle:
        async with _runtime(
                session, signals, config=EgoBesideMindConfig(directions_max=2), lifecycle=lifecycle,
        ) as runtime:
            receipt = runtime.think("a\n\nb\n\nc\n\nd")
            lifecycle.fire_step_end(StepEndEvent(turn=1, step=1, seq=1))
            await _drain(lifecycle, runtime)

    assert len(session.calls) == 2
    assert receipt == "beside-think #1: 2 direction(s), 2 dropped by the cap"


def test_think_refuses_an_empty_group():
    lifecycle = EgoLifecycleBus()
    runtime = _runtime(_Session(), [], lifecycle=lifecycle)

    with pytest.raises(ValueError):
        runtime.think(" \n\n\t\n\n ")


@pytest.mark.asyncio
async def test_no_live_session_is_reported_as_content_not_silence():
    lifecycle = EgoLifecycleBus()
    signals: list = []

    async with lifecycle:
        async with _runtime(None, signals, lifecycle=lifecycle) as runtime:
            runtime.think("a")
            lifecycle.fire_step_end(StepEndEvent(turn=1, step=1, seq=1))
            await _drain(lifecycle, runtime)

    text = signals[0].messages[0].to_content_string()
    assert '<error i="1">' in text


@pytest.mark.asyncio
async def test_exit_discards_the_step_listener():
    lifecycle = EgoLifecycleBus()
    session = _Session()
    signals: list = []

    async with lifecycle:
        async with _runtime(session, signals, lifecycle=lifecycle):
            pass
        lifecycle.fire_step_end(StepEndEvent(turn=1, step=1, seq=1))
        await asyncio.sleep(0)

    assert session.calls == []


@pytest.mark.asyncio
async def test_a_tool_call_is_content_not_silence():
    """A round that asks to run something said nothing in prose — that must not read as empty."""
    lifecycle = EgoLifecycleBus()
    session = _Session([{
        "text": "",
        "tool_calls": [{"id": "c1", "name": "matrix.nodes.status", "arguments": "{}"}],
        "finish_reason": "tool_calls",
    }])
    signals: list = []

    async with lifecycle:
        async with _runtime(session, signals, lifecycle=lifecycle) as runtime:
            runtime.think("look at the world")
            lifecycle.fire_step_end(StepEndEvent(turn=1, step=1, seq=1))
            await _drain(lifecycle, runtime)

    text = signals[0].messages[0].to_content_string()
    assert '<error i="1">' in text and "matrix.nodes.status" in text


@pytest.mark.asyncio
async def test_an_empty_reply_is_content_not_silence():
    lifecycle = EgoLifecycleBus()
    session = _Session([{"text": "   ", "finish_reason": "stop"}])
    signals: list = []

    async with lifecycle:
        async with _runtime(session, signals, lifecycle=lifecycle) as runtime:
            runtime.think("say nothing")
            lifecycle.fire_step_end(StepEndEvent(turn=1, step=1, seq=1))
            await _drain(lifecycle, runtime)

    text = signals[0].messages[0].to_content_string()
    assert '<error i="1">' in text and "empty" in text


@pytest.mark.asyncio
async def test_the_container_is_one_self_being_asked():
    """The reminder addresses a session of the same self: first person, and the task named."""
    lifecycle = EgoLifecycleBus()
    session = _Session()
    signals: list = []

    async with lifecycle:
        async with _runtime(session, signals, lifecycle=lifecycle) as runtime:
            runtime.think("what do I know")
            lifecycle.fire_step_end(StepEndEvent(turn=1, step=1, seq=1))
            await _drain(lifecycle, runtime)

    container = session.calls[0][0][0][0].text
    assert "THIS SESSION is one of my emanations" in container
    assert "The beside task: what do I know" in container
    assert "You are a concurrent" not in container


class TestBesideFactoryGate:
    """A disabled organ mounts nothing; an enabled one mounts the channel around the runtime."""

    class _Store:
        def __init__(self, config: EgoBesideMindConfig) -> None:
            self._config = config

        def get_or_create(self, conf_type):
            return self._config

    class _Container:
        def __init__(self, items: dict) -> None:
            self._items = items

        def get(self, contract):
            return self._items.get(contract)

    def _factory(self, *, enabled: bool, with_runtime: bool = True):
        from ghoshell_moss.ghosts.dolores.configs import DoloresConfigStore
        from ghoshell_moss.ghosts.dolores.mindset.beside.channel import EgoBesideChannel

        lifecycle = EgoLifecycleBus()
        store = self._Store(EgoBesideMindConfig(enabled=enabled))
        container = self._Container({
            DoloresConfigStore: store,
            BesideRuntime: _runtime(_Session(), [], lifecycle=lifecycle) if with_runtime else None,
        })
        return EgoBesideChannel.factory(container)

    def test_disabled_mounts_nothing(self):
        assert self._factory(enabled=False) is None

    def test_enabled_without_a_runtime_mounts_nothing(self):
        assert self._factory(enabled=True, with_runtime=False) is None

    def test_enabled_with_a_runtime_builds_the_channel(self):
        channel = self._factory(enabled=True)
        assert channel is not None
        assert channel.name() == "beside"


def test_a_blank_line_separates_directions():
    """The separator is a blank line — visible in the text itself, nothing to remember.

    The old `||` needed the caller to recall a token and to check that no direction contained
    it. A paragraph break is what the text already looks like, and a direction that wants two
    paragraphs is a brief, not a question.
    """
    lifecycle = EgoLifecycleBus()
    runtime = _runtime(_Session(), [], lifecycle=lifecycle)

    receipt = runtime.think("first question\n\nsecond question\n\nthird question")

    assert receipt == "beside-think #1: 3 direction(s)"


def test_the_old_separator_is_now_ordinary_text():
    """A stale `||` habit must not split a direction: the token is just text now."""
    lifecycle = EgoLifecycleBus()
    runtime = _runtime(_Session(), [], lifecycle=lifecycle)

    receipt = runtime.think("either this || that")

    assert receipt == "beside-think #1: 1 direction(s)"


def test_soft_line_breaks_stay_inside_one_direction():
    """A paragraph may wrap over several lines; only a blank line starts the next direction.

    This is a legal one-direction call, and the receipt is the only place its shape is visible.
    """
    lifecycle = EgoLifecycleBus()
    runtime = _runtime(_Session(), [], lifecycle=lifecycle)

    receipt = runtime.think("第一段\n第二段\n第三段")

    assert receipt == "beside-think #1: 1 direction(s)"


def test_the_doc_renders_the_configured_cap():
    """The cap is config, and the doc is where it is read before a call — so the number must
    come from the config, never from a literal in the docstring."""
    from ghoshell_moss.ghosts.dolores.mindset.beside.channel import build_beside_channel

    lifecycle = EgoLifecycleBus()
    runtime = _runtime(
        _Session(), [], config=EgoBesideMindConfig(directions_max=7), lifecycle=lifecycle,
    )

    command = build_beside_channel(runtime).build.get_own_command("think")

    assert command is not None
    assert "7 at max" in command.meta().interface
