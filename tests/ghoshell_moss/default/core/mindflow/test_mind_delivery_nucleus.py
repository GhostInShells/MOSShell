"""MindDeliveryNucleus — buffer dozens of beside-path results, deliver them together.

Protocol under test:
- every buffered impulse is ``role='mind'`` and ``mode=next``;
- ``peek`` returns the highest priority without consuming;
- ``attended`` merges every buffered message into one carrier;
- ``suppress`` keeps the buffer for the next re-rank;
- ``clear`` is explicit.
"""

import pytest

from ghoshell_moss.core.blueprint.mindflow import ChallengeMode, Priority
from ghoshell_moss.core.mindflow.mind_delivery_nucleus import (
    MindDeliveryNucleus,
    MindDeliveryNucleusMeta,
    MindDeliverySignalMeta,
    new_mind_delivery_signal,
)


def test_signal_meta_name_and_default_priority():
    assert MindDeliverySignalMeta.signal_name() == "mind/delivery"
    assert MindDeliverySignalMeta().to_signal().priority == Priority.NOTICE


def test_meta_exposes_the_signal_type():
    assert list(MindDeliveryNucleusMeta().signals()) == [MindDeliverySignalMeta]


@pytest.mark.asyncio
async def test_add_signal_buffers_role_mind_mode_next():
    nucleus = MindDeliveryNucleus()
    async with nucleus:
        fired = []
        nucleus.with_bus(lambda signal: None, fired.append)
        nucleus.add_signal(new_mind_delivery_signal("first", kind="test"))

        assert len(fired) == 1
        impulse = nucleus.peek()
        assert impulse is not None
        assert impulse.role == "mind"
        assert impulse.mode == ChallengeMode.next.value
        assert [m.to_content_string() for m in impulse.messages] == ["first"]


@pytest.mark.asyncio
async def test_peek_returns_highest_without_consuming():
    nucleus = MindDeliveryNucleus()
    async with nucleus:
        nucleus.with_bus(lambda signal: None, lambda impulse: None)
        nucleus.add_signal(new_mind_delivery_signal("low", priority=Priority.INFO))
        nucleus.add_signal(new_mind_delivery_signal("high", priority=Priority.WARNING))

        first = nucleus.peek()
        second = nucleus.peek()

        assert first is not None
        assert second is not None
        assert [m.to_content_string() for m in first.messages] == ["high"]
        assert first is second


@pytest.mark.asyncio
async def test_attended_merges_all_buffered_messages():
    nucleus = MindDeliveryNucleus()
    async with nucleus:
        nucleus.with_bus(lambda signal: None, lambda impulse: None)
        nucleus.add_signal(new_mind_delivery_signal("one", priority=Priority.INFO))
        nucleus.add_signal(new_mind_delivery_signal("two", priority=Priority.NOTICE))

        carrier = nucleus.peek()
        assert carrier is not None
        merged = nucleus.attended(carrier)

        assert merged is not None
        assert [m.to_content_string() for m in merged.messages] == ["one", "two"]
        assert merged.role == "mind"
        assert merged.mode == ChallengeMode.next.value
        assert merged.priority == Priority.NOTICE
        assert nucleus.peek() is None


@pytest.mark.asyncio
async def test_suppress_keeps_the_buffer_for_rerank():
    nucleus = MindDeliveryNucleus()
    async with nucleus:
        nucleus.with_bus(lambda signal: None, lambda impulse: None)
        nucleus.add_signal(new_mind_delivery_signal("kept", priority=Priority.INFO))

        carrier = nucleus.peek()
        assert carrier is not None
        nucleus.suppress(carrier, carrier)

        again = nucleus.peek()
        assert again is not None
        assert [m.to_content_string() for m in again.messages] == ["kept"]


@pytest.mark.asyncio
async def test_clear_empties_the_buffer():
    nucleus = MindDeliveryNucleus()
    async with nucleus:
        nucleus.with_bus(lambda signal: None, lambda impulse: None)
        nucleus.add_signal(new_mind_delivery_signal("gone"))
        nucleus.clear()
        assert nucleus.peek() is None


@pytest.mark.asyncio
async def test_wrong_signal_name_is_ignored():
    nucleus = MindDeliveryNucleus()
    async with nucleus:
        nucleus.with_bus(lambda signal: None, lambda impulse: None)
        wrong = MindDeliverySignalMeta().to_signal("nope")
        wrong.name = "not/mind/delivery"
        nucleus.add_signal(wrong)
        assert nucleus.peek() is None


@pytest.mark.asyncio
async def test_not_running_does_not_buffer():
    nucleus = MindDeliveryNucleus()
    nucleus.add_signal(new_mind_delivery_signal("before start"))
    assert nucleus.peek() is None
