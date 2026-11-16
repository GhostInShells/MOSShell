"""EgoLifecycleBus — registration, blocking/non-blocking dispatch, isolation, drain."""

from __future__ import annotations

import asyncio

import pytest

from ghoshell_moss.ghosts.dolores.mindset.ego_lifecycle import (
    CommitNoteEvent,
    StartupEvent,
    StepEndEvent,
    ThinkingStartEvent,
)
from ghoshell_moss.ghosts.dolores.mindset.ego_lifecycle_bus import EgoLifecycleBus


@pytest.mark.asyncio
async def test_startup_is_blocking():
    seen: list[str] = []

    async with EgoLifecycleBus() as bus:
        async def listener(event: StartupEvent) -> None:
            await asyncio.sleep(0.01)
            seen.append(event.mode)

        bus.on_startup(listener)
        await bus.fire_startup(StartupEvent(mode="develop"))

    assert seen == ["develop"]


@pytest.mark.asyncio
async def test_thinking_start_is_blocking():
    seen: list[int] = []

    async with EgoLifecycleBus() as bus:
        async def listener(event: ThinkingStartEvent) -> None:
            await asyncio.sleep(0.01)
            seen.append(event.seq)

        bus.on_thinking_start(listener)
        await bus.fire_thinking_start(ThinkingStartEvent(seq=7))

    assert seen == [7]


@pytest.mark.asyncio
async def test_commit_note_is_non_blocking_until_drained():
    seen: list[str] = []
    gate = asyncio.Event()

    async with EgoLifecycleBus() as bus:
        async def listener(event: CommitNoteEvent) -> None:
            await gate.wait()
            seen.append(event.commit_id)

        bus.on_commit_note(listener)
        bus.fire_commit_note(CommitNoteEvent(commit_id="1-1", note="n"))
        assert seen == []
        gate.set()
        await bus.wait_all_done(timeout=1.0)

    assert seen == ["1-1"]


@pytest.mark.asyncio
async def test_discard_removes_listener_and_is_idempotent():
    seen: list[str] = []

    async with EgoLifecycleBus() as bus:
        async def listener(event: StartupEvent) -> None:
            seen.append(event.mode)

        discard = bus.on_startup(listener)
        discard()
        discard()
        await bus.fire_startup(StartupEvent(mode="x"))

    assert seen == []


@pytest.mark.asyncio
async def test_listener_exception_is_isolated():
    seen: list[str] = []

    async with EgoLifecycleBus() as bus:
        async def bad(event: StartupEvent) -> None:
            raise RuntimeError("boom")

        async def good(event: StartupEvent) -> None:
            seen.append(event.mode)

        bus.on_startup(bad)
        bus.on_startup(good)
        await bus.fire_startup(StartupEvent(mode="ok"))

    assert seen == ["ok"]


@pytest.mark.asyncio
async def test_exit_cancels_running_tasks():
    cancelled = asyncio.Event()

    async with EgoLifecycleBus() as bus:
        async def listener(event: CommitNoteEvent) -> None:
            try:
                await asyncio.sleep(10)
            finally:
                cancelled.set()

        bus.on_commit_note(listener)
        bus.fire_commit_note(CommitNoteEvent(commit_id="1-1", note="n"))
        await asyncio.sleep(0)

    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_fire_after_close_is_dropped():
    seen: list[str] = []

    bus = EgoLifecycleBus()
    async with bus:
        async def listener(event: CommitNoteEvent) -> None:
            seen.append(event.commit_id)

        bus.on_commit_note(listener)

    bus.fire_commit_note(CommitNoteEvent(commit_id="1-2", note="n"))
    await asyncio.sleep(0)
    assert seen == []


@pytest.mark.asyncio
async def test_step_end_is_non_blocking_until_drained():
    """The session event dispatcher awaits handlers serially — a listener must never stall it."""
    seen: list[StepEndEvent] = []
    gate = asyncio.Event()

    async with EgoLifecycleBus() as bus:
        async def listener(event: StepEndEvent) -> None:
            await gate.wait()
            seen.append(event)

        bus.on_step_end(listener)
        bus.fire_step_end(StepEndEvent(turn=1, step=2, seq=9))
        assert seen == []
        gate.set()
        await bus.wait_all_done(timeout=1.0)

    assert seen == [StepEndEvent(turn=1, step=2, seq=9)]
