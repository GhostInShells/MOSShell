"""Mock Ghost 测试 — 验证预设替换、Ghost ABC 契约、MockArticulator 收集."""

import asyncio

import pytest
from ghoshell_container import Container

from ghoshell_moss.core.blueprint.ghost import Ghost, GhostMeta
from ghoshell_moss.core.blueprint.mindflow import (
    NucleusMeta,
    Nucleus,
    SignalMeta,
    Moment,
)
from ghoshell_moss.message import Message


# ── test stubs ──────────────────────────────────────


class _StubNucleusMeta(NucleusMeta):
    """测试用 NucleusMeta 桩."""

    def __init__(self, name: str = "stub"):
        self._name = name

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return "stub nucleus"

    def signals(self) -> list[SignalMeta]:
        return []

    def factory(self, container):
        raise NotImplementedError("stub")


# ── helpers ─────────────────────────────────────────


def _mock_meta(**kwargs):
    from ._meta import MockGhostMeta

    return MockGhostMeta(**kwargs)


def _mock_ghost(**kwargs):
    from ._meta import MockGhostMeta
    from ._runtime import MockGhost

    meta = MockGhostMeta()
    return MockGhost(meta=meta, **kwargs)


def _mock_articulator(**kwargs):
    from ._runtime import MockArticulator

    return MockArticulator(**kwargs)


# ── MockGhostMeta ───────────────────────────────────


class TestMockGhostMeta:
    def test_defaults(self):
        meta = _mock_meta()
        assert meta.name() == "mock"
        assert meta.description() == (
            "Mock ghost for testing GhostRuntime without real model calls."
        )
        assert meta.nuclei_metas() == []
        assert meta.providers() == []

    def test_custom_values(self):
        stub = _StubNucleusMeta(name="n1")
        meta = _mock_meta(
            name="test_ghost",
            description="a test ghost",
            nuclei_metas=[stub],
        )
        assert meta.name() == "test_ghost"
        assert meta.description() == "a test ghost"
        assert len(meta.nuclei_metas()) == 1
        assert meta.nuclei_metas()[0].name() == "n1"

    def test_setters_swap_at_runtime(self):
        meta = _mock_meta()
        assert meta.name() == "mock"

        meta._name = "swapped"
        assert meta.name() == "swapped"

        stub = _StubNucleusMeta(name="late")
        meta._nuclei_metas = [stub]
        assert len(meta.nuclei_metas()) == 1

    def test_factory_returns_mock_ghost(self):
        from ._runtime import MockGhost

        meta = _mock_meta()
        ghost = meta.factory(Container())
        assert isinstance(ghost, MockGhost)
        assert isinstance(ghost, Ghost)
        assert ghost.meta is meta

    def test_is_ghost_meta_abc(self):
        meta = _mock_meta()
        assert isinstance(meta, GhostMeta)


# ── MockGhost ───────────────────────────────────────


class TestMockGhost:
    def test_defaults(self):
        ghost = _mock_ghost()
        assert ghost.system_prompt() == ""
        assert ghost.memories() == []
        assert ghost.channel() is None
        assert ghost.mindflow() is None
        assert ghost._articulate_responses == []

    def test_setters_swap_at_runtime(self):
        ghost = _mock_ghost()
        ghost._system_prompt = "you are a ghost"
        ghost._memories = [Message(role="user", content="hello")]
        ghost._articulate_responses = ["cmd:1", "cmd:2"]

        assert ghost.system_prompt() == "you are a ghost"
        assert len(ghost.memories()) == 1
        assert ghost._articulate_responses == ["cmd:1", "cmd:2"]

    def test_is_ghost_abc(self):
        ghost = _mock_ghost()
        assert isinstance(ghost, Ghost)

    def test_lifecycle_no_error(self):
        ghost = _mock_ghost()

        async def run():
            async with ghost:
                pass

        asyncio.run(run())


# ── MockArticulator ─────────────────────────────────


class TestMockArticulator:
    def test_default_moment(self):
        art = _mock_articulator()
        assert isinstance(art.moment, Moment)
        assert art.sent == []
        assert art.aborted is None

    def test_preset_moment(self):
        m = Moment(prompt="test prompt")
        art = _mock_articulator(moment=m)
        assert art.moment.prompt == "test prompt"

    def test_send_nowait_collects(self):
        art = _mock_articulator()
        art.send_nowait("a")
        art.send_nowait("b")
        assert art.sent == ["a", "b"]

    def test_send_logos_collects(self):
        art = _mock_articulator()

        async def feed():
            async def logos():
                yield "x"
                yield "y"

            await art.send_logos(logos())

        asyncio.run(feed())
        assert art.sent == ["x", "y"]

    def test_abort_sets_flag(self):
        art = _mock_articulator()
        art.abort("something wrong")
        assert art.aborted == "something wrong"

    def test_create_task_returns_future(self):
        art = _mock_articulator()

        async def run():
            async def coro():
                return 42

            fut = art.create_task(coro())
            return await asyncio.wait_for(fut, timeout=1)

        result = asyncio.run(run())
        assert result == 42

    def test_flag_returns_event(self):
        art = _mock_articulator()
        ev = art.flag("test")
        assert isinstance(ev, asyncio.Event)
        assert not ev.is_set()
        ev.set()
        assert ev.is_set()

    def test_lifecycle(self):
        art = _mock_articulator()

        async def run():
            async with art:
                pass

        asyncio.run(run())

    def test_set_moment(self):
        art = _mock_articulator()
        new_m = Moment(prompt="updated")
        art.set_moment(new_m)
        assert art.moment.prompt == "updated"


# ── articulate ──────────────────────────────────────


class TestMockGhostArticulate:
    def test_yields_preset_responses_in_order(self):
        ghost = _mock_ghost()
        ghost._articulate_responses = ["hello", " world", "!"]
        art = _mock_articulator()

        async def collect():
            results = []
            async for delta in ghost.articulate(art):
                results.append(delta)
            return results

        results = asyncio.run(collect())
        assert results == ["hello", " world", "!"]

    def test_empty_responses_yields_nothing(self):
        ghost = _mock_ghost()
        art = _mock_articulator()

        async def collect():
            return [delta async for delta in ghost.articulate(art)]

        results = asyncio.run(collect())
        assert results == []

    def test_swap_between_calls_simulates_multi_turn(self):
        ghost = _mock_ghost()
        art = _mock_articulator()

        async def collect():
            return [delta async for delta in ghost.articulate(art)]

        # 第一轮
        ghost._articulate_responses = ["turn1_a", "turn1_b"]
        assert asyncio.run(collect()) == ["turn1_a", "turn1_b"]

        # 模拟第二轮的 CTML 响应被替换
        ghost._articulate_responses = ["turn2_x", "turn2_y", "turn2_z"]
        assert asyncio.run(collect()) == ["turn2_x", "turn2_y", "turn2_z"]

    def test_preset_is_not_consumed(self):
        """articulate() 不修改 articulate_responses，可重复调用获得相同结果."""
        ghost = _mock_ghost()
        ghost._articulate_responses = ["a", "b"]
        art = _mock_articulator()

        async def collect():
            return [delta async for delta in ghost.articulate(art)]

        assert asyncio.run(collect()) == ["a", "b"]
        assert asyncio.run(collect()) == ["a", "b"]
        assert ghost._articulate_responses == ["a", "b"]

    def test_yields_ctml_like_strings(self):
        """模拟真实 CTML 输出场景."""
        ghost = _mock_ghost()
        ghost._articulate_responses = [
            "<command token=",
            '"speak"',
            ' text="hello world"',
            "/>",
        ]
        art = _mock_articulator()

        async def collect():
            return [delta async for delta in ghost.articulate(art)]

        results = asyncio.run(collect())
        assert "".join(results) == '<command token="speak" text="hello world"/>'

    def test_articulate_with_send_nowait_mirrors_real_loop(self):
        """模拟 _articulate_loop 的真实行为: articulate → send_nowait."""
        ghost = _mock_ghost()
        ghost._articulate_responses = ["cmd1", "cmd2", "cmd3"]
        art = _mock_articulator()

        async def real_pattern():
            async for delta in ghost.articulate(art):
                art.send_nowait(delta)

        asyncio.run(real_pattern())
        assert art.sent == ["cmd1", "cmd2", "cmd3"]
