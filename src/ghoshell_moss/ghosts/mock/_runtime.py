import asyncio
from typing import AsyncIterator, Coroutine
from typing_extensions import Self

from ghoshell_moss.core.blueprint.ghost import Ghost, GhostMeta
from ghoshell_moss.core.blueprint.mindflow import Articulator, Mindflow, Moment, Logos
from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.message import Message

__all__ = ["MockGhost", "MockArticulator"]


class MockGhost(Ghost):
    """Mock Ghost — 所有返回值可随时替换，用于测试 GhostRuntime 体系.

    articulate_responses 是核心预设点：每次 articulate() 调用按序 yield 列表中的字符串，
    模拟模型的流式 CTML 输出。测试中可随时替换列表模拟多轮对话。
    """

    def __init__(self, *, meta: GhostMeta):
        self._meta = meta
        self._articulate_responses: list[str] = ["hello world"]
        self._system_prompt: str = ""
        self._memories: list[Message] = []
        self._channel: Channel | None = None
        self._mindflow: Mindflow | None = None

    # ── Ghost ABC ──────────────────────────────────

    @property
    def meta(self) -> GhostMeta:
        return self._meta

    def system_prompt(self) -> str:
        return self._system_prompt

    def memories(self) -> list[Message]:
        return list(self._memories)

    def channel(self) -> Channel | None:
        return self._channel

    def mindflow(self) -> Mindflow | None:
        return self._mindflow

    async def articulate(self, articulator: Articulator) -> AsyncIterator[str]:
        for response in self._articulate_responses:
            yield response

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    # ── observability hooks ──────────────────────────

    def on_articulate_exit(self, articulator, logos, error):
        self._last_logos = logos
        self._last_error = error
        self._last_context = articulator.moment.to_dict()

    def inspect_state(self) -> dict:
        return {
            "articulate_responses_remaining": len(self._articulate_responses),
            "last_logos": getattr(self, "_last_logos", None),
            "last_error": str(getattr(self, "_last_error", None)),
        }

    def inspect_context(self) -> dict:
        return getattr(self, "_last_context", {})

    # ── helpers ─────────────────────────────────────

    def set_articulate_responses(self, responses: list[str]) -> None:
        self._articulate_responses = responses

    def set_system_prompt(self, prompt: str) -> None:
        self._system_prompt = prompt

    def set_memories(self, memories: list[Message]) -> None:
        self._memories = memories

    def set_channel(self, channel: Channel | None) -> None:
        self._channel = channel

    def set_mindflow(self, mindflow: Mindflow | None) -> None:
        self._mindflow = mindflow


class MockArticulator(Articulator):
    """Mock Articulator — 通用测试桩，收集 send_nowait() 调用并暴露可替换的 moment.

    使用方式:
        art = MockArticulator(moment=Moment())
        async for delta in ghost.articulate(art):
            art.send_nowait(delta)
        assert art.sent == ["expected", "deltas"]
    """

    def __init__(self, moment: Moment | None = None):
        self._moment = moment or Moment()
        self._sent: list[str] = []
        self._aborted: str | None = None

    # ── Articulator ABC ────────────────────────────

    @property
    def moment(self) -> Moment:
        return self._moment

    def send_nowait(self, logos_delta: str) -> None:
        self._sent.append(logos_delta)

    async def send_logos(self, logos: Logos) -> None:
        async for delta in logos:
            self._sent.append(delta)

    def abort(self, error: str | Exception | None) -> None:
        self._aborted = str(error) if error else "aborted"

    def create_task(self, cor: Coroutine) -> asyncio.Future:
        return asyncio.ensure_future(cor)

    def flag(self, name: str) -> asyncio.Event:
        return asyncio.Event()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    # ── test helpers ────────────────────────────────

    @property
    def sent(self) -> list[str]:
        return list(self._sent)

    @property
    def aborted(self) -> str | None:
        return self._aborted

    def set_moment(self, moment: Moment) -> None:
        self._moment = moment
