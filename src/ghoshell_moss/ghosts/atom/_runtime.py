from typing import AsyncIterator, TYPE_CHECKING
from typing_extensions import Self
from ghoshell_moss.core.blueprint.ghost import Ghost, GhostMeta
from ghoshell_moss.core.blueprint.channel_builder import Channel, ChannelFactory
from ghoshell_moss.core.blueprint.mindflow import Thinking, Moment
from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger
from ghoshell_container import IoCContainer
from ghoshell_moss.depends import depend_ghost

depend_ghost()
from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest

if TYPE_CHECKING:
    from ._meta import AtomMeta

__all__ = ["Atom"]


class Atom(Ghost):
    """Atom — 最小 Ghost 原型运行时，作为后续所有 Ghost 实现的参照基线.

    上下文由 mindflow 的 Moments 轨迹承载 — 历史从 observer 轨迹重建,
    动态消息只在当前帧最新一份 (不进历史). 已知不做的事（原型范围外）:
    - 上下文超额裁剪: 不做窗口限制，依赖模型自身的 context window
    - 持久化: 轨迹由 Moments 内存持有，重启即丢
    """

    def __init__(
        self,
        meta: "AtomMeta",
        agent: Agent[IoCContainer],
        container: IoCContainer,
        channel: Channel | ChannelFactory | None = None,
    ):
        self._meta = meta
        self._agent = agent
        self._container = container
        self._logger = container.get(LoggerItf) or get_moss_logger()
        self._channel = channel
        self._last_context: dict = {}

    @property
    def meta(self) -> GhostMeta:
        return self._meta

    def channel(self) -> Channel | ChannelFactory | None:
        """Ghost 反身性 channel — Atom 默认 None, echo 等实例可注入 (如 introspect)."""
        return self._channel

    def system_prompt(self) -> str:
        """调试用: 返回 Agent 实际使用的 instruction."""
        return self._meta.build_instruction_from_ioc(self._container)

    # ── 消息协议 ──────────────────────────────────

    def to_model_request(self, moment: Moment) -> ModelRequest:
        """将 Moment 转为 pydantic AI ModelRequest."""
        from ._adapter import moment_to_request
        return moment_to_request(moment)

    # ── 核心循环 ──────────────────────────────────

    def on_thinking_exit(self, thinking: Thinking, error: BaseException | None) -> None:
        self._last_context = {
            "system": self.system_prompt(),
            "history_moments": len(thinking.observer.moments()),
        }

    def inspect_context(self) -> dict:
        return self._last_context

    async def think(self, thinking: Thinking) -> AsyncIterator[str]:
        from ._adapter import moments_to_history

        moment = thinking.moment
        request = self.to_model_request(moment)
        history = moments_to_history(thinking.observer.moments())

        art = thinking.articulator()
        async with art:
            async with self._agent.run_stream(
                user_prompt=request.parts,
                message_history=history,
                deps=self._container,
            ) as stream:
                async for text in stream.stream_text(delta=True):
                    art.send_nowait(text)
                    yield text
            # articulate 自保证: 显式 wait action stop, 保证最后一帧观察落盘.
            if not thinking.is_aborted():
                await art.wait_action_done()

    # ── 生命周期 ──────────────────────────────────

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
