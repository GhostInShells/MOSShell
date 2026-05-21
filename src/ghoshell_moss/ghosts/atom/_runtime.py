from typing import AsyncIterator, TYPE_CHECKING
from typing_extensions import Self
from ghoshell_moss.core.blueprint.ghost import Ghost, GhostMeta
from ghoshell_moss.core.blueprint.mindflow import Articulator, Moment, Reaction
from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger
from ghoshell_moss.message import Message
from ghoshell_container import IoCContainer
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse

if TYPE_CHECKING:
    from ._meta import AtomMeta

__all__ = ["Atom"]


class Atom(Ghost):
    """Atom — 最小 Ghost 原型运行时，作为后续所有 Ghost 实现的参照基线.

    已知不做的事（原型范围外）:
    - 上下文超额裁剪: model_history() 不做窗口限制，依赖模型自身的 context window
    - 持久化: 纯内存历史，重启即丢
    """

    def __init__(
        self,
        meta: "AtomMeta",
        agent: Agent[IoCContainer],
        container: IoCContainer,
    ):
        self._meta = meta
        self._agent = agent
        self._container = container
        self._logger = container.get(LoggerItf) or get_moss_logger()
        self._history: list[ModelMessage] = []
        self._last_context: dict = {}

    @property
    def meta(self) -> GhostMeta:
        return self._meta

    def system_prompt(self) -> str:
        """调试用: 返回 Agent 实际使用的 instruction."""
        return self._meta._build_instruction(self._container)

    # ── 消息协议 ──────────────────────────────────

    def to_model_request(self, moment: Moment) -> ModelRequest:
        """将 Moment 转为 pydantic AI ModelRequest."""
        from ._adapter import moment_to_request
        return moment_to_request(moment)

    def model_history(self) -> list[ModelMessage]:
        """返回当前内存中的对话历史.

        TODO: 不做窗口裁剪，长对话会超出模型 context window.
        """
        return list(self._history)

    def save_model_request(
        self, moment: Moment, response: ModelResponse
    ) -> None:
        """保存本轮交换到内存历史."""
        self._history.append(self.to_model_request(moment))
        self._history.append(response)

    # ── 核心循环 ──────────────────────────────────

    def on_articulate_exit(self, articulator, logos, error) -> None:
        self._last_context = {
            "system": self.system_prompt(),
            "history_turns": len(self._history) // 2,
        }

    def inspect_context(self) -> dict:
        return self._last_context

    async def articulate(self, articulator: Articulator) -> AsyncIterator[str]:
        moment = articulator.moment
        request = self.to_model_request(moment)
        history = self.model_history()
        model_context = history + [request]

        async with self._agent.run_stream(
            message_history=model_context,
            deps=self._container,
        ) as stream:
            async for text in stream.stream_text(delta=True):
                yield text
            self.save_model_request(moment, stream.response)

    # ── 生命周期 ──────────────────────────────────

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
