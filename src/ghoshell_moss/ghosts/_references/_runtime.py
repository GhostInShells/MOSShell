from typing import TYPE_CHECKING, AsyncGenerator, AsyncIterable, AsyncIterator
from typing_extensions import Self
from ghoshell_moss.core.blueprint.ghost import Ghost, GhostMeta
from ghoshell_moss.core.blueprint.mindflow import Logos, Articulator, Moment
from ghoshell_container import IoCContainer
from pydantic_ai import Agent, ModelMessage, ModelRequest, ModelResponse, RequestUsage

if TYPE_CHECKING:
    from ._meta import AtomMeta


class Atom(Ghost):

    def __init__(
            self,
            meta: "AtomMeta",
            pydantic_agent: Agent[IoCContainer],
            container: IoCContainer,
    ):
        self._meta = meta
        self._agent: Agent[IoCContainer] = pydantic_agent
        self._container = container

    @property
    def meta(self) -> GhostMeta:
        return self._meta

    @property
    def agent(self) -> Agent[IoCContainer]:
        return self._agent

    def system_prompt(self) -> str:
        pass

    def to_model_request(self, moment: Moment) -> ModelRequest:
        pass

    def model_history(self) -> list[ModelRequest | ModelResponse]:
        pass

    def save_model_request(self, moment: Moment, response: ModelResponse, usage: RequestUsage) -> None:
        # 索引立刻生效, 同时触发异步任务检查是否要事后压缩.
        # 事后压缩的基础逻辑应该是立刻截断, 然后异步补齐摘要. 这样下一轮 usage 预期会 < n + m (摘要)
        # 如果摘要在执行完毕前完成, 则记忆仍然保持连贯. 否则有 "发呆" 的可能性.
        # 不影响请求前的压缩逻辑?
        pass

    async def articulate(self, articulator: Articulator) -> AsyncIterator[str]:
        moment = articulator.moment
        request = self.to_model_request(moment)
        history = list(self.model_history()) + [request]
        async with self._agent.run_stream(message_history=history, deps=self._container) as stream_result:
            try:
                async for text in stream_result.stream_text(delta=True, debounce_by=None):
                    yield text
            except ValueError:
                # todo
                pass
            finally:
                model_response = stream_result.response
                usage = stream_result.response.usage
                self.save_model_request(moment, model_response, usage)

    async def __aenter__(self) -> Self:
        pass

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
