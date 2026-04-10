import asyncio
from typing import Callable, Coroutine
from typing_extensions import Self
from rich.console import RenderableType
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer
from ghoshell_moss.host.abcd import IHost, IRuntime, ConversationItem
from ghoshell_moss.host.abcd.topics import OutputTopic
import typer
import janus


class TyperCompleter(Completer):
    """
    实现一个基于 typer 的提示体系.
    """
    pass


# 实例化一套工具提示.
app = typer.Typer()


@app.command()
def typer_command_example():
    repl = MOSSRepl.get()

    # 定义闭包.
    async def operator():
        """
        注册
        """
        moss = repl.moss
        # 做一些操作.
        # 然后发送渲染对象.
        repl.output()

    # 发送进链路.
    repl.operate(operator)


class MOSSRepl:
    """
    moss 的 repl 体系.
    """

    def __init__(self, runtime: IRuntime) -> None:
        self.moss = runtime
        self._operator_queue: janus.Queue[MossClosure] = janus.Queue()
        self._output_queue: janus.Queue[ConversationItem] = janus.Queue()
        self._renderer_queue: janus.Queue[RenderableType] = janus.Queue()

    @classmethod
    def get(cls) -> Self:
        """获取进程级别单例."""
        pass

    def output(self, renderable: RenderableType) -> None:
        """
        将一个规划要渲染的对象, 塞入 output 队列.
        没想好是用 rich, 还是放入 ConditionContainer.
        """
        self._renderer_queue.sync_q.put(renderable)

    def operate(self, operator: "MossClosure") -> None:
        self._operator_queue.sync_q.put(operator)

    async def _output_loop(self) -> None:
        subscriber = self.moss.matrix.topics.subscribe_model(OutputTopic)
        async with subscriber:
            while self.moss.is_running():
                topic = await subscriber.poll_model()
                renderable = self._wrap_output_to_renderable(topic.item)
                self.output(renderable)

    def _wrap_output_to_renderable(self, item: ConversationItem) -> RenderableType:
        pass

    async def _moss_runtime_main_loop(self) -> None:
        """
        在这里循环执行 moss runtime.
        """
        operation: asyncio.Task | None = None
        loop = asyncio.get_running_loop()
        async with self.moss as moss:
            while self.moss.is_running():
                operator: MossClosure = await self._operator_queue.async_q.get()

                if operation is not None and not operation.done():
                    operation.cancel()
                    try:
                        await operation
                    except asyncio.CancelledError:
                        pass
                operation = loop.create_task(operator())

    async def _repl_prompt_loop(self) -> None:
        """
        基于 prompt session + completer, 让用户可以异步输入指令, 解析成 operator 执行.
        """
        pass

    def run(self) -> None:
        """
        运行.
        """
        pass


MossClosure = Callable[[], Coroutine[None, None, None]]
