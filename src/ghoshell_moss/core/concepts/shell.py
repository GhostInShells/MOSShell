import asyncio
import contextlib
from abc import ABC, abstractmethod
from typing import Literal, Optional, AsyncIterable, AsyncIterator
from ghoshell_container import IoCContainer

from ghoshell_moss.core.concepts.channel import Channel, ChannelFullPath, ChannelMeta, MutableChannel, ChannelRuntime
from ghoshell_moss.core.concepts.command import Command, CommandTask, CommandToken
from ghoshell_moss.core.concepts.states import StateStore
from ghoshell_moss.core.concepts.interpreter import Interpreter, Interpretation
from ghoshell_moss.core.concepts.topic import Topic, TopicModel, Subscriber, TOPIC_MODEL, SubscribeKeep, TopicService
from ghoshell_moss.message import Message

__all__ = [
    "InterpreterKind",
    "MOSShell",
]

InterpreterKind = Literal["clear", "append", "dry_run"]


class MOSShell(ABC):
    """
    Model-Operated Operating System Shell
    面向模型提供的 Shell, 让 AI 可以操作自身所处的系统.

    这个技术实现的核心目标, 是通过一个双工运行的 Runtime, 为一个持久化智能体提供 Realtime 感知, 交互和控制能力. 以及提供几乎无限的反身性.

    Shell 设计的全双工交互的极简形式:

    创建一个 Shell 实例.
    >>> def create_shell(...) -> MOSShell:
    >>>     ...

    为 Shell 赋予各种 Channel, 其中一些 Channel 是可以有 安装/卸载/打开/关闭 范式的.

    >>> def build_shell(shell: MOSShell, channels: list[Channel]) -> MOSShell:
    >>>     shell.main_channel.import_channels(*channels)
    >>>     return shell

    在这个 Channels 的体系中应该要包含一个完整的 AIOS 范式, 包含:
    + Instructions: AI 自身 instructions 模块的修改.
    + Memories: AI 的记忆体系
    + Mind: 思维管理控制
        - Skills: AI 通过 Skill 管理的注意力机制, 可以专注于做不同的任务.
        - TasksManager:  AI 的多任务管理, 支持树形嵌套, 可以在多个 Tasks 中切换, 并且可以为 task 维护独立上下文.
    + Tools: 可以用的各种工具.
        + Desktops:  AI 自己拥有的桌面软件, 操作它所在的操作系统.
            - Apps: AI 可以管理的本地应用, 每个应用拥有独立的 Runtime.
        - Terminal: AI 可以直接操作和修改的命令行.
        + Assets: AI 可以管理的各种本地资源.
        - Modules: AI 可以在自己的 Runtime 里管理所有可被调用的 python 模块.
    + LAN: 局域网里可以使用的各种工具.
        + HomeAssistant: 智能家居
        + AI Assistants: 可以对话的各种 AI
    + Sencors: 所有可被调用的感知模块.
    + UserInterfaces: 可以和人类交互的各种界面.
    + Bodies: 可以控制的各种物理躯体.

    然后 Shell 运行可以通过 Topic 来进行通讯, 用 CSP 范式来创建持久运行 Agent 逻辑:

    >>> async def main_shell_loop(shell: MOSShell) -> None:
    >>>
    >>>     async def model_create_response() -> AsyncIterable[str]:
    >>>         "模型创建回复的逻辑"
    >>>         ...
    >>>
    >>>     async def receive_input_topic_loop():
    >>>         "持续获取输入消息, 并且消费输入"
    >>>         async with shell.subscribe_topic('input/messages') as subscriber:
    >>>             message = await subscriber.poll()
    >>>             ...  # 解析执行 topic, 发送后续的执行 topic
    >>>
    >>>     async def run_agent_loop():
    >>>         "持续响应 agent 的事件"
    >>>         async with shell.subscribe_topic('agent/event') as subscriber:
    >>>             event = await subscriber.poll()
    >>>             ...  # 解析 event, 确认响应逻辑
    >>>             i: Interpreter = await shell.interpreter('clear')
    >>>             # 获得运行结果.
    >>>             interpretation = i.interpretation()
    >>>             # 使用关键帧生成的解释器, 完成上下文响应.
    >>>             async with interpreter:
    >>>                 # 来执行模型生成.
    >>>                 async for token in model_create_response():
    >>>                     i.feed(token)
    >>>                 i.commit()
    >>>                 ... # 等待 interpreter 结果并执行.
    >>>                 interpretation = await i.wait_stopped()
    >>>
    >>>     # 启动 Shell
    >>>     async with shell:
    >>>           # 执行这些 loop, 直到关键点结束.
    >>>           await asyncio.gather(receive_input_topic_loop(), run_agent_loop())

    在 Shell 能够持续, 稳定运行的情况下, AI (Ghost) 运行在 Shell 中, 持续地与现实世界交互.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def container(self) -> IoCContainer:
        pass

    @property
    @abstractmethod
    def states(self) -> StateStore:
        pass

    @abstractmethod
    def topics(self) -> TopicService:
        pass

    @abstractmethod
    async def pub_topic(
        self,
        topic: Topic | TopicModel,
        *,
        name: str = "",
    ) -> None:
        """
        shell 广播 topic
        """
        pass

    @abstractmethod
    def subscribe_topic_model(
        self,
        model: type[TOPIC_MODEL],
        *,
        name: str = "",
        maxsize: int = 0,
        keep: SubscribeKeep = "latest",
    ) -> Subscriber[TOPIC_MODEL]:
        """
        shell 层监听 topic.
        """
        pass

    @abstractmethod
    def subscribe_topic(
        self,
        name: str,
        *,
        maxsize: int = 0,
        keep: SubscribeKeep = "latest",
    ) -> Subscriber:
        pass

    # --- channels --- #

    @property
    @abstractmethod
    def main_channel(self) -> MutableChannel:
        """
        Shell 自身的主轨. 主轨同时可以用来注册所有的子轨.
        主轨的名称必须是空字符串.
        定位类似于 python 的 __main__ 模块.
        """
        pass

    @property
    @abstractmethod
    def runtime(self) -> ChannelRuntime:
        pass

    # --- runtime methods --- #

    @abstractmethod
    def is_running(self) -> bool:
        """
        shell 是否在运行中.
        """
        pass

    @abstractmethod
    async def wait_connected(self, *channel_paths: str) -> None:
        """
        强行等待指定的轨道, 或者所有的轨道完成连接.
        通常并不是必要的. 只是为了测试.
        """
        pass

    @abstractmethod
    def is_closed(self) -> bool:
        """
        是否已经关闭运行.
        """
        pass

    @abstractmethod
    def is_idle(self) -> bool:
        """
        是否在闲置状态. 闲置状态指的是没有任何 command 在运行.
        """
        pass

    @abstractmethod
    async def wait_until_idle(self, timeout: float | None = None) -> None:
        """
        等待到 shell 所有的 command 运行结束.
        todo: 应该可以指定某个具体的 channel.
        """
        pass

    @abstractmethod
    async def wait_until_closed(self) -> None:
        """
        阻塞等到 Shell 被关闭.
        """
        pass

    @abstractmethod
    def commands(
        self, available_only: bool = True, *, config: dict[ChannelFullPath, ChannelMeta] | None = None
    ) -> dict[ChannelFullPath, dict[str, Command]]:
        """
        当前运行时所有的可用的命令.
        注意, key 是 channel path. 例如 foo.bar:baz 表示 command 来自 channel `foo.bar`, 名称是 'baz'
        """
        pass

    @abstractmethod
    def channel_metas(
        self,
        available: bool = True,
    ) -> dict[ChannelFullPath, ChannelMeta]:
        """
        当前运行状态中的 Channel meta 信息.
        key 是 channel path, 例如 foo.bar
        如果为 '', 表示为主 channel.
        """
        pass

    @abstractmethod
    def meta_instruction(self) -> str:
        pass

    @abstractmethod
    def channel_instructions(self) -> dict[str, list[Message]]:
        pass

    @abstractmethod
    def channel_context_messages(self) -> dict[str, list[Message]]:
        """
        context messages of all the channels.
        """
        pass

    @abstractmethod
    async def get_command(self, chan: str, name: str, /, exec_in_chan: bool = False) -> Optional[Command]:
        """
        获取一个可以运行的 channel command.
        这个语法可以理解为 from channel_path import command_name

        :param chan: channel 的 path, 例如 foo.bar
        :param name: command name
        :param exec_in_chan: 表示这个 command 在像函数一样调用时, 仍然会发送 command task 到 channel 中.
        :return: None 表示命令不存在.
        """
        pass

    # --- interpret --- #

    @abstractmethod
    def interpreting(self) -> Optional[Interpreter]:
        pass

    @contextlib.asynccontextmanager
    async def interpreter_in_ctx(
        self,
        kind: InterpreterKind = "clear",
        *,
        meta_instruction: str | None = None,
        stream_id: Optional[str] = None,
        config: Optional[dict[ChannelFullPath, ChannelMeta]] = None,
        clear_after_exit: bool = False,
        ignore_wrong_command: bool = False,
    ) -> AsyncIterator[Interpreter]:
        """
        简单的语法糖.
        """
        interpreter = await self.interpreter(
            kind=kind,
            meta_instruction=meta_instruction,
            stream_id=stream_id,
            config=config,
            clear_after_exit=clear_after_exit,
            ignore_wrong_command=ignore_wrong_command,
        )
        async with interpreter:
            yield interpreter

    @abstractmethod
    async def interpreter(
        self,
        kind: InterpreterKind = "clear",
        *,
        meta_instruction: str | None = None,
        stream_id: Optional[str] = None,
        config: Optional[dict[ChannelFullPath, ChannelMeta]] = None,
        prepare_timeout: float = 2.0,
        ignore_wrong_command: bool = False,
        token_replacements: dict[str, str] | None = None,
        clear_after_exit: bool = False,
    ) -> Interpreter:
        """
        实例化一个 interpreter 用来做解释.
        :param kind: 实例化 Interpreter 时的前置行为:
                    clear 表示清空所有运行中命令.
                    defer_clear 表示延迟清空, 但一旦有新命令, 就会被清空.
                    run 表示正常运行.
                    dry_run 表示 interpreter 虽然会正常执行, 但不会把生成的 command task 推送给 shell.

        :param stream_id: 设置一个指定的 stream id,
                     interpreter 整个运行周期生成的 command token 都会用它做标记.

        :param config: 如果传入了动态的 channel metas,
                    则运行时可用的命令由真实命令和这里传入的 channel metas 取交集.
                    是一种动态修改运行时能力的办法.

        :param prepare_timeout: 准备过度阶段允许的时间.
        :param meta_instruction: 可以用来替换系统默认的 moss 语法 prompt. 通常只在调试时需要修改.

        :param ignore_wrong_command: 遇到了幻想的 command 也不会解析错误.

        :param token_replacements: 根据 key 替换 interpreter feed 获得的一部分 token, 将之替换为 value.
                    这种做法可以用 instruction 里的 token 置换输出时的 token. 响应速度和费用能够有调整.

                    假设用 n 个代理 token, 平均每个代理 token 消耗是 m, 代理掉 v 个token, 在 t 次多轮对话中平均使用了 k 个代理 token.
                    t 轮 instruction 多消耗的 token: n * m * t
                    t 轮输出实际减少的 tokens:  (v - m) * k * t
                    所以 (v - m) * k * 3 > n * m    就有正收益.
                    假设 m = 1, v = 10, k=3, n=20,  每轮多消耗 20 个点,  每轮减少 80 个点开销. 大意如此.

        :param clear_after_exit: clear undone tasks after exit.
        """
        pass

    async def parse_text_to_command_tokens(
        self,
        text: str | AsyncIterable[str],
    ) -> AsyncIterable[CommandToken]:
        """
        语法糖, 用来展示如何把文本生成 command tokens.
        """
        interpreter = await self.interpreter("dry_run")
        if isinstance(text, str):

            async def generate():
                yield text

            text_stream = generate()
        else:
            text_stream = text
        async for token in interpreter.aparse_text_to_command_tokens(text_stream):
            if token is None:
                break
            yield token

    async def parse_tokens_to_command_tasks(
        self,
        tokens: AsyncIterable[CommandToken],
        *,
        ignore_wrong_command: bool = False,
    ) -> AsyncIterable[CommandTask]:
        """
        语法糖, 用来展示如何将 command tokens 生成 command tasks.
        """
        _token_queue = asyncio.Queue[CommandToken | None]()
        _task_queue = asyncio.Queue[CommandTask | None | Exception]()
        interpreter = await self.interpreter("dry_run", ignore_wrong_command=ignore_wrong_command)

        async def sender():
            try:
                async for token in tokens:
                    await _token_queue.put(token)
                    await asyncio.sleep(0.0)
            except Exception as e:
                raise e
            finally:
                _token_queue.put_nowait(None)

        sender_task = asyncio.create_task(sender())
        consumer_task = asyncio.create_task(
            interpreter.parse_tokens_to_command_tasks(_token_queue, _task_queue.put_nowait),
        )
        try:
            while True:
                item = await _task_queue.get()
                if item is None:
                    break
                yield item
                await asyncio.sleep(0.0)
            await consumer_task
        finally:
            if not sender_task.done():
                sender_task.cancel()

    async def parse_text_to_tasks(
        self,
        text: str | AsyncIterable[str] | list[str],
        *,
        ignore_wrong_command: bool = False,
    ) -> AsyncIterable[CommandTask]:
        """
        语法糖, 用来展示如何将 text 直接生成 command tasks
        """

        async def generate_text():
            if isinstance(text, str):
                yield text
                return
            elif isinstance(text, list):
                for content in text:
                    yield content
                return
            else:
                async for content in text:
                    yield content

        tokens = self.parse_text_to_command_tokens(generate_text())
        async for task in self.parse_tokens_to_command_tasks(tokens, ignore_wrong_command=ignore_wrong_command):
            yield task

    # --- runtime methods --- #

    @abstractmethod
    def push_task(self, *tasks: CommandTask) -> None:
        """
        添加 task 到运行时. 这些 task 会阻塞在 Channel Runtime 队列中直到获取执行机会.
        """
        pass

    @abstractmethod
    async def stop_interpretation(self) -> Optional[Interpretation]:
        """
        临时实现的中断方法. 原理设计有问题.
        todo: 重新设计 shell 的中断逻辑.
        """
        pass

    @abstractmethod
    async def clear(self) -> None:
        """
        清空所有的命令.
        注意 clear 是树形广播的, clear 一个 父 channel 也会 clear 所有的子 channel.
        """
        pass

    async def start(self) -> None:
        """
        启动 Shell 的 runtime.
        """
        await self.__aenter__()

    async def close(self) -> None:
        """
        shell 停止运行.
        """
        await self.__aexit__(None, None, None)

    @abstractmethod
    async def __aenter__(self):
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
