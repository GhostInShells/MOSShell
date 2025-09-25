from abc import ABC, abstractmethod
from typing import List, Iterable, Dict, Literal, Optional
from typing_extensions import Self
from ghoshell_moss.concepts.channel import Channel, ChannelMeta
from ghoshell_moss.concepts.interpreter import Interpreter
from ghoshell_moss.concepts.command import Command, CommandTask
from ghoshell_container import IoCContainer


class OutputStream(ABC):
    """
    shell 发送文本的专用模块.
    """
    id: str
    """所有文本片段都有独立的全局唯一id, 通常是 command_part_id"""

    @abstractmethod
    def buffer(self, text: str, *, complete: bool = False) -> None:
        """
        添加文本片段到输出流里.
        由于文本可以通过 tts 生成语音, 而 tts 有独立的耗时, 所以通常一边解析 command token 一边 buffer 到 tts 中.
        而音频允许播放的时间则会靠后, 必须等上一段完成后才能开始播放下一段.

        :param text: 文本片段
        :type complete: 输出流是否已经结束.
        """
        pass

    def commit(self) -> None:
        self.buffer("", complete=True)

    @abstractmethod
    def output_start(self) -> None:
        """
        允许文本片段开始播放. 这时可能文本片段本身都未生成完, 如果是流式的 tts, 则可以一边 buffer, 一边 tts, 一边播放. 三者并行.
        """
        pass

    @abstractmethod
    def wait_done(self, timeout: float | None = None) -> None:
        """
        阻塞等待到文本输出完毕. 当文本输出是一个独立的模块时, 需要依赖这个函数实现阻塞.
        """
        pass

    @abstractmethod
    def is_done(self) -> bool:
        pass

    @abstractmethod
    def as_command_task(self, commit: bool = False) -> Optional[CommandTask]:
        """
        将 wait done 转化为一个 command task.
        这个 command task 通常在主轨 (channel name == "") 中运行.
        """
        pass

    @abstractmethod
    def close(self):
        pass


class Output(ABC):
    """
    文本输出模块. 通常和语音输出模块结合.
    """

    @abstractmethod
    def new_stream(self, *, batch_id: Optional[str] = None) -> OutputStream:
        """
        创建一个新的输出流, 第一个 stream 应该设置为 play
        """
        pass

    @abstractmethod
    def outputted(self) -> List[str]:
        pass

    @abstractmethod
    def clear(self) -> List[str]:
        """
        清空所有输出中的 output
        """
        pass


class ChannelRuntime(ABC):
    """
    管理 channel 的所有的 command task 运行时状态, 包括阻塞, 执行, 等待.
    """

    @abstractmethod
    def name(self) -> str:
        """
        channel 的名称.
        """
        pass

    @abstractmethod
    async def append(self, *tasks: CommandTask) -> None:
        """
        添加 task 到运行时的队列中.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        清空所有的运行任务和运行中的任务.
        """
        pass

    @abstractmethod
    def defer_clear(self) -> None:
        """
        设置 channel 为软重启. 当有一个属于当前 channel runtime 的 task 推送进来时, 清空自身和所有子节点.
        """
        pass

    @abstractmethod
    def clear_pending(self) -> int:
        """
        清空自身和子节点队列中的任务.
        """
        pass

    @abstractmethod
    def cancel_running(self) -> None:
        """
        取消正在运行的所有任务, 包括自身和子节点的任务.
        """
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """
        runtime 是否在运行中.
        """
        pass

    @abstractmethod
    def is_busy(self) -> bool:
        """
        是否正在运行任务, 或者队列中存在任务.
        """
        pass

    @abstractmethod
    async def get_child(self, name: str) -> "ChannelRuntime":
        """
        获取子 channel 的 runtime.
        :raise: KeyError if channel doesn't exist.
        """
        pass

    @abstractmethod
    async def start(self) -> None:
        """
        运行直到结束.
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """
        停止 runtime 运行.
        """
        pass


class ShellRuntime(ABC):

    @abstractmethod
    def is_running(self) -> bool:
        pass

    @abstractmethod
    def is_idle(self) -> bool:
        pass

    @abstractmethod
    def interpret(self, kind: Literal['clear', 'defer_clear', 'try'] = "clear") -> Interpreter:
        pass

    @abstractmethod
    async def append(self, *commands: CommandTask) -> None:
        pass

    @abstractmethod
    async def clear(self, *chans: str) -> None:
        pass

    @abstractmethod
    async def defer_clear(self, *chans: str) -> None:
        pass

    @abstractmethod
    async def system_prompt(self) -> str:
        pass

    @abstractmethod
    async def commands(self) -> Dict[str, List[Command]]:
        """
        get commands from shell
        """
        pass

    @abstractmethod
    async def wait_until_idle(self, timeout: float | None = None) -> None:
        pass

    @abstractmethod
    async def start(self) -> None:
        pass

    @abstractmethod
    async def stop(self) -> None:
        pass

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
        return self


class MOSSShell(ABC):
    """
    Model-Operated System Shell
    面向模型提供的 Shell, 让 AI 可以操作自身所处的系统.
    """

    container: IoCContainer

    @abstractmethod
    def name(self) -> str:
        """
        shell 的名称.
        """
        pass

    @property
    @abstractmethod
    def runtime(self) -> ShellRuntime:
        """
        shell 的运行时状态管理.
        """
        pass

    @abstractmethod
    def channels(self) -> Dict[str, ChannelMeta]:
        """
        返回所有的 Channel Meta 信息.
        """
        pass

    @abstractmethod
    def register(self, parent: str = "", *channels: Channel) -> None:
        """
        注册 channel.
        """
        pass

    @abstractmethod
    def configure(self, *metas: ChannelMeta) -> None:
        """
        配置 channel meta, 运行时会以配置的 channel meta 为准.
        """
        pass

    @abstractmethod
    def with_output(self, output: Output) -> None:
        """
        注册 Output 对象.
        """
        pass

    # --- properties --- #

    @property
    @abstractmethod
    def main(self) -> Channel:
        """
        Shell 自身的主轨. 主轨同时可以用来注册所有的子轨.
        """
        pass

    # --- runtime --- #

    @abstractmethod
    def is_running(self) -> bool:
        """
        shell 是否在运行中.
        """
        pass

    # --- lifecycle --- #

    @abstractmethod
    async def run(self) -> ShellRuntime:
        """
        生成 runtime 对象.
        """
        pass
