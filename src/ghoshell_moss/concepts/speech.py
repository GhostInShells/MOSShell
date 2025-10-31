from abc import ABC, abstractmethod
from typing import List, Optional, Literal, Dict, Callable, Any
from ghoshell_moss.concepts.command import CommandTask
from pydantic import BaseModel, Field
import numpy as np
from enum import Enum

__all__ = [
    'Speech', 'SpeechStream',
    'AudioType',
    'StreamAudioPlayer',
    'StreamTTS', 'StreamTTSBatch', 'StreamTTSSession',
]


class SpeechStream(ABC):
    """
    shell 发送文本的专用模块. 本身是非阻塞的.
    """
    id: str
    """所有文本片段都有独立的全局唯一id, 通常是 command_token.part_id"""

    cmd_task: Optional[CommandTask] = None
    """stream 生成的 command task"""

    committed: bool = False
    """是否完成了这个 stream 的提交. """

    def buffer(self, text: str, *, complete: bool = False) -> None:
        """
        添加文本片段到输出流里.
        由于文本可以通过 tts 生成语音, 而 tts 有独立的耗时, 所以通常一边解析 command token 一边 buffer 到 tts 中.
        而音频允许播放的时间则会靠后, 必须等上一段完成后才能开始播放下一段.

        :param text: 文本片段
        :type complete: 输出流是否已经结束.
        """
        if self.committed:
            # 不 buffer.
            return
        if text:
            # 文本不为空.
            self._buffer(text)
            if self.cmd_task is not None:
                # buffer 到 cmd task
                self.cmd_task.tokens = self.buffered()
        if complete:
            # 提交.
            self.commit()

    @abstractmethod
    def _buffer(self, text: str) -> None:
        """
        真实的 buffer 逻辑,
        """
        pass

    def commit(self) -> None:
        if self.committed:
            return
        self.committed = True
        self._commit()

    @abstractmethod
    def _commit(self) -> None:
        """真实的结束 stream 讯号. 如果 stream 通过 tts 实现, 这个讯号会通知 tts 完成输出. """
        pass

    def as_command_task(self, commit: bool = False) -> Optional[CommandTask]:
        """
        将 speech stream 转化为一个 command task, 使之可以发送到 Shell 中阻塞.
        """
        from ghoshell_moss.concepts.command import BaseCommandTask, CommandMeta, CommandWrapper
        if self.cmd_task is not None:
            return self.cmd_task

        if commit:
            # 是否要标记提交. stream 可能在生成 task 的时候, 还没有完成内容的提交.
            self.commit()

        async def _speech_lifecycle() -> None:
            try:
                # 标记开始播放.
                self.start()
                # 等待输入结束, 播放结束.
                await self.wait()
            finally:
                # 关闭播放.
                self.close()

        meta = CommandMeta(
            name="__speech__",
            # 默认主轨运行.
            chan="",
        )

        command = CommandWrapper(meta, _speech_lifecycle)
        task = BaseCommandTask.from_command(
            command,
        )
        task.cid = self.id
        # 添加默认的 tokens.
        task.tokens = self.buffered()
        self.cmd_task = task
        return task

    @abstractmethod
    def buffered(self) -> str:
        """
        返回已经缓冲的文本内容, 可能经过了加工.
        """
        pass

    @abstractmethod
    async def wait(self) -> None:
        """
        阻塞等待到播放完成. start & commit 是两个必要的开关.
        commit 意味着文本片段生成完毕.
        start 意味着允许开始播放.
        """
        pass

    @abstractmethod
    def start(self) -> None:
        """
        start to output
        """
        pass

    @abstractmethod
    def close(self):
        """
        关闭一个 Stream.
        """
        pass


class Speech(ABC):
    """
    文本输出模块. 通常和语音输出模块结合.
    """

    @abstractmethod
    def new_stream(self, *, batch_id: Optional[str] = None) -> SpeechStream:
        """
        创建一个新的输出流, 第一个 stream 应该设置为 play
        """
        pass

    @abstractmethod
    def outputted(self) -> List[str]:
        """
        清空之前生成的文本片段, speech 必须能感知到所有输出.
        """
        pass

    @abstractmethod
    def clear(self) -> List[str]:
        """
        清空所有输出中的 output
        """
        pass


class AudioType(Enum):
    PCM_S16LE = 's16le'
    PCM_F32LE = 'float32le'


class StreamAudioPlayer(ABC):
    """
    音频播放的极简抽象.
    底层可能是 pyaudio, pulseaudio 或者别的实现.
    """

    audio_type: AudioType
    channel: int
    sample_rate: int

    @abstractmethod
    async def start(self) -> None:
        """
        启动 audio player.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        关闭连接
        """
        pass

    async def __aenter__(self):
        await self.start()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    @abstractmethod
    async def clear(self) -> None:
        """
        清空当前输入的可播放片段, 立刻终止当前的播放内容.
        """
        pass

    @abstractmethod
    async def add(
            self,
            chunk: np.ndarray,
            *,
            audio_type: AudioType,
            rate: int,
            channel: int = 1,
    ) -> float:
        """
        添加音频片段. 关于音频的参数, 用来方便做转码 (根据底层实现判断转码的必要性)

        注意: 这个接口是非阻塞的, 通常会立刻返回. 方便提前把流式的音频片段都 buffer 好.

        :return: 返回一个 second 为单位的时间戳, 每一个音频片段插入后, 会根据音频播放的时间计算一个新的播放结束时间.
        """
        pass

    @abstractmethod
    async def wait_play_done(self, timeout: float | None = None) -> None:
        """
        等待所有输入的音频片段播放结束.
        实际上可能是阻塞到这个结束时间.
        """
        pass

    @abstractmethod
    def is_playing(self) -> bool:
        """
        返回当前是否在播放.
        有可能在运行中, 但没有任何音频输入.
        """
        pass

    @abstractmethod
    def is_closed(self) -> bool:
        """
        音频输入是否已经关闭了.
        """
        pass


class TTSInfo(BaseModel):
    rate: int = Field(description="accept rate of the audio chunk")
    """音频片段的 rate"""

    channels: int = Field(description="channel number of the audio chunk")
    """通道数"""

    format: Literal['pcm16'] = Field(
        default="pcm16", description="format of the audio chunk",
    )

    config_schema: Optional[Dict] = Field(
        default=None,
        description="音频可配置项的 schema, 可以给大模型看. 为 None 表示没有可配置项"
    )

    configs: Dict[str, Dict] = Field(
        default_factory=dict,
        description="音频模块可选的配置项"
    )
    default_conf_key: str = Field(
        default="",
        description="默认的配置项的 key"
    )


class StreamTTSBatch(ABC):
    """
    流式 tts 的批次. 简单解释一下批次的含义.

    假设有云端的 TTS 服务, 可以流式地解析 tts, 这样会创建一个 connection, 比如 websocket connection.
    这个 connection 并不是只能解析一段文本,  它可以分批 (可能并行, 可能不并行) 解析多端文本, 生成多个音频流.
    之所以要多个音频流, 因为它们的播放控制逻辑可能并不一致. 比如一段段被播放.

    我们的抽象需要屏蔽掉并行或不并行. 单独用 Batch 来代表一个流式输出串中的很多断.
    """

    @abstractmethod
    async def start(self):
        """
        启动这个 Batch
        """
        pass

    @abstractmethod
    def with_audio_callback(self, callback: Callable[[np.ndarray | None], None]) -> None:
        """
        设置音频的回调路径. 不会清空已有的.
        """
        pass

    @abstractmethod
    async def feed(self, text: str):
        """
        提交新的文本片段.
        """
        pass

    @abstractmethod
    def commit(self):
        """
        结束提交.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        结束这个 batch.
        """
        pass

    @abstractmethod
    async def wait_until_done(self, timeout: float | None = None):
        """
        阻塞等待这个 batch 结束.
        """
        pass


class StreamTTSSession(ABC):
    """
    单个音频 TTS 的 session.
    """

    @abstractmethod
    async def new_batch(self, batch_id: str, callback: Callable[[np.ndarray | None], None]) -> StreamTTSBatch:
        """
        创建一个 batch.
        """
        pass

    @abstractmethod
    async def clear(self) -> None:
        """
        关闭所有的 batch 解析.
        """
        pass

    @abstractmethod
    def is_closed(self) -> bool:
        """
        session 是否关闭.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        关闭所有的 batch, 结束这个 tts 流.
        """
        pass


class StreamTTS(ABC):
    """
    实现一个可拆卸的 TTS 模块, 用来解析文本到语音.
    名义上是 Stream TTS, 实际上也可以不是.
    要求支持 asyncio 的 api, 但具体实现可以配合多线程.
    """

    @abstractmethod
    async def start_session(self, session_id: str = "") -> StreamTTSSession:
        """
        启动一个 session, 所有的解析都在这个 session 里.
        """
        pass

    @abstractmethod
    def get_info(self) -> TTSInfo:
        """
        返回 TTS 的配置项.
        这些配置项应该决定了 tts 的音色, 效果, 音量, 语速等各种参数. 每种不同实现, 底层的参数也会不一样.
        """
        pass

    @abstractmethod
    def use_config(self, config_key: str) -> None:
        """
        选择一个配置好的 config key.
        """
        pass

    @abstractmethod
    def set_config(self, config: Dict[str, Any]) -> None:
        """
        设置输出的 config
        """
        pass
