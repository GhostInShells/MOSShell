from abc import ABC, abstractmethod
from typing import List, Optional, Literal, Dict, Callable, Any
from ghoshell_moss.concepts.command import CommandTask
from pydantic import BaseModel, Field
import numpy as np
from enum import Enum

__all__ = [
    'Speech', 'SpeechStream',
    'AudioFormat',
    'StreamAudioPlayer',
    'TTS', 'TTSBatch',
    'TTSInfo', 'TTSAudioCallback',
]


class SpeechStream(ABC):
    """
    Speech 创建的单个 Stream.
    Shell 发送文本的专用模块. 是对语音或文字输出的高阶抽象.
    一个 speech 可以同时创建多个 stream, 但执行 tts 的顺序按先后排列.
    """

    def __init__(
            self,
            id: str,  # 所有文本片段都有独立的全局唯一id, 通常是 command_token.part_id
            cmd_task: Optional[CommandTask] = None,  # stream 生成的 command task
            committed: bool = False,  # 是否完成了这个 stream 的提交
    ):
        self.id = id
        self.cmd_task = cmd_task
        self.committed = committed

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
                await self.start()
                # 等待输入结束, 播放结束.
                await self.wait()
            finally:
                # 关闭播放.
                await self.close()

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
    async def start(self) -> None:
        """
        start to output
        """
        pass

    @abstractmethod
    async def close(self):
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
    async def clear(self) -> List[str]:
        """
        清空所有输出中的 output
        """
        pass

    @abstractmethod
    async def start(self) -> None:
        pass

    @abstractmethod
    async def close(self) -> None:
        pass

    async def __aenter__(self):
        await self.start()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    @abstractmethod
    async def wait_closed(self) -> None:
        pass

    async def run_until_closed(self) -> None:
        async with self:
            await self.wait_closed()


class AudioFormat(Enum):
    PCM_S16LE = 's16le'
    PCM_F32LE = 'float32le'


class StreamAudioPlayer(ABC):
    """
    音频播放的极简抽象.
    底层可能是 pyaudio, pulseaudio 或者别的实现.
    """

    audio_type: AudioFormat
    channels: int
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
    def add(
            self,
            chunk: np.ndarray,
            *,
            audio_type: AudioFormat,
            rate: int,
            channels: int = 1,
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

    def on_play(self, callback: Callable[[np.ndarray], None]) -> None:
        pass

    def on_play_done(self, callback: Callable[[], None]) -> None:
        pass


class TTSInfo(BaseModel):
    """
    反映出 tts 生成音频的参数, 用于播放时做数据的转换.
    """
    sample_rate: int = Field(description="音频的采样率")
    """音频片段的 rate"""

    channels: int = Field(default=1, description="音频的通道数")

    audio_format: str = Field(
        default=AudioFormat.PCM_S16LE.value, description="音频的默认格式, 还没设计好所有类型.",
    )

    voice_schema: Optional[Dict] = Field(
        default=None,
        description="声音的 schema, 通常用来给模型看"
    )

    voices: Dict[str, Dict] = Field(
        default_factory=dict,
        description="声音的可选项"
    )
    current_voice: str = Field(
        default="",
        description="当前的声音"
    )


_SampleRate = int
_Channels = int
TTSAudioCallback = Callable[[np.ndarray], None]


class TTSBatch(ABC):
    """
    流式 tts 的批次. 简单解释一下批次的含义.

    假设有云端的 TTS 服务, 可以流式地解析 tts, 这样会创建一个 connection, 比如 websocket connection.
    这个 connection 并不是只能解析一段文本,  它可以分批 (可能并行, 可能不并行) 解析多段文本, 生成多个音频流.

    而这里的 tts batch, 就是用来理解多个音频流已经阻塞生成完毕.
    """

    @abstractmethod
    def batch_id(self) -> str:
        """
        唯一 id.
        """
        pass

    @abstractmethod
    def with_callback(self, callback: TTSAudioCallback) -> None:
        """
        设置一个 callback 取代已经存在的.
        当音频数据生成后, 就会直接回调这个 callback.
        """
        pass

    @abstractmethod
    def feed(self, text: str):
        """
        提交新的文本片段.
        """
        pass

    @abstractmethod
    def commit(self):
        """
        结束文本片段的提交. tts 应该要能生成文本完整的音频.
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
        阻塞等待这个 batch 结束. 包含两种情况:
        1. closed: 被提前关闭.
        2. done: 按逻辑顺序是先完成 commit 后, 再完成 tts, 才能算 done.
        """
        pass


class TTS(ABC):
    """
    实现一个可拆卸的 TTS 模块, 用来解析文本到语音.
    名义上是 Stream TTS, 实际上也可以不是.
    要求支持 asyncio 的 api, 但具体实现可以配合多线程.
    """

    @abstractmethod
    def new_batch(self, batch_id: str = "", *, callback: TTSAudioCallback | None = None) -> TTSBatch:
        """
        创建一个 batch.
        这个 batch 有独立的生命周期阻塞逻辑 (wait until done)
        可以用来感知到 tts 是否已经完成了.
        完成的音频数据会发送给 callback. callback 应该要立刻播放音频.
        """
        pass

    @abstractmethod
    async def clear(self) -> None:
        """
        清空所有进行中的 tts batch.
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
    def use_voice(self, config_key: str) -> None:
        """
        选择一个配置好的音色.
        :param config_key: 与 tts_info 中一致.
        """
        pass

    @abstractmethod
    def set_voice(self, config: Dict[str, Any]) -> None:
        """
        设置一个临时的 voice config.
        """
        pass

    @abstractmethod
    async def start(self) -> None:
        """
        启动 tts 服务. 理论上一创建 Batch 就会尽快进行解析.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        关闭 tts 服务.
        """
        pass

    async def __aenter__(self):
        await self.start()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
