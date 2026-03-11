import asyncio
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional, AsyncIterator, Callable, TypedDict, AsyncIterable

import numpy as np
from pydantic import BaseModel, Field
from typing_extensions import Self
from ghoshell_moss.core.concepts.command import CommandTask, PyCommand, Command
from ghoshell_moss.core.concepts.channel import ChannelCtx
import json

__all__ = [
    "AudioFormat",
    "Speech",
    "SpeechStream",
    "StreamAudioPlayer",
    "TTS",
    "TTSItem",
    "TTSAudioCallback",
    "TTSBatch",
    "TTSInfo",
    "TTSSpeech",
]


class SpeechStream(ABC):
    """
    Speech 创建的单个 Stream.
    Shell 发送文本的专用模块. 是对语音或文字输出的高阶抽象.
    一个 speech 可以同时创建多个 stream, 但执行 tts 的顺序按先后排列.

    实现这个 SpeechStream 可以创建多种音频 Channel.
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

    def feed(self, text: str, *, complete: bool = False) -> None:
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
    async def fail(self, err: Exception) -> None:
        """
        根据异常的类型, 决定 stream 是否要终止.
        """
        pass

    @abstractmethod
    def _buffer(self, text: str) -> None:
        """
        真实的 buffer 逻辑,
        """
        pass

    def commit(self) -> None:
        """
        必须可重入.
        """
        if self.committed:
            return
        self.committed = True
        self._commit()

    @abstractmethod
    def _commit(self) -> None:
        """真实的结束 stream 讯号. 如果 stream 通过 tts 实现, 这个讯号会通知 tts 完成输出."""
        pass

    def as_command_task(self, commit: bool = False, chan: str = "") -> Optional[CommandTask]:
        """
        将 speech stream 转化为一个 command task, 使之可以发送到 Shell 中阻塞.
        这种使用方法, 假设 Stream 是独立在外部完成 feed & commit.
        """
        from ghoshell_moss.core.concepts.command import BaseCommandTask, CommandMeta, CommandWrapper

        if self.cmd_task is not None:
            # 只生成一个 task.
            return self.cmd_task

        if commit:
            # 是否要标记提交. stream 可能在生成 task 的时候, 还没有完成内容的提交.
            self.commit()

        meta = CommandMeta(
            name="__speak__",
            # 默认主轨运行.
            chan=chan,
            blocking=True,
        )
        start_synthesis = self.start_synthesis

        async def partial(*args, **kwargs) -> tuple[list, dict]:
            # 启动 tts 合成.
            nonlocal start_synthesis
            await start_synthesis()
            start_synthesis = None
            return list(args), kwargs

        command = CommandWrapper(meta, self.say, partial=partial)
        task = BaseCommandTask.from_command(
            command,
            chan_=chan,
            cid=self.id,
        )
        # 添加默认的 tokens.
        self.cmd_task = task
        return task

    @abstractmethod
    def buffered(self) -> str:
        """
        返回已经缓冲的文本内容, 可能经过了加工.
        """
        pass

    @abstractmethod
    async def wait_played(self) -> None:
        """
        阻塞等待到播放完成或结束. start & commit & play & closed 四元条件构成.
        - commit: 文本被全部提交.
        - synthesis: 允许开始 tts 合成. 文本没提交完, 也可以开始解析.
        - play: 允许音频开始播放.
        以上三个参数可以乱序调用.
        为何如此呢?

        1. 纯流式交互中, 文本输入, tts 解析, 音频播放三者均为并行的.
        2. 新的 Stream 只有在 Play 的时候, 才会关闭上一个 Stream. 所以上一个 Stream 未完成, 新的 Stream 也可以 synthesis
        3. 文本没有完成 commit 时, synthesis 和 play 都不能结束.
        4. close 时, 所有流程一起结束.

        - close 如果 stream 关闭了, 则等待也终止.
        """
        pass

    @abstractmethod
    async def start_synthesis(self) -> None:
        """
        允许开始解析输入文本.
        要求这个函数可重入.
        """
        pass

    @abstractmethod
    async def start_play(self) -> Self:
        """
        允许播放声音. 在允许播放声音的同时, 上一个 Stream 必须被关闭.
        """
        pass

    @abstractmethod
    async def close(self):
        """
        关闭, 结束 speech stream.
        要求这个函数可重入.
        """
        pass

    @abstractmethod
    def is_closed(self) -> bool:
        """
        是否已经运行结束.
        """
        pass

    async def say(self) -> None:
        """
        播放文本的完整生命周期.
        """
        if self.is_closed():
            return
        async with self:
            # 不会主动 commit.
            # 如果没有开始解析, 这时要开启.
            await self.start_synthesis()
            # 如果没有允许播放, 这时要允许播放.
            await self.start_play()
            await self.wait_played()

    async def speak(self, chunks__: AsyncIterable[str]) -> None:
        """
        完整的生命周期展示.
        """
        async with self:
            # 开启解析
            await self.start_synthesis()
            # 开启执行.
            await self.start_play()
            async for chunk in chunks__:
                self.feed(chunk)
            # speak 会保证 commit.
            self.commit()
            await self.wait_played()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            await self.fail(exc_val)
        await self.close()

    @abstractmethod
    def close_sync(self) -> None:
        """
        需要支持同步调用.
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
    def is_running(self) -> bool:
        pass

    @abstractmethod
    async def clear(self) -> list[str]:
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
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    @abstractmethod
    async def wait_closed(self) -> None:
        pass

    async def run_until_closed(self) -> None:
        async with self:
            await self.wait_closed()


class AudioFormat(Enum):
    PCM_S16LE = "s16le"
    PCM_F32LE = "float32le"


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

    @abstractmethod
    def on_play(self, callback: Callable[[np.ndarray], None]) -> None:
        raise NotImplementedError

    @abstractmethod
    def on_play_done(self, callback: Callable[[], None]) -> None:
        raise NotImplementedError


class TTSInfo(BaseModel):
    """
    反映出 tts 生成音频的参数, 用于播放时做数据的转换.
    """

    sample_rate: int = Field(description="音频的采样率")
    """音频片段的 rate"""

    channels: int = Field(default=1, description="音频的通道数")

    audio_format: str = Field(
        default=AudioFormat.PCM_S16LE.value,
        description="音频的默认格式, 还没设计好所有类型.",
    )

    voice_schema: Optional[dict] = Field(default=None, description="声音的 schema, 通常用来给模型看")

    tones: dict[str, str] = Field(default_factory=dict, description="tone name and description")
    current_tone: str = Field(default="", description="当前的声音")


_SampleRate = int
_Channels = int
TTSAudioCallback = Callable[[np.ndarray], None]


class TTSItem(TypedDict):
    """
    tts item 的数据.
    """

    text: str
    audio: np.ndarray  # 音频片段.
    sample_rate: int  # 对齐 sample rate
    audio_format: str  # 对齐 AudioFormat
    channels: int  # 对齐 Channels.
    tone: str  # 对齐 tone
    voice: dict  # 对齐 voice


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
    async def start(self) -> None:
        """
        正式启动 Batch 的 TTS 过程.
        """
        pass

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    @abstractmethod
    async def close(self) -> None:
        """
        结束这个 batch.
        """
        pass

    @abstractmethod
    def is_committed(self) -> bool:
        """
        是否提交了文本.
        """
        pass

    @abstractmethod
    def is_closed(self) -> bool:
        """
        是否运行结束.
        """
        pass

    @abstractmethod
    def is_started(self) -> bool:
        """
        开始运行 tts 逻辑.
        """
        pass

    @abstractmethod
    async def items(self) -> AsyncIterable[TTSItem]:
        """
        返回生成的音频片段.
        :return AsyncIterable[TTSItem]: 音频片段.
        """
        pass

    @abstractmethod
    async def wait_done(self, timeout: float | None = None):
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
    def new_batch(
        self,
        batch_id: str = "",
        *,
        callback: TTSAudioCallback | None = None,
        tone: str | None = None,
        voice: dict | None = None,
    ) -> TTSBatch:
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
    def use_tone(self, config_key: str) -> None:
        """
        选择一个配置好的音色.
        :param config_key: 与 tts_info 中一致.
        """
        pass

    @abstractmethod
    def current_tone(self) -> str:
        pass

    @abstractmethod
    def set_voice(self, config: dict[str, Any]) -> None:
        """
        设置一个临时的 voice config.
        """
        pass

    @abstractmethod
    def get_voice(self) -> dict[str, Any]:
        """
        返回当前的 voice 配置.
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


class TTSSpeech(Speech, ABC):
    """
    支持 TTS 的 speech.
    同样也能提供各种特殊的 command.
    """

    @abstractmethod
    def tts(self) -> TTS:
        pass

    @abstractmethod
    def player(self) -> StreamAudioPlayer:
        pass

    @abstractmethod
    def new_tts_stream(self, batch: TTSBatch) -> SpeechStream:
        pass

    def commands(self) -> list[Command]:
        """
        返回 TTS Speech 默认支持的命令.
        """
        tts = self.tts()
        tts_info = tts.get_info()
        voice_schema_str = json.dumps(tts_info.voice_schema, ensure_ascii=False, indent=0)

        def say_doc() -> str:
            current_voice = tts.get_voice()
            current_tone = tts.current_tone()
            tones = tts_info.tones
            tone_descriptions = []
            for _tone, description in tones.items():
                tone_descriptions.append(f"`{_tone}`: {description}")
            tone_descriptions_str = ";".join(tone_descriptions)

            return (
                f"使用指定的声音状态说话. 当它在 __main__ channel 时, 默认可以省略. \n"
                f":param voice: 声音的速度, 音调等. json 结构, json schema 是 {voice_schema_str}\n "
                f"  你当前的声音状态是: {json.dumps(current_voice, ensure_ascii=False)}.\n"
                f":param as_default: 将本轮设置的声音状态变成默认.\n"
                f":param chunks__: 你说话的文本内容. \n"
                f":param tone: 切换使用的音色. 默认为当前音色\n"
                f"  当前的音色是 `{current_tone}`"
                f"  当前可以使用的音色: {tone_descriptions_str}\n"
            )

        async def say_partial(
            chunks__,
            voice: dict | None = None,
            as_default: bool = False,
            tone: str = "",
        ) -> tuple[list, dict]:
            """
            预先准备 say 的逻辑.
            """
            if as_default:
                if voice:
                    tts.set_voice(voice)
                if tone:
                    tts.use_tone(tone)
            batch = self.tts().new_batch(voice=voice, tone=tone)
            stream = self.new_tts_stream(batch)

            async def run_tts_batch() -> None:
                try:
                    nonlocal chunks__
                    # 允许开启解析.
                    await stream.start_synthesis()
                    async for chunk in chunks__:
                        if stream.is_closed():
                            return
                        stream.feed(chunk)
                except Exception as e:
                    await stream.fail(e)
                finally:
                    stream.commit()

            # 开始异步运行.
            _ = asyncio.create_task(run_tts_batch())
            return [], dict(voice=voice, chunks__=stream, as_default=as_default)

        async def say(chunks__, voice: dict | None = None, as_default: bool = False, tone: str = "") -> None:
            """
            实际上拿到的 chunks__ 是一个 stream.
            """
            if not isinstance(chunks__, SpeechStream):
                raise ValueError(f"System error: Chunks is not prepared")
            await chunks__.say()

        say_command = PyCommand(
            say,
            doc=say_doc,
            partial=say_partial,
        )

        return [say_command]
