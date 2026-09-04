import asyncio
import logging
from typing import Optional, Callable, Coroutine

import numpy as np
from ghoshell_common.contracts import LoggerItf
from ghoshell_moss.message import unique_id

from ghoshell_moss.contracts.speech import (
    TTS,
    AudioFormat,
    PlaybackSample,
    TTSSpeech,
    SpeechStream,
    StreamAudioPlayer,
    TTSBatch,
)
from ghoshell_moss.core.helpers.asyncio_utils import ThreadSafeEvent


class TTSSpeechStream(SpeechStream):
    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        audio_format: AudioFormat | str,
        channels: int,
        sample_rate: int,
        player: StreamAudioPlayer,
        tts_batch: TTSBatch,
        logger: LoggerItf,
    ):
        batch_id = tts_batch.batch_id()
        super().__init__(id=batch_id)

        self.logger = logger
        self.committed = False
        self._sample_rate = sample_rate
        self._running_loop = loop
        self._audio_type = AudioFormat(audio_format) if isinstance(audio_format, str) else audio_format
        self._channels = channels
        self._tts_batch = tts_batch
        self._player = player
        self._text_buffer = ""
        self._started = False
        self._playing = False
        self._playing_loop_task: Optional[asyncio.Task] = None
        self._play_done_event = asyncio.Event()
        self._closed_event = ThreadSafeEvent()
        self._has_audio_data = False
        self._log_prefix = "[TTSSpeechStream id=%s] " % batch_id

    def _buffer(self, text: str) -> None:
        self._text_buffer += text
        self._tts_batch.feed(text)

    def _commit(self) -> None:
        self._tts_batch.commit()

    async def fail(self, err: Exception) -> None:
        if not isinstance(err, asyncio.CancelledError):
            self.logger.exception("%s stream failed: %s", self._log_prefix, err)
            await self.close()

    def buffered(self) -> str:
        return self._text_buffer

    async def wait_played(self) -> None:
        if not self._started:
            return
        if self._closed_event.is_set():
            return

        # 先等 tts 解析完成.
        await self._tts_batch.wait_done()
        # 等待 play done 完成.
        await self._play_done_event.wait()
        self.logger.info("%s wait play done", self._log_prefix)

    async def start_synthesis(self) -> None:
        if self._started:
            return
        self._started = True
        self.logger.info("%s Starting TTS stream", self._log_prefix)
        await self._tts_batch.start()

    def is_closed(self) -> bool:
        return self._closed_event.is_set()

    def on_sample(self, callback: Callable[[PlaybackSample], None]) -> Callable[[], None]:
        """订阅 player 的真实播放样本, 只回调属于本 stream 的片段.

        player.observe 是全局回调 — 这里包一层 stream_id 过滤, 只把本 stream
        (id == batch_id) 的样本交给 callback. 返回 player 的 disposer,
        say() 结束时会调用摘除.
        """

        def _match(sample: PlaybackSample) -> None:
            if sample.stream_id == self.id:
                callback(sample)

        return self._player.observe(_match)

    async def _play_loop(self) -> None:
        try:
            await self._player.clear()
            if not self._started:
                await self.start_synthesis()
            self.logger.debug("%s start new audio playing", self._log_prefix)
            index = 0
            async for item in self._tts_batch.items():
                # 每个 item 的 text 随音频进 player, sample 借此自解释真实播放文本.
                self._player.add(
                    data=item["audio"],
                    channels=self._channels,
                    audio_type=self._audio_type,
                    rate=self._sample_rate,
                    stream_id=self.id,
                    fragment_id=f"{self.id}:{index}",
                    text=item.get("text", ""),
                )
                index += 1
                await asyncio.sleep(0)
                self.logger.debug("%s add audio %d bytes", self._log_prefix, len(item["audio"]))
            await self._player.wait_play_done()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.exception("%s play failed: %s", self._log_prefix, e)
        finally:
            self._play_done_event.set()
            # 冗余的 clear.
            await self._player.clear()

    async def start_play(self) -> None:
        if self._playing:
            return
        self.logger.info("%s Starting playing TTS stream", self._log_prefix)
        self._playing = True
        self._playing_loop_task = asyncio.create_task(self._play_loop())

    async def close(self):
        if self._closed_event.is_set():
            return
        if not self._started:
            return
        self._closed_event.set()
        self.logger.info("%s close TTS stream", self._log_prefix)
        if self._playing_loop_task is not None:
            self._playing_loop_task.cancel()
            try:
                await self._playing_loop_task
            except asyncio.CancelledError:
                pass
        # 防止有未关闭的 wait.
        self._play_done_event.set()
        await asyncio.gather(self._tts_batch.close(), self._player.clear())

    def close_sync(self) -> None:
        """从任意线程调度异步 close 到事件循环线程 (线程安全)."""
        if self._running_loop is None:
            return
        asyncio.run_coroutine_threadsafe(self.close(), self._running_loop)


class BaseTTSSpeech(TTSSpeech):
    def __init__(
        self,
        *,
        player: StreamAudioPlayer,
        tts: TTS,
        logger: Optional[LoggerItf] = None,
    ):
        self.logger = logger or logging.getLogger("moss")
        self._player = player
        self._tts = tts
        self._tts_info = tts.get_info()
        self._outputted: list[str] = []
        self._log_prefix = "[BaseTTSSpeech]"
        self._running_loop: Optional[asyncio.AbstractEventLoop] = None
        self._starting = False
        self._started = False
        self._closing = False
        self._closed_event = ThreadSafeEvent()

    def tts(self) -> TTS:
        return self._tts

    def player(self) -> StreamAudioPlayer:
        return self._player

    def new_stream(self, *, batch_id: Optional[str] = None) -> SpeechStream:
        batch_id = batch_id or unique_id()
        tts_batch = self._tts.new_batch(batch_id=batch_id)
        return self.new_tts_stream(tts_batch)

    def new_tts_stream(self, batch: TTSBatch) -> SpeechStream:
        stream = TTSSpeechStream(
            loop=self._running_loop,
            audio_format=self._tts_info.audio_format,
            channels=self._tts_info.channels,
            sample_rate=self._tts_info.sample_rate,
            player=self._player,
            tts_batch=batch,
            logger=self.logger,
        )
        return stream

    def is_running(self) -> bool:
        return self._started and not self._closing

    def _check_running(self):
        if not self._started or self._closing:
            raise RuntimeError("TTS Speech is not running")

    def outputted(self) -> list[str]:
        if not self.is_running():
            return []
        return self._outputted

    async def clear(self) -> list[str]:
        if not self.is_running():
            return []
        self.logger.info("%s clear", self._log_prefix)
        outputted = self._outputted.copy()
        self._outputted.clear()
        return outputted

    async def start(self) -> None:
        if self._starting:
            return
        self._starting = True
        self._running_loop = asyncio.get_running_loop()
        await self._player.start()
        await self._tts.start()
        self.logger.info("%s started", self._log_prefix)
        self._started = True

    async def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        await self.clear()
        # 关闭 tts
        await self._tts.close()
        # 关闭 player.
        await self._player.close()
        self._closed_event.set()
        self.logger.info("%s is closed", self._log_prefix)

    async def wait_closed(self) -> None:
        await self._closed_event.wait()
