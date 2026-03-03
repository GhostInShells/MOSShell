import asyncio
import logging
from typing import Optional, Callable, Coroutine

import numpy as np
from ghoshell_common.contracts import LoggerItf
from ghoshell_common.helpers import uuid

from ghoshell_moss.core.concepts.speech import (
    TTS,
    AudioFormat,
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
            close_last: Optional[Callable[[], Coroutine[None, None, None]]] = None,
    ):
        batch_id = tts_batch.batch_id()
        super().__init__(id=batch_id)

        self.logger = logger
        self.cmd_task = None
        self.committed = False
        self._sample_rate = sample_rate
        self._running_loop = loop
        self._audio_type = AudioFormat(audio_format) if isinstance(audio_format, str) else audio_format
        self._channels = channels
        self._tts_batch = tts_batch
        self._player = player
        self._close_last = close_last
        self._text_buffer = ""
        self._audio_buffer = []
        self._starting = False
        self._started_event = ThreadSafeEvent()
        self._closed_event = ThreadSafeEvent()
        self._has_audio_data = False
        self._log_prefix = "[TTSSpeechStream id=%s] " % batch_id

        # 注册 callback 回调.
        tts_batch.with_callback(self._audio_callback)

    def _buffer(self, text: str) -> None:
        self._text_buffer += text
        self._tts_batch.feed(text)

    def _commit(self) -> None:
        self._tts_batch.commit()

    def buffered(self) -> str:
        return self._text_buffer

    def _audio_callback(self, data: np.ndarray) -> None:
        if data is None:
            return
        self._has_audio_data = True
        if not self._started_event.is_set():
            self._audio_buffer.append(data)
        else:
            self._player.add(
                data,
                channels=self._channels,
                audio_type=self._audio_type,
                rate=self._sample_rate,
            )

    async def wait(self) -> None:
        await self._tts_batch.wait_done()
        self.logger.info("%s wait batch done", self._log_prefix)
        if self._has_audio_data:
            await self._player.wait_play_done()
        self.logger.info("%s wait play done", self._log_prefix)

    async def astart(self) -> None:
        if self._starting:
            await self._started_event.wait()
            return
        self.logger.info("%s Starting TTS stream", self._log_prefix)
        self._starting = True
        if self._close_last:
            # 确认关闭上一个.
            await self._close_last()
            self._close_last = None
        for data in self._audio_buffer:
            # 将 buffer 的内容
            self._player.add(
                data,
                channels=self._channels,
                audio_type=self._audio_type,
                rate=self._sample_rate,
            )
        self._audio_buffer.clear()
        self._started_event.set()

    async def aclose(self):
        if self._closed_event.is_set():
            return
        self.logger.info("%s close TTS stream", self._log_prefix)
        self._closed_event.set()
        self._audio_buffer.clear()
        close_all = [self._tts_batch.close()]
        if self._close_last:
            close_all.append(self._close_last())
            self._close_last = None
        if self._started_event.is_set():
            close_all.append(self._player.clear())
        done = await asyncio.gather(*close_all, return_exceptions=True)
        for t in done:
            if isinstance(t, Exception):
                self.logger.error("%s close stream failed: %s", t)

    def close(self) -> None:
        self._running_loop.create_task(self.aclose)


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
        # self._streams: dict[str, SpeechStream] = {}
        self._last_stream: Optional[TTSSpeechStream] = None
        self._log_prefix = "[BaseTTSSpeech]"
        self._running_loop: Optional[asyncio.AbstractEventLoop] = None
        self._starting = False
        self._started = False
        self._closing = False
        self._closed_event = ThreadSafeEvent()

    def tts(self) -> TTS:
        return self._tts

    def new_stream(self, *, batch_id: Optional[str] = None) -> SpeechStream:
        batch_id = batch_id or uuid()
        tts_batch = self._tts.new_batch(batch_id=batch_id)
        close_last = None
        if self._last_stream:
            close_last = self._last_stream.aclose
        stream = TTSSpeechStream(
            loop=self._running_loop,
            audio_format=self._tts_info.audio_format,
            channels=self._tts_info.channels,
            sample_rate=self._tts_info.sample_rate,
            player=self._player,
            tts_batch=tts_batch,
            logger=self.logger,
            close_last=close_last,
        )
        self._last_stream = stream
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
        if self._last_stream:
            await self._last_stream.aclose()
            self._last_stream = None
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
        await self._tts.close()
        await self._player.close()
        self._closed_event.set()
        self.logger.info("%s is closed", self._log_prefix)

    async def wait_closed(self) -> None:
        await self._closed_event.wait()
