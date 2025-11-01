from typing import List, Optional, Dict

from ghoshell_moss.concepts.speech import (
    Speech, SpeechStream, StreamAudioPlayer,
    TTS, TTSBatch,
    AudioFormat,
)
from ghoshell_moss.helpers.asyncio_utils import ThreadSafeEvent
from ghoshell_common.contracts import LoggerItf
from ghoshell_common.helpers import uuid
import numpy as np
import logging
import asyncio


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
        self.cmd_task = None
        self.committed = False
        self._sample_rate = sample_rate
        self._running_loop = loop
        self._audio_type = AudioFormat(audio_format) if isinstance(audio_format, str) else audio_format
        self._channels = channels
        self._tts_batch = tts_batch
        self._player = player
        self._text_buffer = ""
        self._audio_buffer = []
        self._starting = False
        self._started_event = ThreadSafeEvent()
        self._has_audio_data = False

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
        await self._tts_batch.wait_until_done()
        if self._has_audio_data:
            await self._player.wait_play_done()

    async def start(self) -> None:
        if self._starting:
            await self._started_event.wait()
            return
        self._starting = True
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

    async def close(self):
        await self._tts_batch.close()
        self._audio_buffer.clear()
        if self._started_event.is_set():
            await self._player.clear()


class TTSSpeech(Speech):

    def __init__(
            self,
            *,
            player: StreamAudioPlayer,
            tts: TTS,
            logger: Optional[LoggerItf] = None,
    ):
        self.logger = logger or logging.getLogger("StreamTTSSpeech")
        self._player = player
        self._tts = tts
        self._tts_info = tts.get_info()
        self._outputted: List[str] = []
        self._streams: Dict[str, SpeechStream] = {}

        self._running_loop: Optional[asyncio.AbstractEventLoop] = None
        self._starting = False
        self._started = False
        self._closing = False
        self._closed_event = ThreadSafeEvent()

    def new_stream(self, *, batch_id: Optional[str] = None) -> SpeechStream:
        batch_id = batch_id or uuid()
        tts_batch = self._tts.new_batch(batch_id=batch_id)
        stream = TTSSpeechStream(
            loop=self._running_loop,
            audio_format=self._tts_info.audio_format,
            channels=self._tts_info.channels,
            sample_rate=self._tts_info.sample_rate,
            player=self._player,
            tts_batch=tts_batch,
            logger=self.logger,
        )
        self._streams[stream.id] = stream
        return stream

    def _check_running(self):
        if not self._started or self._closing:
            raise RuntimeError("TTS Speech is not running")

    def outputted(self) -> List[str]:
        self._check_running()
        return self._outputted

    async def clear(self) -> List[str]:
        self._check_running()
        outputted = self._outputted.copy()
        self._outputted = []
        streams = self._streams.copy()
        self._streams.clear()
        close_all = []
        for stream in streams.values():
            close_all.append(stream.close())
        await asyncio.gather(*close_all)
        return outputted

    async def start(self) -> None:
        if self._starting:
            return
        self._starting = True
        self._running_loop = asyncio.get_running_loop()
        await self._player.start()
        await self._tts.start()
        self._started = True

    async def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        await self._tts.close()
        await self._player.close()
        self._closed_event.set()

    async def wait_closed(self) -> None:
        await self._closed_event.wait()
