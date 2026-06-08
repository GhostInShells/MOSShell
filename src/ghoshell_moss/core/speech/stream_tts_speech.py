import asyncio
import logging
import time
from typing import Optional, Callable, Coroutine

import numpy as np
from ghoshell_common.contracts import LoggerItf
from ghoshell_moss.message import unique_id

from ghoshell_moss.contracts.audio import AudioRuntimeTopic
from ghoshell_moss.contracts.speech import (
    TTS,
    AudioFormat,
    TTSSpeech,
    SpeechStream,
    SpeechTopic,
    StreamAudioPlayer,
    TTSBatch,
)
from ghoshell_moss.core.concepts.topic import TopicService
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
            topic_service: TopicService | None = None,
            speaker_name: str = "Ghost",
    ):
        batch_id = tts_batch.batch_id()
        super().__init__(id=batch_id)

        self.logger = logger
        self._topic_service = topic_service
        self._speaker_name = speaker_name
        self.cmd_task = None
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

    async def _play_loop(self) -> None:
        try:
            await self._player.clear()
            if not self._started:
                await self.start_synthesis()
            self.logger.debug("%s start new audio playing", self._log_prefix)
            async for item in self._tts_batch.items():
                # 将 buffer 的内容
                data = item["audio"]
                self._player.add(
                    data,
                    channels=self._channels,
                    audio_type=self._audio_type,
                    rate=self._sample_rate,
                )
                await asyncio.sleep(0)
                self.logger.debug("%s add audio %d bytes", self._log_prefix, len(data))
            await self._player.wait_play_done()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.exception("%s play failed: %s", self._log_prefix, e)
        finally:
            self._play_done_event.set()
            # 冗余的 clear.
            await self._player.clear()
            # Broadcast playback end so the audio pipeline knows TTS is done.
            if self._topic_service is not None:
                self._topic_service.pub(
                    AudioRuntimeTopic(
                        running=False,
                        stream_key=self.id,
                        device_name="speaker",
                        device_explain=f"TTS output ({self._speaker_name})",
                    ),
                    creator="tts",
                )
                self.logger.info("%s Broadcasted AudioRuntimeTopic(running=False)", self._log_prefix)

    async def start_play(self) -> None:
        if self._playing:
            return
        self.logger.info("%s Starting playing TTS stream", self._log_prefix)
        self._playing = True
        self._playing_loop_task = asyncio.create_task(self._play_loop())

        # Broadcast SpeechTopic so the audio pipeline knows Ghost is speaking.
        if self._topic_service is not None:
            speech_topic = SpeechTopic(
                text=self._text_buffer,
                speaker_id="ghost",
                speaker_name=self._speaker_name,
                role="ghost",
                batch_id=self.id,
                timestamp=time.monotonic(),
            )
            self._topic_service.pub(speech_topic, creator="tts")
            self.logger.info("%s Broadcasted SpeechTopic: %s", self._log_prefix, self._text_buffer[:50])

            # Broadcast runtime start so listener knows TTS playback has begun.
            self._topic_service.pub(
                AudioRuntimeTopic(
                    running=True,
                    stream_key=self.id,
                    device_name="speaker",
                    device_explain=f"TTS output ({self._speaker_name})",
                    started_at=time.monotonic(),
                ),
                creator="tts",
            )
            self.logger.info("%s Broadcasted AudioRuntimeTopic(running=True)", self._log_prefix)

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
        self._running_loop.create_task(self.close)


class BaseTTSSpeech(TTSSpeech):
    def __init__(
            self,
            *,
            player: StreamAudioPlayer,
            tts: TTS,
            topic_service: TopicService | None = None,
            speaker_name: str = "Ghost",
            logger: Optional[LoggerItf] = None,
    ):
        self.logger = logger or logging.getLogger("moss")
        self._topic_service = topic_service
        self._speaker_name = speaker_name
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
            topic_service=self._topic_service,
            speaker_name=self._speaker_name,
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
