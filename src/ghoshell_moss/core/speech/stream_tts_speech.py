import asyncio
import logging
import time
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
    SpeechClause,
    SpeechSegment,
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
        clause_callbacks: Optional[list[Callable[[SpeechClause], None]]] = None,
        segment_callbacks: Optional[list[Callable[[SpeechSegment], None]]] = None,
        on_close: Optional[Callable[[], None]] = None,
        last_stream: Optional[SpeechStream] = None,
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
        # 对齐: 真实播放时长累加 (worker 线程), clause 数组由 batch 累积 (event loop).
        self._clause_callbacks: list[Callable[[SpeechClause], None]] = list(clause_callbacks or [])
        self._segment_callbacks: list[Callable[[SpeechSegment], None]] = list(segment_callbacks or [])
        self._played_duration = 0.0
        self._clause_cursor = 0
        self._sample_disposer: Optional[Callable[[], None]] = None
        self._audio_chunks: list[np.ndarray] = []
        self._on_close = on_close
        self._last_stream = last_stream

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

    def played_text(self) -> str:
        """已被真实播放出去的文本 — 由 clause 对齐推出, 不是已喂文本.

        只有 ``_clause_cursor`` 之前的 clause 才算播出声 (见 ``_accumulate_sample``:
        played_duration 追过 clause 末尾才推游标), 因此中断时最后一句未播完的 clause
        不计入. batch 没有 clause 能力 (服务端不给字幕) 时返回空串.
        """
        clauses = self._tts_batch.clauses()
        if not clauses:
            return ""
        return "".join(clause.text for clause in clauses[:self._clause_cursor])

    def on_sample(self, callback: Callable[[PlaybackSample], None]) -> Callable[[], None]:
        """订阅 player 的真实播放样本, 只回调属于本 stream 的片段.

        player.observe 是全局回调 — 这里包一层 stream_id 过滤, 只把本 stream
        (id == batch_id) 的样本交给 callback. 返回 player 的 disposer,
        say() 结束时会调用摘除.
        """

        def _match(sample: PlaybackSample) -> None:
            if sample.segment_id == self.id:
                callback(sample)

        return self._player.observe(_match)

    @staticmethod
    def _clause_end(clause: SpeechClause) -> float:
        """clause 的绝对结束时间 (session 内), 无词时视为 0 (立即对齐)."""
        return clause.words[-1].end_time if clause.words else 0.0

    def _accumulate_sample(self, sample: PlaybackSample) -> None:
        """对齐触发: 累加真实播放时长, 追到 clause 边界才发 on_clause.

        运行在 player 的 observe 回调 (audio worker 线程). TTS 合成总是快于播放,
        clause 数组 (batch 累积) 先于其音频播放到位; 这里用 played_duration 对
        clause 的 words[].end_time, 判断"这句真的播完了"才回调, 而非 TTS 返回即回调.
        """
        self._played_duration += sample.duration
        clauses = self._tts_batch.clauses()
        while self._clause_cursor < len(clauses):
            clause = clauses[self._clause_cursor]
            if self._played_duration < self._clause_end(clause):
                break
            clause.timestamp = sample.timestamp
            for callback in self._clause_callbacks:
                callback(clause)
            self._clause_cursor += 1

    def _emit_segment(self) -> None:
        """segment 播放结束 (event loop): 发 on_segment, 带完整 segment 结果."""
        if not self._segment_callbacks:
            return
        clauses = self._tts_batch.clauses()
        audio = np.concatenate(self._audio_chunks).tobytes() if self._audio_chunks else b""
        segment = SpeechSegment(
            segment_id=self.stream_id,
            timestamp=time.time(),
            text=self.buffered(),
            clauses=clauses,
            audio=audio,
            sample_rate=self._sample_rate,
            channels=self._channels,
            interrupted=self._clause_cursor < len(clauses),
        )
        for callback in self._segment_callbacks:
            callback(segment)

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
                    chunk=item["audio"],
                    channels=self._channels,
                    audio_type=self._audio_type,
                    rate=self._sample_rate,
                    stream_id=self.id,
                    fragment_id=f"{self.id}:{index}",
                    text=item.get("text", ""),
                )
                self._audio_chunks.append(item["audio"])
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
            # 播放结束后发 segment 结果 (clause 已在播放过程中逐句对齐触发).
            self._emit_segment()
            # 冗余的 clear.
            await self._player.clear()

    async def start_play(self) -> None:
        if self._playing:
            return
        self.logger.info("%s Starting playing TTS stream", self._log_prefix)
        self._playing = True
        if self._last_stream is not None and not self._last_stream.is_closed():
            await self._last_stream.close()
        # 内部订阅真实播放样本, 累加 played_duration 供 clause 结果融合.
        self._sample_disposer = self.on_sample(self._accumulate_sample)
        self._playing_loop_task = asyncio.create_task(self._play_loop())

    async def close(self):
        if self._closed_event.is_set():
            return
        if not self._started:
            self._notify_closed()
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
        if self._sample_disposer is not None:
            self._sample_disposer()
            self._sample_disposer = None
        await asyncio.gather(self._tts_batch.close(), self._player.clear())
        self._notify_closed()

    def _notify_closed(self) -> None:
        if self._on_close is not None:
            on_close = self._on_close
            self._on_close = None
            on_close()

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
        self._streams: dict[str, SpeechStream] = {}
        self._log_prefix = "[BaseTTSSpeech]"
        self._running_loop: Optional[asyncio.AbstractEventLoop] = None
        self._starting = False
        self._started = False
        self._closing = False
        self._closed_event = ThreadSafeEvent()
        self._clause_callbacks: list[Callable[[SpeechClause], None]] = []
        self._segment_callbacks: list[Callable[[SpeechSegment], None]] = []
        self._last_stream: Optional[SpeechStream] = None

    def tts(self) -> TTS:
        return self._tts

    def player(self) -> StreamAudioPlayer:
        return self._player

    def new_segment(self, *, batch_id: Optional[str] = None) -> SpeechStream:
        batch_id = batch_id or unique_id()
        tts_batch = self._tts.new_batch(batch_id=batch_id)
        return self.new_tts_stream(tts_batch)

    def on_clause(self, callback: Callable[[SpeechClause], None]) -> Callable[[], None]:
        """注册 clause 结果回调: 每个新 segment 播放完成时逐句回调其 SpeechClause."""
        self._clause_callbacks.append(callback)

        def _dispose() -> None:
            if callback in self._clause_callbacks:
                self._clause_callbacks.remove(callback)

        return _dispose

    def on_segment(self, callback: Callable[[SpeechSegment], None]) -> Callable[[], None]:
        """注册 segment 结果回调: 每个新 segment 播放结束时回调其 SpeechSegment."""
        self._segment_callbacks.append(callback)

        def _dispose() -> None:
            if callback in self._segment_callbacks:
                self._segment_callbacks.remove(callback)

        return _dispose

    def new_tts_stream(self, batch: TTSBatch) -> SpeechStream:
        stream = TTSSpeechStream(
            loop=self._running_loop,
            audio_format=self._tts_info.audio_format,
            channels=self._tts_info.channels,
            sample_rate=self._tts_info.sample_rate,
            player=self._player,
            tts_batch=batch,
            logger=self.logger,
            clause_callbacks=self._clause_callbacks,
            segment_callbacks=self._segment_callbacks,
            on_close=lambda: self._streams.pop(stream.id, None),
            last_stream=self._last_stream,
        )
        self._streams[stream.id] = stream
        self._last_stream = stream
        return stream

    def is_running(self) -> bool:
        return self._started and not self._closing

    def _check_running(self):
        if not self._started or self._closing:
            raise RuntimeError("TTS Speech is not running")

    async def clear(self) -> list[str]:
        if not self.is_running():
            return []
        self.logger.info("%s clear", self._log_prefix)
        streams = list(self._streams.values())
        outputted = [s.buffered() for s in streams]
        if streams:
            await asyncio.gather(*(s.close() for s in streams), return_exceptions=True)
        self._streams.clear()
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
        # 关闭 tts
        await self._tts.close()
        # 关闭 player.
        await self._player.close()
        self._closed_event.set()
        self.logger.info("%s is closed", self._log_prefix)

    async def wait_closed(self) -> None:
        await self._closed_event.wait()
