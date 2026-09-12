"""Volcengine SAUC ASR 实现 — 豆包大模型流式识别 (bigmodel_async)。

模型 (对齐官方 SDK 与文档):
  一条 WS = 一次「说话」(turn) = 一个 segment。
    - 服务端 VAD 分句: 静音超过 end_window_size 判停 → 分句 (utterance)。
      enable_nonstream=true 时, 非流式二次识别的分句带 definite=true (最终确定)。
    - 端侧 commit / 音频断: 发 is_last=True (负序号) → 服务端回 is_last_package=true。
    - is_last_package=true 是「整个流的最后一个响应包」→ 切段 (整个 turn 的 audio+text) → 流结束。

因此: definite (分句) 与 is_last_package (流结束) 是两个不同层级, 不混用。
"""
import asyncio
import contextlib
import inspect
import logging
import time
from typing import AsyncIterable, Awaitable, Callable, Optional

import numpy as np
import websockets
from ghoshell_common.contracts import LoggerItf
from ghoshell_common.helpers import uuid

from ghoshell_moss.contracts.asr import (
    ASR,
    ASRInfo,
    Clause,
    RecognitionStream,
    RecognitionPhase,
    RecognitionEvent,
    RecognitionSegment,
)

from .config import VolcengineSaucConfig, VolcengineSaucParams, VolcengineSaucCorpus
from .protocol import (
    PayloadMsg,
    Response,
    connect,
    create_audio_only_request,
    create_init_request,
    nparray_to_bytes,
    parse_response,
)

__all__ = ["VolcengineSaucASR"]

# 收包短超时: 周期性醒来判超时/关闭, 避免 recv 无限阻塞.
_RECV_TIMEOUT = 1.0


class VolcengineSaucASR(ASR):
    """豆包大模型流式识别。recognize() 返回一条识别流, 每流一条 WS。"""

    def __init__(
            self,
            config: VolcengineSaucConfig,
            *,
            logger: Optional[LoggerItf] = None,
    ):
        # ASR 实例私有副本 — 调参只改自己的副本, 不回流 config store.
        self._config = config.model_copy(deep=True)
        self._corpus = VolcengineSaucCorpus()
        self._logger = logger or logging.getLogger("moss")
        self._log_prefix = "[VolcengineSaucASR]"
        self._closed = False
        self._error_callback: Callable[[Exception], None] | None = None

    # ── ASR contract ──

    def get_info(self) -> ASRInfo:
        return ASRInfo(
            sample_rate=self._config.sample_rate,
            bits=self._config.bits,
            channel=self._config.channel,
            vad_end_window_ms=self._config.params.end_window_size,
            params_schema=VolcengineSaucParams.model_json_schema(),
            params=self._config.params.model_dump(),
        )

    def configure(self, params: dict) -> None:
        self._config.params = VolcengineSaucParams.model_validate(params)

    def on_error(self, callback: Callable[[Exception], None]) -> None:
        self._error_callback = callback

    def recognize(
            self,
            audio_chunks: AsyncIterable[np.ndarray],
            *,
            stream_id: str | None = None,
    ) -> RecognitionStream:
        if self._closed:
            raise RuntimeError("ASR is closed")
        return _VolcengineSaucRecognitionStream(
            config=self._config,
            corpus=self._corpus,
            audio_chunks=audio_chunks,
            stream_id=stream_id,
            logger=self._logger,
            log_prefix=self._log_prefix,
            error_callback=self._error_callback,
        )

    async def close(self) -> None:
        self._closed = True
        self._logger.info("%s closed", self._log_prefix)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    # ── 火山专属面 (不进 ASR 抽象) ──

    def configure_corpus(self, corpus: VolcengineSaucCorpus) -> None:
        """配置热词/上下文 (火山专属面)。修改后下一次 recognize() 的 init 带上最新 corpus。"""
        self._corpus = corpus.model_copy(deep=True)


class _VolcengineSaucRecognitionStream(RecognitionStream):
    """一条识别流 = 整条音频输入, 内部逐 turn 建 WS。

    1 stream = n segment。音频输入为界: ``while 音频未耗尽: 开 WS → 喂音频 →
    commit/音频断发 NEG → 收尾包 → 切段 → 下一 turn``。WS 是 turn 的实现细节,
    不是流的生命周期。懒开: 首个 __anext__ 才建 session。
    """

    def __init__(
            self,
            *,
            config: VolcengineSaucConfig,
            audio_chunks: AsyncIterable[np.ndarray],
            stream_id: str | None,
            logger: LoggerItf,
            log_prefix: str,
            error_callback: Callable[[Exception], None] | None,
            corpus: VolcengineSaucCorpus | None = None,
    ):
        self._config = config
        self._corpus = corpus if corpus is not None else VolcengineSaucCorpus()
        self._audio_chunks = audio_chunks
        self._logger = logger
        self._log_prefix = log_prefix
        self._error_callback = error_callback

        self._stream_id = stream_id or uuid()
        self._request_id = uuid()
        self._segment_id = uuid()
        self._queue: asyncio.Queue[RecognitionEvent | None] = asyncio.Queue()
        self._started = False
        self._session_task: asyncio.Task | None = None
        self._commit_event = asyncio.Event()
        self._input_done = False
        self._closed = False
        self._fatal_error: Exception | None = None
        self._on_segment_callback: Callable[[RecognitionSegment], None] | None = None
        self._event_creating_callbacks: list[Callable[[RecognitionEvent], Awaitable[None] | None]] = []

        # 当前 segment 的 audio + text (整个 turn 一份).
        self._current_audio: list[np.ndarray] = []
        self._segment_text = ""
        self._total_samples = 0

        # 已吐出的 definite 句数 (result_type=full 时服务端全量返回, 靠它去重).
        self._emitted_clauses = 0

        # FIRST / partial 去重状态 (每 segment 一份).
        self._first_emitted = False
        self._last_text = ""

        # 尾包等待状态 (每 turn 一份): NEG 已发 → 收包侧只等尾包, 超宽限则兜底切段.
        self._tail_requested = False
        self._tail_deadline = 0.0

    # ── RecognitionStream contract ──

    @property
    def stream_id(self) -> str:
        return self._stream_id

    def on_segment(self, callback: Callable[[RecognitionSegment], None]) -> None:
        self._on_segment_callback = callback

    def on_event_creating(
            self,
            callback: Callable[[RecognitionEvent], Awaitable[None] | None],
    ) -> None:
        self._event_creating_callbacks.append(callback)

    def commit(self) -> None:
        """结束当前 segment: 让 send loop 发 NEG, 拿尾包后切段开下一 turn."""
        self._commit_event.set()

    async def close(self) -> None:
        """主动关闭 (public-internal): 停止监听, 结束迭代, 不产尾包."""
        self._closed = True
        task = self._session_task
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def is_input_done(self) -> bool:
        return self._input_done

    def __aiter__(self) -> RecognitionStream:
        return self

    async def __anext__(self) -> RecognitionEvent:
        if not self._started:
            self._started = True
            self._session_task = asyncio.create_task(self._run_session())
        result = await self._queue.get()
        if result is None:
            if self._fatal_error is not None:
                raise self._fatal_error
            raise StopAsyncIteration
        return result

    # ── session ──

    async def _run_session(self) -> None:
        backoff = self._config.connect_backoff_initial
        consecutive = 0
        try:
            while not self._closed and not self._input_done:
                self._commit_event.clear()
                if await self._run_turn():
                    self._advance_segment()
                    backoff = self._config.connect_backoff_initial
                    consecutive = 0
                else:
                    consecutive += 1
                    if consecutive >= self._config.max_connect_retries:
                        self._fatal_error = RuntimeError(
                            f"ASR connect failed {consecutive} consecutive times"
                        )
                        self._report_error(self._fatal_error)
                        return
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, self._config.connect_backoff_cap)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._fatal_error = e
            self._report_error(e)
        finally:
            await self._queue.put(None)

    async def _run_turn(self) -> bool:
        """跑一个 turn: 开 WS → init → send/receive 并发 → 尾包后切段退出.

        返回 True=正常跑完; connect/init 失败返回 False (由 _run_session 退避重试).
        """
        self._request_id = uuid()
        send_task: asyncio.Task | None = None
        receive_task: asyncio.Task | None = None
        try:
            async with await connect(self._config, self._request_id) as ws:
                await ws.send(create_init_request(self._segment_id, self._config, self._corpus))
                send_task = asyncio.create_task(self._send_loop(ws))
                receive_task = asyncio.create_task(self._receive_loop(ws))
                await receive_task
        except asyncio.CancelledError:
            raise
        except Exception:
            # connect/init 失败: 静默返回 False, 由 _run_session 退避重试.
            return False
        finally:
            for t in (send_task, receive_task):
                if t is not None and not t.done():
                    t.cancel()
            for t in (send_task, receive_task):
                if t is not None:
                    with contextlib.suppress(asyncio.CancelledError):
                        await t
        return True

    def _advance_segment(self) -> None:
        """切段后重置 per-turn 状态, 分配新 segment_id."""
        self._segment_id = uuid()
        self._first_emitted = False
        self._last_text = ""
        self._emitted_clauses = 0
        self._current_audio = []
        self._segment_text = ""
        self._total_samples = 0
        self._tail_requested = False
        self._tail_deadline = 0.0

    # ── loops ──

    async def _send_loop(self, ws) -> None:
        # init (create_init_request) 已占用 seq=1, 音频从 seq=2 起递增.
        seq = 2
        committed = False
        try:
            async for audio in self._audio_chunks:
                if self._closed:
                    break
                arr = np.asarray(audio).ravel()
                self._current_audio.append(arr)
                self._total_samples += arr.size
                await ws.send(create_audio_only_request(nparray_to_bytes(arr), seq, is_last=False))
                seq += 1
                if self._commit_event.is_set():
                    self._commit_event.clear()
                    # commit: 发 is_last (负序号) 后不再喂音频, 等尾包切段开下一 turn.
                    await ws.send(create_audio_only_request(b"", seq, is_last=True))
                    seq += 1
                    committed = True
                    self._tail_requested = True
                    self._tail_deadline = time.monotonic() + self._config.tail_grace_seconds
                    break

            if not committed and not self._closed:
                # 音频耗尽: 最后一次 last package, 之后整条流自然结束.
                await ws.send(create_audio_only_request(b"", seq, is_last=True))
                self._input_done = True
                self._tail_requested = True
                self._tail_deadline = time.monotonic() + self._config.tail_grace_seconds
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._report_error(e)

    async def _receive_loop(self, ws) -> None:
        try:
            while True:
                try:
                    data = await asyncio.wait_for(ws.recv(), timeout=_RECV_TIMEOUT)
                except asyncio.TimeoutError:
                    if self._tail_requested and time.monotonic() > self._tail_deadline:
                        # NEG 已发但服务端不回尾包也不断连 → 兜底切段.
                        await self._queue.put(self._tail_result(text=self._segment_text, error="tail timeout"))
                        self._cut_segment()
                        break
                    continue
                except websockets.exceptions.ConnectionClosed:
                    await self._queue.put(self._tail_result(text=self._segment_text, error="connection closed"))
                    self._cut_segment()
                    break

                if not data:
                    continue

                response: Response = parse_response(data)

                if response.code != 0:
                    error_msg = f"server error {response.code}: {response.error_msg}"
                    self._logger.error("%s %s, request=%s", self._log_prefix, error_msg, self._request_id)
                    await self._queue.put(self._tail_result(text=self._segment_text, error=error_msg))
                    self._cut_segment()
                    break

                for chunk in self._parse_result(response.payload_msg):
                    await self.dispatch_event_creating(chunk)
                    await self._queue.put(chunk)

                if response.is_last_package:
                    # 尾包: 切段 → 退出本 turn (由 _run_session 决定是否开下一 turn).
                    await self._queue.put(self._tail_result(text=self._segment_text))
                    self._cut_segment()
                    break

        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._report_error(e)
            await self._queue.put(self._tail_result(text=self._segment_text, error=str(e)))
            self._cut_segment()

    # ── event creating (commit-decision hook) ──

    async def dispatch_event_creating(self, event: RecognitionEvent) -> None:
        """public-internal: 事件创建分派 — 由 _receive_loop 在 parse 后、put 前调用,
        也供测试验证两种回调形态的分派行为.

        awaitable callback → inline await (blocking consumption point);
        sync callback → offload to a thread (non-blocking, parallel).
        """
        for cb in list(self._event_creating_callbacks):
            try:
                if inspect.iscoroutinefunction(cb):
                    await cb(event)
                else:
                    await asyncio.to_thread(cb, event)
            except Exception:
                self._logger.exception("%s on_event_creating callback failed", self._log_prefix)

    # ── parse ──

    def _parse_result(self, msg: PayloadMsg) -> list[RecognitionEvent]:
        """把一个响应 payload 解析成 first/clause/partial 结果。不切段。"""
        result = msg.result
        text = result.text
        self._segment_text = text
        chunks: list[RecognitionEvent] = []

        # FIRST: 第一个有语义的包 (text 非空), 每 segment 一次.
        if not self._first_emitted and text:
            self._first_emitted = True
            self._last_text = text
            chunks.append(RecognitionEvent(
                stream_id=self._stream_id,
                segment_id=self._segment_id,
                phase=RecognitionPhase.FIRST,
                text=text,
            ))

        # CLAUSE: 新定稿的分句, 每个只发一次; 签发后不可变 (修正只落 tail).
        definite = [u for u in result.utterances if u.definite]
        new = definite[self._emitted_clauses:]
        self._emitted_clauses = len(definite)
        for u in new:
            self._last_text = text
            chunks.append(RecognitionEvent(
                stream_id=self._stream_id,
                segment_id=self._segment_id,
                phase=RecognitionPhase.CLAUSE,
                text=text,
                clause=Clause(
                    text=u.text,
                    start_ms=u.start_time,
                    end_ms=u.end_time,
                    additional=u.additions,
                ),
            ))

        # PARTIAL: text 相对上次有变化才发 (相邻相同压掉).
        if result.utterances and not result.utterances[-1].definite:
            if text and text != self._last_text:
                self._last_text = text
                chunks.append(RecognitionEvent(
                    stream_id=self._stream_id,
                    segment_id=self._segment_id,
                    phase=RecognitionPhase.PARTIAL,
                    text=text,
                ))
        return chunks

    # ── segment ──

    def _tail_result(self, *, text: str, error: str = "") -> RecognitionEvent:
        return RecognitionEvent(
            stream_id=self._stream_id,
            segment_id=self._segment_id,
            phase=RecognitionPhase.TAIL,
            text=text,
            error=error,
        )

    def _stream_ms(self) -> int:
        return int(self._total_samples / self._config.sample_rate * 1000)

    def _cut_segment(self) -> None:
        audio = (
            np.concatenate(self._current_audio)
            if self._current_audio
            else np.array([], dtype=np.int16)
        )
        segment = RecognitionSegment(
            id=self._segment_id,
            stream_id=self._stream_id,
            text=self._segment_text,
            start_ms=0,
            end_ms=self._stream_ms(),
            sample_rate=self._config.sample_rate,
            bits=self._config.bits,
            channel=self._config.channel,
            audio=audio,
            offset_ms=0,
        )
        self._emit_segment(segment)

    def _emit_segment(self, segment: RecognitionSegment) -> None:
        if self._on_segment_callback is not None:
            try:
                self._on_segment_callback(segment)
            except Exception:
                self._logger.exception("%s on_segment callback failed", self._log_prefix)

    # ── internals ──

    def _report_error(self, e: Exception) -> None:
        self._logger.error("%s recognition error: %s", self._log_prefix, e)
        if self._error_callback is not None:
            try:
                self._error_callback(e)
            except Exception:
                self._logger.exception("%s error callback failed", self._log_prefix)
