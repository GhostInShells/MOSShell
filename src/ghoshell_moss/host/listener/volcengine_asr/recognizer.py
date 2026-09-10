import asyncio
import contextlib
import json
import logging
from typing import AsyncIterable, Callable, Optional

import numpy as np
import websockets
from ghoshell_common.contracts import LoggerItf
from ghoshell_common.helpers import uuid

from ghoshell_moss.contracts.asr import (
    ASR,
    ASRInfo,
    RecognitionStream,
    RecognitionPhase,
    RecognitionResult,
    RecognitionSegment,
)

from .config import VolcengineASRConfig, VolcengineASRParams
from .protocol import (
    ResponseMessageType,
    connect,
    nparray_to_bytes,
    parse_response,
    send_audio,
    send_init_request,
)

__all__ = ["VolcengineASR"]

# 收包短超时: 无尾包时周期性醒来检查 input_done, 让"音频断→自然退出"不依赖服务端及时回尾包.
_RECV_TIMEOUT = 1.0


class VolcengineASR(ASR):
    """火山引擎大模型 ASR 实现。recognize() 返回一条连续识别 stream, 每 stream 一条 WS."""

    def __init__(
        self,
        config: VolcengineASRConfig,
        *,
        logger: Optional[LoggerItf] = None,
    ):
        self._config = config
        self._logger = logger or logging.getLogger("moss")
        self._log_prefix = "[VolcengineASR]"
        self._closed = False
        self._error_callback: Callable[[Exception], None] | None = None

    # ── ASR contract: get_info / configure / on_error / recognize / close ──

    def get_info(self) -> ASRInfo:
        return ASRInfo(
            sample_rate=self._config.sample_rate,
            bits=self._config.bits,
            channel=self._config.channel,
            params_schema=VolcengineASRParams.model_json_schema(),
            params=self._config.params.model_dump(),
        )

    def configure(self, params: dict) -> None:
        self._config.params = VolcengineASRParams.model_validate(params)

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
        return _VolcengineRecognitionStream(
            config=self._config,
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


class _VolcengineRecognitionStream(RecognitionStream):
    """一条连续识别 stream: 音频流 → partial/clause/tail 结果流.

    1 stream = n segment. commit() 发负序号讯号 (切一段, WS 不关), 音频断 = 最后一次
    commit, 拿到尾包后自然结束. 懒开: WS 在首次 __anext__ 时建立.
    """

    def __init__(
        self,
        *,
        config: VolcengineASRConfig,
        audio_chunks: AsyncIterable[np.ndarray],
        stream_id: str | None,
        logger: LoggerItf,
        log_prefix: str,
        error_callback: Callable[[Exception], None] | None,
    ):
        self._config = config
        self._audio_chunks = audio_chunks
        self._logger = logger
        self._log_prefix = log_prefix
        self._error_callback = error_callback

        self._stream_id = stream_id or uuid()
        self._connection_id = uuid()
        self._segment_id = uuid()
        self._queue: asyncio.Queue[RecognitionResult | None] = asyncio.Queue()
        self._started = False
        self._session_task: asyncio.Task | None = None
        self._commit_event = asyncio.Event()
        self._input_done = False
        self._on_segment_callback: Callable[[RecognitionSegment], None] | None = None

        # 当前 segment 的 audio + text (有界: 一段的量), 每个 tail 切分一次.
        self._current_audio: list[np.ndarray] = []
        self._segment_text = ""
        self._total_samples = 0
        self._buffer_offset_ms = 0

        self._emitted_clauses = 0
        self._final_text = ""

    # ── RecognitionStream contract ──

    @property
    def stream_id(self) -> str:
        return self._stream_id

    def on_segment(self, callback: Callable[[RecognitionSegment], None]) -> None:
        self._on_segment_callback = callback

    def commit(self) -> None:
        self._commit_event.set()

    def is_input_done(self) -> bool:
        return self._input_done

    def __aiter__(self) -> RecognitionStream:
        return self

    async def __anext__(self) -> RecognitionResult:
        if not self._started:
            self._started = True
            self._session_task = asyncio.create_task(self._run_session())
        result = await self._queue.get()
        if result is None:
            raise StopAsyncIteration
        return result

    # ── session ──

    async def _run_session(self) -> None:
        try:
            async with await connect(self._config, self._connection_id) as ws:
                await send_init_request(ws, self._config, self._connection_id)
                send_task = asyncio.create_task(self._send_loop(ws))
                receive_task = asyncio.create_task(self._receive_loop(ws))
                await receive_task
                if not send_task.done():
                    send_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await send_task
        except Exception as e:
            self._report_error(e)
            await self._queue.put(self._tail_result(text=self._final_text, error=str(e)))
        finally:
            self._input_done = True
            await self._queue.put(None)

    # ── loops ──

    async def _send_loop(self, ws) -> None:
        seq = 1
        try:
            async for audio in self._audio_chunks:
                arr = np.asarray(audio).ravel()
                self._current_audio.append(arr)
                self._total_samples += arr.size
                await send_audio(ws, nparray_to_bytes(arr), seq, is_last=False)
                seq += 1
                if self._commit_event.is_set():
                    self._commit_event.clear()
                    # commit 讯号: 负序号切一段, WS 不关, 继续喂下一段.
                    await send_audio(ws, b"", seq, is_last=True)
                    seq += 1

            # 音频流结束: 最后一次 commit.
            await send_audio(ws, b"", seq, is_last=True)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._report_error(e)
        finally:
            self._input_done = True

    async def _receive_loop(self, ws) -> None:
        try:
            while True:
                try:
                    data = await asyncio.wait_for(ws.recv(), timeout=_RECV_TIMEOUT)
                except asyncio.TimeoutError:
                    # 无尾包时周期性醒来: 音频已断则自然退出.
                    if self._input_done:
                        break
                    continue
                except websockets.exceptions.ConnectionClosed:
                    await self._queue.put(
                        self._tail_result(text=self._final_text, error="connection closed")
                    )
                    break

                if not data:
                    continue

                response = parse_response(data)

                if response.message_type == ResponseMessageType.server_error:
                    error_msg = f"server error {response.error_code}: {response.payload}"
                    self._logger.error(
                        "%s %s, connection=%s",
                        self._log_prefix,
                        error_msg,
                        self._connection_id,
                    )
                    await self._queue.put(self._tail_result(text=self._final_text, error=error_msg))
                    break

                elif response.message_type == ResponseMessageType.server_ack:
                    continue

                elif response.message_type == ResponseMessageType.full_server_response:
                    if response.is_last:
                        self._final_text = self._extract_text(response.payload)
                        self._segment_text = self._final_text
                        await self._queue.put(self._tail_result(text=self._final_text))
                        self._cut_segment()
                        if self._input_done:
                            break
                        # 否则: commit 尾包, 切段后继续收下一段.
                    else:
                        for chunk in self._parse_utterances(response.payload):
                            await self._queue.put(chunk)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._report_error(e)

    # ── parse ──

    def _extract_text(self, payload: str) -> str:
        try:
            data = json.loads(payload)
            return data.get("result", {}).get("text", "")
        except Exception:
            return ""

    def _parse_utterances(self, payload: str) -> list[RecognitionResult]:
        """非尾包响应 → 逐分句吐 clause, 末尾未 definite 的吐 partial. 不切 segment."""
        try:
            data = json.loads(payload)
            result = data.get("result", {})
            text = result.get("text", "")
            self._final_text = text
            # 更新式: result.text 是全文 replace, 分句会校正之前的误识别 — 直接覆盖, 不 append.
            self._segment_text = text
            utterances = result.get("utterances", [])
            chunks: list[RecognitionResult] = []

            definite = [u for u in utterances if u.get("definite")]
            new = definite[self._emitted_clauses:]
            self._emitted_clauses = len(definite)
            for u in new:
                chunks.append(RecognitionResult(
                    stream_id=self._stream_id,
                    segment_id=self._segment_id,
                    phase=RecognitionPhase.CLAUSE,
                    text=u.get("text", ""),
                    start_ms=u.get("start_time", 0),
                    end_ms=u.get("end_time", 0),
                ))

            if utterances and not utterances[-1].get("definite"):
                chunks.append(RecognitionResult(
                    stream_id=self._stream_id,
                    segment_id=self._segment_id,
                    phase=RecognitionPhase.PARTIAL,
                    text=text,
                ))
            return chunks
        except Exception as e:
            self._logger.warning(
                "%s failed to parse result: %s, payload=%s",
                self._log_prefix,
                e,
                payload[:200],
            )
            return []

    # ── segment 切分 (每 tail 一次: 整段 audio + 累积 text 带走, segment_id 递增) ──

    def _tail_result(self, *, text: str, error: str = "") -> RecognitionResult:
        return RecognitionResult(
            stream_id=self._stream_id,
            segment_id=self._segment_id,
            phase=RecognitionPhase.TAIL,
            text=text,
            error=error,
        )

    def _stream_ms(self) -> int:
        return int(self._total_samples / self._config.sample_rate * 1000)

    def _cut_segment(self) -> None:
        start_ms = self._buffer_offset_ms
        end_ms = self._stream_ms()
        audio = (
            np.concatenate(self._current_audio)
            if self._current_audio
            else np.array([], dtype=np.int16)
        )
        segment = RecognitionSegment(
            id=self._segment_id,
            stream_id=self._stream_id,
            text=self._segment_text,
            start_ms=start_ms,
            end_ms=end_ms,
            sample_rate=self._config.sample_rate,
            bits=self._config.bits,
            channel=self._config.channel,
            audio=audio,
            offset_ms=self._buffer_offset_ms,
        )
        self._emit_segment(segment)

        self._current_audio.clear()
        self._segment_text = ""
        self._buffer_offset_ms = end_ms
        self._segment_id = uuid()

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
