import asyncio
import json
import logging
import os
import time
from typing import AsyncIterable, Optional

import numpy as np
import websockets
from ghoshell_common.contracts import LoggerItf
from ghoshell_common.helpers import uuid

from ghoshell_moss.contracts.asr import ASR, ASRResult

from .config import VolcengineASRConfig
from .protocol import (
    ResponseMessageType,
    connect,
    nparray_to_bytes,
    parse_response,
    send_audio,
    send_init_request,
)

__all__ = ["VolcengineASR"]

_ASR_ERROR_PREFIX = "__VOLCENGINE_ASR_ERROR__:"


class VolcengineASR(ASR):
    """火山引擎大模型 ASR 实现。每次 recognize 独立建立 WebSocket 连接。"""

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

    async def recognize(
        self,
        audio_chunks: AsyncIterable[np.ndarray],
    ) -> AsyncIterable[ASRResult]:
        if self._closed:
            raise RuntimeError("ASR is closed")

        connection_id = uuid()
        self._logger.info(
            "%s starting recognition, connection=%s",
            self._log_prefix,
            connection_id,
        )

        # 结果队列，receive_loop 生产，recognize 消费
        result_queue: asyncio.Queue[Optional[ASRResult]] = asyncio.Queue()

        async with await connect(self._config, connection_id) as ws:
            resolved = self._config.resolve_env()
            self._logger.info(
                "%s websocket connected, connection=%s url=%s resource=%s model=%s sample_rate=%s auth=%s logid=%s",
                self._log_prefix,
                connection_id,
                resolved.url,
                resolved.resource_id,
                resolved.model_name,
                resolved.sample_rate,
                "api_key" if resolved.api_key else "app_token",
                self._response_header(ws, "X-Tt-Logid") or "-",
            )
            await send_init_request(ws, self._config, connection_id)
            self._logger.info(
                "%s init request sent, connection=%s end_window=%sms force_to_speech=%sms",
                self._log_prefix,
                connection_id,
                resolved.end_window_size,
                resolved.force_to_speech_time,
            )

            send_task = asyncio.create_task(
                self._send_loop(ws, audio_chunks, connection_id)
            )
            receive_task = asyncio.create_task(
                self._receive_loop(ws, result_queue, connection_id)
            )

            try:
                while True:
                    result = await result_queue.get()
                    if result is None:
                        break
                    yield result
                    if result.is_final:
                        break
            finally:
                send_task.cancel()
                receive_task.cancel()
                try:
                    await send_task
                except asyncio.CancelledError:
                    pass
                try:
                    await receive_task
                except asyncio.CancelledError:
                    pass

        self._logger.info(
            "%s recognition done, connection=%s",
            self._log_prefix,
            connection_id,
        )

    async def _send_loop(
        self,
        ws: websockets.ClientConnection,
        audio_chunks: AsyncIterable[np.ndarray],
        connection_id: str,
    ) -> None:
        seq = 1
        sent_packets = 0
        sent_bytes = 0
        last_log_at = time.monotonic()
        resolved = self._config.resolve_env()
        samples_per_packet = max(
            1,
            int(resolved.sample_rate * resolved.audio_packet_ms / 1000) * max(1, resolved.channel),
        )
        pending = np.array([], dtype=np.int16)

        async def _send_pcm_packet(pcm: np.ndarray) -> None:
            nonlocal seq, sent_packets, sent_bytes, last_log_at
            if pcm.size == 0:
                return
            compressed = nparray_to_bytes(pcm.astype(np.int16, copy=False))
            seq = await send_audio(ws, compressed, seq, is_last=False)
            sent_packets += 1
            sent_bytes += int(pcm.nbytes)
            now = time.monotonic()
            if sent_packets == 1 or sent_packets % 25 == 0 or now - last_log_at >= 5.0:
                self._logger.info(
                    "%s audio sent, connection=%s packets=%d pcm_bytes=%d packet_ms=%d last_shape=%s last_dtype=%s",
                    self._log_prefix,
                    connection_id,
                    sent_packets,
                    sent_bytes,
                    resolved.audio_packet_ms,
                    tuple(pcm.shape),
                    str(pcm.dtype),
                )
                last_log_at = now

        try:
            async for audio in audio_chunks:
                pcm = np.asarray(audio, dtype=np.int16).reshape(-1)
                if pcm.size == 0:
                    continue
                pending = np.concatenate((pending, pcm))
                while pending.size >= samples_per_packet:
                    packet = pending[:samples_per_packet]
                    pending = pending[samples_per_packet:]
                    await _send_pcm_packet(packet)

            # 音频流结束，发送尾包
            if pending.size:
                await _send_pcm_packet(pending)
            self._logger.debug(
                "%s sending final audio packet, connection=%s",
                self._log_prefix,
                connection_id,
            )
            final_packet = nparray_to_bytes(np.array([], dtype=np.int16))
            seq = await send_audio(ws, final_packet, seq, is_last=True)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._logger.exception(
                "%s send loop error: %s, connection=%s",
                self._log_prefix,
                e,
                connection_id,
            )

    async def _receive_loop(
        self,
        ws: websockets.ClientConnection,
        result_queue: asyncio.Queue[Optional[ASRResult]],
        connection_id: str,
    ) -> None:
        received_count = 0
        try:
            while True:
                try:
                    data = await ws.recv()
                except websockets.exceptions.ConnectionClosed:
                    break

                if not data:
                    continue

                response = parse_response(data)
                received_count += 1
                if received_count <= 3 or response.message_type == ResponseMessageType.server_error:
                    self._logger.info(
                        "%s response received, connection=%s count=%d type=%s sequence=%s is_last=%s payload_len=%d",
                        self._log_prefix,
                        connection_id,
                        received_count,
                        response.message_type,
                        response.sequence,
                        response.is_last,
                        len(response.payload or ""),
                    )

                if response.message_type == ResponseMessageType.server_error:
                    message = (response.payload or "").strip()
                    self._logger.error(
                        "%s server error: code=%s message=%s connection=%s",
                        self._log_prefix,
                        response.error_code,
                        message[:500],
                        connection_id,
                    )
                    # 通用 ASR 合约不应把服务端错误伪装成用户说的话；默认只
                    # 返回空 final，让调用方结束本轮识别并查看日志。aEther
                    # listener 需要把错误码发布到运行时诊断 topic，因此通过
                    # VOLCENGINE_BM_ASR_PROPAGATE_ERRORS=1 显式启用哨兵文本。
                    text = ""
                    if os.environ.get("VOLCENGINE_BM_ASR_PROPAGATE_ERRORS") == "1":
                        text = f"{_ASR_ERROR_PREFIX}{response.error_code}|{message}"
                    await result_queue.put(ASRResult(text=text, is_final=True))
                    break

                elif response.message_type == ResponseMessageType.server_ack:
                    continue

                elif response.message_type == ResponseMessageType.full_server_response:
                    result = self._parse_result(response.payload)
                    if result is not None:
                        await result_queue.put(result)
                        if result.is_final:
                            break

        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._logger.exception(
                "%s receive loop error: %s, connection=%s",
                self._log_prefix,
                e,
                connection_id,
            )
        finally:
            # sentinel
            await result_queue.put(None)

    def _parse_result(self, payload: str) -> Optional[ASRResult]:
        try:
            data = json.loads(payload)
            text = data.get("result", {}).get("text", "")
            utterances = data.get("result", {}).get("utterances", [])
            is_final = any(
                bool(u.get("definite", False)) for u in utterances
            )
            return ASRResult(text=text, is_final=is_final)
        except Exception as e:
            self._logger.warning(
                "%s failed to parse result: %s, payload=%s",
                self._log_prefix,
                e,
                payload[:200],
            )
            return None

    @staticmethod
    def _response_header(ws: websockets.ClientConnection, name: str) -> str:
        response = getattr(ws, "response", None)
        headers = getattr(response, "headers", None)
        if not headers:
            return ""
        try:
            return str(headers.get(name, ""))
        except Exception:
            return ""

    async def close(self) -> None:
        self._closed = True
        self._logger.info("%s closed", self._log_prefix)
