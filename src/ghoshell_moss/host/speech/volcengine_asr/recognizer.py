import asyncio
import json
import logging
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
            await send_init_request(ws, self._config, connection_id)

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
        try:
            async for audio in audio_chunks:
                compressed = nparray_to_bytes(audio)
                await send_audio(ws, compressed, seq, is_last=False)
                seq += 1

            # 音频流结束，发送尾包
            self._logger.debug(
                "%s sending final audio packet, connection=%s",
                self._log_prefix,
                connection_id,
            )
            await send_audio(ws, b"", seq, is_last=True)
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
        try:
            while True:
                try:
                    data = await ws.recv()
                except websockets.exceptions.ConnectionClosed:
                    break

                if not data:
                    continue

                response = parse_response(data)

                if response.message_type == ResponseMessageType.server_error:
                    self._logger.error(
                        "%s server error: %s, connection=%s",
                        self._log_prefix,
                        response.error_code,
                        connection_id,
                    )
                    await result_queue.put(ASRResult(text="", is_final=True))
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
            is_final = False
            if utterances and len(utterances) > 0:
                # definite 表示是一个确定的分句
                is_final = bool(utterances[0].get("definite", False))
            return ASRResult(text=text, is_final=is_final)
        except Exception as e:
            self._logger.warning(
                "%s failed to parse result: %s, payload=%s",
                self._log_prefix,
                e,
                payload[:200],
            )
            return None

    async def close(self) -> None:
        self._closed = True
        self._logger.info("%s closed", self._log_prefix)
