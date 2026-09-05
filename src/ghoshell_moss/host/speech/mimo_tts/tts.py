import asyncio
import base64
import json
import logging
from typing import Any, Optional, AsyncIterator
import httpx
import numpy as np

from ghoshell_common.contracts import LoggerItf
from ghoshell_moss.contracts.speech import (
    TTS,
    TTSBatch,
    TTSItem,
    TTSInfo,
    TTSAudioCallback,
    AudioFormat,
    speech_tail,
)
from ghoshell_moss.core.helpers.asyncio_utils import ThreadSafeEvent
from ghoshell_moss.message import unique_id
from .config import MiMoSpeakerConf, MiMoTTSConf

__all__ = [
    "MiMoSpeakerConf",
    "MiMoTTSConf",
    "MiMoTTSBatch",
    "MiMoTTS",
]


class MiMoTTSBatch(TTSBatch):
    """MiMo HTTP TTS 批次 — 每个 batch 独立发起 HTTP 请求。"""

    def __init__(
            self,
            *,
            loop: asyncio.AbstractEventLoop,
            speaker: MiMoSpeakerConf,
            batch_id: str,
            model: str,
            base_url: str,
            api_key: str,
            sample_rate: int,
            stream: bool,
            request_timeout: float,
            tone: str,
            logger: LoggerItf,
            callback: Optional[TTSAudioCallback] = None,
    ):
        self._speaker_conf = speaker
        self._model = model
        self._base_url = base_url
        self._api_key = api_key
        self._sample_rate = sample_rate
        self._stream = stream
        self._request_timeout = request_timeout
        self._tone = tone
        self._logger = logger
        self._callback: Optional[TTSAudioCallback] = callback

        self._batch_id = batch_id or unique_id()
        self._committed = False
        self._running_loop = loop

        self._started = ThreadSafeEvent()
        self._done = ThreadSafeEvent()
        self._exception: Optional[Exception] = None

        self._chunks: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
        self._text_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._http_task: Optional[asyncio.Task] = None
        self._full_text: str = ""
        self._log_prefix = f"[MiMoTTSBatch][id={self._batch_id} tone={self._tone}]"

    def batch_id(self) -> str:
        return self._batch_id

    def feed(self, text: str):
        if self._done.is_set():
            return
        if text:
            self._running_loop.call_soon_threadsafe(self._text_queue.put_nowait, text)

    def commit(self):
        if self._committed:
            return
        self._committed = True
        self._logger.info("%s commited", self._log_prefix)
        self._running_loop.call_soon_threadsafe(self._text_queue.put_nowait, None)

    def is_committed(self) -> bool:
        return self._committed

    def is_closed(self) -> bool:
        return self._done.is_set()

    def is_started(self) -> bool:
        return self._started.is_set()

    async def start(self) -> None:
        self._started.set()

    def with_callback(self, callback: TTSAudioCallback) -> None:
        self._callback = callback

    async def wait_done(self, timeout: float | None = None):
        if timeout is not None and timeout > 0.0:
            await asyncio.wait_for(self._done.wait(), timeout=timeout)
        else:
            await self._done.wait()
        if self._exception is not None:
            raise self._exception

    async def close(self) -> None:
        if self._done.is_set():
            return
        self._done.set()
        if self._http_task is not None and not self._http_task.done():
            self._http_task.cancel()
            try:
                await self._http_task
            except asyncio.CancelledError:
                pass
        self._chunks.put_nowait(None)
        self._logger.info("%s closed", self._log_prefix)

    def fail(self, err: Exception) -> None:
        self._exception = err
        if not self._done.is_set():
            self._done.set()
        self._chunks.put_nowait(None)

    async def items(self) -> AsyncIterator[TTSItem]:
        # 拿不到 text-音频对齐时, 在最后一帧附上尾帧文本 (已喂文本尾部), 供 stopped_message 消费.
        prev_chunk = None
        while True:
            chunk = await self._chunks.get()
            if chunk is None:
                if prev_chunk is not None:
                    yield TTSItem(
                        tone=self._tone,
                        voice={},
                        audio_format=AudioFormat.PCM_S16LE.value,
                        channels=1,
                        sample_rate=self._sample_rate,
                        audio=prev_chunk,
                        text=speech_tail(self._full_text),
                    )
                return
            if prev_chunk is not None:
                yield TTSItem(
                    tone=self._tone,
                    voice={},
                    audio_format=AudioFormat.PCM_S16LE.value,
                    channels=1,
                    sample_rate=self._sample_rate,
                    audio=prev_chunk,
                    text="",
                )
            prev_chunk = chunk

    def _run_http(self):
        """Threadsafe entry: schedule _execute_http on the event loop."""
        if self._http_task is None:
            self._http_task = asyncio.create_task(self._execute_http())

    async def _execute_http(self):
        """后台 HTTP 请求：等待 start 和 commit，然后发送请求。"""
        try:
            await self._started.wait()
            if self._done.is_set():
                return

            # 收集文本直到 commit
            texts: list[str] = []
            while True:
                text = await self._text_queue.get()
                if text is None:
                    break
                texts.append(text)
            full_text = "".join(texts)
            self._full_text = full_text
            if not full_text.strip():
                self._logger.warning("%s empty text, skipping", self._log_prefix)
                self._chunks.put_nowait(None)
                self._done.set()
                return

            url = f"{self._base_url}/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "api-key": MiMoTTSConf.unwrap_env(self._api_key),
            }
            body = {
                "model": self._model,
                "messages": [
                    {"role": "user", "content": ""},
                    {"role": "assistant", "content": full_text},
                ],
                "audio": {
                    "voice": self._speaker_conf.voice,
                    "format": "pcm16" if self._stream else "wav",
                },
                "stream": self._stream,
            }

            self._logger.info("%s sending request to %s", self._log_prefix, url)
            async with httpx.AsyncClient(timeout=self._request_timeout) as client:
                if self._stream:
                    await self._stream_request(client, url, headers, body)
                else:
                    await self._nonstream_request(client, url, headers, body)
        except asyncio.CancelledError:
            self._logger.info("%s cancelled", self._log_prefix)
        except Exception as e:
            self._logger.exception("%s HTTP request failed: %s", self._log_prefix, e)
            self.fail(e)
        finally:
            if not self._done.is_set():
                self._done.set()
            self._chunks.put_nowait(None)
            self._logger.info("%s done", self._log_prefix)

    async def _stream_request(
            self,
            client: httpx.AsyncClient,
            url: str,
            headers: dict,
            body: dict,
    ):
        has_data = False
        async with client.stream("POST", url, json=body, headers=headers) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if self._done.is_set():
                    break
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    self._logger.warning("%s invalid JSON in SSE: %s", self._log_prefix, payload[:100])
                    continue
                choices = data.get("choices")
                if not choices:
                    continue
                audio = choices[0].get("delta", {}).get("audio")
                if not audio:
                    continue
                audio_b64 = audio.get("data", "")
                if not audio_b64:
                    continue
                try:
                    audio_bytes = base64.b64decode(audio_b64)
                except Exception:
                    self._logger.warning("%s base64 decode failed", self._log_prefix)
                    continue
                if len(audio_bytes) % 2 != 0:
                    self._logger.warning("%s odd audio bytes: %d", self._log_prefix, len(audio_bytes))
                    continue
                np_chunk = np.frombuffer(audio_bytes, dtype=np.int16)
                if not has_data:
                    self._logger.info("%s received first audio chunk", self._log_prefix)
                    has_data = True
                await self._chunks.put(np_chunk)
                if self._callback:
                    self._callback(np_chunk)

    async def _nonstream_request(
            self,
            client: httpx.AsyncClient,
            url: str,
            headers: dict,
            body: dict,
    ):
        response = await client.post(url, json=body, headers=headers)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices")
        if not choices:
            self._logger.warning("%s no choices in response", self._log_prefix)
            return
        audio = choices[0].get("message", {}).get("audio")
        if not audio:
            self._logger.warning("%s no audio in response", self._log_prefix)
            return
        audio_b64 = audio.get("data", "")
        if not audio_b64:
            self._logger.warning("%s no audio data in response", self._log_prefix)
            return
        audio_bytes = base64.b64decode(audio_b64)
        np_chunk = np.frombuffer(audio_bytes, dtype=np.int16)
        self._logger.info("%s received audio chunk: %d bytes", self._log_prefix, len(audio_bytes))
        await self._chunks.put(np_chunk)
        if self._callback:
            self._callback(np_chunk)


class MiMoTTS(TTS):
    """MiMo HTTP TTS 服务。每个 batch 独立 HTTP 请求，无持久连接。"""

    def __init__(
            self,
            *,
            conf: MiMoTTSConf | None = None,
            logger: LoggerItf | None = None,
    ):
        self._logger = logger or logging.getLogger("moss")
        self._log_prefix = "[MiMoTTS] "

        self._conf = conf or MiMoTTSConf()
        self._current_speaker: str = self._conf.default_speaker
        self._current_speaker_conf: MiMoSpeakerConf = self._conf.default_speaker_conf()

        self._starting = False
        self._started = False
        self._closing = False
        self._closed_event = ThreadSafeEvent()
        self._running_loop: Optional[asyncio.AbstractEventLoop] = None

        self._active_batches: set[MiMoTTSBatch] = set()
        self._default_tts_info = self.get_info()

    # ---- TTS contract ----

    def get_info(self) -> TTSInfo:
        return self._conf.to_tts_info(self._current_speaker)

    def use_tone(self, config_key: str) -> None:
        if config_key not in self._conf.speakers:
            raise LookupError(f"voice '{config_key}' not found")
        self._current_speaker = config_key
        self._current_speaker_conf = self._conf.speakers[config_key].model_copy(deep=True)

    def current_tone(self) -> str:
        return self._current_speaker

    def set_voice(self, config: dict[str, Any]) -> None:
        # MiMo uses text-based style control, not parametric voice config
        pass

    def get_voice(self) -> dict[str, Any]:
        return {}

    def new_batch(
            self,
            batch_id: str = "",
            *,
            callback: TTSAudioCallback | None = None,
            voice: dict[str, Any] | None = None,
            tone: str | None = None,
    ) -> TTSBatch:
        self._check_running()
        speaker_conf = self._current_speaker_conf
        if tone is not None and tone != self._current_speaker:
            speaker_conf = self._conf.speakers.get(tone, speaker_conf)
        batch = MiMoTTSBatch(
            loop=self._running_loop,
            speaker=speaker_conf,
            batch_id=batch_id,
            model=self._conf.model,
            base_url=self._conf.base_url,
            api_key=self._conf.api_key,
            sample_rate=self._conf.sample_rate,
            stream=self._conf.stream,
            request_timeout=self._conf.request_timeout,
            tone=tone or self._current_speaker,
            logger=self._logger,
            callback=callback,
        )
        batch._run_http()
        self._active_batches.add(batch)
        return batch

    async def start(self) -> None:
        if self._starting:
            return
        self._starting = True
        self._running_loop = asyncio.get_running_loop()
        self._started = True
        self._logger.info("%s started", self._log_prefix)

    async def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        batches = list(self._active_batches)
        self._active_batches.clear()
        for batch in batches:
            await batch.close()
        self._closed_event.set()
        self._logger.info("%s closed", self._log_prefix)

    async def clear(self) -> None:
        self._check_running()
        batches = list(self._active_batches)
        self._active_batches.clear()
        for batch in batches:
            await batch.close()

    def _check_running(self):
        if not self._started or self._closing:
            raise RuntimeError("MiMoTTS is not running")

    async def wait_closed(self) -> None:
        await self._closed_event.wait()
