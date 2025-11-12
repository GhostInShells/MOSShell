import logging
from typing import Optional, Any

import numpy as np
from typing_extensions import Literal
from pydantic import Field
from ghoshell_moss.speech.volcengine_tts.protocol import (
    start_connection, start_session, finish_session, finish_connection,
    receive_message, cancel_session, task_request,
    EventType, wait_for_event, MsgType,
)
from ghoshell_moss.concepts.speech import TTS, TTSBatch, TTSInfo, TTSAudioCallback, AudioFormat
from ghoshell_moss.helpers.asyncio_utils import ThreadSafeEvent
from ghoshell_common.helpers import uuid
from ghoshell_common.contracts import LoggerItf
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK
from websockets import connect, ClientConnection
from collections import deque
import os
import json
import asyncio

__all__ = [
    'ChineseVoiceEmotion', 'EnglishVoiceEmotion',
    'SpeakerConf', 'SpeakerInfo', 'SpeakerTypes', 'VoiceConf',
    'VolcengineTTSConf',
    'VolcengineTTS', 'VolcengineTTSBatch',
]

ChineseVoiceEmotion = Literal[
    "happy",  # 开心
    "sad",  # 悲伤
    "angry",  # 生气
    "surprised",  # 惊讶
    "fear",  # 恐惧
    "hate",  # 厌恶
    "excited",  # 激动
    "coldness",  # 冷漠
    "neutral",  # 中性
    "depressed",  # 沮丧
    "lovey-dovey",  # 撒娇
    "shy",  # 害羞
    "comfort",  # 安慰鼓励
    "tension",  # 咆哮/焦急
    "tender",  # 温柔
    "storytelling",  # 讲故事 / 自然讲述
    "radio",  # 情感电台
    "magnetic",  # 磁性
    "advertising",  # 广告营销
    "vocal-fry",  # 气泡音
    "ASMR",  # 低语
    "news",  # 新闻播报
    "entertainment",  # 娱乐八卦
    "dialect"  # 方言
]

# 英文音色及其对应的情感参数
EnglishVoiceEmotion = Literal[
    "neutral",  # 中性
    "happy",  # 愉悦
    "angry",  # 愤怒
    "sad",  # 悲伤
    "excited",  # 兴奋
    "chat",  # 对话 / 闲聊
    "ASMR",  # 低语
    "warm",  # 温暖
    "affectionate",  # 深情
    "authoritative"  # 权威
]

from typing import Dict, Literal
from pydantic import BaseModel


# 定义 Speaker 信息模型
class SpeakerInfo(BaseModel):
    display_name: str
    language: str
    supports_english: bool
    use_case: str

    def description(self) -> str:
        return f"language: ({self.language}), support english: {self.supports_english}, use case: {self.use_case}"


# 定义所有 Speaker 类型
SpeakerTypes = Literal[
    "vivi",
    "zh_male_dayi_saturn_bigtts",
    "zh_female_mizai_saturn_bigtts",
    "zh_female_jitangnv_saturn_bigtts",
    "zh_female_meilinvyou_saturn_bigtts",
    "zh_female_santongyongns_saturn_bigtts",
    "zh_male_ruyayichen_saturn_bigtts",
    "saturn_zh_female_keainvsheng_tob",
    "saturn_zh_female_tiaopigongzhu_tob",
    "saturn_zh_male_shuanglangshaonian_tob",
    "saturn_zh_male_tiancaitongzhuo_tob",
    "saturn_zh_female_cancan_tob"
]

# 创建 Speaker 信息字典
SPEAKER_INFO_MAP: Dict[SpeakerTypes, SpeakerInfo] = {
    "vivi": SpeakerInfo(
        display_name="vivi",
        language="中文、英语",
        supports_english=False,
        use_case="视频配音"
    ),
    "zh_male_dayi_saturn_bigtts": SpeakerInfo(
        display_name="大壹",
        language="中文",
        supports_english=False,
        use_case="视频配音"
    ),
    "zh_female_mizai_saturn_bigtts": SpeakerInfo(
        display_name="黑猫侦探社咪仔",
        language="中文",
        supports_english=False,
        use_case="视频配音"
    ),
    "zh_female_jitangnv_saturn_bigtts": SpeakerInfo(
        display_name="鸡汤女",
        language="中文",
        supports_english=False,
        use_case="视频配音"
    ),
    "zh_female_meilinvyou_saturn_bigtts": SpeakerInfo(
        display_name="魅力女友",
        language="中文",
        supports_english=False,
        use_case="视频配音"
    ),
    "zh_female_santongyongns_saturn_bigtts": SpeakerInfo(
        display_name="流畅女声",
        language="中文",
        supports_english=False,
        use_case="视频配音"
    ),
    "zh_male_ruyayichen_saturn_bigtts": SpeakerInfo(
        display_name="儒雅逸辰",
        language="中文",
        supports_english=False,
        use_case="角色扮演"
    ),
    "saturn_zh_female_keainvsheng_tob": SpeakerInfo(
        display_name="可爱女生",
        language="中文",
        supports_english=False,
        use_case="角色扮演"
    ),
    "saturn_zh_female_tiaopigongzhu_tob": SpeakerInfo(
        display_name="调皮公主",
        language="中文",
        supports_english=False,
        use_case="角色扮演"
    ),
    "saturn_zh_male_shuanglangshaonian_tob": SpeakerInfo(
        display_name="爽朗少年",
        language="中文",
        supports_english=False,
        use_case="角色扮演"
    ),
    "saturn_zh_male_tiancaitongzhuo_tob": SpeakerInfo(
        display_name="天才同桌",
        language="中文",
        supports_english=False,
        use_case="角色扮演"
    ),
    "saturn_zh_female_cancan_tob": SpeakerInfo(
        display_name="知性灿灿",
        language="中文",
        supports_english=False,
        use_case="角色扮演"
    )
}

# 获取所有 Speaker 类型的列表
ALL_SPEAKER_TYPES = list(SPEAKER_INFO_MAP.keys())


class User(BaseModel):
    uid: str = Field(default="", description="")


class AudioParams(BaseModel):
    format: Literal["mp3", "pcm", "ogg_opus"] = Field(default="pcm")
    sample_rate: int = Field(
        default=44100, description="8000,16000,22050,24000,32000,44100,48000")
    loudness_rate: Optional[int] = Field(default=0)
    speech_rate: Optional[int] = Field(default=0)
    emotion: Optional[ChineseVoiceEmotion] = Field(default="neutral")


class ReqParams(BaseModel):
    audio_params: AudioParams = Field(default_factory=AudioParams)
    speaker: str = Field(default="zh_female_cancan_mars_bigtts")
    additions: Optional[str] = Field(default=None)


class Session(BaseModel):
    """
    session 数据.
    """
    user: User = Field(default_factory=User)
    event: int = EventType.StartSession.value
    req_params: ReqParams = Field(default_factory=ReqParams)

    def to_payload_bytes(self) -> bytes:
        config = self
        data = config.model_dump_json(exclude_none=True)
        return data.encode()

    def to_payload_str(self) -> str:
        config = self
        data = config.model_dump_json(exclude_none=True)
        return data

    def to_request_payload_bytes(self, text: str) -> bytes:
        data = self.model_dump(exclude_none=True)
        data["req_params"]["text"] = text
        data["event"] = EventType.TaskRequest.value
        j = json.dumps(data, ensure_ascii=False)
        return j.encode()


class VoiceConf(BaseModel):
    speech_rate: Optional[int] = Field(
        default=None,
        description="语速，取值范围[-50,100]，100代表2.0倍速，-50代表0.5倍数. 0是正常", ge=-50, le=100,
    )
    loudness_rate: Optional[int] = Field(
        default=None,
        description="音量，取值范围[-50,100]，100代表2.0倍音量，-50代表0.5倍音量. 0是正常", ge=-50, le=100,
    )
    emotion: Optional[ChineseVoiceEmotion] = Field(
        default=None,
        description="声音情绪, 拥有多种可选择的声音情绪."
    )


class SpeakerConf(BaseModel):
    """
    角色配置, 可以更改.
    """
    tone: str = Field(default="saturn_zh_female_cancan_tob")
    description: str = Field(default="", description="角色的描述")
    resource_id: Optional[str] = Field(default=None, description="使用声音复刻的独立的资源")
    voice: VoiceConf = Field(
        default_factory=VoiceConf,
        description="声音配置"
    )

    def to_voice_conf(self) -> Dict:
        return self.model_dump(exclude={"resource_id"})


_Head = Dict[str, Any]
_Url = str


class VolcengineTTSConf(BaseModel):
    """
    火山引擎 tts 基础配置.
    """
    app_key: str = Field(default="$VOLCENGINE_STREAM_TTS_APP")
    access_token: str = Field(default="$VOLCENGINE_STREAM_TTS_ACCESS_TOKEN")
    resource_id: str = Field(default="seed-tts-2.0", description="官方的默认资源")
    sample_rate: int = Field(default=44100, description="生成音频的采样率要求.")
    audio_format: Literal['pcm'] = Field(default="pcm", description="默认可用的数据格式")

    disconnect_on_idle: int = Field(
        default=100,
        description="闲置多少秒后退出",
    )

    disable_markdown_filter: bool = Field(
        default=True, description="支持朗读 markdown 格式. ")
    url: str = Field(
        default="wss://openspeech.bytedance.com/api/v3/tts/bidirection",
        description="火山的流式语音模型的地址",
    )

    speakers: Dict[str, SpeakerConf] = Field(
        default_factory=lambda: {
            name: SpeakerConf(tone=name, description=speaker_info.description())
            for name, speaker_info in SPEAKER_INFO_MAP.items()
        },
        description="the speakers list",
    )
    default_speaker: str = Field(
        default="default",
        description="the default speaker",
    )

    @classmethod
    def unwrap_env(cls, value: str, default: str = "") -> str:
        if value.startswith('$'):
            return os.environ.get(value[1:], default)
        return value or default

    def default_speaker_conf(self) -> SpeakerConf:
        conf = self.speakers.get(self.default_speaker, None)
        if conf is not None:
            return conf.model_copy(deep=True)
        conf = SpeakerConf()
        return conf

    def gen_header(self, *, connection_id: str = "", resource_id: Optional[str] = None) -> _Head:
        """
        :return: (header, url)
        """
        connection_id = connection_id or uuid()
        ws_header = {
            "X-Api-App-Key": self.unwrap_env(self.app_key),
            "X-Api-Access-Key": self.unwrap_env(self.access_token),
            "X-Api-Resource-Id": resource_id or self.resource_id,
            "X-Api-Connect-Id": connection_id,
        }
        return ws_header

    def to_session(self, speaker: SpeakerConf) -> Session:
        # 生成 additions.
        additions_data = dict(
            disable_markdown_filter=self.disable_markdown_filter,
        )
        additions = json.dumps(additions_data)
        return Session(
            speaker=speaker.tone,
            req_params=dict(
                audio_params=AudioParams(
                    format=self.audio_format,
                    sample_rate=self.sample_rate,
                    loudness_rate=speaker.voice.loudness_rate,
                    speech_rate=speaker.voice.speech_rate,
                    emotion=speaker.voice.emotion,
                ),
                speaker=speaker.tone,
                additions=additions,
            ),
        )

    def to_tts_info(self, current_voice: str = "") -> TTSInfo:
        return TTSInfo(
            sample_rate=self.sample_rate,
            channels=1,
            audio_format=AudioFormat.PCM_S16LE.value,
            voice_schema=VoiceConf.model_json_schema(),
            voices={key: value.to_voice_conf() for key, value in self.speakers.items()},
            current_voice=current_voice or self.default_speaker,
        )


class VolcengineTTSBatch(TTSBatch):

    def __init__(
            self,
            *,
            loop: asyncio.AbstractEventLoop,
            speaker: SpeakerConf,
            batch_id: str = "",
            callback: Optional[TTSAudioCallback] = None,
    ):
        self.speaker = speaker
        self.callback = callback
        self.started = ThreadSafeEvent()
        self.committed = False
        self.done = ThreadSafeEvent()
        self.text_buffer = ""
        self.exception: Optional[Exception] = None
        self._running_loop = loop
        self._has_valid_text = False
        self._batch_id = batch_id or uuid()
        self._text_lock = asyncio.Lock()
        self.texts: asyncio.Queue[str | None] = asyncio.Queue()

    def batch_id(self) -> str:
        return self._batch_id

    def with_callback(self, callback: TTSAudioCallback) -> None:
        self.callback = callback

    def fail(self, reason: str) -> None:
        self.exception = RuntimeError(reason)
        self.done.set()
        self.commit()

    def feed(self, text: str):
        self.text_buffer += text
        # 已经有过数据了.
        if self._has_valid_text:
            self._running_loop.call_soon_threadsafe(self.texts.put_nowait, text)
        # 这里只能 lstrip
        elif stripped := self.text_buffer.lstrip():
            self._running_loop.call_soon_threadsafe(self.texts.put_nowait, stripped)
            self._has_valid_text = True

    def commit(self):
        self.committed = True
        self._running_loop.call_soon_threadsafe(self.texts.put_nowait, None)
        if not self.text_buffer.strip():
            self.done.set()

    async def close(self) -> None:
        self.commit()
        self.done.set()

    async def wait_until_done(self, timeout: float | None = None):
        if timeout is not None and timeout > 0.0:
            await asyncio.wait_for(self.done.wait(), timeout=timeout)
        else:
            await self.done.wait()
        # 抛出异常.
        if self.exception is not None:
            raise self.exception


class VolcengineTTS(TTS):

    def __init__(
            self,
            *,
            conf: VolcengineTTSConf | None = None,
            logger: LoggerItf | None = None,
    ):
        self.logger = logger or logging.getLogger("volcengine.tts")

        # ---- 配置状态 --- #
        # 当前生成的 Batches.
        self._conf = conf or VolcengineTTSConf()
        self._current_speaker: str = self._conf.default_speaker
        self._current_speaker_conf: SpeakerConf = self._conf.default_speaker_conf()

        # --- runtime --- #
        self._starting = False
        self._started = True
        self._running_loop: Optional[asyncio.AbstractEventLoop] = None
        self._main_loop_task: Optional[asyncio.Task] = None
        self._closing_event = ThreadSafeEvent()
        self._closed_event = ThreadSafeEvent()

        self._tts_connection_conf = self._current_speaker_conf

        self._pending_batches_queue: asyncio.Queue[VolcengineTTSBatch] = asyncio.Queue()
        self._unfinished_batches: deque[VolcengineTTSBatch] = deque()
        self._running_batch: Optional[VolcengineTTSBatch] = None
        self._has_any_batch_event = asyncio.Event()

        self._consume_pending_batches_task: Optional[asyncio.Task] = None

    def get_info(self) -> TTSInfo:
        return self._conf.to_tts_info(self._current_speaker)

    def use_voice(self, config_key: str) -> None:
        if config_key not in self._conf.speakers:
            raise LookupError(f"The voice {config_key} not found")
        conf = self._conf.speakers[config_key]
        self._current_speaker = config_key
        self._current_speaker_conf = conf.model_copy(deep=True)

    def set_voice(self, config: Dict[str, Any]) -> None:
        voice = VoiceConf(**config)
        self._current_speaker_conf.voice = voice

    def _check_running(self) -> None:
        if not self._started or self._closing_event.is_set():
            raise RuntimeError("TTS is closed")

    def new_batch(self, batch_id: str = "", *, callback: TTSAudioCallback | None = None) -> TTSBatch:
        self._check_running()
        batch = self._create_batch(batch_id, callback)
        self._pending_batches_queue.put_nowait(batch)
        self._has_any_batch_event.set()
        return batch

    def _create_batch(self, batch_id: str = "", callback: TTSAudioCallback | None = None) -> VolcengineTTSBatch:
        speaker_conf = self._current_speaker_conf
        tts_batch = VolcengineTTSBatch(
            loop=self._running_loop,
            speaker=speaker_conf,
            batch_id=batch_id,
            callback=callback,
        )
        return tts_batch

    async def _main_loop(self):
        """ tts main connection loop"""
        # 没有关闭前, 一直执行这个循环.
        try:
            while not self._closing_event.is_set():
                if len(self._unfinished_batches) > 0:
                    batch = self._unfinished_batches.popleft()
                else:
                    # 等待到有 batch 存在.
                    await self._has_any_batch_event.wait()
                    if self._pending_batches_queue.empty():
                        # 发现实际上没有 batch.
                        self._has_any_batch_event.clear()
                        continue
                    # 等待一个 connection loop 完成. 要求不会抛出任何异常. 除了 cancel.
                    batch = await self._pending_batches_queue.get()
                # 这个 loop 会持续消费 batch, 直到超过等待时间还没有新 batch 为止.
                task = asyncio.create_task(self._start_consuming_batch_loop(batch))
                # 创建一个可以 cancel 的 task. 它自己应该不要抛出 cancel 异常.
                self._consume_pending_batches_task = task
                # 阻塞等待这个消费循环结束.
                try:
                    await task
                finally:
                    self._consume_pending_batches_task = None
        except asyncio.CancelledError:
            # 不需要记录.
            pass
        finally:
            self._consume_pending_batches_task = None
            self.logger.info("TTS main loop is closed")

    async def _start_consuming_batch_loop(self, batch: VolcengineTTSBatch):
        try:
            if batch.done.is_set():
                # 已经被关闭了.
                return
            speaker = batch.speaker
            # 当前火山的 resource id
            resource_id = speaker.resource_id or self._conf.resource_id
            connection_id = uuid()
            header = self._conf.gen_header(connection_id=connection_id, resource_id=resource_id)
            url = self._conf.url
            # 创建初始连接.
            self.logger.info("prepare to connect to %s with header %s", url, header)
            async with connect(url, additional_headers=header) as ws:
                # 建连确认.
                await start_connection(ws)
                self.logger.debug("start connection %s", connection_id)
                # 接受确认的事件. 完成握手.
                await wait_for_event(
                    ws,
                    MsgType.FullServerResponse,
                    EventType.ConnectionStarted,
                )
                self.logger.debug("connection %s started", connection_id)

                # 消费完第一个 batch.
                goon = await self._consume_batch_in_connection(batch, connection=ws, current_resource_id=resource_id)
                # 消费后续的 batch.
                if goon:
                    await self._consume_pending_batches(connection=ws, resource_id=resource_id)
                # 全部结束了, 就退出来. 等待外层继续调度.
                self.logger.info("consume batch loop %s is done", connection_id)
                # 发送退出信号. 不等待握手了.
                await finish_connection(ws)

        except ConnectionClosedOK:
            self.logger.info("TTS connection closed ok")
        except ConnectionClosed:
            self.logger.info("TTS connection closed")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.logger.exception(e)

    async def _consume_batch_in_connection(
            self,
            batch: VolcengineTTSBatch,
            connection: ClientConnection,
            current_resource_id: str,
    ) -> bool:
        if batch.done.is_set():
            return True
        batch_id = batch.batch_id()
        try:
            self._running_batch = batch
            resource_id = batch.speaker.resource_id or current_resource_id
            if resource_id != current_resource_id:
                # 连接不一致, 将未完成的 batch 入队, 关闭整个连接.
                self._unfinished_batches.append(batch)
                return False

            session = self._conf.to_session(batch.speaker)
            # 开启 session.
            await start_session(
                connection, session.to_payload_bytes(), batch_id,
            )
            # 等待拿到 session 启动的事件.
            await wait_for_event(
                connection, MsgType.FullServerResponse, EventType.SessionStarted
            )
            # 开始发送文本的流程.
            send_task = asyncio.create_task(self._send_batch_text_to_server(batch, session, connection))
            # 开始接受音频的流程.
            receive_task = asyncio.create_task(self._receive_batch_audio_from_server(batch, connection))
            # 等两个都完成, 才能进入下一步.
            send_and_receive = asyncio.gather(send_task, receive_task)
            batch_closed = asyncio.create_task(batch.done.wait())
            done, pending = await asyncio.wait([send_and_receive, batch_closed], return_when=asyncio.FIRST_COMPLETED)
            if batch_closed in done:
                if not receive_task.done():
                    receive_task.cancel()
                if not send_task.done():
                    send_task.cancel()
            await send_and_receive

            # 正常完成返回 true
            return True
        except ValueError as e:
            # todo: log update
            self.logger.exception(e)
        finally:
            batch.done.set()
            self._running_batch = None

    async def _send_batch_text_to_server(
            self,
            batch: VolcengineTTSBatch,
            session: Session,
            connection: ClientConnection,
    ) -> None:
        batch_id = batch.batch_id()
        try:
            while not batch.done.is_set():
                # 发送文本.
                text = await batch.texts.get()
                if text is None:
                    # 拿到了毒丸.
                    break
                # 发送给服务端.
                payload = session.to_request_payload_bytes(text)
                await task_request(
                    connection,
                    payload,
                    session_id=batch_id,
                )
            if batch.committed:
                await finish_session(connection, batch_id)
            else:
                # 提前被中断了, 都没有正确提交.
                await cancel_session(connection, batch_id)

        except asyncio.CancelledError:
            pass
        except (ConnectionClosedOK, ConnectionClosed):
            raise
        except Exception as e:
            self.logger.exception(e)
            batch.fail(str(e))
            # 特殊的错误, 则关闭 batch.
            await batch.close()
        finally:
            self.logger.info("batch %s send text done", batch_id)

    async def _receive_batch_audio_from_server(
            self,
            batch: VolcengineTTSBatch,
            connection: ClientConnection,
    ) -> None:
        callback = batch.callback
        try:
            batch_id = batch.batch_id()
            while not batch.done.is_set():
                msg = await receive_message(connection)
                self.logger.debug("session %s receive message %s", batch_id, msg)
                if msg.type == MsgType.Error:
                    self.logger.error(f"batch %s received error message {msg}", batch_id)
                    batch.done.set()
                    break
                elif msg.type == MsgType.FullServerResponse:
                    if msg.event in {EventType.SessionFinished, EventType.SessionCanceled}:
                        # todo: log
                        self.logger.info("session finished %s", batch_id)
                        # break the loop
                        break
                    elif msg.event == EventType.TTSSentenceStart:
                        # todo: 首包埋点.
                        pass
                    elif msg.event == EventType.TTSSentenceEnd:
                        # todo: 尾包埋点.
                        pass

                if msg.type == MsgType.AudioOnlyServer:
                    # 首包
                    audio_data = msg.payload
                    if msg.session_id != batch_id:
                        self.logger.info("session id mismatch %s to batch %s", msg.session_id, batch_id)
                        continue
                    if len(audio_data) > 0 and callback:
                        # todo: 先写死是 int16
                        np_data = np.frombuffer(audio_data, dtype=np.int16)
                        callback(np_data)
            self.logger.info("batch %s receive task done", batch_id)
        finally:
            # batch 永远要设置为关闭.
            batch.done.set()

    async def _consume_pending_batches(self, connection: ClientConnection, resource_id: str) -> None:
        while not self._closing_event.is_set():
            # 尝试获取一个最新的 batch.
            try:
                # 等待下一个 batch, 直到超时为止. 超时后关闭 connection, 避免浪费服务端.
                batch = await asyncio.wait_for(self._pending_batches_queue.get(), timeout=self._conf.disconnect_on_idle)
                # 完成对这个 batch 的执行.
                goon = await self._consume_batch_in_connection(batch, connection, resource_id)
                if not goon:
                    # 这个 loop 不行了, 要从头开始运行.
                    break
            except asyncio.TimeoutError:
                # 超时还没拿到新的 batch, tts 就关闭 connection 了.
                self.logger.info(
                    "close connection after disconnect timeout %s",
                    self._conf.disconnect_on_idle,
                )
                return

    async def clear(self) -> None:
        self._check_running()
        # 清空通知事件.
        self._has_any_batch_event.clear()
        # 清空老队列.
        old_queue = self._pending_batches_queue
        self._pending_batches_queue = asyncio.Queue()
        # 取消运行中的消费对象.
        if self._consume_pending_batches_task is not None:
            self._consume_pending_batches_task.cancel()
        # 设置所有元素为废弃. 速度应该挺快的.
        while not old_queue.empty():
            batch = await old_queue.get()
            await batch.close()

    async def start(self) -> None:
        if self._starting:
            return
        self._starting = True
        self._running_loop = asyncio.get_running_loop()
        self._main_loop_task = asyncio.create_task(self._main_loop())
        self._started = True

    async def close(self) -> None:
        if self._closing_event.is_set():
            return
        self._closing_event.set()
        if self._main_loop_task is not None:
            self._main_loop_task.cancel()
            try:
                await self._main_loop_task
            except asyncio.CancelledError:
                pass
            finally:
                self._main_loop_task = None
        self._closed_event.set()
