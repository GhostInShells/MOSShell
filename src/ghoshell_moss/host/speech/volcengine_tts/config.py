import orjson as json
import os
from enum import IntEnum
from typing import Any, Literal, Optional

from ghoshell_moss.message import unique_id
from pydantic import Field

from ghoshell_moss.contracts.speech import AudioFormat, TTSInfo

from pydantic import BaseModel

__all__ = [
    'ChineseVoiceEmotion', 'EnglishVoiceEmotion',
    'EventType',
    'SpeakerInfo', 'SPEAKER_INFO_MAP',
    'Session', 'AudioParams', 'ReqParams',
    'SpeakerConf', 'SpeakerTypes',
    'VolcengineTTSConf', 'User',
    'VoiceConf',
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
    "dialect",  # 方言
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
    "authoritative",  # 权威
]


class EventType(IntEnum):
    """Event type enumeration"""

    None_ = 0  # Default event

    # 1 ~ 49 Upstream Connection events
    StartConnection = 1
    StartTask = 1  # Alias of StartConnection
    FinishConnection = 2
    FinishTask = 2  # Alias of FinishConnection

    # 50 ~ 99 Downstream Connection events
    ConnectionStarted = 50  # Connection established successfully
    TaskStarted = 50  # Alias of ConnectionStarted
    ConnectionFailed = 51  # Connection failed (possibly due to authentication failure)
    TaskFailed = 51  # Alias of ConnectionFailed
    ConnectionFinished = 52  # Connection ended
    TaskFinished = 52  # Alias of ConnectionFinished

    # 100 ~ 149 Upstream Session events
    StartSession = 100
    CancelSession = 101
    FinishSession = 102

    # 150 ~ 199 Downstream Session events
    SessionStarted = 150
    SessionCanceled = 151
    SessionFinished = 152
    SessionFailed = 153
    UsageResponse = 154  # Usage response
    ChargeData = 154  # Alias of UsageResponse

    # 200 ~ 249 Upstream general events
    TaskRequest = 200
    UpdateConfig = 201

    # 250 ~ 299 Downstream general events
    AudioMuted = 250

    # 300 ~ 349 Upstream TTS events
    SayHello = 300

    # 350 ~ 399 Downstream TTS events
    TTSSentenceStart = 350
    TTSSentenceEnd = 351
    TTSResponse = 352
    TTSEnded = 359
    PodcastRoundStart = 360
    PodcastRoundResponse = 361
    PodcastRoundEnd = 362

    # 450 ~ 499 Downstream ASR events
    ASRInfo = 450
    ASRResponse = 451
    ASREnded = 459

    # 500 ~ 549 Upstream dialogue events
    ChatTTSText = 500  # (Ground-Truth-Alignment) text for speech synthesis

    # 550 ~ 599 Downstream dialogue events
    ChatResponse = 550
    ChatEnded = 559

    # 650 ~ 699 Downstream dialogue events
    # Events for source (original) language subtitle
    SourceSubtitleStart = 650
    SourceSubtitleResponse = 651
    SourceSubtitleEnd = 652
    # Events for target (translation) language subtitle
    TranslationSubtitleStart = 653
    TranslationSubtitleResponse = 654
    TranslationSubtitleEnd = 655

    def __str__(self) -> str:
        return self.name or f"EventType({self.value})"


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
    # Saturn 系列
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
    "saturn_zh_female_cancan_tob",
        # 豆包 Seed-TTS 2.0 系列
    "zh_female_vv_uranus_bigtts",
    "zh_female_xiaohe_uranus_bigtts",
    "zh_male_m191_uranus_bigtts",
    "zh_male_taocheng_uranus_bigtts",
    "en_male_tim_uranus_bigtts",
    "en_female_dacey_uranus_bigtts",
    "en_female_stokie_uranus_bigtts",
    "zh_male_liufei_uranus_bigtts",
    "zh_female_qingxinnvsheng_uranus_bigtts",
    "zh_female_cancan_uranus_bigtts",
    "zh_female_sajiaoxuemei_uranus_bigtts",
    "zh_female_tianmeixiaoyuan_uranus_bigtts",
    "zh_female_tianmeitaozi_uranus_bigtts",
    "zh_female_shuangkuaisisi_uranus_bigtts",
    "zh_female_peiqi_uranus_bigtts",
]

# 创建 Speaker 信息字典
SPEAKER_INFO_MAP: dict[SpeakerTypes, SpeakerInfo] = {
    "zh_male_dayi_saturn_bigtts": SpeakerInfo(
        display_name="大壹", language="中文", supports_english=False, use_case="视频配音"
    ),
    "zh_female_mizai_saturn_bigtts": SpeakerInfo(
        display_name="黑猫侦探社咪仔", language="中文", supports_english=False, use_case="视频配音"
    ),
    "zh_female_jitangnv_saturn_bigtts": SpeakerInfo(
        display_name="鸡汤女", language="中文", supports_english=False, use_case="视频配音"
    ),
    "zh_female_meilinvyou_saturn_bigtts": SpeakerInfo(
        display_name="魅力女友", language="中文", supports_english=False, use_case="视频配音"
    ),
    "zh_female_santongyongns_saturn_bigtts": SpeakerInfo(
        display_name="流畅女声", language="中文", supports_english=False, use_case="视频配音"
    ),
    "zh_male_ruyayichen_saturn_bigtts": SpeakerInfo(
        display_name="儒雅逸辰", language="中文", supports_english=False, use_case="角色扮演"
    ),
    "saturn_zh_female_keainvsheng_tob": SpeakerInfo(
        display_name="可爱女生", language="中文", supports_english=False, use_case="角色扮演"
    ),
    "saturn_zh_female_tiaopigongzhu_tob": SpeakerInfo(
        display_name="调皮公主", language="中文", supports_english=False, use_case="角色扮演"
    ),
    "saturn_zh_male_shuanglangshaonian_tob": SpeakerInfo(
        display_name="爽朗少年", language="中文", supports_english=False, use_case="角色扮演"
    ),
    "saturn_zh_male_tiancaitongzhuo_tob": SpeakerInfo(
        display_name="天才同桌", language="中文", supports_english=False, use_case="角色扮演"
    ),
    "saturn_zh_female_cancan_tob": SpeakerInfo(
        display_name="知性灿灿", language="中文", supports_english=False, use_case="角色扮演"
    ),
    # 豆包 Seed-TTS 2.0 音色
    "zh_female_vv_uranus_bigtts": SpeakerInfo(
        display_name="vivi 2.0", language="中文", supports_english=False, use_case="通用场景"
    ),
    "zh_female_xiaohe_uranus_bigtts": SpeakerInfo(
        display_name="小何", language="中文", supports_english=False, use_case="通用场景"
    ),
    "zh_male_m191_uranus_bigtts": SpeakerInfo(
        display_name="云舟", language="中文", supports_english=False, use_case="通用场景"
    ),
    "zh_male_taocheng_uranus_bigtts": SpeakerInfo(
        display_name="小天", language="中文", supports_english=False, use_case="通用场景"
    ),
    "en_male_tim_uranus_bigtts": SpeakerInfo(
        display_name="Tim", language="English", supports_english=True, use_case="通用场景"
    ),
    "en_female_dacey_uranus_bigtts": SpeakerInfo(
        display_name="Dacey", language="English", supports_english=True, use_case="通用场景"
    ),
    "en_female_stokie_uranus_bigtts": SpeakerInfo(
        display_name="Stokie", language="English", supports_english=True, use_case="通用场景"
    ),
    "zh_male_liufei_uranus_bigtts": SpeakerInfo(
        display_name="刘飞 2.0", language="中文", supports_english=False, use_case="通用场景"
    ),
    "zh_female_qingxinnvsheng_uranus_bigtts": SpeakerInfo(
        display_name="清新女声 2.0", language="中文", supports_english=False, use_case="通用场景"
    ),
    "zh_female_cancan_uranus_bigtts": SpeakerInfo(
        display_name="知性灿灿 2.0", language="中文", supports_english=False, use_case="角色扮演"
    ),
    "zh_female_sajiaoxuemei_uranus_bigtts": SpeakerInfo(
        display_name="撒娇学妹 2.0", language="中文", supports_english=False, use_case="角色扮演"
    ),
    "zh_female_tianmeixiaoyuan_uranus_bigtts": SpeakerInfo(
        display_name="甜美小源 2.0", language="中文", supports_english=False, use_case="通用场景"
    ),
    "zh_female_tianmeitaozi_uranus_bigtts": SpeakerInfo(
        display_name="甜美桃子 2.0", language="中文", supports_english=False, use_case="通用场景"
    ),
    "zh_female_shuangkuaisisi_uranus_bigtts": SpeakerInfo(
        display_name="爽快思思 2.0", language="中文", supports_english=False, use_case="通用场景"
    ),
    "zh_female_peiqi_uranus_bigtts": SpeakerInfo(
        display_name="佩奇猪 2.0", language="中文", supports_english=False, use_case="视频配音"
    ),
}

# 获取所有 Speaker 类型的列表
ALL_SPEAKER_TYPES = list(SPEAKER_INFO_MAP.keys())


class User(BaseModel):
    uid: str = Field(default="", description="")


class AudioParams(BaseModel):
    format: Literal["mp3", "pcm", "ogg_opus"] = Field(default="pcm")
    sample_rate: int = Field(default=44100, description="8000,16000,22050,24000,32000,44100,48000")
    loudness_rate: Optional[int] = Field(default=0)
    speech_rate: Optional[int] = Field(default=0)
    emotion: Optional[ChineseVoiceEmotion] = Field(default="neutral")


class ReqParams(BaseModel):
    audio_params: AudioParams = Field(default_factory=AudioParams)
    speaker: str = Field(default="zh_female_cancan_mars_bigtts")
    model: Optional[str] = Field(default=None,
                                 description="TTS 2.0 model: seed-tts-2.0-standard / seed-tts-2.0-expressive")
    additions: Optional[str] = Field(default=None)


class Session(BaseModel):
    """
    session 数据.
    """

    user: User = Field(default_factory=User)
    namespace: str = Field(default="BidirectionalTTS")
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
        j = json.dumps(data)
        return j


class VoiceConf(BaseModel):
    speech_rate: Optional[int] = Field(
        default=None,
        description="语速，取值范围[-50,100]，100代表2.0倍速，-50代表0.5倍数. 0是正常",
        ge=-50,
        le=100,
    )
    loudness_rate: Optional[int] = Field(
        default=None,
        description="音量，取值范围[-50,100]，100代表2.0倍音量，-50代表0.5倍音量. 0是正常",
        ge=-50,
        le=100,
    )
    emotion: Optional[ChineseVoiceEmotion] = Field(default=None, description="声音情绪, 拥有多种可选择的声音情绪.")


class SpeakerConf(BaseModel):
    """
    角色配置, 可以更改.
    """

    tone: str = Field(default="saturn_zh_female_cancan_tob")
    description: str = Field(default="", description="角色的描述")
    resource_id: Optional[str] = Field(default=None, description="使用声音复刻的独立的资源")
    voice: VoiceConf = Field(default_factory=VoiceConf, description="声音配置")

    def to_voice_conf(self) -> dict:
        return self.model_dump(exclude={"resource_id"})


_Head = dict[str, Any]
_Url = str


class VolcengineTTSConf(BaseModel):
    """
    火山引擎 tts 基础配置.
    """

    app_key: str = Field(default="$VOLCENGINE_STREAM_TTS_APP")
    access_token: str = Field(default="$VOLCENGINE_STREAM_TTS_ACCESS_TOKEN")
    api_key: str = Field(default="$VOLCENGINE_STREAM_TTS_API_KEY", description="新版控制台 API Key")
    resource_id: str = Field(default="seed-tts-2.0", description="官方的默认资源")
    sample_rate: int = Field(default=44100, description="生成音频的采样率要求.")
    audio_format: Literal["pcm"] = Field(default="pcm", description="默认可用的数据格式")

    disconnect_on_idle: int = Field(
        default=300,
        description="闲置多少秒后退出",
    )

    disable_markdown_filter: bool = Field(default=True, description="支持朗读 markdown 格式. ")
    model: Optional[str] = Field(default=None,
                                 description="TTS 2.0 引擎: seed-tts-2.0-standard 或 seed-tts-2.0-expressive")
    url: str = Field(
        default="wss://openspeech.bytedance.com/api/v3/tts/bidirection",
        description="火山的流式语音模型的地址",
    )

    speakers: dict[str, SpeakerConf] = Field(
        default_factory=lambda: {
            speaker_info.display_name: SpeakerConf(tone=name, description=speaker_info.description())
            for name, speaker_info in SPEAKER_INFO_MAP.items()
        },
        description="the speakers list. 可以自行配置. ",
    )
    default_speaker: str = Field(
        default="知性灿灿",
        description="the default speaker",
    )

    @classmethod
    def unwrap_env(cls, value: str, default: str = "") -> str:
        if value.startswith("$"):
            return os.environ.get(value[1:], default)
        return value or default

    def default_speaker_conf(self) -> SpeakerConf:
        conf = self.speakers.get(self.default_speaker, None)
        if conf is not None:
            return conf.model_copy(deep=True)
        conf = SpeakerConf()
        return conf

    def gen_header(self, *, connection_id: str = "", resource_id: Optional[str] = None) -> _Head:
        connection_id = connection_id or unique_id()
        app_key = self.unwrap_env(self.app_key)
        # 旧版鉴权 header 始终发送（兼容新旧控制台）
        ws_header = {
            "X-Api-App-Key": app_key,
            "X-Api-App-Id": app_key,
            "X-Api-Access-Key": self.unwrap_env(self.access_token),
            "X-Api-Resource-Id": resource_id or self.resource_id,
            "X-Api-Request-Id": unique_id(),
            "X-Api-Connect-Id": connection_id,
        }
        # 新版控制台 API Key（与旧版兼容共存）
        api_key = self.unwrap_env(self.api_key)
        if api_key:
            ws_header["X-Api-Key"] = api_key
        return ws_header

    def to_session(self, speaker: SpeakerConf) -> Session:
        # 生成 additions.
        additions_data = {
            "disable_markdown_filter": self.disable_markdown_filter,
        }
        additions = json.dumps(additions_data).decode()
        return Session(
            req_params=ReqParams(
                audio_params=AudioParams(
                    format=self.audio_format,
                    sample_rate=self.sample_rate,
                    loudness_rate=speaker.voice.loudness_rate,
                    speech_rate=speaker.voice.speech_rate,
                    emotion=speaker.voice.emotion,
                ),
                speaker=speaker.tone,
                model=self.model,
                additions=additions,
            ),
        )

    def to_tts_info(self, current_tone: str = "") -> TTSInfo:
        return TTSInfo(
            sample_rate=self.sample_rate,
            channels=1,
            audio_format=AudioFormat.PCM_S16LE.value,
            voice_schema=VoiceConf.model_json_schema(),
            tones={key: value.description for key, value in self.speakers.items()},
            current_tone=current_tone or self.default_speaker,
        )
