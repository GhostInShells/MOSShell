"""Volcengine SAUC 大模型流式识别 — 二进制 WS 协议 (bigmodel_async)。

对齐官方 SDK (sauc_python) 与文档 https://docs.volcengine.com/docs/6561/1354869?lang=zh

请求侧: header(4B) + seq(4B) + payload_size(4B) + gzip(payload)
响应侧: header(4B) + [seq(4B)] + [event(4B)] + payload_size(4B) + gzip(payload_msg)
响应 header 的 message_type_specific_flags:
  bit0 (0x01) 是否带 payload_sequence
  bit1 (0x02) 是否 is_last_package (最后一个响应包)
  bit2 (0x04) 是否带 event (会话事件)
"""
import enum
import gzip
import io
import json
import struct
from typing import Optional

import numpy as np
import websockets
from ghoshell_common.helpers import uuid
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from .config import VolcengineSaucConfig, VolcengineSaucCorpus

__all__ = [
    "ResponseMessageType",
    "Response",
    "AudioInfo",
    "Word",
    "Utterance",
    "Result",
    "PayloadMsg",
    "connect",
    "create_init_request",
    "create_audio_only_request",
    "nparray_to_bytes",
    "parse_response",
    "parse_payload",
]


class _Protocol:
    PROTOCOL_VERSION = 0x01
    DEFAULT_HEADER_SIZE = 0x01  # 单位: 4 字节

    FULL_CLIENT_REQUEST = 0x01
    AUDIO_ONLY_REQUEST = 0x02
    FULL_SERVER_RESPONSE = 0x09
    SERVER_ERROR_RESPONSE = 0x0F

    NO_SEQUENCE = 0x00
    POS_SEQUENCE = 0x01
    NEG_SEQUENCE = 0x02
    NEG_WITH_SEQUENCE = 0x03

    NO_SERIALIZATION = 0x00
    JSON = 0x01

    NO_COMPRESSION = 0x00
    GZIP = 0x01

    @staticmethod
    def gzip_compress(data: bytes) -> bytes:
        if not data:
            return b""
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb") as f:
            f.write(data)
        return buf.getvalue()

    @staticmethod
    def gzip_decompress(data: bytes) -> bytes:
        if not data:
            return b""
        buf = io.BytesIO(data)
        with gzip.GzipFile(fileobj=buf, mode="rb") as f:
            return f.read()


async def connect(config: VolcengineSaucConfig, request_id: str = "") -> websockets.ClientConnection:
    config = config
    request_id = request_id or uuid()
    headers = {
        "X-Api-Key": config.api_key,
        "X-Api-Resource-Id": config.resource_id,
        "X-Api-Request-Id": request_id,
    }
    return await websockets.connect(config.url, additional_headers=headers)


def _request_frame(message_type: int, flags: int, seq: int, payload: bytes, compression: int) -> bytes:
    header = bytearray(4)
    header[0] = (_Protocol.PROTOCOL_VERSION << 4) | _Protocol.DEFAULT_HEADER_SIZE
    header[1] = (message_type << 4) | flags
    header[2] = (_Protocol.JSON << 4) | compression
    header[3] = 0x00
    return b"".join([
        bytes(header),
        struct.pack(">i", seq),
        struct.pack(">I", len(payload)),
        payload,
    ])


def create_init_request(uid: str, config: VolcengineSaucConfig, corpus: VolcengineSaucCorpus | None = None) -> bytes:
    """构造 full client request。uid = segment_id, 每次识别流开局发一次。

    corpus (热词/上下文) 是动态的, 从 ASR 实例传入, 不进 config。
    """
    p = config.params
    request: dict = {
        "model_name": config.model_name,
        "enable_nonstream": p.enable_nonstream,
        "enable_itn": p.enable_itn,
        "enable_punc": p.enable_punc,
        "enable_ddc": p.enable_ddc,
        "show_utterances": p.show_utterances,
        "result_type": p.result_type,
        "end_window_size": p.end_window_size,
        "force_to_speech_time": p.force_to_speech_time,
        "vad_segment_duration": p.vad_segment_duration,
    }
    # 热词 / 上下文 (火山专属面, 动态, 不进 config)
    c = corpus or VolcengineSaucCorpus()
    if c.boosting_table_name:
        request["boosting_table_name"] = c.boosting_table_name
    if c.boosting_table_id:
        request["boosting_table_id"] = c.boosting_table_id
    ctx = c.context_payload()
    if ctx:
        request["context"] = json.dumps(ctx, ensure_ascii=False)

    payload = {
        "audio": {
            "format": "pcm",
            "codec": "raw",
            "rate": config.sample_rate,
            "bits": config.bits,
            "channel": config.channel,
        },
        "request": request,
    }
    payload_bytes = _Protocol.gzip_compress(json.dumps(payload).encode("utf-8"))
    return _request_frame(
        _Protocol.FULL_CLIENT_REQUEST, _Protocol.POS_SEQUENCE, 1, payload_bytes, _Protocol.GZIP,
    )


def nparray_to_bytes(audio: np.ndarray) -> bytes:
    return _Protocol.gzip_compress(audio.astype(np.int16).tobytes())


def create_audio_only_request(audio: bytes, seq: int, is_last: bool = False) -> bytes:
    """构造 audio only request。is_last=True 时发负序号 (端侧 commit / 流结束)。"""
    seq_value = -seq if is_last else seq
    flags = _Protocol.NEG_WITH_SEQUENCE if is_last else _Protocol.POS_SEQUENCE
    # 空 payload 标 GZIP 会导致服务端解压 EOF, 用 NO_COMPRESSION.
    compression = _Protocol.NO_COMPRESSION if len(audio) == 0 else _Protocol.GZIP
    return _request_frame(_Protocol.AUDIO_ONLY_REQUEST, flags, seq_value, audio, compression)


# ── 响应数据结构 (payload_msg) ──


class AudioInfo(BaseModel):
    """音频相关信息。"""
    duration: int = 0


class Word(BaseModel):
    """词级时间戳 (ms, 流相对)。"""
    start_time: int = 0
    end_time: int = 0
    text: str = ""


class Utterance(BaseModel):
    """一条分句。

    definite=true 表示该分句最终确定 (仅 enable_nonstream=true 时二次识别结果携带)。
    additions 承载火山返回的丰富概念 (说话人/情绪/性别/年龄/语种/语速/音量/来源...),
    这些是可选槽位 —— 未开启对应能力时缺省为空/None。
    """

    additions: dict = Field(default_factory=dict)
    definite: bool = False
    start_time: int = 0
    end_time: int = 0
    text: str = ""
    words: list[Word] = Field(default_factory=list)

    # ── 预留兼容槽位 (从 additions 提取, 允许缺省) ──
    source: str = ""  # 分句来源: stream / nonstream
    speaker_id: str = ""  # 说话人 ID (enable_speaker_info)
    speech_rate: Optional[float] = None  # 语速 token/s (show_speech_rate)
    volume: Optional[float] = None  # 音量 dB (show_volume)
    emotion: str = ""  # 情绪 (enable_emotion_detection)
    gender: str = ""  # 性别 (enable_gender_detection)
    age: str = ""  # 年龄 (enable_age_detection)
    language: str = ""  # 语种/场景标签 (enable_lid, 含 singing_* 唱歌)

    @model_validator(mode="after")
    def _extract_additions(self) -> Self:
        a = self.additions or {}
        for field, key in (
                ("source", "source"),
                ("speaker_id", "speaker_id"),
                ("speech_rate", "speech_rate"),
                ("volume", "volume"),
                ("emotion", "emotion"),
                ("gender", "gender"),
                ("age", "age"),
                ("language", "language"),
        ):
            if key in a and getattr(self, field) in (None, ""):
                setattr(self, field, a[key])
        return self


class Result(BaseModel):
    """识别结果。text 是全量 (accumulated), utterances 是分句数组。"""
    additions: dict = Field(default_factory=dict)
    text: str = ""
    utterances: list[Utterance] = Field(default_factory=list)


class PayloadMsg(BaseModel):
    """响应数据主体 (payload_msg)。"""
    audio_info: AudioInfo = Field(default_factory=AudioInfo)
    result: Result = Field(default_factory=Result)


class Response(BaseModel):
    """解码后的响应包 (header + payload_msg)。

    is_last_package 来自 header bit1, 表示整个流的最后一个响应包。
    code 非 0 表示识别失败。
    """

    code: int = 0
    event: int = 0
    is_last_package: bool = False
    payload_sequence: int = 0
    payload_size: int = 0
    payload_msg: PayloadMsg = Field(default_factory=PayloadMsg)
    error_msg: str = ""  # 错误响应: 原始 JSON (结构不同于 PayloadMsg, 不建模, 避免丢信息)


class ResponseMessageType(str, enum.Enum):
    full_server_response = "full_server_response"
    server_error = "server_error"


def parse_response(data: bytes) -> Response:
    """按官方 SDK 的 parse_response 解析二进制响应。

    header_size 以 4 字节为单位 (data[0] & 0x0f), payload 从 header_size*4 开始,
    依次是 [seq(4B)] [event(4B)] payload_size(4B) payload_msg。
    """
    header_size = data[0] & 0x0F
    message_type = (data[1] >> 4) & 0x0F
    flags = data[1] & 0x0F
    serialization = (data[2] >> 4) & 0x0F
    compression = data[2] & 0x0F

    payload = data[header_size * 4:]

    response = Response()
    if flags & 0x01:  # 带 payload_sequence
        response.payload_sequence = struct.unpack(">i", payload[:4])[0]
        payload = payload[4:]
    if flags & 0x02:  # is_last_package
        response.is_last_package = True
    if flags & 0x04:  # 带 event
        response.event = struct.unpack(">i", payload[:4])[0]
        payload = payload[4:]

    if message_type == _Protocol.FULL_SERVER_RESPONSE:
        response.code = 0
        response.payload_size = struct.unpack(">I", payload[:4])[0]
        payload = payload[4:]
    elif message_type == _Protocol.SERVER_ERROR_RESPONSE:
        response.code = struct.unpack(">i", payload[:4])[0]
        response.payload_size = struct.unpack(">I", payload[4:8])[0]
        payload = payload[8:]
    else:
        return response

    if not payload:
        return response

    if compression == _Protocol.GZIP:
        try:
            payload = _Protocol.gzip_decompress(payload)
        except Exception:
            return response

    if serialization == _Protocol.JSON:
        text = payload.decode("utf-8")
        if response.code != 0:
            # 错误响应: payload 是错误信息 JSON, 结构不同于 PayloadMsg, 原样保留.
            response.error_msg = text
        else:
            try:
                response.payload_msg = parse_payload(text)
            except Exception:
                pass
    return response


def parse_payload(payload: str) -> PayloadMsg:
    """把 payload_msg 的 JSON 字符串解析成结构化 PayloadMsg。"""
    return PayloadMsg.model_validate_json(payload)
