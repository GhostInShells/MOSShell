import enum
import gzip
import io
import json
import struct
from typing import NamedTuple, Optional

import numpy as np
import websockets
from ghoshell_common.helpers import uuid
from pydantic import BaseModel, Field

from .config import VolcengineASRConfig


class _Protocol:
    PROTOCOL_VERSION = 0x01
    DEFAULT_HEADER_SIZE = 0x01

    FULL_CLIENT_REQUEST = 0x01
    AUDIO_ONLY_REQUEST = 0x02
    FULL_SERVER_RESPONSE = 0x09
    SERVER_ACK = 0x0B
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
    def get_header(
        message_type: int,
        message_type_specific_flags: int,
        serial_method: int,
        compression_type: int,
        reserved_data: int = 0,
    ) -> bytes:
        header = bytearray(4)
        header[0] = (_Protocol.PROTOCOL_VERSION << 4) | _Protocol.DEFAULT_HEADER_SIZE
        header[1] = (message_type << 4) | message_type_specific_flags
        header[2] = (serial_method << 4) | compression_type
        header[3] = reserved_data
        return bytes(header)

    @staticmethod
    def int_to_bytes(value: int) -> bytes:
        return struct.pack(">i", value)

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


async def connect(config: VolcengineASRConfig, connection_id: str = "") -> websockets.ClientConnection:
    config = config.resolve_env()
    connection_id = connection_id or uuid()
    headers = {
        "X-Api-App-Key": config.appid,
        "X-Api-Access-Key": config.token,
        "X-Api-Resource-Id": config.resource_id,
        "X-Api-Connect-Id": connection_id,
    }
    return await websockets.connect(config.url, additional_headers=headers)


def create_init_request(uid: str, config: VolcengineASRConfig) -> tuple[bytes, int]:
    payload = {
        "user": {"uid": uid},
        "audio": {
            "format": "pcm",
            "sample_rate": config.sample_rate,
            "bits": config.bits,
            "channel": config.channel,
            "codec": "raw",
        },
        "request": {
            "model_name": config.model_name,
            "enable_punc": config.params.enable_punc,
            "end_window_size": config.params.end_window_size,
            "force_to_speech_time": config.params.force_to_speech_time,
            "show_utterances": True,
        },
    }
    payload_str = json.dumps(payload)
    payload_bytes = _Protocol.gzip_compress(payload_str.encode("utf-8"))
    seq = 1
    seq_bytes = _Protocol.int_to_bytes(seq)
    header = _Protocol.get_header(
        _Protocol.FULL_CLIENT_REQUEST,
        _Protocol.POS_SEQUENCE,
        _Protocol.JSON,
        _Protocol.GZIP,
    )
    payload_size = _Protocol.int_to_bytes(len(payload_bytes))
    return header + seq_bytes + payload_size + payload_bytes, seq


def nparray_to_bytes(audio: np.ndarray) -> bytes:
    audio_bytes = audio.tobytes()
    return _Protocol.gzip_compress(audio_bytes)


def create_audio_only_request(audio: bytes, seq: int, is_last: bool = False) -> tuple[bytes, int]:
    seq += 1
    seq_value = -seq if is_last else seq
    seq_bytes = _Protocol.int_to_bytes(seq_value)
    payload_size = _Protocol.int_to_bytes(len(audio))
    message_flags = _Protocol.NEG_WITH_SEQUENCE if is_last else _Protocol.POS_SEQUENCE
    # Empty payload with GZIP flag triggers "ungzip payload: EOF" on the server.
    # Use NO_COMPRESSION when there is nothing to decompress.
    compression = _Protocol.NO_COMPRESSION if len(audio) == 0 else _Protocol.GZIP
    header = _Protocol.get_header(
        _Protocol.AUDIO_ONLY_REQUEST,
        message_flags,
        _Protocol.NO_SERIALIZATION,
        compression,
    )
    return b"".join([header, seq_bytes, payload_size, audio]), seq


async def send_init_request(ws: websockets.ClientConnection, config: VolcengineASRConfig, uid: str) -> None:
    message, _ = create_init_request(uid, config)
    await ws.send(message)


async def send_audio(ws: websockets.ClientConnection, audio: bytes, seq: int, is_last: bool = False) -> None:
    message, _ = create_audio_only_request(audio, seq, is_last)
    await ws.send(message)


class ResponseMessageType(str, enum.Enum):
    full_server_response = "full_server_response"
    server_error = "server_error"
    server_ack = "server_ack"


class Response(NamedTuple):
    sequence: int
    message_type: Optional[ResponseMessageType]
    error_code: Optional[int]
    is_last: bool
    payload: str


def parse_response(data: bytes) -> Response:
    message_type = (data[1] >> 4) & 0x0F
    message_type_specific_flags = data[1] & 0x0F
    message_compression = data[2] & 0x0F

    # 序列号字段是否存在于 header 之后由 bit0 (hasSeq) 决定. 无序号尾包 (0b0010) 时
    # header 后直接是 payload_size; 固定偏移会把 gzip payload 前 4 字节当长度, 错位丢包.
    has_sequence = bool(message_type_specific_flags & 0x01)
    seq_size = 4 if has_sequence else 0
    sequence = struct.unpack(">i", data[4:4 + seq_size])[0] if has_sequence else 0
    size_offset = 4 + seq_size
    payload_size = struct.unpack(">I", data[size_offset:size_offset + 4])[0]
    payload_offset = size_offset + 4
    payload = (
        data[payload_offset:payload_offset + payload_size]
        if len(data) >= payload_offset + payload_size
        else data[payload_offset:]
    )

    is_last_package = bool(message_type_specific_flags & 0x02)

    if message_type == _Protocol.FULL_SERVER_RESPONSE:
        if message_compression == _Protocol.GZIP:
            decompressed = _Protocol.gzip_decompress(payload)
            payload_str = decompressed.decode("utf-8")
        else:
            payload_str = payload.decode("utf-8")
        return Response(
            sequence=sequence,
            message_type=ResponseMessageType.full_server_response,
            error_code=None,
            is_last=is_last_package,
            payload=payload_str,
        )
    elif message_type == _Protocol.SERVER_ACK:
        return Response(
            sequence=sequence,
            message_type=ResponseMessageType.server_ack,
            error_code=None,
            is_last=False,
            payload="",
        )
    elif message_type == _Protocol.SERVER_ERROR_RESPONSE:
        code = int.from_bytes(payload[:4], "big", signed=False)
        payload_msg = payload[8:]
        if message_compression == _Protocol.GZIP:
            payload_msg = gzip.decompress(payload_msg)
        return Response(
            sequence=sequence,
            message_type=ResponseMessageType.server_error,
            error_code=code,
            is_last=is_last_package,
            payload=payload_msg.decode("utf-8", errors="replace"),
        )
    else:
        return Response(
            sequence=-1,
            message_type=ResponseMessageType.server_error,
            error_code=-1,
            is_last=False,
            payload="unknown error",
        )


# ── full_server_response 的 JSON payload 数据结构 ──
# 发现路径: 火山引擎「大模型流式语音识别 API」— https://docs.volcengine.com/docs/6561/1354869?lang=zh
# 实测抓包 (2026-09-12):
#   {"audio_info":{"duration":11000},
#    "result":{"additions":{"log_id":"..."},"text":"测试测试。",
#              "utterances":[{"additions":{...},"definite":true,"end_time":2432,"start_time":0,
#                             "text":"测试测试。","words":[{"end_time":1400,"start_time":1320,"text":"测试"},...]},
#                            {"additions":{...},"definite":false,"end_time":-1,"start_time":-1,"text":""}]}}
#
# 关键语义:
#   - result.text 是「全量 replace」(非 delta)，每帧都带完整文本。
#   - utterances[] 里 definite=true 的那条 = VAD 判停后的一句稳定句 (句边界/tail)。
#   - 末尾 definite=false、start_time=-1 的那条是空占位 (进行中)。
#   - start_time/end_time 是流相对时间戳 (ms)；audio_info.duration 是已上传音频时长 (ms)。


class AudioInfo(BaseModel):
    """payload.audio_info — 已上传音频时长 (ms)。"""
    duration: int = 0


class Word(BaseModel):
    """词级时间戳 (ms, 流相对)。"""
    start_time: int = 0
    end_time: int = 0
    text: str = ""


class Utterance(BaseModel):
    """一条 utterance — 分句单元。definite=true 表示该句已稳定 (VAD 判停)。"""
    additions: dict = Field(default_factory=dict)
    definite: bool = False
    start_time: int = 0
    end_time: int = 0
    text: str = ""
    words: list[Word] = Field(default_factory=list)


class Result(BaseModel):
    """识别结果 — text 是全量 replace，utterances 是分句。"""
    additions: dict = Field(default_factory=dict)
    text: str = ""
    utterances: list[Utterance] = Field(default_factory=list)


class ResponsePayload(BaseModel):
    """full_server_response 的 JSON payload 结构化模型。"""
    audio_info: AudioInfo = Field(default_factory=AudioInfo)
    result: Result = Field(default_factory=Result)


def parse_payload(payload: str) -> ResponsePayload:
    """把 full_server_response 的 JSON 字符串解析成结构化 ResponsePayload。"""
    return ResponsePayload.model_validate_json(payload)
