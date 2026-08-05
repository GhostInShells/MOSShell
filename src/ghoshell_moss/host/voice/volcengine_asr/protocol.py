import enum
import gzip
import io
import json
import struct
from typing import NamedTuple, Optional

import numpy as np
import websockets
from ghoshell_common.helpers import uuid

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
            "enable_punc": config.enable_punc,
            "end_window_size": config.end_window_size,
            "force_to_speech_time": 1000,
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
    header = _Protocol.get_header(
        _Protocol.AUDIO_ONLY_REQUEST,
        message_flags,
        _Protocol.NO_SERIALIZATION,
        _Protocol.GZIP,
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

    sequence = struct.unpack(">i", data[4:8])[0]
    payload_size = struct.unpack(">I", data[8:12])[0]
    payload = data[12:12 + payload_size] if len(data) >= 12 + payload_size else data[12:]

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
