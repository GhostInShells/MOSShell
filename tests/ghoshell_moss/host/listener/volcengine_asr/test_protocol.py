import pytest

from ghoshell_moss.host.listener.volcengine_asr.config import VolcengineASRConfig
from ghoshell_moss.host.listener.volcengine_asr.protocol import (
    _Protocol,
    create_audio_only_request,
    create_init_request,
    parse_response,
)


class TestVolcengineASRConfig:
    def test_resolve_env_literal(self):
        conf = VolcengineASRConfig(appid="literal_appid", token="literal_token")
        resolved = conf.resolve_env()
        assert resolved.appid == "literal_appid"
        assert resolved.token == "literal_token"


class TestProtocol:
    def test_create_init_request_structure(self):
        config = VolcengineASRConfig(
            appid="test_app",
            token="test_token",
            sample_rate=16000,
        )
        msg, seq = create_init_request("uid-123", config)
        assert seq == 1
        assert len(msg) >= 12  # header + seq + size
        # header[0] = version + header_size
        assert (msg[0] >> 4) == _Protocol.PROTOCOL_VERSION

    def test_create_audio_only_request_sequence(self):
        msg, seq = create_audio_only_request(b"audio", 5, is_last=False)
        assert seq == 6
        assert len(msg) >= 12

    def test_create_audio_only_request_is_last(self):
        msg, seq = create_audio_only_request(b"audio", 5, is_last=True)
        assert seq == 6
        # is_last uses NEG_WITH_SEQUENCE
        assert (msg[1] & 0x0F) == _Protocol.NEG_WITH_SEQUENCE

    def test_parse_server_ack(self):
        header = _Protocol.get_header(
            _Protocol.SERVER_ACK,
            _Protocol.NO_SEQUENCE,
            _Protocol.JSON,
            _Protocol.NO_COMPRESSION,
        )
        seq = _Protocol.int_to_bytes(0)
        size = _Protocol.int_to_bytes(0)
        data = header + seq + size
        resp = parse_response(data)
        assert resp.message_type.value == "server_ack"
        assert resp.error_code is None

    def test_parse_server_error(self):
        header = _Protocol.get_header(
            _Protocol.SERVER_ERROR_RESPONSE,
            _Protocol.POS_SEQUENCE,
            _Protocol.JSON,
            _Protocol.NO_COMPRESSION,
        )
        seq = _Protocol.int_to_bytes(1)
        # error code 1234 + padding + empty msg
        payload = b"\x00\x00\x04\xd2" + b"\x00" * 4 + b"err"
        size = _Protocol.int_to_bytes(len(payload))
        data = header + seq + size + payload
        resp = parse_response(data)
        assert resp.message_type.value == "server_error"
        assert resp.error_code == 1234

    def test_parse_full_server_response_last_package_without_sequence(self):
        # 无序号尾包 (flags=0b0010): header 后无 sequence 字段, 直接 payload_size.
        header = _Protocol.get_header(
            _Protocol.FULL_SERVER_RESPONSE,
            _Protocol.NEG_SEQUENCE,
            _Protocol.JSON,
            _Protocol.GZIP,
        )
        payload = _Protocol.gzip_compress(b'{"result":{"text":"hello"}}')
        size = _Protocol.int_to_bytes(len(payload))
        data = header + size + payload
        resp = parse_response(data)
        assert resp.message_type.value == "full_server_response"
        assert resp.is_last is True
        assert resp.payload == '{"result":{"text":"hello"}}'

    def test_parse_full_server_response_last_package_with_negative_sequence(self):
        # 带序号尾包 (flags=0b0011): header 后带负序号 sequence.
        header = _Protocol.get_header(
            _Protocol.FULL_SERVER_RESPONSE,
            _Protocol.NEG_WITH_SEQUENCE,
            _Protocol.JSON,
            _Protocol.GZIP,
        )
        seq = _Protocol.int_to_bytes(-5)
        payload = _Protocol.gzip_compress(b'{"result":{"text":"hello"}}')
        size = _Protocol.int_to_bytes(len(payload))
        data = header + seq + size + payload
        resp = parse_response(data)
        assert resp.message_type.value == "full_server_response"
        assert resp.is_last is True
        assert resp.sequence == -5
        assert resp.payload == '{"result":{"text":"hello"}}'
