"""Pure-logic tests for ServiceKeyspace, ServiceKeyExpr, and query envelope.

No zenoh router required — all paths are string-level assertions.
"""

import pytest

from ghoshell_moss.matrix.operator._utils import (
    ServiceKeyspace,
    ServiceKeyExpr,
    _META_KEY,
)
from ghoshell_moss.matrix.operator.zenoh_service_terminal import (
    _encode_query_payload,
    _decode_query_payload,
)

_NETWORK_NS = "MOSS/matrix/scopes/local"


# -- ServiceKeyspace -----------------------------------------------------


class TestServiceKeyspace:

    @pytest.fixture
    def ks(self) -> ServiceKeyspace:
        return ServiceKeyspace(_NETWORK_NS)

    def test_services_ns_is_scoped(self, ks: ServiceKeyspace) -> None:
        assert ks.services_ns == f"{_NETWORK_NS}/services"
        assert ks.services_ns_prefix == ks.services_ns + "/"

    def test_per_service_returns_expr(self, ks: ServiceKeyspace) -> None:
        expr = ks.per_service("host/main", "webview")
        assert isinstance(expr, ServiceKeyExpr)
        assert expr.address == "host/main"
        assert expr.kind == "webview"
        assert expr.dotted == "host.main"

    # -- wildcards -------------------------------------------------------

    def test_kind_live_wildcard(self, ks: ServiceKeyspace) -> None:
        wc = ks.kind_live_wildcard("webview")
        assert wc == f"{_NETWORK_NS}/services/**/webview/live"

    def test_kind_query_meta_wildcard(self, ks: ServiceKeyspace) -> None:
        wc = ks.kind_query_meta_wildcard("webview")
        assert wc == f"{_NETWORK_NS}/services/**/webview/query/{_META_KEY}"

    def test_kind_pub_wildcard(self, ks: ServiceKeyspace) -> None:
        wc = ks.kind_pub_wildcard("webview", "badge")
        assert wc == f"{_NETWORK_NS}/services/**/webview/pub/badge"

    # -- parse_key -------------------------------------------------------

    def test_parse_key_valid(self, ks: ServiceKeyspace) -> None:
        parsed = ks.parse_key(
            f"{_NETWORK_NS}/services/global.host.main/webview/pub/badge"
        )
        assert parsed == ("global.host.main", "webview", "pub", "badge")

    def test_parse_key_with_sub_key(self, ks: ServiceKeyspace) -> None:
        parsed = ks.parse_key(
            f"{_NETWORK_NS}/services/a.b/resource/pub/image/generated"
        )
        assert parsed == ("a.b", "resource", "pub", "image/generated")

    def test_parse_key_wrong_prefix(self, ks: ServiceKeyspace) -> None:
        assert ks.parse_key("MOSS/matrix/scopes/other/services/a/b/q/x") is None

    def test_parse_key_too_short(self, ks: ServiceKeyspace) -> None:
        assert ks.parse_key(f"{_NETWORK_NS}/services/a") is None
        assert ks.parse_key(f"{_NETWORK_NS}/services/a/b") is None
        assert ks.parse_key(f"{_NETWORK_NS}/services/a/b/q") is None

    # -- parse_live_identity ---------------------------------------------

    def test_parse_live_identity_valid(self, ks: ServiceKeyspace) -> None:
        parsed = ks.parse_live_identity("global.host.main/webview/live")
        assert parsed == ("global.host.main", "webview")

    def test_parse_live_identity_not_live(self, ks: ServiceKeyspace) -> None:
        assert ks.parse_live_identity("global.host.main/webview/pub/x") is None

    def test_parse_live_identity_single_segment_ambiguous(self, ks: ServiceKeyspace) -> None:
        # "webview/live" has no '/' between addr and kind — cannot disambiguate
        assert ks.parse_live_identity("webview/live") is None

    def test_parse_live_identity_minimal(self, ks: ServiceKeyspace) -> None:
        assert ks.parse_live_identity("a/webview/live") == ("a", "webview")


# -- ServiceKeyExpr ------------------------------------------------------


class TestServiceKeyExpr:

    @pytest.fixture
    def ks(self) -> ServiceKeyspace:
        return ServiceKeyspace(_NETWORK_NS)

    def test_address_normalization_single_slash(self, ks: ServiceKeyspace) -> None:
        expr = ks.per_service("host/main", "webview")
        assert expr.dotted == "host.main"

    def test_address_normalization_multi_slash(self, ks: ServiceKeyspace) -> None:
        expr = ks.per_service("global/host/main", "webview")
        assert expr.dotted == "global.host.main"

    def test_address_no_slash_unchanged(self, ks: ServiceKeyspace) -> None:
        expr = ks.per_service("simple", "webview")
        assert expr.dotted == "simple"

    def test_base_structure(self, ks: ServiceKeyspace) -> None:
        expr = ks.per_service("host/main", "webview")
        assert expr.base == f"{_NETWORK_NS}/services/host.main/webview"

    # -- key builders ----------------------------------------------------

    def test_query_key(self, ks: ServiceKeyspace) -> None:
        expr = ks.per_service("host/main", "webview")
        assert expr.query_key("status") == (
            f"{_NETWORK_NS}/services/host.main/webview/query/status"
        )

    def test_pub_key(self, ks: ServiceKeyspace) -> None:
        expr = ks.per_service("host/main", "webview")
        assert expr.pub_key("badge") == (
            f"{_NETWORK_NS}/services/host.main/webview/pub/badge"
        )

    def test_listen_key(self, ks: ServiceKeyspace) -> None:
        expr = ks.per_service("host/main", "webview")
        assert expr.listen_key("command") == (
            f"{_NETWORK_NS}/services/host.main/webview/listen/command"
        )

    def test_meta_query_key_uses_META_KEY(self, ks: ServiceKeyspace) -> None:
        expr = ks.per_service("host/main", "webview")
        assert expr.meta_query_key() == (
            f"{_NETWORK_NS}/services/host.main/webview/query/{_META_KEY}"
        )

    def test_live_key_no_suffix(self, ks: ServiceKeyspace) -> None:
        expr = ks.per_service("host/main", "webview")
        assert expr.live_key == f"{_NETWORK_NS}/services/host.main/webview/live"
        assert not expr.live_key.endswith("/")

    # -- key prefix properties -------------------------------------------

    def test_prefixes_end_with_slash(self, ks: ServiceKeyspace) -> None:
        expr = ks.per_service("host/main", "webview")
        assert expr.query_prefix.endswith("/")
        assert expr.pub_prefix.endswith("/")
        assert expr.listen_prefix.endswith("/")

    def test_business_key_appended_after_prefix(self, ks: ServiceKeyspace) -> None:
        expr = ks.per_service("h", "k")
        assert expr.query_key("x") == expr.query_prefix + "x"
        assert expr.pub_key("x") == expr.pub_prefix + "x"
        assert expr.listen_key("x") == expr.listen_prefix + "x"

    # -- roundtrip: build → parse ----------------------------------------

    def test_key_roundtrip(self, ks: ServiceKeyspace) -> None:
        expr = ks.per_service("host/main", "webview")
        full_key = expr.pub_key("badge")
        parsed = ks.parse_key(full_key)
        assert parsed == ("host.main", "webview", "pub", "badge")

    def test_key_roundtrip_sub_key(self, ks: ServiceKeyspace) -> None:
        expr = ks.per_service("host/main", "resource")
        full_key = expr.query_key("image/generated")
        parsed = ks.parse_key(full_key)
        assert parsed == ("host.main", "resource", "query", "image/generated")


# -- Query envelope ------------------------------------------------------


class TestQueryEnvelope:

    def test_roundtrip_with_params(self) -> None:
        encoded = _encode_query_payload("caller-1", b"hello")
        caller, params = _decode_query_payload(encoded)
        assert caller == "caller-1"
        assert params == b"hello"

    def test_roundtrip_params_none(self) -> None:
        encoded = _encode_query_payload("caller-2", None)
        caller, params = _decode_query_payload(encoded)
        assert caller == "caller-2"
        assert params is None

    def test_roundtrip_params_empty_bytes(self) -> None:
        """b'' must survive the roundtrip, not be collapsed to None."""
        encoded = _encode_query_payload("caller-3", b"")
        caller, params = _decode_query_payload(encoded)
        assert caller == "caller-3"
        assert params == b""

    def test_roundtrip_empty_caller(self) -> None:
        encoded = _encode_query_payload("", b"data")
        caller, params = _decode_query_payload(encoded)
        assert caller == ""
        assert params == b"data"

    def test_decoded_payload_is_valid_json(self) -> None:
        encoded = _encode_query_payload("addr", b"\x00\xff\xab")
        # must parse without error
        import json
        d = json.loads(encoded)
        assert set(d.keys()) == {"caller", "params"}

    def test_decode_missing_caller_defaults_empty(self) -> None:
        caller, params = _decode_query_payload(b'{"params": "00ff"}')
        assert caller == ""
        assert params == b"\x00\xff"

    def test_decode_missing_params_defaults_none(self) -> None:
        caller, params = _decode_query_payload(b'{"caller": "x"}')
        assert caller == "x"
        assert params is None
