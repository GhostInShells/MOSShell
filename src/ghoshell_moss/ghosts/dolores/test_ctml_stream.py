"""CtmlArgumentStream — the streaming unescaper must be exact.

The streamed CTML is the authoritative value (there is no re-send from the parsed tool arguments),
so the only honest way to claim correctness is to compare against ``json.loads`` over a wide corpus
of strings and chunk boundaries — including the boundaries that split an escape or a surrogate pair.
"""

import json
import random

import pytest

from ._ctml_stream import CtmlArgumentStream

# characters worth throwing at the unescaper: CTML syntax, every single-char escape, BMP and astral
# code points (the latter become \uXXXX / surrogate pairs under ensure_ascii), plus a lone surrogate.
_CHARS = (
    list('<>:|/="\' abcXYZ019 \n\t') * 6
    + ['"', '\\', '/', '\b', '\f', '\n', '\r', '\t'] * 3
    + ['中', 'é', '→', '😀', '𝕏', '\ud83d']
)


def _render(value: str, *, ascii_only: bool, indent: int | None) -> str:
    return json.dumps({"ctml": value}, ensure_ascii=ascii_only, indent=indent)


def _split(text: str, sizes) -> list[str]:
    chunks, i = [], 0
    for size in sizes:
        if i >= len(text):
            break
        chunks.append(text[i:i + size])
        i += size
    if i < len(text):
        chunks.append(text[i:])
    return chunks


def _drain(chunks) -> tuple[CtmlArgumentStream, str]:
    stream = CtmlArgumentStream()
    return stream, ''.join(stream.add(chunk) for chunk in chunks)


@pytest.mark.parametrize("ascii_only", [True, False])
@pytest.mark.parametrize("indent", [None, 2])
@pytest.mark.parametrize("chunk_size", [1, 2, 3, 17])
def test_roundtrip_matches_json_loads(ascii_only, indent, chunk_size):
    """dump → split at every `chunk_size` → decode: identical to the original value."""
    rng = random.Random(20260923)
    for _ in range(60):
        value = ''.join(rng.choices(_CHARS, k=rng.randint(0, 40)))
        args = _render(value, ascii_only=ascii_only, indent=indent)
        assert json.loads(args)["ctml"] == value  # the corpus itself is honest

        stream, decoded = _drain([args[i:i + chunk_size] for i in range(0, len(args), chunk_size)])
        assert decoded == value
        assert stream.value == value
        assert stream.complete
        assert stream.matched


def test_roundtrip_with_random_chunk_boundaries():
    """Random splits — escapes and surrogate pairs land across boundaries by accident, not by design."""
    rng = random.Random(1)
    for _ in range(200):
        value = ''.join(rng.choices(_CHARS, k=rng.randint(0, 80)))
        args = _render(value, ascii_only=True, indent=None)
        sizes = [rng.randint(1, 6) for _ in range(len(args) + 1)]

        stream, decoded = _drain(_split(args, sizes))
        assert decoded == value
        assert stream.complete


def test_escape_split_across_every_offset():
    """Each escape, chopped at every possible offset, still decodes."""
    value = '"\\/\b\f\n\r\té\U0001f600—end'
    args = _render(value, ascii_only=True, indent=None)
    for cut in range(len(args) + 1):
        stream, decoded = _drain([args[:cut], args[cut:]])
        assert decoded == value, f"cut={cut}"


def test_astral_pair_and_lone_surrogate():
    """Surrogate pairs fuse into one code point; a lone surrogate survives as-is (json.loads parity)."""
    for value in ['😀', 'a😀b', '\ud83d', 'x\ud83d y', '😀😀']:
        args = _render(value, ascii_only=True, indent=None)
        _, decoded = _drain([args[i:i + 1] for i in range(len(args))])
        assert decoded == json.loads(args)["ctml"] == value


def test_empty_value():
    stream, decoded = _drain(list(_render('', ascii_only=True, indent=None)))
    assert decoded == ''
    assert stream.value == ''
    assert stream.complete


def test_value_holding_raw_delimiters():
    """The CTML's own quotes and braces are escaped in JSON — they must not be mistaken for the end."""
    value = '{"ctml":"nested"} <say>hi</say>'
    args = _render(value, ascii_only=True, indent=None)
    _, decoded = _drain([args[i:i + 3] for i in range(0, len(args), 3)])
    assert decoded == value


def test_trailing_backslash_before_closing_quote():
    value = 'cmd__text\\\\'
    args = _render(value, ascii_only=True, indent=None)
    _, decoded = _drain([args[i:i + 1] for i in range(len(args))])
    assert decoded == value


@pytest.mark.parametrize("args", ['{"other": 1}', '{"ctml": 5}', '{"ctml": null}', 'plain text'])
def test_shape_mismatch_degrades_instead_of_streaming(args):
    """Anything that is not {"ctml": "…"} emits nothing and flips `matched` — the caller falls back to tool/call."""
    stream, decoded = _drain([args[i:i + 2] for i in range(0, len(args), 2)])
    assert decoded == ''
    assert stream.value == ''
    assert not stream.complete
    assert not stream.matched


def test_empty_stream_is_not_a_mismatch():
    """No deltas yet: nothing matched, nothing contradicted — the caller just keeps waiting."""
    stream = CtmlArgumentStream()
    assert stream.add(None) == ''
    assert not stream.complete
    assert stream.matched


def test_add_ignores_none_and_empty_deltas():
    """dsh may emit a delta with no arguments; it must be a no-op, not a state transition."""
    stream = CtmlArgumentStream()
    assert stream.add(None) == ''
    assert stream.add('') == ''
    assert ''.join(stream.add(chunk) for chunk in ['{"ctm', None, 'l": "a', '', 'b"}']) == 'ab'
    assert stream.complete


def test_malformed_escape_flips_failed():
    """非法转义 (\\x) → failed, 停止消费, 不当作 content 发出去."""
    stream = CtmlArgumentStream()
    out = stream.add('{"ctml": "a\\x')
    assert stream.failed
    assert stream.matched  # 前缀是匹配的, 是 body 里的转义非法
    # 已经发出去的前缀仍保留, 但之后的输入被丢弃.
    assert out == 'a'
    assert stream.add('yz"}') == ''


def test_malformed_unicode_flips_failed():
    """非法 \\u (非 hex) → failed, 停止消费."""
    stream = CtmlArgumentStream()
    stream.add('{"ctml": "\\u12G4')
    assert stream.failed


def test_valid_escapes_do_not_fail():
    """合法转义不触发 failed (回归)."""
    stream = CtmlArgumentStream()
    assert stream.add('{"ctml": "a\\nb\\t\\u0041"') == 'a\nb\tA'
    assert stream.complete
    assert not stream.failed
