"""Tests for the @ file protocol — message_from_prompt / message_from_file
and the moss → pydantic-ai conversion (llms.pydantic_ai_adapter.conversion).

The @ protocol: a prompt may carry ``@path`` references inline (like a
mention) — ``@`` + a bare non-whitespace run, or a quoted ``@"path with
spaces"``. Resolved files become Messages; unresolvable references stay inline
as the plain string the caller typed. ``expose_file_meta`` (external flag)
wraps files with the meta layer (tag="file" + path/type/size) — discardable,
rendered by as_contents.
"""

from pathlib import Path

from ghoshell_moss.llms.pydantic_ai_adapter.conversion import message_to_parts, messages_to_parts
from ghoshell_moss.message import Message
from ghoshell_moss.message.prompt import message_from_file, message_from_prompt


def _tree(tmp_path: Path) -> Path:
    (tmp_path / "notes.md").write_text("# Notes\nbody\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("print(1)\n", encoding="utf-8")
    (tmp_path / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    return tmp_path


def _text(blocks) -> str:
    """Flatten the text content of parsed messages, in order."""
    out = []
    for block in blocks:
        for content in block.contents:
            if content.get("type") == "text":
                out.append(content["text"])
    return "".join(out)


# ── message_from_prompt ────────────────────────────────────────────────


def test_prompt_plain_text_single_message(tmp_path: Path):
    blocks = message_from_prompt("just text", base_dir=tmp_path)
    assert len(blocks) == 1
    assert blocks[0].meta.tag == ""
    assert blocks[0].contents[0]["type"] == "text"
    assert blocks[0].contents[0]["text"] == "just text"


def test_prompt_inline_ref_resolves(tmp_path: Path):
    """A bare @ref resolves inline (like a mention), not only at line start."""
    _tree(tmp_path)
    blocks = message_from_prompt("see @notes.md here", base_dir=tmp_path)
    assert _text(blocks) == "see # Notes\nbody\n here"


def test_prompt_at_within_word_stays_text(tmp_path: Path):
    """@ preceded by a non-whitespace char (e.g. an email) is not a ref."""
    _tree(tmp_path)
    blocks = message_from_prompt("email me@example.com", base_dir=tmp_path)
    assert len(blocks) == 1
    assert blocks[0].contents[0]["text"] == "email me@example.com"


def test_prompt_resolves_text_file(tmp_path: Path):
    _tree(tmp_path)
    blocks = message_from_prompt("analyze:\n@app.py\nnow", base_dir=tmp_path)
    assert blocks[1].contents[0]["type"] == "text"
    assert "print(1)" in blocks[1].contents[0]["text"]
    assert _text(blocks) == "analyze:\nprint(1)\n\nnow"


def test_prompt_unresolvable_stays_inline(tmp_path: Path):
    _tree(tmp_path)
    blocks = message_from_prompt("@missing.txt", base_dir=tmp_path)
    assert len(blocks) == 1
    assert blocks[0].contents[0]["text"] == "@missing.txt"


def test_prompt_resolves_trailing_sentence_punct(tmp_path: Path):
    """Trailing sentence punctuation is stripped so ``@app.py.`` resolves."""
    _tree(tmp_path)
    blocks = message_from_prompt("@app.py.", base_dir=tmp_path)
    assert "print(1)" in _text(blocks)


def test_prompt_resolves_trailing_comma(tmp_path: Path):
    _tree(tmp_path)
    blocks = message_from_prompt("@app.py,", base_dir=tmp_path)
    assert "print(1)" in _text(blocks)


def test_prompt_unresolvable_mention_keeps_literal(tmp_path: Path):
    """An @mention that is not a file stays literal (token + following text)."""
    _tree(tmp_path)
    blocks = message_from_prompt("@alice said hi.", base_dir=tmp_path)
    assert _text(blocks) == "@alice said hi."


def test_prompt_quoted_path_with_spaces(tmp_path: Path):
    f = tmp_path / "my file.txt"
    f.write_text("hello space\n", encoding="utf-8")
    blocks = message_from_prompt('@"my file.txt"', base_dir=tmp_path)
    assert len(blocks) == 1
    assert blocks[0].contents[0]["text"] == "hello space\n"


def test_prompt_multiple_refs_one_line(tmp_path: Path):
    _tree(tmp_path)
    blocks = message_from_prompt("see @notes.md and @app.py", base_dir=tmp_path)
    assert "body" in _text(blocks)
    assert "print(1)" in _text(blocks)


def test_prompt_expose_file_meta(tmp_path: Path):
    tree = _tree(tmp_path)
    blocks = message_from_prompt("@notes.md", base_dir=tree, expose_file_meta=True)
    assert len(blocks) == 1
    msg = blocks[0]
    assert msg.meta.tag == "file"
    assert msg.meta.attributes["path"] == "notes.md"
    assert msg.meta.attributes["type"] == "text/markdown"
    assert msg.meta.attributes["size"]
    assert msg.contents[0]["type"] == "text"
    # meta layer renders as XML file block
    rendered = "".join(msg.content_as_string(c) for c in msg.as_contents(with_meta=True))
    assert rendered.startswith("<file path=\"notes.md\"")
    assert "body" in rendered
    assert "</file>" in rendered


def test_prompt_image_without_meta_bare_content(tmp_path: Path):
    _tree(tmp_path)
    blocks = message_from_prompt("@logo.png", base_dir=tmp_path)
    assert len(blocks) == 1
    assert blocks[0].meta.tag == ""
    assert blocks[0].contents[0]["type"] == "image"


def test_message_from_file_unsupported_or_missing(tmp_path: Path):
    _tree(tmp_path)
    assert message_from_file("@missing.txt", base_dir=tmp_path) is None
    # image without expose_file_meta → bare content; with → wrapped
    bare = message_from_file("logo.png", base_dir=tmp_path)
    assert bare is not None and bare.meta.tag == ""
    wrapped = message_from_file("logo.png", base_dir=tmp_path, expose_file_meta=True)
    assert wrapped is not None and wrapped.meta.tag == "file"


# ── conversion (moss Message → pydantic-ai parts) ──────────────────────


def test_message_to_parts_order_preserved():
    msg = Message.new()
    msg.with_content("before")
    msg.with_content({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="}})
    msg.with_content("after")
    parts = message_to_parts(msg)
    # image → BinaryContent (protocol-neutral base64 carrier), not ImageUrl
    assert [type(p).__name__ for p in parts] == ["TextContent", "BinaryContent", "TextContent"]
    assert parts[0].content == "before"
    assert parts[2].content == "after"
    assert parts[1].media_type == "image/png"
    assert parts[1].data == b"hello"


def test_messages_to_parts_flattens_list():
    a = Message.new().with_content("a")
    b = Message.new().with_content("b")
    parts = messages_to_parts([a, b])
    assert [p.content for p in parts] == ["a", "b"]
