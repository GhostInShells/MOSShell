from pathlib import Path

from ghoshell_moss.message import Message, Text, MessageMeta, Base64Image


def test_base64_image_from_file_labels_by_bytes_not_extension(tmp_path: Path):
    """from_file 的 media_type 取自字节头: 扩展名撒谎 (JPEG 存成 .png) 时不跟着错."""
    mislabeled = tmp_path / "photo.png"
    mislabeled.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 16)
    assert Base64Image.from_file(mislabeled).source["media_type"] == "image/jpeg"

    png = tmp_path / "logo.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    assert Base64Image.from_file(png).source["media_type"] == "image/png"


def test_message_baseline():
    msg = Message.new()
    msg.with_content(*[Text.new("hello").to_content()])
    assert len(msg.contents) == 1


def test_message_meta_attributes_str():
    meta = MessageMeta()
    assert 'created' in meta.gen_attributes_str()


def test_message_unmarshal():
    msg = Message.new().with_content(Base64Image.from_binary(data=bytes(), media_type='image/jpeg'))

    image = Base64Image.from_content(msg.contents[0])
    assert 'image/jpeg' in image.data_url


def test_as_contents_tag_wrapping_with_join_text():
    """有 tag 时 as_contents(join_text=True) 应该用 XML 标签包裹内容。"""
    msg = Message.new(tag="thinking", timestamp=False).with_content("推理内容")
    results = list(msg.as_contents(with_meta=True, join_text=True))
    text = "".join(c.get("text", "") for c in results)
    assert "<thinking" in text
    assert "推理内容" in text
    assert "</thinking>" in text


def test_as_contents_tag_wrapping_without_join_text():
    """join_text=False 时同样应该包裹标签。"""
    msg = Message.new(tag="observation", timestamp=False).with_content("感知数据")
    results = list(msg.as_contents(with_meta=True, join_text=False))
    texts = [c.get("text", "") for c in results]
    assert any("<observation" in t for t in texts)
    assert any("感知数据" in t for t in texts)
    assert any("</observation>" in t for t in texts)


def test_as_contents_no_tag_no_wrapping():
    """tag 为空时不包裹标签。"""
    msg = Message.new(tag="", timestamp=False).with_content("普通消息")
    results = list(msg.as_contents(with_meta=True, join_text=True))
    text = "".join(c.get("text", "") for c in results)
    assert text == "普通消息"


def test_as_contents_with_meta_false_no_wrapping():
    """with_meta=False 时即使有 tag 也不包裹。"""
    msg = Message.new(tag="thinking", timestamp=False).with_content("内容")
    results = list(msg.as_contents(with_meta=False, join_text=True))
    text = "".join(c.get("text", "") for c in results)
    assert text == "内容"


def test_as_contents_join_text_preserves_image_order():
    """join_text=True 合并相邻 text 时, image 块保持相对顺序, 不越位到前导 text 之前."""
    img = Base64Image.from_base64('image/png', 'aGVsbG8=')
    msg = (
        Message.new(tag="percepts", timestamp=False)
        .with_content("前导")
        .with_content(img)
        .with_content("后缀")
    )
    results = list(msg.as_contents(with_meta=True, join_text=True))
    assert [c.get("type") for c in results] == ['text', 'image', 'text']
    assert "前导" in results[0].get("text", "")
    assert "后缀" not in results[0].get("text", "")
