import io
import pytest
from PIL import Image as PILImage

from ghoshell_moss.core import new_ctml_shell, new_channel, PyChannel
from ghoshell_moss.message import Message, Text
from ghoshell_moss.message.contents.images import Base64Image


def _red_image(w=100, h=50) -> PILImage.Image:
    return PILImage.new("RGB", (w, h), color="red")


def _image_to_bytes(img: PILImage.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# -- baseline: command returns PIL Image directly ------------------------


@pytest.mark.asyncio
async def test_shell_image_baseline():
    """shell 执行命令, 直接拿到 PIL Image 对象."""
    shell = new_ctml_shell()

    @shell.main_channel.build.command()
    async def get_image() -> PILImage.Image:
        return _red_image(100, 50)

    async with shell:
        async with await shell.interpreter() as interpreter:
            interpreter.feed("<get_image />")
            interpreter.commit()
            tasks = await interpreter.wait_tasks()
            assert len(tasks) == 1
            task = list(tasks.values())[0]
            assert task.success()
            result = task.result()
            assert isinstance(result, PILImage.Image)
            assert result.size == (100, 50)


@pytest.mark.asyncio
async def test_shell_image_return_bytes():
    """shell 执行返回图片 bytes 的命令."""
    shell = new_ctml_shell()

    @shell.main_channel.build.command()
    async def get_image_bytes() -> bytes:
        img = _red_image(64, 64)
        return _image_to_bytes(img)

    async with shell:
        async with await shell.interpreter() as interpreter:
            interpreter.feed("<get_image_bytes />")
            interpreter.commit()
            tasks = await interpreter.wait_tasks()
            assert len(tasks) == 1
            task = list(tasks.values())[0]
            assert task.success()
            result = task.result()
            assert isinstance(result, bytes)
            # bytes 可以还原为 PIL Image
            restored = PILImage.open(io.BytesIO(result))
            assert restored.size == (64, 64)


# -- channel with multiple image commands --------------------------------


@pytest.mark.asyncio
async def test_shell_image_in_sub_channel():
    """子 channel 中的 image 命令, shell 可以从子 channel 路径调用."""
    shell = new_ctml_shell()
    img_chan = new_channel("img")

    @img_chan.build.command()
    async def capture() -> bytes:
        img = _red_image(200, 100)
        return _image_to_bytes(img, fmt="JPEG")

    shell.main_channel.import_channels(img_chan)

    async with shell:
        async with await shell.interpreter() as interpreter:
            interpreter.feed("<img:capture />")
            interpreter.commit()
            tasks = await interpreter.wait_tasks()
            assert len(tasks) == 1
            task = list(tasks.values())[0]
            assert task.success()
            result = task.result()
            assert isinstance(result, bytes)
            restored = PILImage.open(io.BytesIO(result))
            assert restored.size == (200, 100)
            assert restored.format == "JPEG"


@pytest.mark.asyncio
async def test_shell_image_with_args():
    """image 命令接受参数动态生成图片."""
    shell = new_ctml_shell()

    @shell.main_channel.build.command()
    async def create_image(width: int, height: int, color: str = "blue") -> bytes:
        img = PILImage.new("RGB", (width, height), color=color)
        return _image_to_bytes(img)

    async with shell:
        async with await shell.interpreter() as interpreter:
            interpreter.feed('<create_image width="128" height="64" color="green" />')
            interpreter.commit()
            tasks = await interpreter.wait_tasks()
            assert len(tasks) == 1
            task = list(tasks.values())[0]
            assert task.success()
            result = task.result()
            restored = PILImage.open(io.BytesIO(result))
            assert restored.size == (128, 64)


# -- context_messages returning image ------------------------------------


@pytest.mark.asyncio
async def test_context_messages_with_pil_image():
    """channel 的 context_messages 返回 PIL Image, meta.context 中应有 Base64Image 消息."""
    chan = PyChannel(name="vision")

    @chan.build.context_messages
    async def snapshot() -> list:
        return [_red_image(320, 240)]

    async with chan.bootstrap() as runtime:
        meta = runtime.self_meta()
        assert len(meta.context) == 1
        msg = meta.context[0]
        assert isinstance(msg, Message)
        assert len(msg.contents) == 1
        content = msg.contents[0]
        # PIL Image → Base64Image.from_pil_image().to_content() → Content dict
        assert Base64Image.match(content)
        assert content["source"]["media_type"] == "image/png"
        assert content["source"]["type"] == "base64"
        assert len(content["source"]["data"]) > 0


@pytest.mark.asyncio
async def test_shell_context_messages_with_image():
    """shell 引入带 context_messages 的 channel, refresh 后 meta 中包含图片."""
    shell = new_ctml_shell()
    vision_chan = new_channel("vision")

    @vision_chan.build.context_messages
    async def snapshot() -> list:
        return [_red_image(640, 480)]

    shell.main_channel.import_channels(vision_chan)

    async with shell:
        await shell.refresh_metas()
        metas = shell.channel_metas()
        assert "vision" in metas
        vision_meta = metas["vision"]
        assert len(vision_meta.context) == 1
        msg = vision_meta.context[0]
        assert isinstance(msg, Message)
        assert len(msg.contents) == 1
        content = msg.contents[0]
        assert Base64Image.match(content)
        assert content["source"]["media_type"] == "image/png"
        assert len(content["source"]["data"]) > 0


@pytest.mark.asyncio
async def test_context_messages_with_image_and_text():
    """context_messages 混合返回 PIL Image 和 str, 均被正确包装为 Message."""
    chan = PyChannel(name="multi")

    @chan.build.context_messages
    async def mixed() -> list:
        return [_red_image(100, 100), "camera snapshot at 30fps"]

    async with chan.bootstrap() as runtime:
        meta = runtime.self_meta()
        # _wrap_messages 将连续的 raw item 合并到一个 Message 中
        assert len(meta.context) == 1
        msg = meta.context[0]
        assert len(msg.contents) == 2
        # 第一条 content 是 image
        assert Base64Image.match(msg.contents[0])
        assert msg.contents[0]["source"]["media_type"] == "image/png"
        # 第二条 content 是 text
        assert "camera snapshot" in Message.content_as_string(msg.contents[1])


# -- serialization round-trip (跨进程传输场景) ---------------------------


@pytest.mark.asyncio
async def test_context_messages_image_survives_serialization():
    """模拟跨进程序列化/反序列化: Message → JSON → dict → Message, image content 不丢失."""
    chan = PyChannel(name="vision")

    @chan.build.context_messages
    async def snapshot() -> list:
        return [_red_image(320, 240)]

    async with chan.bootstrap() as runtime:
        meta = runtime.self_meta()
        original_msg = meta.context[0]

        # 序列化为 JSON 再还原（模拟跨进程传输）
        json_str = original_msg.to_json()
        raw = Message.model_validate_json(json_str)
        assert isinstance(raw, Message)

        # 反序列化后的 content 是 plain dict，但 from_content 应能还原
        for content in raw.contents:
            # Text.from_content 和 Base64Image.from_content 都需要 type 键
            if text := Text.from_content(content):
                continue
            elif base64_image := Base64Image.from_content(content):
                assert base64_image.source["media_type"] == "image/png"
                assert len(base64_image.source["data"]) > 0
                # data_url 可用于 pydantic-ai ImageUrl
                assert base64_image.data_url.startswith("data:image/png;base64,")
                break
            else:
                pytest.fail(f"content lost after deserialization: {content}")


def test_image_content_survives_message_roundtrip():
    """非 async: 直接验证 wrap_content → Message → JSON → 反序列化 → from_content 全链路."""
    img = _red_image(64, 64)
    msg = Message.new().with_content(img)

    # 序列化往返
    raw = Message.model_validate_json(msg.to_json())

    assert len(raw.contents) == 1
    content = raw.contents[0]
    # 反序列化后是 Content dict，必须有 type 键才能被 from_content 匹配
    assert "type" in content
    assert content["type"] == "image"
    # Base64Image.from_content 能正常工作
    restored = Base64Image.from_content(content)
    assert restored is not None
    assert restored.data_url.startswith("data:image/png;base64,")


# -- integration: full proxy/provider bridge with context_messages image --


@pytest.mark.asyncio
async def test_context_messages_image_through_bridge():
    """context_messages 返回 PIL Image，通过 thread bridge 全链路序列化传输后 proxy 侧可拿到图片."""
    from ghoshell_moss.core.duplex.thread_channel import create_thread_bridge

    chan = PyChannel(name="vision")

    @chan.build.context_messages
    async def snapshot() -> list:
        return [_red_image(320, 240)]

    provider, proxy = create_thread_bridge("proxy")

    async with provider.arun(chan):
        async with proxy.bootstrap() as runtime:
            await runtime.wait_connected()
            assert runtime.is_running()

            # 刷新 metas，触发 context_messages 通过 bridge 序列化传输
            await runtime.refresh_metas()

            metas = runtime.metas()
            # provider 的 root channel 在 proxy 侧 key 为 ""，name 被 overwrite 为 proxy name
            root_meta = metas.get("") or metas.get("vision")
            assert root_meta is not None
            assert len(root_meta.context) == 1

            msg = root_meta.context[0]
            assert isinstance(msg, Message)
            assert len(msg.contents) == 1

            content = msg.contents[0]
            # 全链路序列化/反序列化后 Content dict 的 type 和 source 必须存活
            assert content.get("type") == "image"
            source = content.get("source") or {}
            assert source.get("media_type") == "image/png"
            assert source.get("type") == "base64"
            assert len(source.get("data", "")) > 0

            # from_content 还原为 Base64Image
            restored = Base64Image.from_content(content)
            assert restored is not None
            assert restored.data_url.startswith("data:image/png;base64,")
