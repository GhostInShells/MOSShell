import asyncio

import pytest

from ghoshell_moss import Text, Message
from ghoshell_moss_contrib.agent.chat.queue import QueueChat


@pytest.mark.asyncio
async def test_queue_chat():
    input_q = asyncio.Queue()
    output_q = asyncio.Queue()
    chat = QueueChat(input_q, output_q)
    runner = asyncio.create_task(chat.run())

    # 等待启动消息
    msg = await output_q.get()
    assert msg.role == "system"
    assert len(msg.contents) == 1
    text = Text.from_content(msg.contents[0])
    assert text is not None
    assert text.text == "队列聊天已启动"

    # 发送一条消息
    input_q.put_nowait(Message.new().with_content(Text(text="你好")))
    # 等待回复消息
    msg = await output_q.get()
    assert msg.role == "user"
    assert len(msg.contents) == 1
    text = Text.from_content(msg.contents[0])
    assert text is not None
    assert text.text == "你好"

    chat.close()
    await runner
    assert chat.is_closed.is_set()



