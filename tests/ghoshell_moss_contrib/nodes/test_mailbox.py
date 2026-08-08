"""Tests for ghoshell_moss_contrib.nodes.mailbox — MailboxBridge + MCP integration."""
import asyncio

import pytest

from ghoshell_moss_contrib.nodes.mailbox import MailboxBridge, _Envelope


# ---------------------------------------------------------------------------
# MailboxBridge unit tests
# ---------------------------------------------------------------------------

class TestMailboxBridge:
    def test_create_envelope(self):
        bridge = MailboxBridge()
        env = bridge.create("hello")
        assert isinstance(env, _Envelope)
        assert len(env.task_id) == 12
        assert env.message == "hello"
        assert env.reply is None

    def test_check_before_reply(self):
        bridge = MailboxBridge()
        env = bridge.create("msg")
        assert bridge.check(env.task_id) is None

    def test_post_and_check(self):
        bridge = MailboxBridge()
        env = bridge.create("msg")
        assert bridge.post(env.task_id, "response") is True
        assert bridge.check(env.task_id) == "response"

    def test_post_unknown_id(self):
        assert MailboxBridge().post("nonexistent", "x") is False

    def test_check_unknown_id(self):
        assert MailboxBridge().check("nonexistent") is None

    @pytest.mark.asyncio
    async def test_wait_immediate(self):
        bridge = MailboxBridge()
        env = bridge.create("msg")
        bridge.post(env.task_id, "instant")
        reply = await bridge.wait(env.task_id, timeout=5.0)
        assert reply == "instant"

    @pytest.mark.asyncio
    async def test_wait_delayed(self):
        bridge = MailboxBridge()
        env = bridge.create("msg")

        async def delay():
            await asyncio.sleep(0.05)
            bridge.post(env.task_id, "delayed")

        asyncio.create_task(delay())
        reply = await bridge.wait(env.task_id, timeout=5.0)
        assert reply == "delayed"

    @pytest.mark.asyncio
    async def test_wait_timeout(self):
        bridge = MailboxBridge()
        env = bridge.create("msg")
        reply = await bridge.wait(env.task_id, timeout=0.01)
        assert reply is None

    @pytest.mark.asyncio
    async def test_wait_unknown_id(self):
        reply = await MailboxBridge().wait("nonexistent", timeout=1.0)
        assert reply is None

    def test_post_already_replied_overwrites(self):
        bridge = MailboxBridge()
        env = bridge.create("msg")
        bridge.post(env.task_id, "first")
        bridge.post(env.task_id, "second")
        assert bridge.check(env.task_id) == "second"

    @pytest.mark.asyncio
    async def test_gc_removes_stale(self):
        bridge = MailboxBridge(ttl=0.01)
        env = bridge.create("msg")
        bridge.post(env.task_id, "done")
        await asyncio.sleep(0.05)
        bridge.create("trigger gc")
        assert env.task_id not in bridge._envelopes

    def test_gc_keeps_unreplied(self):
        bridge = MailboxBridge(ttl=0.01)
        env = bridge.create("msg")
        bridge.create("trigger gc")
        assert env.task_id in bridge._envelopes


# ---------------------------------------------------------------------------
# MCP integration: send/pull cycle via streamable HTTP client
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mcp_send_pull_cycle():
    """Full cycle: send → pull(waiting) → post reply → pull(reply)."""
    from mcp.server.mcpserver import MCPServer
    from mcp.client.session_group import ClientSessionGroup, StreamableHttpParameters

    bridge = MailboxBridge()
    mcp = MCPServer("test-mailbox")
    _PORT = 20779

    @mcp.tool()
    async def send(message: str, wait: float = 0) -> str:
        env = bridge.create(message)
        if wait > 0:
            reply = await bridge.wait(env.task_id, timeout=wait)
            if reply is not None:
                return reply
            return f"[pending] task_id: {env.task_id}"
        return f"[sent] task_id: {env.task_id}"

    @mcp.tool()
    async def pull(task_id: str) -> str:
        reply = bridge.check(task_id)
        if reply is not None:
            return reply
        return f"[waiting] no reply yet for {task_id}"

    server_task = asyncio.create_task(
        mcp.run_streamable_http_async(
            host="127.0.0.1", port=_PORT, stateless_http=True,
        )
    )
    await asyncio.sleep(0.5)

    try:
        params = StreamableHttpParameters(url=f"http://127.0.0.1:{_PORT}/mcp")
        async with ClientSessionGroup() as sg:
            session = await sg.connect_to_server(params)
            await session.initialize()

            tools = await session.list_tools()
            tool_names = {t.name for t in tools.tools}
            assert "send" in tool_names
            assert "pull" in tool_names

            # send -> should get a task_id
            result = await session.call_tool("send", {"message": "hello"})
            text = result.content[0].text
            assert "task_id:" in text
            task_id = text.split("task_id: ")[1].split("\n")[0]

            # pull before reply
            result = await session.call_tool("pull", {"task_id": task_id})
            assert "waiting" in result.content[0].text.lower()

            # ghost replies
            bridge.post(task_id, "ghost says hi")

            # pull after reply
            result = await session.call_tool("pull", {"task_id": task_id})
            assert result.content[0].text == "ghost says hi"
    finally:
        server_task.cancel()
        try:
            await server_task
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_send_blocking_wait():
    """send(wait=N) blocks and returns reply when ghost responds in time."""
    from mcp.server.mcpserver import MCPServer
    from mcp.client.session_group import ClientSessionGroup, StreamableHttpParameters

    bridge = MailboxBridge()
    mcp = MCPServer("test-wait-mailbox")
    _PORT = 20780

    @mcp.tool()
    async def send(message: str, wait: float = 0) -> str:
        env = bridge.create(message)
        if wait > 0:
            reply = await bridge.wait(env.task_id, timeout=wait)
            if reply is not None:
                return reply
            return f"[pending] task_id: {env.task_id}"
        return f"[sent] task_id: {env.task_id}"

    @mcp.tool()
    async def pull(task_id: str) -> str:
        reply = bridge.check(task_id)
        if reply is not None:
            return reply
        return f"[waiting] {task_id}"

    server_task = asyncio.create_task(
        mcp.run_streamable_http_async(
            host="127.0.0.1", port=_PORT, stateless_http=True,
        )
    )
    await asyncio.sleep(0.5)

    try:
        params = StreamableHttpParameters(url=f"http://127.0.0.1:{_PORT}/mcp")
        async with ClientSessionGroup() as sg:
            session = await sg.connect_to_server(params)
            await session.initialize()

            # Arrange a delayed reply
            async def delay_reply():
                await asyncio.sleep(0.2)
                for tid in list(bridge._envelopes):
                    bridge.post(tid, "delayed response")

            asyncio.create_task(delay_reply())
            result = await session.call_tool("send", {"message": "hi", "wait": 2.0})
            text = result.content[0].text
            assert "delayed response" in text
    finally:
        server_task.cancel()
        try:
            await server_task
        except (asyncio.CancelledError, Exception):
            pass
