"""Mailbox bridge — bidirectional request-reply between MCP agents and MOSS ghost.

This module provides the :class:`MailboxBridge` state object and a
:func:`serve_mailbox` helper that wires it into a Matrix channel +
MCPServer in one call.  The node entry point (:file:`nodes/mailbox/main.py`)
is a thin shell over this module.

Dependencies (mcp) are lazily checked at serve time so the module is
always importable.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field

from ghoshell_moss.core.blueprint.channel_builder import new_channel
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.depends import depend_mcp
from ghoshell_moss.message import Message
from ghoshell_moss.signals import NotifySignalMeta


# ---------------------------------------------------------------------------
# Shared bridge
# ---------------------------------------------------------------------------

@dataclass
class _Envelope:
    """One request-reply cycle."""
    task_id: str
    message: str
    reply: str | None = None
    replied: asyncio.Event = field(default_factory=asyncio.Event)
    created_at: float = field(default_factory=time.monotonic)


class MailboxBridge:
    """Request-reply buffer bridging MCP tools (agent side) and the
    ghost-side channel command (``mailbox:reply``).

    *Agent → ghost*: ``create()`` stores a pending envelope, ``wait()`` /
    ``check()`` retrieve the ghost's reply.

    *Ghost → agent*: ``post()`` fills the envelope with the reply content,
    waking any waiter.

    Envelopes with completed replies are garbage-collected after *ttl*
    seconds (default 300).
    """

    def __init__(self, ttl: float = 300):
        self._ttl = ttl
        self._envelopes: dict[str, _Envelope] = {}

    # -- agent side ----------------------------------------------------------

    def create(self, message: str) -> _Envelope:
        """Create a pending request.  Returns the envelope with a task_id."""
        self._gc()
        task_id = uuid.uuid4().hex[:12]
        env = _Envelope(task_id=task_id, message=message)
        self._envelopes[task_id] = env
        return env

    async def wait(self, task_id: str, timeout: float) -> str | None:
        """Block until a reply arrives or *timeout* seconds pass."""
        env = self._envelopes.get(task_id)
        if env is None:
            return None
        try:
            await asyncio.wait_for(env.replied.wait(), timeout=timeout)
            return env.reply
        except asyncio.TimeoutError:
            return None

    def check(self, task_id: str) -> str | None:
        """Non-blocking poll for a reply."""
        env = self._envelopes.get(task_id)
        return env.reply if env else None

    # -- ghost side ----------------------------------------------------------

    def post(self, task_id: str, content: str) -> bool:
        """Post a reply from the ghost.  Returns False for unknown ids."""
        self._gc()
        env = self._envelopes.get(task_id)
        if env is None:
            return False
        env.reply = content
        env.replied.set()
        return True

    # -- housekeeping --------------------------------------------------------

    def _gc(self):
        now = time.monotonic()
        stale = [
            tid for tid, e in self._envelopes.items()
            if e.reply and (now - e.created_at) > self._ttl
        ]
        for tid in stale:
            del self._envelopes[tid]


# ---------------------------------------------------------------------------
# Node wiring
# ---------------------------------------------------------------------------

async def serve_mailbox(
    matrix: Matrix,
    bridge: MailboxBridge,
    *,
    server_name: str = "ghost-mailbox",
    host: str = "127.0.0.1",
    port: int = 0,
) -> None:
    """Wire *bridge* into a Matrix channel + MCPServer and serve both concurrently.

    The MCPServer exposes ``send`` and ``pull`` tools for external agents.
    The Matrix channel exposes a ``reply`` command for the ghost to call
    via CTML (``mailbox:reply(id=..., content=...)``).

    Both block until shutdown; the first to exit cancels the other.

    *port=0* lets the OS choose a free port (recommended).
    """
    depend_mcp()  # lazy — only checked when serve is called
    from mcp.server.mcpserver import MCPServer  # noqa: F811

    mcp = MCPServer(server_name)

    # -- MCP tools (agent side) ----------------------------------------------

    @mcp.tool()
    async def send(message: str, wait: float = 0) -> str:
        """Send a message to the ghost.  Returns a task_id for polling.

        Set wait>0 to block up to that many seconds for an immediate
        reply.  Otherwise call pull(task_id) to check for the ghost's
        response.
        """
        env = bridge.create(message)
        signal = NotifySignalMeta().to_signal(
            Message.new().with_content(f"[mailbox:{env.task_id}] {message}"),
            description=(
                f"mailbox message — reply with CTML: "
                f"<mailbox:reply task_id=\"{env.task_id}\">your reply</mailbox:reply>"
            ),
        )
        matrix.session.add_signal(signal)

        if wait > 0:
            reply = await bridge.wait(env.task_id, timeout=wait)
            if reply is not None:
                return reply
            return (
                f"[pending] no reply within {wait}s.\n"
                f"task_id: {env.task_id}\n"
                f"poll: pull(task_id=\"{env.task_id}\")"
            )

        return (
            f"[sent] task_id: {env.task_id}\n"
            f"poll: pull(task_id=\"{env.task_id}\")"
        )

    @mcp.tool()
    async def pull(task_id: str) -> str:
        """Poll for a ghost reply by task_id.  Returns the reply content
        when ready, or a waiting status otherwise.
        """
        reply = bridge.check(task_id)
        if reply is not None:
            return reply
        return f"[waiting] no reply yet for {task_id}"

    @mcp.tool()
    async def wait_reply(task_id: str, timeout: float = 60.0) -> str:
        """Block until the ghost replies to task_id, or timeout seconds pass.

        Returns the reply content when ready, or a pending status on
        timeout.  The wait is event-driven (no busy polling) — the MCP
        server holds the request until the ghost's reply arrives.
        """
        reply = await bridge.wait(task_id, timeout=timeout)
        if reply is not None:
            return reply
        return f"[pending] no reply within {timeout}s for {task_id}"

    # -- Matrix channel (ghost side) -----------------------------------------

    chan = new_channel(
        name="mailbox",
        description=(
            "Reply to MCP agent messages. "
            "Call reply(task_id, text__) — open-close form with the "
            "task_id from the agent's signal."
        ),
    )

    @chan.build.command(always_observe=False)
    async def reply(task_id: str, text__: str) -> str:
        """Reply to an MCP agent message by task_id.

        The task_id comes from the [mailbox:xxx] prefix in the signal body.
        Use open-close form so the body is free-form text: wrap XML-like
        content in <![CDATA[ ... ]]> — no attribute escaping needed.
        """
        ok = bridge.post(task_id, text__)
        if ok:
            return f"[mailbox] reply delivered for {task_id}"
        return f"[mailbox] unknown task_id: {task_id}"

    await asyncio.gather(
        matrix.provide_channel(chan),
        mcp.run_streamable_http_async(
            host=host, port=port, stateless_http=True,
        ),
    )
