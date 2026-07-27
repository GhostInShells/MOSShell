"""Text Blocks channel — shared text-block carrier for human-model collaboration.

Usage::

    from ghoshell_text_blocks.store import BlockStore
    from ghoshell_text_blocks.screen import ScreenAddr, ScreenPush
    from ghoshell_text_blocks.channel import new_text_blocks_channel

    store = BlockStore()
    addr = ScreenAddr(host="127.0.0.1", port=8765)
    screen = ScreenPush(store, addr)

    chan = new_text_blocks_channel(
        name="blocks", store=store, addr=addr, screen=screen,
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ghoshell_moss.core.blueprint.channel_builder import (
    CommandUtil,
    MutableChannel,
    Message,
    new_channel,
)
from ghoshell_moss.core.concepts.channel import Channel

if TYPE_CHECKING:
    from ghoshell_text_blocks.store import BlockStore
    from ghoshell_text_blocks.screen import ScreenAddr, ScreenPush


def new_text_blocks_channel(
    *,
    name: str = "blocks",
    store: BlockStore,
    addr: ScreenAddr,
    screen: ScreenPush,
) -> Channel:
    """Create a text-blocks channel wired to *store* and *screen*."""

    chan = new_channel(
        name=name,
        description=(
            "shared text-block surface: stream text via __content__ to "
            "create numbered blocks; humans edit in browser, submit "
            "produces unified diffs. use read_block(id) to re-sync."
        ),
    )

    # -- instruction: only address + interaction rules --

    @chan.build.instruction
    def instruction() -> str:
        return (
            f"surface: {addr.url}\n"
            f"humans open this in a browser. blocks you stream appear in "
            f"real time (lock=g, streaming). humans see content grow but "
            f"cannot edit until you seal (done=True).\n"
            f"sealed blocks become human-editable. human edits arrive as "
            f"signals with unified diffs.\n"
            f"use done(id) to release lock after content/revise with "
            f"done=False.\n"
            f"use read_file(path) to bring a local file onto the surface.\n"
            f"use dump() to export to tmp/text_blocks_{{uid}}/ by default."
        )

    # -- context_messages: block summary + recent actions --

    @chan.build.context_messages
    async def context() -> list[Message]:
        summary = store.summary()
        actions = store.action_log.recent(5)
        lines = [summary]
        for a in actions:
            lines.append(f"  [{a.kind}] {a.summary}")
        return [Message.new().with_content("\n".join(lines))]

    # -- content: streaming text -> new block --

    @chan.build.content_command
    async def content(chunks__, title: str = "", done: bool = True) -> str:
        """stream text -> new block, live to screen; returns block id.

        set done=False to keep the lock for later append/revise.
        """
        bid = store.create(source="g", title=title, lock="g")
        await screen.block_start(bid)
        async for chunk in chunks__:
            store.append_to_current(bid, chunk)
            await screen.push_chunk(bid, chunk)
        if done:
            store.seal(bid)
            await screen.block_done(bid)
            return CommandUtil.observe(f"#{bid} sealed")
        else:
            await screen.block_held(bid)
            return CommandUtil.observe(f"#{bid} streaming, lock held")

    # -- done: release lock --

    @chan.build.command()
    async def done(block_id: int) -> str:
        """release lock and seal the block."""
        store.seal(block_id)
        await screen.block_done(block_id)
        return CommandUtil.observe(f"#{block_id} sealed")

    # -- revise: model rewrites an existing block --

    @chan.build.command()
    async def revise(block_id: int, chunks__, done: bool = True) -> str:
        """model rewrites a block. creates new version (source=g)."""
        store.acquire_lock(block_id, "g")
        store.get(block_id).new_version("g", "")
        await screen.block_start(block_id)
        async for chunk in chunks__:
            store.append_to_current(block_id, chunk)
            await screen.push_chunk(block_id, chunk)
        if done:
            store.seal(block_id)
            await screen.block_done(block_id)
            return CommandUtil.observe(f"#{block_id} revised")
        else:
            await screen.block_held(block_id)
            return CommandUtil.observe(f"#{block_id} streaming, lock held")

    # -- append: continue writing to a held block --

    @chan.build.command()
    async def append(block_id: int, chunks__) -> str:
        """append text to the current version. requires lock=g."""
        block = store.get(block_id)
        if block is None:
            return CommandUtil.observe(f"#{block_id} not found")
        if block.lock != "g":
            return CommandUtil.observe(f"#{block_id}: lock not held")
        async for chunk in chunks__:
            store.append_to_current(block_id, chunk)
            await screen.push_chunk(block_id, chunk)
        return CommandUtil.observe(f"appended to #{block_id}")

    # -- replace_line: surgical line replacement --

    @chan.build.command()
    async def replace_line(
        block_id: int,
        line_no: int,
        new_text: str,
        count: int = 1,
    ) -> str:
        """replace `count` lines starting at line_no with new_text. requires lock=g."""
        block = store.get(block_id)
        if block is None:
            return CommandUtil.observe(f"#{block_id} not found")
        if block.lock != "g":
            return CommandUtil.observe(f"#{block_id}: lock not held")
        replaced = block.replace_lines(line_no, count, new_text)
        await screen.push_block(block_id, block.content)
        return CommandUtil.observe(
            f"#{block_id}:{line_no} replaced ({replaced} lines)"
        )

    # -- read commands --

    @chan.build.command()
    async def read_block(block_id: int, version: int | None = None) -> str:
        """read block content with line numbers. version=None = latest."""
        block = store.get(block_id)
        if block is None:
            return f"block #{block_id} not found"
        return block.with_line_numbers

    @chan.build.command()
    async def list_blocks(self) -> str:
        """list all blocks: id, source, status, lock, version count, title."""
        return store.index()

    # -- file bridge --

    @chan.build.command()
    async def read_file(path: str, title: str = "") -> str:
        """read a local file onto the surface as a new sealed block."""
        import aiofiles
        async with aiofiles.open(path, "r") as f:
            content = await f.read()
        bid = store.create(source="g", title=title or path, content=content)
        await screen.push_block(bid, content)
        from ghoshell_text_blocks.store import Action as StoreAction
        store.action_log.record(StoreAction(
            kind="block_read_file", block_id=bid,
            summary=f"g: read_file {path} -> #{bid}",
        ))
        return CommandUtil.observe(f"#{bid} <- {path}")

    # -- dump --

    @chan.build.command()
    async def dump(
        path: str = "",
        ids: list[int] | None = None,
    ) -> str:
        """export blocks. default: tmp/text_blocks_{uid}/"""
        result = store.dump(path, ids)
        return CommandUtil.observe(
            f"{result.count} blocks -> {result.path}"
        )

    # -- lifecycle --

    @chan.build.startup
    async def startup():
        await screen.serve()

    @chan.build.close
    async def close():
        await screen.shutdown()

    return chan
