import tempfile
from pathlib import Path

import pytest

from ghoshell_moss.core.concepts.errors import CommandError
from ghoshell_moss.channels.notebook_channel import new_notebook_channel


@pytest.fixture
def tmpdir():
    with tempfile.TemporaryDirectory() as d:
        yield d


# ---- 基线 ---- #

@pytest.mark.asyncio
async def test_write_and_read(tmpdir):
    chan = new_notebook_channel(tmpdir)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        assert runtime.get_command("write") is not None
        assert runtime.get_command("read") is not None

        await runtime.execute_command("write", kwargs={"name": "note.md", "text__": "# Hello"})
        result = await runtime.execute_command("read", kwargs={"name": "note.md"})
        assert result == "# Hello"


@pytest.mark.asyncio
async def test_append(tmpdir):
    chan = new_notebook_channel(tmpdir)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("write", kwargs={"name": "log.txt", "text__": "line1"})
        await runtime.execute_command("append", kwargs={"name": "log.txt", "text__": "\nline2"})
        result = await runtime.execute_command("read", kwargs={"name": "log.txt"})
        assert result == "line1\nline2"


@pytest.mark.asyncio
async def test_list_pages(tmpdir):
    chan = new_notebook_channel(tmpdir)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("write", kwargs={"name": "a.md", "text__": "a"})
        await runtime.execute_command("write", kwargs={"name": "b.md", "text__": "b"})
        pages = await runtime.execute_command("list_pages")
        assert set(pages) == {"a.md", "b.md"}


@pytest.mark.asyncio
async def test_delete(tmpdir):
    chan = new_notebook_channel(tmpdir)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("write", kwargs={"name": "tmp.md", "text__": "x"})
        result = await runtime.execute_command("delete", kwargs={"name": "tmp.md"})
        assert "Deleted" in result
        result = await runtime.execute_command("read", kwargs={"name": "tmp.md"})
        assert "not found" in result


# ---- 路径安全 ---- #

@pytest.mark.asyncio
async def test_traversal_rejected(tmpdir):
    chan = new_notebook_channel(tmpdir)
    async with chan.bootstrap() as runtime:
        with pytest.raises(CommandError, match="Invalid"):
            await runtime.execute_command("write", kwargs={"name": "../escape.md", "text__": "x"})


@pytest.mark.asyncio
async def test_absolute_path_rejected(tmpdir):
    chan = new_notebook_channel(tmpdir)
    async with chan.bootstrap() as runtime:
        with pytest.raises(CommandError, match="Invalid"):
            await runtime.execute_command("read", kwargs={"name": "/etc/passwd"})


# ---- context_messages ---- #

@pytest.mark.asyncio
async def test_context_shows_tree(tmpdir):
    chan = new_notebook_channel(tmpdir)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("write", kwargs={"name": "notes/design.md", "text__": "x"})
        await runtime.refresh_metas()
        meta = runtime.self_meta()
        assert len(meta.context) > 0
        context_text = "".join(
            c.get("text", "") for c in meta.context[0].contents if c.get("type") == "text"
        )
        assert "design.md" in context_text
        assert "notes/" in context_text


# ---- 子目录支持 ---- #

@pytest.mark.asyncio
async def test_nested_directories(tmpdir):
    chan = new_notebook_channel(tmpdir)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("write", kwargs={"name": "a/b/c.md", "text__": "deep"})
        result = await runtime.execute_command("read", kwargs={"name": "a/b/c.md"})
        assert result == "deep"
        assert Path(tmpdir, "a", "b", "c.md").read_text() == "deep"


# ---- 空 notebook ---- #

@pytest.mark.asyncio
async def test_empty_notebook_context(tmpdir):
    chan = new_notebook_channel(tmpdir)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        meta = runtime.self_meta()
        assert len(meta.context) > 0
        context_text = "".join(
            c.get("text", "") for c in meta.context[0].contents if c.get("type") == "text"
        )
        assert "empty" in context_text.lower()


# ---- 写入覆盖 ---- #

@pytest.mark.asyncio
async def test_write_overwrites(tmpdir):
    chan = new_notebook_channel(tmpdir)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("write", kwargs={"name": "x.md", "text__": "v1"})
        await runtime.execute_command("write", kwargs={"name": "x.md", "text__": "v2"})
        result = await runtime.execute_command("read", kwargs={"name": "x.md"})
        assert result == "v2"
