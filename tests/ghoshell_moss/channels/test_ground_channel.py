"""Tests for ground_channel — 一个 home 一条命令.

测的是 channel 对协议的唯一增量承诺: 把「某个位置的渲染」推进 ghost 的活上下文.
命令面只有 `render`; 编辑归文件工具, 发现归 GROUND.md 自己, 边界归 home.
"""

from __future__ import annotations

import pytest

from ghoshell_moss.channels.ground_channel import new_ground_channel


@pytest.fixture
def home(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "hot.py").write_text("def hot():\n    return 1\n")
    (tmp_path / "GROUND.md").write_text(
        "---\n"
        "name: test\n"
        "pins:\n"
        "- label: hot\n"
        "  verb: file\n"
        "  arguments:\n"
        "    path: data/hot.py\n"
        "  description: hot spot\n"
        "---\n"
        "\n"
        "# Test Ground\n"
        "body line\n",
        encoding="utf-8",
    )
    return tmp_path


def _chan(home, **kw):
    return new_ground_channel(home, **kw)


@pytest.mark.asyncio
async def test_exposes_exactly_one_command(home):
    """命令面就是一个 render —— 其余能力各有其主 (文件工具 / 协议本身)."""
    chan = _chan(home)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        meta = runtime.self_meta()
    assert {c.name for c in meta.commands} == {"render"}


@pytest.mark.asyncio
async def test_render_defaults_to_the_home_root(home):
    chan = _chan(home)
    async with chan.bootstrap() as runtime:
        out = await runtime.execute_command("render")
    assert 'name="test"' in out
    assert "body line" in out


@pytest.mark.asyncio
async def test_render_a_subdirectory_walks(home):
    (home / "data" / "inner").mkdir()
    chan = _chan(home)
    async with chan.bootstrap() as runtime:
        out = await runtime.execute_command("render", kwargs={"path": "data/inner"})
    assert "<cwd" in out
    assert "body line" not in out


@pytest.mark.asyncio
async def test_render_outside_the_home_is_an_error_face(home):
    """边界是这个 home: 越界不是一个空结果, 而是一句说清楚的话."""
    chan = _chan(home)
    async with chan.bootstrap() as runtime:
        out = await runtime.execute_command("render", kwargs={"path": "../.."})
    assert 'status="error"' in out
    assert "outside boundary" in out


@pytest.mark.asyncio
async def test_render_through_a_template(home):
    tables = home / ".grounds"
    tables.mkdir()
    (tables / "lens.md").write_text("---\nname: lens\n---\n\nLENS BODY\n", encoding="utf-8")
    chan = _chan(home)
    async with chan.bootstrap() as runtime:
        out = await runtime.execute_command("render", kwargs={"template": "lens"})
    assert "LENS BODY" in out
    assert "body line" not in out


@pytest.mark.asyncio
async def test_instruction_names_the_home(home):
    chan = _chan(home)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        meta = runtime.self_meta()
    assert str(home.resolve()) in (meta.instruction or "")
