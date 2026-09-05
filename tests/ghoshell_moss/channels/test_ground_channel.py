"""Tests for ground_channel — 认知场的运行时落点 (virtual children 模型).

Covers:
- 命令注册 (open / close / render / pin_* / spec / validate / templates)
- open → virtual child (instruction=meta, help=帧); close → 撤下
- render 无状态 peek (body + pin 内容), meta flag
- pin_file 增改落盘 + always_show; 目标不存在报告不创建
- spec / validate 诊断
- open_on_start 启动时开场面
"""

from __future__ import annotations

import pytest

from ghoshell_moss.channels.ground_channel import new_ground_channel
from ghoshell_moss.ground import DefaultGroundSet


@pytest.fixture
def ground(tmp_path):
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
        "body line\n"
    )
    return tmp_path


def _chan(ground, **kw):
    gs = DefaultGroundSet(workspace_root=ground)
    # 测试默认展开编辑命令组, 直接覆盖全命令面; 折叠行为另测.
    kw.setdefault("edit", True)
    return new_ground_channel(gs, workspace_root=ground, **kw)


def _meta_by_name(runtime, name):
    for m in runtime.metas().values():
        if m.name == name:
            return m
    return None


@pytest.mark.asyncio
async def test_commands_registered(ground):
    chan = _chan(ground)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        names = {c.name for c in runtime.self_meta().commands}
        assert {"open", "close", "render", "edit", "pin_file", "pin_glob",
                "pin_frontmatter", "pin_ls", "pin_exec", "pin_law",
                "spec", "validate", "templates"} <= names


@pytest.mark.asyncio
async def test_instruction_explains_ground_but_does_not_list_commands(ground):
    chan = _chan(ground)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        instruction = runtime.self_meta().instruction
        assert "GROUND.md" in instruction
        # 红线: instruction 不重复罗列命令
        assert "pin_file(" not in instruction


@pytest.mark.asyncio
async def test_open_adds_virtual_child_with_meta_and_frame(ground):
    (ground / "sub").mkdir()
    (ground / "sub" / "GROUND.md").write_text("---\nname: sub\n---\n# Sub Ground\n")
    chan = _chan(ground)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        await runtime.execute_command("open", kwargs={"directory": "sub"})
        await runtime.refresh_metas()
        child = _meta_by_name(runtime, "sub")
        assert child is not None
        assert "cd " in child.instruction            # meta 在 instruction
        assert "# Sub Ground" in child.notice          # 帧在 help


@pytest.mark.asyncio
async def test_close_removes_virtual_child(ground):
    (ground / "sub").mkdir()
    (ground / "sub" / "GROUND.md").write_text("---\nname: sub\n---\n# Sub Ground\n")
    chan = _chan(ground)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        await runtime.execute_command("open", kwargs={"directory": "sub"})
        result = await runtime.execute_command("close", kwargs={"label": "sub"})
        assert "closed sub" in result
        await runtime.refresh_metas()
        assert _meta_by_name(runtime, "sub") is None


@pytest.mark.asyncio
async def test_render_peek_shows_frame(ground):
    chan = _chan(ground)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command("render", kwargs={"directory": "."})
        assert "# Test Ground" in result
        assert "def hot():" in result
        assert "hot spot" in result


@pytest.mark.asyncio
async def test_render_meta_flag_adds_identity(ground):
    chan = _chan(ground)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command("render", kwargs={"directory": ".", "meta": True})
        assert "cd " in result and "pins:" in result


@pytest.mark.asyncio
async def test_render_nonexistent_dir_reported(ground):
    chan = _chan(ground)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command("render", kwargs={"directory": "nope"})
        assert "not a directory" in result or "no GROUND.md" in result


@pytest.mark.asyncio
async def test_pin_file_reports_missing_ground_file(ground):
    chan = _chan(ground)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command(
            "pin_file", kwargs={"ground_file": "missing.md", "label": "x", "path": "y"}
        )
        assert "no such GROUND.md" in result
        assert not (ground / "missing.md").exists()


@pytest.mark.asyncio
async def test_pin_file_writes_with_always_show(ground):
    chan = _chan(ground)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command(
            "pin_file",
            kwargs={"ground_file": ".", "label": "greet", "path": "data/hot.py",
                    "always_show": True, "description": "greet"},
        )
        assert "pinned file:greet" in result
        text = (ground / "GROUND.md").read_text()
        assert "label: greet" in text
        assert "label: hot" in text               # 原有 pin 保留
        assert "always_show: true" in text        # always_show 落盘


@pytest.mark.asyncio
async def test_spec_returns_specification(ground):
    chan = _chan(ground)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command("spec")
        assert "GROUND" in result


@pytest.mark.asyncio
async def test_validate_reports_invalid(ground):
    bad = ground / "bad.md"
    bad.write_text("---\npins:\n- verb: file\n  arguments:\n    path: x\n---\n")
    chan = _chan(ground)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command("validate", kwargs={"filepath": "bad.md"})
        assert "[ERROR]" in result
        assert "missing 'label'" in result


@pytest.mark.asyncio
async def test_validate_warns_on_unknown_verb(ground):
    bad = ground / "bad.md"
    bad.write_text("---\npins:\n- label: x\n  verb: unknown_verb\n---\n")
    chan = _chan(ground)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command("validate", kwargs={"filepath": "bad.md"})
        assert "[WARN]" in result
        assert "unknown verb" in result


@pytest.mark.asyncio
async def test_validate_rejects_long_label(ground):
    # label 超 63 字符须报 ERROR, 与 contract 的 Pin.label max_length=63 对齐.
    bad = ground / "bad.md"
    bad.write_text("---\npins:\n- label: " + "a" * 64 + "\n  verb: file\n  arguments:\n    path: x\n---\n")
    chan = _chan(ground)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command("validate", kwargs={"filepath": "bad.md"})
        assert "[ERROR]" in result
        assert "63" in result


@pytest.mark.asyncio
async def test_templates_lists_or_reports_empty(ground):
    chan = _chan(ground)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command("templates")
        assert "templates" in result or "found" in result


@pytest.mark.asyncio
async def test_open_on_start_injects_child(ground):
    child = ground / "child"
    child.mkdir()
    (child / "GROUND.md").write_text("---\nname: child\n---\n# Child Ground\n")
    chan = _chan(ground, open_on_start=["child"])
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        assert _meta_by_name(runtime, "child") is not None


@pytest.mark.asyncio
async def test_edit_folds_commands_by_default(ground):
    # 默认 edit=False: 编辑命令组折叠, 只留 open/close/render/edit.
    gs = DefaultGroundSet(workspace_root=ground)
    chan = new_ground_channel(gs, workspace_root=ground)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        names = {c.name for c in runtime.self_meta().commands}
        assert {"open", "close", "render", "edit"} <= names
        assert "pin_file" not in names
        assert "spec" not in names
        assert "validate" not in names
        assert "templates" not in names


@pytest.mark.asyncio
async def test_edit_command_unfolds_gated_commands(ground):
    gs = DefaultGroundSet(workspace_root=ground)
    chan = new_ground_channel(gs, workspace_root=ground)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        assert "pin_file" not in {c.name for c in runtime.self_meta().commands}
        await runtime.execute_command("edit", kwargs={"on": True})
        await runtime.refresh_metas()
        names = {c.name for c in runtime.self_meta().commands}
        assert "pin_file" in names
        assert "spec" in names
        # 折叠回来
        await runtime.execute_command("edit", kwargs={"on": False})
        await runtime.refresh_metas()
        assert "pin_file" not in {c.name for c in runtime.self_meta().commands}
