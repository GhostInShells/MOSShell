"""Tests for ground_channel — 认知场的运行时落点 (one set, one channel).

Covers:
- 一个 channel 一个 GroundSet: 子面是 root 的 ``inline:``, 根帧按 render_root 决定去处
- gate: 子场默认只进目录, ``mount_child`` 才进 facade
- 命令面分两层: 主面只有 ``refresh`` (读侧), 落盘/诊断收在 ``admin`` 子通道
- open 的边界 = home: 越界目录拒绝
- 契约 classmethod: peek_dir / validate_text / templates_at / upsert_pin / unpin_at
- pin_file 增改落盘 + always_show; 目标不存在报告不创建
- open_on_start 启动时开场面 (进目录, 不自动挂)
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
    return new_ground_channel(gs, **kw)


def _meta_by_name(runtime, name):
    for m in runtime.metas().values():
        if m.name == name:
            return m
    return None


async def _mount(runtime, name: str):
    """挂一个 gated 子通道, 返回它的 runtime —— 子通道的命令只能挂在树上才调得到."""
    await runtime.execute_command("mount_child", kwargs={"name": name})
    await runtime.refresh_metas()
    return runtime.fetch_sub_runtime(name)


# -- 命令面分层 -------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_surface_is_thin(ground):
    """主面只有 refresh + gate 自带的 mount/unmount; 其余都在 admin 里."""
    chan = _chan(ground)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        names = {c.name for c in runtime.self_meta().commands}
        # unmount_child 只在有挂载时才出现 (available 谓词) —— 挂着才需要卸.
        assert names == {"refresh", "mount_child"}
        assert {"render", "validate", "templates", "spec"}.isdisjoint(names)


@pytest.mark.asyncio
async def test_admin_holds_the_write_surface(ground):
    chan = _chan(ground)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        admin = await _mount(runtime, "admin")
        assert admin is not None
        names = {c.name for c in admin.self_meta().commands}
        assert {"open", "close", "stats", "ground_run", "edit",
                "pin_file", "pin_glob", "pin_frontmatter", "pin_ls", "pin_exec",
                "pin_law", "unpin"} <= names


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
async def test_instruction_can_be_cleared(ground):
    """次要的 ground channel 传空字符串 —— instruction 整段不出现."""
    gs = DefaultGroundSet(workspace_root=ground)
    chan = new_ground_channel(gs, instruction="")
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        assert not runtime.self_meta().instruction


# -- gate: 目录在, 内容不在 ---------------------------------------------------


def _with_child(ground):
    (ground / "child").mkdir()
    (ground / "child" / "GROUND.md").write_text("---\nname: child\n---\n# Child Ground\n")
    (ground / "GROUND.md").write_text(
        "---\nname: test\ninline: [child]\n---\n# Test Ground\nbody line\n"
    )


@pytest.mark.asyncio
async def test_children_are_catalogued_not_mounted(ground):
    _with_child(ground)
    chan = _chan(ground)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        catalog = runtime.self_meta().named_notices.get("gated_children", "")
        assert "child (closed)" in catalog
        assert _meta_by_name(runtime, "child") is None


@pytest.mark.asyncio
async def test_mounting_a_child_brings_its_frame(ground):
    _with_child(ground)
    chan = _chan(ground)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        child = await _mount(runtime, "child")
        assert child is not None
        meta = child.self_meta()
        assert "# Child Ground" in meta.named_notices.get("frame", "")
        # 冷数据 (身份) 与热数据 (帧) 都在 named notice — instruction 无 delta 载体.
        assert "cd " in meta.named_notices.get("meta", "")
        assert "cd " not in (meta.instruction or "")


@pytest.mark.asyncio
async def test_unmount_hides_the_frame_again(ground):
    _with_child(ground)
    chan = _chan(ground)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        await _mount(runtime, "child")
        await runtime.execute_command("unmount_child", kwargs={"name": "child"})
        await runtime.refresh_metas()
        assert _meta_by_name(runtime, "child") is None
        catalog = runtime.self_meta().named_notices.get("gated_children", "")
        assert "child (closed)" in catalog


# -- open / close -----------------------------------------------------------


@pytest.mark.asyncio
async def test_open_returns_frame_and_catalogues_the_field(ground):
    (ground / "sub").mkdir()
    (ground / "sub" / "GROUND.md").write_text("---\nname: sub\n---\n# Sub Ground\n")
    chan = _chan(ground)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        admin = await _mount(runtime, "admin")
        result = await admin.execute_command("open", kwargs={"directory": "sub"})
        assert "opened sub" in result
        assert "# Sub Ground" in result
        await runtime.refresh_metas()
        assert _meta_by_name(runtime, "sub") is None      # 进目录, 不自动挂
        assert "sub (closed)" in runtime.self_meta().named_notices["gated_children"]


@pytest.mark.asyncio
async def test_open_refuses_a_directory_outside_the_home(ground):
    chan = _chan(ground)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        admin = await _mount(runtime, "admin")
        with pytest.raises(Exception) as exc:
            await admin.execute_command("open", kwargs={"directory": "../outside"})
        assert "outside" in str(exc.value)


@pytest.mark.asyncio
async def test_open_refuses_the_admin_label(ground):
    (ground / "admin").mkdir()
    (ground / "admin" / "GROUND.md").write_text("---\nname: admin\n---\n# Admin\n")
    chan = _chan(ground)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        admin = await _mount(runtime, "admin")
        with pytest.raises(Exception) as exc:
            await admin.execute_command("open", kwargs={"directory": "admin"})
        assert "reserved" in str(exc.value)


@pytest.mark.asyncio
async def test_close_removes_the_field_from_the_catalog(ground):
    (ground / "sub").mkdir()
    (ground / "sub" / "GROUND.md").write_text("---\nname: sub\n---\n# Sub Ground\n")
    chan = _chan(ground)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        admin = await _mount(runtime, "admin")
        await admin.execute_command("open", kwargs={"directory": "sub"})
        result = await admin.execute_command("close", kwargs={"label": "sub"})
        assert "closed sub" in result
        await runtime.refresh_metas()
        catalog = runtime.self_meta().named_notices.get("gated_children", "")
        assert "sub" not in catalog


# -- 契约 classmethod --------------------------------------------------------


@pytest.mark.asyncio
async def test_peek_dir_shows_frame(ground):
    result = await DefaultGroundSet.peek_dir(ground)
    assert "# Test Ground" in result
    assert "def hot():" in result
    assert "hot spot" in result


@pytest.mark.asyncio
async def test_peek_meta_adds_identity(ground):
    result = await DefaultGroundSet.peek_meta(ground)
    assert "cd " in result and "pins:" in result


@pytest.mark.asyncio
async def test_peek_nonexistent_dir_reported(ground):
    result = await DefaultGroundSet.peek_dir(ground / "nope")
    assert "not a directory" in result or "no GROUND.md" in result


@pytest.mark.asyncio
async def test_validate_reports_invalid(ground):
    bad = ground / "bad.md"
    bad.write_text("---\npins:\n- verb: file\n  arguments:\n    path: x\n---\n")
    result = DefaultGroundSet.validate_text(bad.read_text())
    assert "[ERROR]" in result
    assert "missing 'label'" in result


@pytest.mark.asyncio
async def test_validate_warns_on_unknown_verb(ground):
    bad = ground / "bad.md"
    bad.write_text("---\npins:\n- label: x\n  verb: unknown_verb\n---\n")
    result = DefaultGroundSet.validate_text(bad.read_text())
    assert "[WARN]" in result
    assert "unknown verb" in result


@pytest.mark.asyncio
async def test_validate_rejects_long_label(ground):
    # label 超 63 字符须报 ERROR, 与 contract 的 Pin.label max_length=63 对齐.
    bad = ground / "bad.md"
    bad.write_text("---\npins:\n- label: " + "a" * 64 + "\n  verb: file\n  arguments:\n    path: x\n---\n")
    result = DefaultGroundSet.validate_text(bad.read_text())
    assert "[ERROR]" in result
    assert "63" in result


@pytest.mark.asyncio
async def test_templates_at_discovers_project_templates(ground):
    gem = ground / ".grounds" / "basic.md"
    gem.parent.mkdir()
    gem.write_text("---\ndescription: a template\n---\n# t\n")
    names = [t.name for t in DefaultGroundSet.templates_at(ground)]
    assert "basic" in names


# -- 启动 / 渲染 / 指令覆盖 ---------------------------------------------------


@pytest.mark.asyncio
async def test_open_on_start_catalogues_the_field(ground):
    child = ground / "child"
    child.mkdir()
    (child / "GROUND.md").write_text("---\nname: child\n---\n# Child Ground\n")
    chan = _chan(ground, open_on_start=["child"])
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        assert "child (closed)" in runtime.self_meta().named_notices["gated_children"]


@pytest.mark.asyncio
async def test_child_frame_cached_until_refresh(ground):
    """帧缓存契约: 只 refresh_metas 不重渲染 (吃缓存), 显式 refresh 才重读."""
    (ground / "sub").mkdir()
    (ground / "sub" / "note.md").write_text("v1\n")
    (ground / "sub" / "GROUND.md").write_text(
        "---\nname: sub\npins:\n"
        "- label: note\n  verb: file\n  arguments:\n    path: note.md\n"
        "---\n# Sub Ground\n"
    )
    chan = _chan(ground)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        admin = await _mount(runtime, "admin")
        await admin.execute_command("open", kwargs={"directory": "sub"})
        child = await _mount(runtime, "sub")
        assert "v1" in child.self_meta().named_notices.get("frame", "")

        # 磁盘变了, 但只 refresh_metas → 帧仍是缓存值
        (ground / "sub" / "note.md").write_text("v2\n")
        await runtime.refresh_metas()
        assert "v2" not in child.self_meta().named_notices.get("frame", "")

        # 显式 refresh → 帧更新, 缓存重读
        await runtime.execute_command("refresh", kwargs={"label": "sub"})
        await runtime.refresh_metas()
        assert "v2" in child.self_meta().named_notices.get("frame", "")


@pytest.mark.asyncio
async def test_root_not_rendered_by_default(ground):
    _with_child(ground)
    chan = _chan(ground)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        notice = runtime.self_meta().notice
        assert "# Test Ground" not in notice  # root 默认不渲染进 notice


@pytest.mark.asyncio
async def test_render_root_renders_root_in_notice(ground):
    _with_child(ground)
    gs = DefaultGroundSet(workspace_root=ground)
    chan = new_ground_channel(gs, edit=True, render_root=True)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        notice = runtime.self_meta().notice
        assert "# Test Ground" in notice


@pytest.mark.asyncio
async def test_supplied_root_frame_is_reused(ground):
    """送进来的 root 帧直接吃, 不在 startup 重渲染一次 (exec pin 不该跑两遍)."""
    gs = DefaultGroundSet(workspace_root=ground)
    chan = new_ground_channel(gs, root_frame="PRE-RENDERED")
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        admin = await _mount(runtime, "admin")
        stats = await admin.execute_command("stats")
        assert "rendered at" in stats


@pytest.mark.asyncio
async def test_root_refresh_calls_back(ground):
    """refresh(root) 把新帧交给 instruction 的持有者."""
    seen: list[str] = []
    gs = DefaultGroundSet(workspace_root=ground)
    chan = new_ground_channel(gs, on_root_refresh=seen.append)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        await runtime.execute_command("refresh")
        assert seen and "# Test Ground" in seen[-1]


@pytest.mark.asyncio
async def test_instruction_override(ground):
    gs = DefaultGroundSet(workspace_root=ground)
    chan = new_ground_channel(gs, edit=True, instruction="CUSTOM INSTRUCTION")
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        assert runtime.self_meta().instruction == "CUSTOM INSTRUCTION"


@pytest.mark.asyncio
async def test_materialize_error_surfaced_in_notice(ground):
    (ground / "GROUND.md").write_text("---\ninline: [missing]\n---\n# Test Ground\n")
    chan = _chan(ground)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        assert "missing" in runtime.self_meta().notice


# -- admin 的落盘面 / CLI runner ---------------------------------------------


@pytest.mark.asyncio
async def test_pin_file_reports_missing_ground_file(ground):
    chan = _chan(ground)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        admin = await _mount(runtime, "admin")
        result = await admin.execute_command(
            "pin_file", kwargs={"ground_file": str(ground / "missing.md"), "label": "x", "path": "y"}
        )
        assert "no such GROUND.md" in result
        assert not (ground / "missing.md").exists()


@pytest.mark.asyncio
async def test_pin_file_writes_with_always_show(ground):
    chan = _chan(ground)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        admin = await _mount(runtime, "admin")
        result = await admin.execute_command(
            "pin_file",
            kwargs={"ground_file": str(ground), "label": "greet", "path": "data/hot.py",
                    "always_show": True, "description": "greet"},
        )
        assert "pinned file:greet" in result
        text = (ground / "GROUND.md").read_text()
        assert "label: greet" in text
        assert "label: hot" in text               # 原有 pin 保留
        assert "always_show: true" in text        # always_show 落盘


@pytest.mark.asyncio
async def test_unpin_removes_a_pin(ground):
    chan = _chan(ground)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        admin = await _mount(runtime, "admin")
        result = await admin.execute_command(
            "unpin", kwargs={"ground_file": str(ground), "label": "hot"}
        )
        assert "unpinned hot" in result
        assert "label: hot" not in (ground / "GROUND.md").read_text()


@pytest.mark.asyncio
async def test_unpin_reports_unknown_label(ground):
    chan = _chan(ground)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        admin = await _mount(runtime, "admin")
        result = await admin.execute_command(
            "unpin", kwargs={"ground_file": str(ground), "label": "nope"}
        )
        assert "no pin 'nope'" in result
        assert "label: hot" in (ground / "GROUND.md").read_text()  # 不写盘


@pytest.mark.asyncio
async def test_edit_face_refuses_relative_paths(ground):
    """编辑面只认绝对路径 —— 相对谁没有唯一答案, 与其猜不如报错."""
    chan = _chan(ground)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        admin = await _mount(runtime, "admin")
        result = await admin.execute_command(
            "pin_file", kwargs={"ground_file": "GROUND.md", "label": "x", "path": "y"}
        )
        assert "absolute" in result


@pytest.mark.asyncio
async def test_edit_folds_commands_by_default(ground):
    gs = DefaultGroundSet(workspace_root=ground)
    chan = new_ground_channel(gs)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        admin = await _mount(runtime, "admin")
        names = {c.name for c in admin.self_meta().commands}
        assert {"open", "close", "stats", "ground_run", "edit"} <= names
        assert "pin_file" not in names
        assert "unpin" not in names


@pytest.mark.asyncio
async def test_edit_command_unfolds_gated_commands(ground):
    gs = DefaultGroundSet(workspace_root=ground)
    chan = new_ground_channel(gs)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        admin = await _mount(runtime, "admin")
        assert "pin_file" not in {c.name for c in admin.self_meta().commands}
        await admin.execute_command("edit", kwargs={"on": True})
        await runtime.refresh_metas()
        names = {c.name for c in admin.self_meta().commands}
        assert "pin_file" in names
        assert "unpin" in names
        # 折叠回来
        await admin.execute_command("edit", kwargs={"on": False})
        await runtime.refresh_metas()
        assert "pin_file" not in {c.name for c in admin.self_meta().commands}


@pytest.mark.asyncio
async def test_ground_run_proxies_the_cli(ground):
    """读侧归 CLI: runner 在 home 下跑 ``moss ground <verb>`` 并带回 stdout."""
    chan = _chan(ground)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        admin = await _mount(runtime, "admin")
        result = await admin.execute_command("ground_run", kwargs={"text__": "verbs"})
        assert "file" in result and "exec" in result
