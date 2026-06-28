"""Desktop Phase 1 acceptance 测试.

覆盖 .design §10 的全部 acceptance 边界:
- 12 原语行为
- ReadHistory protocol + 进程内缺省实现
- read-before-write 守卫
- 统一输出截断 + tmp_path 不重复截断
- 反思路径白名单触发 ReflectionHint
- Pin 注册 / 查询 / 移除 / LRU 淘汰
- ProcessManager 注入 vs 裸 subprocess cwd 一致
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ghoshell_moss.core.desktop import (
    DefaultDesktop,
    InProcessReadHistory,
    ReadBeforeWriteError,
    PathOutsideRootError,
    ReflectionHint,
    DEFAULT_REFLECTION_PATHS,
)
from ghoshell_moss.contracts.desktop import ReadHistory


# ================================================================
# fixtures
# ================================================================


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "a.md").write_text("# A\n\nhello\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "b.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    (tmp_path / "src" / "c.py").write_text("x = 2\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def desk(root: Path) -> DefaultDesktop:
    return DefaultDesktop(root=root)


# ================================================================
# 导航层: cd / pwd + 边界
# ================================================================


def test_pwd_starts_at_root(desk: DefaultDesktop, root: Path):
    assert desk.pwd() == str(root.resolve())


def test_cd_within_root(desk: DefaultDesktop, root: Path):
    target = desk.cd("src")
    assert Path(target) == (root / "src").resolve()
    assert desk.pwd() == str((root / "src").resolve())


def test_cd_rejects_outside_root(desk: DefaultDesktop, tmp_path: Path):
    with pytest.raises(PathOutsideRootError):
        desk.cd("..")


def test_cd_to_nonexistent_dir(desk: DefaultDesktop):
    with pytest.raises(FileNotFoundError):
        desk.cd("nope")


# ================================================================
# 发现层: tree / glob / grep
# ================================================================


def test_tree_returns_structure(desk: DefaultDesktop):
    t = desk.tree(depth=2)
    assert t.type == "dir"
    names = {c.name for c in (t.children or [])}
    assert "src" in names and "a.md" in names


def test_tree_ignores_hidden(desk: DefaultDesktop, root: Path):
    (root / ".secret").mkdir()
    (root / ".secret" / "x").write_text("x")
    t = desk.tree(depth=2)
    names = {c.name for c in (t.children or [])}
    assert ".secret" not in names


def test_glob_matches_files(desk: DefaultDesktop):
    matches = desk.glob("src/*.py")
    assert "src/b.py" in matches
    assert "src/c.py" in matches


def test_grep_finds_pattern(desk: DefaultDesktop):
    results = desk.grep("def foo")
    assert len(results) == 1
    assert results[0].file == "src/b.py"
    assert results[0].line == 1


def test_grep_invalid_regex(desk: DefaultDesktop):
    with pytest.raises(ValueError):
        desk.grep("[unclosed")


# ================================================================
# 读取层: read / frontmatter
# ================================================================


def test_read_returns_lines(desk: DefaultDesktop):
    fc = desk.read("a.md")
    assert fc.total_lines == 3
    assert fc.start_line == 1
    assert fc.lines[0] == (1, "# A")


def test_read_offset_limit(desk: DefaultDesktop):
    fc = desk.read("a.md", offset=1, limit=1)
    assert fc.start_line == 2
    assert fc.lines == [(2, "")]


def test_read_marks_history(desk: DefaultDesktop, root: Path):
    desk.read("a.md")
    # 后续 write 不再抛
    desk.write("a.md", "# A v2\n")


def test_read_nonexistent(desk: DefaultDesktop):
    with pytest.raises(FileNotFoundError):
        desk.read("nope.txt")


def test_frontmatter_basic(desk: DefaultDesktop, root: Path):
    (root / "fm.md").write_text("---\ntitle: hi\nstatus: ok\n---\n\nbody\n", encoding="utf-8")
    meta = desk.frontmatter("fm.md")
    assert meta == {"title": "hi", "status": "ok"}


def test_frontmatter_keys_filter(desk: DefaultDesktop, root: Path):
    (root / "fm.md").write_text("---\ntitle: hi\nstatus: ok\n---\n", encoding="utf-8")
    meta = desk.frontmatter("fm.md", "title")
    assert meta == {"title": "hi"}


def test_frontmatter_no_yaml(desk: DefaultDesktop):
    assert desk.frontmatter("a.md") is None


# ================================================================
# 写入层: write / edit + read-before-write
# ================================================================


def test_write_new_file_ok(desk: DefaultDesktop, root: Path):
    hint = desk.write("new.txt", "hi")
    assert hint is None
    assert (root / "new.txt").read_text() == "hi"


def test_write_existing_requires_read(desk: DefaultDesktop):
    with pytest.raises(ReadBeforeWriteError):
        desk.write("a.md", "rewritten")


def test_write_after_read_ok(desk: DefaultDesktop, root: Path):
    desk.read("a.md")
    desk.write("a.md", "rewritten")
    assert (root / "a.md").read_text() == "rewritten"


def test_edit_requires_read(desk: DefaultDesktop):
    with pytest.raises(ReadBeforeWriteError):
        desk.edit("a.md", "# A", "# A v2")


def test_edit_after_read_ok(desk: DefaultDesktop, root: Path):
    desk.read("a.md")
    line, hint = desk.edit("a.md", "# A", "# A v2")
    assert line == 1
    assert hint is None
    assert (root / "a.md").read_text().startswith("# A v2")


def test_edit_old_not_found(desk: DefaultDesktop):
    desk.read("a.md")
    with pytest.raises(ValueError, match="not found"):
        desk.edit("a.md", "ZZZ", "")


def test_edit_old_ambiguous(desk: DefaultDesktop, root: Path):
    (root / "dup.txt").write_text("xx\nxx\n")
    desk.read("dup.txt")
    with pytest.raises(ValueError, match="matches 2 times"):
        desk.edit("dup.txt", "xx", "y")


# ================================================================
# ReadHistory 注入
# ================================================================


class _RecordingHistory:
    def __init__(self) -> None:
        self.reads: list[Path] = []
        self._set: set[Path] = set()

    def has_read(self, path: Path) -> bool:
        return path in self._set

    def mark_read(self, path: Path) -> None:
        self.reads.append(path)
        self._set.add(path)


def test_read_history_injection(root: Path):
    history = _RecordingHistory()
    d = DefaultDesktop(root=root, read_history=history)
    assert isinstance(history, ReadHistory)  # protocol check
    d.read("a.md")
    assert (root / "a.md").resolve() in history._set


def test_in_process_read_history_default(root: Path):
    d = DefaultDesktop(root=root)
    # 实现私有 — 通过 write 失败前后的行为间接验证
    with pytest.raises(ReadBeforeWriteError):
        d.write("a.md", "x")
    d.read("a.md")
    d.write("a.md", "x")


def test_in_process_read_history_protocol():
    h = InProcessReadHistory()
    p = Path("/tmp/x")
    assert not h.has_read(p)
    h.mark_read(p)
    assert h.has_read(p)


# ================================================================
# 统一输出截断 + tmp_path 不重复截断
# ================================================================


def test_read_truncation_writes_tmp(root: Path):
    big = "\n".join(f"line{i}" for i in range(500))
    (root / "big.txt").write_text(big, encoding="utf-8")
    d = DefaultDesktop(root=root)
    fc = d.read("big.txt", limit=500)
    assert fc.truncated
    assert fc.tmp_path is not None
    assert Path(fc.tmp_path).exists()


def test_tmp_path_read_does_not_truncate(root: Path):
    big = "\n".join(f"line{i}" for i in range(500))
    (root / "big.txt").write_text(big, encoding="utf-8")
    d = DefaultDesktop(root=root)
    fc1 = d.read("big.txt", limit=500)
    fc2 = d.read(fc1.tmp_path, limit=10000)
    # tmp 路径读 — 永不截断, 即使新读出的内容仍超阈值
    assert not fc2.truncated
    assert fc2.tmp_path is None


def test_custom_tmp_root_outside(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.md").write_text("hi")
    tmp = tmp_path / "external_tmp"
    d = DefaultDesktop(root=root, tmp_root=tmp)
    assert d.tmp_root == tmp.resolve()
    big = "\n".join(f"L{i}" for i in range(500))
    (root / "big.txt").write_text(big)
    fc = d.read("big.txt", limit=500)
    assert fc.truncated
    assert Path(fc.tmp_path).parent == tmp.resolve()


# ================================================================
# 反思路径白名单 → ReflectionHint
# ================================================================


def test_reflection_hit_top_level_file(root: Path):
    (root / "CLAUDE.md").write_text("# rules\n")
    d = DefaultDesktop(root=root)
    d.read("CLAUDE.md")
    hint = d.write("CLAUDE.md", "# rules v2\n")
    assert hint is not None
    assert hint.path == "CLAUDE.md"
    assert hint.severity == "instruction"
    assert hint.recommend_commit


def test_reflection_hit_directory_prefix(root: Path):
    (root / ".moss").mkdir()
    (root / ".moss" / "manifests.yml").write_text("---\n")
    d = DefaultDesktop(root=root)
    d.read(".moss/manifests.yml")
    hint = d.write(".moss/manifests.yml", "---\nx: 1\n")
    assert hint is not None
    assert hint.severity == "config"


def test_reflection_miss_for_normal_file(desk: DefaultDesktop):
    desk.read("a.md")
    hint = desk.write("a.md", "ordinary content\n")
    assert hint is None


def test_reflection_edit_returns_hint(root: Path):
    (root / "DESKTOP.md").write_text("alpha\nbeta\n")
    d = DefaultDesktop(root=root)
    d.read("DESKTOP.md")
    line, hint = d.edit("DESKTOP.md", "alpha", "gamma")
    assert line == 1
    assert hint is not None
    assert hint.severity == "instruction"


def test_reflection_custom_paths(root: Path):
    (root / "custom.conf").write_text("v=1\n")
    d = DefaultDesktop(root=root, reflection_paths={"custom.conf": "config"})
    d.read("custom.conf")
    hint = d.write("custom.conf", "v=2\n")
    assert hint is not None and hint.severity == "config"


# ================================================================
# Pin: 注册 / 查询 / 移除 / LRU
# ================================================================


def test_pin_register_via_meta_param(desk: DefaultDesktop):
    desk.tree(depth=1, _pin=True)
    pins = desk.pinned()
    assert len(pins) == 1
    assert pins[0].command_name == "tree"


def test_pin_idempotent_same_args(desk: DefaultDesktop):
    desk.glob("*.md", _pin=True)
    desk.glob("*.md", _pin=True)
    assert len(desk.pinned()) == 1


def test_pin_distinct_args(desk: DefaultDesktop):
    desk.glob("*.md", _pin=True)
    desk.glob("*.py", _pin=True)
    assert len(desk.pinned()) == 2


def test_unpin_removes(desk: DefaultDesktop):
    desk.tree(depth=1, _pin=True)
    pin_id = desk.pinned()[0].id
    desk.unpin(pin_id)
    assert desk.pinned() == []


def test_unpin_unknown_raises(desk: DefaultDesktop):
    with pytest.raises(KeyError):
        desk.unpin("not-a-real-id")


def test_pin_lru_eviction(root: Path):
    d = DefaultDesktop(root=root, max_pins=2)
    d.glob("a", _pin=True)
    d.glob("b", _pin=True)
    d.glob("c", _pin=True)
    pins = d.pinned()
    assert len(pins) == 2
    args = [p.args_preview for p in pins]
    assert any("'b'" in a for a in args)
    assert any("'c'" in a for a in args)
    assert not any("'a'" in a for a in args)


def test_pin_budget_warning_when_full(root: Path):
    d = DefaultDesktop(root=root, max_pins=2)
    d.glob("a", _pin=True)
    d.glob("b", _pin=True)
    pins = d.pinned()
    assert all(p.pin_budget_warning for p in pins)


def test_pin_lru_refresh_on_repin(root: Path):
    d = DefaultDesktop(root=root, max_pins=2)
    d.glob("a", _pin=True)
    d.glob("b", _pin=True)
    d.glob("a", _pin=True)  # 重 pin a 应将 a 移到末尾, 下一个新 pin 应淘汰 b
    d.glob("c", _pin=True)
    pins = d.pinned()
    args = [p.args_preview for p in pins]
    assert any("'a'" in a for a in args)
    assert any("'c'" in a for a in args)
    assert not any("'b'" in a for a in args)


@pytest.mark.asyncio
async def test_pin_refresh_reexecutes(desk: DefaultDesktop, root: Path):
    desk.glob("*.md", _pin=True)
    (root / "fresh.md").write_text("hi")
    await desk.refresh()
    pin = desk.pinned()[0]
    assert "fresh.md" in pin.last_preview


# ================================================================
# instruction + DESKTOP.md 覆盖
# ================================================================


def test_default_instruction(desk: DefaultDesktop, root: Path):
    txt = desk.instruction()
    assert str(root.resolve()) in txt
    assert "read-before-write" in txt or "read it first" in txt


def test_desktop_md_overrides(root: Path):
    (root / "DESKTOP.md").write_text("CUSTOM INSTRUCTION", encoding="utf-8")
    d = DefaultDesktop(root=root)
    assert d.instruction() == "CUSTOM INSTRUCTION"


# ================================================================
# exec — 裸 subprocess 路径
# ================================================================


@pytest.mark.asyncio
async def test_exec_returns_stdout(desk: DefaultDesktop):
    result = await desk.exec("echo hello")
    assert result.exit_code == 0
    assert result.stdout.strip() == "hello"
    assert not result.killed


@pytest.mark.asyncio
async def test_exec_cwd_follows_pwd(desk: DefaultDesktop, root: Path):
    desk.cd("src")
    result = await desk.exec("pwd")
    assert result.stdout.strip() == str((root / "src").resolve())


@pytest.mark.asyncio
async def test_exec_timeout_kills(desk: DefaultDesktop):
    result = await desk.exec("sleep 5", timeout=0.3)
    assert result.killed


@pytest.mark.asyncio
async def test_exec_bg_returns_task_id(desk: DefaultDesktop):
    result = await desk.exec("echo bg", _bg=True)
    assert result.task_id is not None
    # 等待 runner 启动并完成
    import asyncio as _a
    await _a.sleep(0.5)
    tasks = desk.tasks()
    assert any(t.id == result.task_id for t in tasks)
    await desk.shutdown()


@pytest.mark.asyncio
async def test_task_read_and_cancel(desk: DefaultDesktop):
    result = await desk.exec("echo content", _bg=True)
    import asyncio as _a
    await _a.sleep(0.5)
    tasks = desk.tasks()
    target = next(t for t in tasks if t.id == result.task_id)
    out = await target.read()
    assert "content" in out
    await target.cancel()


# ================================================================
# ProcessManager 注入路径 — cwd 一致性
# ================================================================


@pytest.mark.asyncio
async def test_exec_via_process_manager_cwd(root: Path):
    """ProcessManager 注入 vs 裸 subprocess 行为等价 (cwd 一致)."""
    from ghoshell_moss.core.process_manager._impl import ProcessManagerImpl

    async with ProcessManagerImpl(root=root, output_path=root / "tmp" / "pm") as pm:
        d = DefaultDesktop(root=root, process_manager=pm)
        d.cd("src")
        result_pm = await d.exec("pwd")

    d2 = DefaultDesktop(root=root)
    d2.cd("src")
    result_raw = await d2.exec("pwd")

    assert result_pm.stdout.strip() == result_raw.stdout.strip()
    assert result_pm.exit_code == result_raw.exit_code == 0


# ================================================================
# 路径越界 (绝对路径访问)
# ================================================================


def test_read_outside_root_rejected(desk: DefaultDesktop, tmp_path: Path):
    outside = tmp_path.parent / "outside_root.txt"
    outside.write_text("x")
    try:
        with pytest.raises(PathOutsideRootError):
            desk.read(str(outside))
    finally:
        outside.unlink(missing_ok=True)


def test_tmp_path_absolute_accessible(root: Path):
    d = DefaultDesktop(root=root)
    big = "\n".join(f"L{i}" for i in range(500))
    (root / "big.txt").write_text(big)
    fc = d.read("big.txt", limit=500)
    # tmp_path 是绝对路径, 出 root, 但应可读
    fc2 = d.read(fc.tmp_path, limit=10)
    assert fc2.lines[0] == (1, "L0")
