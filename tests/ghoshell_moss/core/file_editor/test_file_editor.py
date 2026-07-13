"""Contract acceptance tests — DefaultFileEditor 走 tmp_path 真 IO.

覆盖 (契约验收, 不测 vendor 实现细节):

- 5 个动词 (view / create / str_replace / insert / undo_edit) 的 happy path
- 每个动词的边界与异常
- 异常翻译 — 契约层永远收 FileEditorError 家族, 不透传 vendor ToolError
- 结果结构 — FileEditorResult 字段填充正确
- Undo 栈行为 (LIFO, 多次编辑多次 undo, 空栈抛错)

不覆盖:

- Vendor 实现细节 (FileHistoryManager 磁盘/内存策略, encoding 检测算法)
- Channel 装配 (未来 desktop channel / 独立 file editor channel 落地时才做)
- 大文件 / 编码边界 (vendor 已测过, 这里做一个 smoke)
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ghoshell_moss.contracts.file_editor import (
    FileEditor,
    FileEditorError,
    FileEditorResult,
    FileValidationError,
    NoEditHistoryError,
    ParameterInvalidError,
    ParameterMissingError,
)
from ghoshell_moss.core.file_editor import DefaultFileEditor


# -- fixtures ------------------------------------------------------------

@pytest.fixture
def editor() -> DefaultFileEditor:
    """Fresh editor per test — undo history is per-instance memory."""
    return DefaultFileEditor()


@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    """A 5-line text file for view / edit tests."""
    p = tmp_path / "sample.txt"
    p.write_text("alpha\nbeta\ngamma\ndelta\nepsilon\n")
    return p


# -- contract shape ------------------------------------------------------

class TestContractShape:
    """DefaultFileEditor implements the FileEditor ABC and produces the
    documented result / exception types."""

    def test_is_file_editor(self, editor: DefaultFileEditor):
        assert isinstance(editor, FileEditor)

    def test_all_exceptions_share_base(self):
        for cls in (
            ParameterMissingError,
            ParameterInvalidError,
            FileValidationError,
            NoEditHistoryError,
        ):
            assert issubclass(cls, FileEditorError)


# -- view -----------------------------------------------------------------

class TestView:
    def test_view_full_file(self, editor: DefaultFileEditor, sample_file: Path):
        r = editor.view(sample_file)
        assert isinstance(r, FileEditorResult)
        assert r.prev_exist is True
        assert r.old_content is None
        assert r.new_content is None
        assert "alpha" in r.output
        assert "epsilon" in r.output
        # cat -n style (line-numbered)
        assert "\t" in r.output

    def test_view_range(self, editor: DefaultFileEditor, sample_file: Path):
        r = editor.view(sample_file, view_range=[2, 3])
        assert "beta" in r.output
        assert "gamma" in r.output
        assert "alpha" not in r.output
        assert "delta" not in r.output

    def test_view_range_to_end(
        self, editor: DefaultFileEditor, sample_file: Path
    ):
        # Vendor allows -1 as sentinel for "to end of file"
        r = editor.view(sample_file, view_range=[4, -1])
        assert "delta" in r.output
        assert "epsilon" in r.output
        assert "alpha" not in r.output

    def test_view_range_out_of_bounds(
        self, editor: DefaultFileEditor, sample_file: Path
    ):
        with pytest.raises(ParameterInvalidError) as ei:
            editor.view(sample_file, view_range=[100, 200])
        assert ei.value.parameter == "view_range"

    def test_view_range_malformed(
        self, editor: DefaultFileEditor, sample_file: Path
    ):
        with pytest.raises(ParameterInvalidError):
            editor.view(sample_file, view_range=[1])  # not a pair

    def test_view_directory_rejected(
        self, editor: DefaultFileEditor, tmp_path: Path
    ):
        # Directory listing is NOT this tool's job — bash/glob covers it.
        with pytest.raises(ParameterInvalidError) as ei:
            editor.view(tmp_path)
        assert ei.value.parameter == "path"

    def test_view_missing_path(
        self, editor: DefaultFileEditor, tmp_path: Path
    ):
        with pytest.raises(ParameterInvalidError):
            editor.view(tmp_path / "does_not_exist.txt")

    def test_view_relative_path_rejected(
        self, editor: DefaultFileEditor
    ):
        # Vendor requires absolute paths.
        with pytest.raises(ParameterInvalidError):
            editor.view("relative/path.txt")

    def test_view_binary_rejected(
        self, editor: DefaultFileEditor, tmp_path: Path
    ):
        bp = tmp_path / "bin"
        bp.write_bytes(b"\x00\x01\x02hello")
        with pytest.raises(FileValidationError) as ei:
            editor.view(bp)
        assert "binary" in ei.value.reason.lower()

    def test_view_str_path_accepted(
        self, editor: DefaultFileEditor, sample_file: Path
    ):
        # Contract advertises str | Path.
        r = editor.view(str(sample_file))
        assert "alpha" in r.output


# -- create ---------------------------------------------------------------

class TestCreate:
    def test_create_new_file(
        self, editor: DefaultFileEditor, tmp_path: Path
    ):
        p = tmp_path / "new.txt"
        r = editor.create(p, "hello world\n")
        assert isinstance(r, FileEditorResult)
        assert r.prev_exist is False
        assert r.new_content == "hello world\n"
        assert p.read_text() == "hello world\n"

    def test_create_empty_file(
        self, editor: DefaultFileEditor, tmp_path: Path
    ):
        p = tmp_path / "empty.txt"
        editor.create(p, "")
        assert p.exists()
        assert p.read_text() == ""

    def test_create_existing_rejected(
        self, editor: DefaultFileEditor, sample_file: Path
    ):
        with pytest.raises(ParameterInvalidError) as ei:
            editor.create(sample_file, "overwrite")
        assert ei.value.parameter == "path"

    def test_create_missing_file_text(
        self, editor: DefaultFileEditor, tmp_path: Path
    ):
        # ABC signature makes file_text mandatory, but the underlying
        # engine also validates — this is defense in depth via the adapter.
        with pytest.raises(TypeError):
            editor.create(tmp_path / "x.txt")  # type: ignore[call-arg]

    def test_create_missing_parent_dir(
        self, editor: DefaultFileEditor, tmp_path: Path
    ):
        # Documented behavior: caller must mkdir the parent.
        with pytest.raises(FileEditorError):
            editor.create(tmp_path / "nope" / "x.txt", "content")


# -- str_replace ----------------------------------------------------------

class TestStrReplace:
    def test_str_replace_unique(
        self, editor: DefaultFileEditor, sample_file: Path
    ):
        r = editor.str_replace(sample_file, "gamma", "GAMMA")
        assert r.old_content is not None and "gamma" in r.old_content
        assert r.new_content is not None and "GAMMA" in r.new_content
        assert sample_file.read_text() == "alpha\nbeta\nGAMMA\ndelta\nepsilon\n"

    def test_str_replace_multiline(
        self, editor: DefaultFileEditor, sample_file: Path
    ):
        r = editor.str_replace(sample_file, "beta\ngamma", "MIDDLE")
        assert "MIDDLE" in sample_file.read_text()

    def test_str_replace_multi_occurrence_rejected(
        self, editor: DefaultFileEditor, tmp_path: Path
    ):
        p = tmp_path / "dup.txt"
        p.write_text("cat\ndog\ncat\n")
        # Two occurrences of "cat" — vendor requires uniqueness.
        with pytest.raises(FileEditorError):
            editor.str_replace(p, "cat", "CAT")

    def test_str_replace_no_match(
        self, editor: DefaultFileEditor, sample_file: Path
    ):
        with pytest.raises(FileEditorError):
            editor.str_replace(sample_file, "not-in-file", "replacement")

    def test_str_replace_same_str_rejected(
        self, editor: DefaultFileEditor, sample_file: Path
    ):
        with pytest.raises(ParameterInvalidError) as ei:
            editor.str_replace(sample_file, "alpha", "alpha")
        assert ei.value.parameter == "new_str"

    def test_str_replace_delete_via_empty_new(
        self, editor: DefaultFileEditor, sample_file: Path
    ):
        # Empty new_str == deletion.
        editor.str_replace(sample_file, "gamma\n", "")
        assert "gamma" not in sample_file.read_text()

    def test_str_replace_output_has_snippet(
        self, editor: DefaultFileEditor, sample_file: Path
    ):
        r = editor.str_replace(sample_file, "gamma", "GAMMA")
        assert "GAMMA" in r.output  # snippet shown
        assert "cat -n" in r.output or "\t" in r.output  # line-numbered

    def test_str_replace_directory_rejected(
        self, editor: DefaultFileEditor, tmp_path: Path
    ):
        with pytest.raises(ParameterInvalidError):
            editor.str_replace(tmp_path, "x", "y")


# -- insert ---------------------------------------------------------------

class TestInsert:
    def test_insert_at_line(
        self, editor: DefaultFileEditor, sample_file: Path
    ):
        # Insert after line 2 (i.e. between beta and gamma)
        editor.insert(sample_file, 2, "INSERTED")
        lines = sample_file.read_text().splitlines()
        assert lines[2] == "INSERTED"
        assert lines[1] == "beta"
        assert lines[3] == "gamma"

    def test_insert_at_beginning(
        self, editor: DefaultFileEditor, sample_file: Path
    ):
        # insert_line=0 → before any existing content
        editor.insert(sample_file, 0, "FIRST")
        lines = sample_file.read_text().splitlines()
        assert lines[0] == "FIRST"
        assert lines[1] == "alpha"

    def test_insert_at_end(
        self, editor: DefaultFileEditor, sample_file: Path
    ):
        editor.insert(sample_file, 5, "LAST")
        assert sample_file.read_text().endswith("LAST\n")

    def test_insert_multiline(
        self, editor: DefaultFileEditor, sample_file: Path
    ):
        editor.insert(sample_file, 1, "one\ntwo\nthree")
        lines = sample_file.read_text().splitlines()
        assert lines[1:4] == ["one", "two", "three"]

    def test_insert_out_of_bounds(
        self, editor: DefaultFileEditor, sample_file: Path
    ):
        with pytest.raises(ParameterInvalidError) as ei:
            editor.insert(sample_file, 999, "x")
        assert ei.value.parameter == "insert_line"

    def test_insert_negative_rejected(
        self, editor: DefaultFileEditor, sample_file: Path
    ):
        with pytest.raises(ParameterInvalidError):
            editor.insert(sample_file, -1, "x")


# -- undo_edit ------------------------------------------------------------

class TestUndoEdit:
    def test_undo_after_str_replace(
        self, editor: DefaultFileEditor, sample_file: Path
    ):
        original = sample_file.read_text()
        editor.str_replace(sample_file, "gamma", "GAMMA")
        assert "GAMMA" in sample_file.read_text()
        editor.undo_edit(sample_file)
        assert sample_file.read_text() == original

    def test_undo_after_insert(
        self, editor: DefaultFileEditor, sample_file: Path
    ):
        original = sample_file.read_text()
        editor.insert(sample_file, 1, "NEW")
        editor.undo_edit(sample_file)
        assert sample_file.read_text() == original

    def test_undo_after_create(
        self, editor: DefaultFileEditor, tmp_path: Path
    ):
        # create IS logged to history — undoing it rewrites to prior content,
        # but there's no "prior content" so undo restores to file_text of the
        # first history entry (which is the created content itself). This is
        # inherited vendor behavior; we just document it doesn't crash.
        p = tmp_path / "n.txt"
        editor.create(p, "content")
        r = editor.undo_edit(p)
        assert isinstance(r, FileEditorResult)

    def test_undo_lifo(
        self, editor: DefaultFileEditor, sample_file: Path
    ):
        editor.str_replace(sample_file, "alpha", "A")
        editor.str_replace(sample_file, "beta", "B")
        # Undo restores in reverse order.
        editor.undo_edit(sample_file)
        text = sample_file.read_text()
        assert "A" in text and "beta" in text  # first edit still present
        editor.undo_edit(sample_file)
        assert sample_file.read_text() == "alpha\nbeta\ngamma\ndelta\nepsilon\n"

    def test_undo_empty_history_raises(
        self, editor: DefaultFileEditor, sample_file: Path
    ):
        with pytest.raises(NoEditHistoryError) as ei:
            editor.undo_edit(sample_file)
        assert str(sample_file) in ei.value.path

    def test_undo_history_per_file(
        self, editor: DefaultFileEditor, tmp_path: Path
    ):
        p1 = tmp_path / "a.txt"
        p2 = tmp_path / "b.txt"
        p1.write_text("A0\n")
        p2.write_text("B0\n")
        editor.str_replace(p1, "A0", "A1")
        # Undo on p2 should fail — history is per-file.
        with pytest.raises(NoEditHistoryError):
            editor.undo_edit(p2)

    def test_undo_history_isolated_between_editors(
        self, tmp_path: Path
    ):
        # History is per-instance (per-process); a fresh editor has no history.
        ed1 = DefaultFileEditor()
        ed2 = DefaultFileEditor()
        p = tmp_path / "shared.txt"
        p.write_text("v0\n")
        ed1.str_replace(p, "v0", "v1")
        with pytest.raises(NoEditHistoryError):
            ed2.undo_edit(p)


# -- exception translation -----------------------------------------------

class TestExceptionTranslation:
    """Every failure path emits a FileEditorError subclass; no vendor
    exception should escape the adapter."""

    def test_missing_path_gives_parameter_invalid(
        self, editor: DefaultFileEditor, tmp_path: Path
    ):
        with pytest.raises(ParameterInvalidError):
            editor.view(tmp_path / "ghost.txt")

    def test_create_over_existing_gives_parameter_invalid(
        self, editor: DefaultFileEditor, sample_file: Path
    ):
        with pytest.raises(ParameterInvalidError):
            editor.create(sample_file, "x")

    def test_all_failures_are_file_editor_errors(
        self, editor: DefaultFileEditor, tmp_path: Path, sample_file: Path
    ):
        # Sample of every documented failure — none should leak vendor types.
        cases = [
            lambda: editor.view(tmp_path / "no.txt"),
            lambda: editor.view(tmp_path),
            lambda: editor.create(sample_file, "x"),
            lambda: editor.str_replace(sample_file, "not-there", "x"),
            lambda: editor.str_replace(sample_file, "alpha", "alpha"),
            lambda: editor.insert(sample_file, 999, "x"),
            lambda: editor.undo_edit(tmp_path / "no.txt"),
        ]
        for c in cases:
            with pytest.raises(FileEditorError):
                c()


# -- workspace_root informational hint -----------------------------------

class TestWorkspaceRoot:
    def test_workspace_root_is_optional(self, tmp_path: Path):
        # None disables the hint; behavior otherwise identical.
        ed = DefaultFileEditor(workspace_root=None)
        p = tmp_path / "x.txt"
        ed.create(p, "hi")
        assert p.read_text() == "hi"

    def test_workspace_root_accepts_path(self, tmp_path: Path):
        ed = DefaultFileEditor(workspace_root=tmp_path)
        p = tmp_path / "x.txt"
        ed.create(p, "hi")
        assert p.read_text() == "hi"

    def test_workspace_root_does_not_enforce_boundary(self, tmp_path: Path):
        # K1: file editor does NOT enforce spatial boundary. Grounds does.
        # A path outside workspace_root is still accepted as long as it's
        # absolute and valid.
        sub = tmp_path / "sub"
        sub.mkdir()
        outside = tmp_path / "outside.txt"
        ed = DefaultFileEditor(workspace_root=sub)
        ed.create(outside, "hi")
        assert outside.read_text() == "hi"


# -- max_file_size_mb -----------------------------------------------------

class TestMaxFileSize:
    def test_file_size_limit_enforced(self, tmp_path: Path):
        # 1 MB limit, write a 2 MB file → validation error on view.
        ed = DefaultFileEditor(max_file_size_mb=1)
        p = tmp_path / "big.txt"
        p.write_text("x" * (2 * 1024 * 1024))
        with pytest.raises(FileValidationError) as ei:
            ed.view(p)
        assert "too large" in ei.value.reason.lower()
