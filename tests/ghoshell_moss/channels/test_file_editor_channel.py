"""File editor channel tests — 走 test_ctml_v1.py 范式.

覆盖策略: 每个测试用真实 CTML 字符串触发, 走 parse -> dispatch -> execute
全链路. 文件 IO 用 tmp_path fixture, editor 是 test-scoped 实例 (undo 隔离).

## 落地时验证到的 CTML 姿态 (给下一实例)

这几条只写在 spec 里而无 example, 单测把它们钉死了:

1. **CDATA 包 JSON in text__ 跑得通** — 见 test 4. str_replace 的双 blob
   (old_str / new_str) 放不进单一 attribute, 走 `<![CDATA[{...JSON...}]]>`
   免了 XML 转义, JSON 内部用 `\\n` 逃逸多行, 与 Claude Code Edit tool
   的 JSON schema 先验完全对齐. 唯一隐患是 payload 里出现 `]]>` — 实际
   代码里罕见, 未撞到再想.

2. **mixed attribute + text__ 跑得通** — 见 test 5. `insert` 用 int
   attribute (`insert_line="1"`) 混 text__ CDATA body, CTML 一起 dispatch,
   handler 各自拿到. 是"attribute 传短参、text__ 传长 blob"的验证.

3. **`view_range="1,3"` 经 ast.literal_eval → tuple(1,3)** — 见 test 2.
   CTML 默认对 attribute 值走 literal_eval, 逗号分隔会变 tuple 而非 str.
   Channel 层 `_parse_view_range` 三种形状 (str/tuple/list) 都接, 模型
   不用记 `:str` 后缀.

## Task 断言的注意

- **task.result() 抛异常时是 None** — 想验 exception 就用 `.exception()`,
  别用 `.result() is None` 反推. 踩过一次, 见 test 7/8 的 view_tasks 过滤.
- **tasks[0] 不一定是"业务首个"** — `<_>` scope 本身产生 task 且排在业务
  前面. 用 `caller_name().startswith("channel:cmd")` 过滤才稳.
- **ObserveError 中断同轨 sibling** — 见 test 7. `CommandUtil.raise_observe`
  抛出后同 scope 后续 task 走 cancel 或有 exception, sibling 未跑到.
"""

import json

import pytest
from ghoshell_container import IoCContainer, Container

from ghoshell_moss.channels.file_editor_channel import (
    build_file_editor_channel,
    new_file_editor_channel,
)
from ghoshell_moss.contracts.file_editor import FileEditor
from ghoshell_moss.core.ctml import ctml_shell_test
from ghoshell_moss.core.file_editor import DefaultFileEditor


# -- fixtures ----------------------------------------------------------------


@pytest.fixture
def editor():
    return DefaultFileEditor()


@pytest.fixture
def chan(editor):
    return new_file_editor_channel(editor)


# -- 1. view (whole file, attribute-only baseline) ---------------------------


@pytest.mark.asyncio
async def test_view_whole_file(tmp_path, chan):
    f = tmp_path / "hello.py"
    f.write_text("line 1\nline 2\nline 3\n")

    tasks = await ctml_shell_test(
        chan,
        ctml=f'<file_editor:view path="{f}"/>',
    )
    assert len(tasks) == 1
    result = tasks[0].result()
    assert "line 1" in result
    assert "line 3" in result


# -- 2. view with range (CTML ast.literal_eval → tuple form) -----------------


@pytest.mark.asyncio
async def test_view_with_range(tmp_path, chan):
    f = tmp_path / "many.py"
    f.write_text("\n".join(f"line {i}" for i in range(1, 11)) + "\n")

    tasks = await ctml_shell_test(
        chan,
        ctml=f'<file_editor:view path="{f}" view_range="1,3"/>',
    )
    result = tasks[0].result()
    assert "line 1" in result
    assert "line 3" in result
    assert "line 4" not in result


# -- 3. create + CDATA text__ (single blob) ----------------------------------


@pytest.mark.asyncio
async def test_create_with_cdata_text(tmp_path, chan):
    f = tmp_path / "new.py"
    ctml = f"""<file_editor:create path="{f}"><![CDATA[
def hello():
    print("hi <world>")
]]></file_editor:create>"""

    tasks = await ctml_shell_test(chan, ctml=ctml)
    assert len(tasks) == 1
    result = tasks[0].result()
    assert "created" in result.lower() or "File" in result
    content = f.read_text()
    assert 'print("hi <world>")' in content


# -- 4. str_replace + CDATA + JSON (double blob, the trickiest one) ----------


@pytest.mark.asyncio
async def test_str_replace_with_cdata_json(tmp_path, chan, editor):
    f = tmp_path / "target.py"
    f.write_text('def hello():\n    print("hi")\n')

    payload = json.dumps({
        "old_str": '    print("hi")',
        "new_str": '    print("hello world")',
    })
    ctml = f"""<file_editor:str_replace path="{f}"><![CDATA[
{payload}
]]></file_editor:str_replace>"""

    tasks = await ctml_shell_test(chan, ctml=ctml)
    assert len(tasks) == 1
    assert tasks[0].exception() is None, tasks[0].exception()
    content = f.read_text()
    assert 'print("hello world")' in content
    assert 'print("hi")' not in content


# -- 5. insert (mixed attribute int + text__ CDATA body) ---------------------


@pytest.mark.asyncio
async def test_insert_mixed_int_and_text(tmp_path, chan):
    f = tmp_path / "target.py"
    f.write_text("line A\nline B\n")

    ctml = f"""<file_editor:insert path="{f}" insert_line="1"><![CDATA[inserted line]]></file_editor:insert>"""

    tasks = await ctml_shell_test(chan, ctml=ctml)
    assert tasks[0].exception() is None, tasks[0].exception()
    content = f.read_text()
    # inserted AFTER line 1 (0-indexed anchor)
    assert content.splitlines() == ["line A", "inserted line", "line B"]


# -- 6. undo_edit round-trip -------------------------------------------------


@pytest.mark.asyncio
async def test_undo_edit_round_trip(tmp_path, chan):
    f = tmp_path / "target.py"
    f.write_text("original\n")

    payload = json.dumps({"old_str": "original", "new_str": "modified"})
    ctml = f"""
    <_>
      <file_editor:str_replace path="{f}"><![CDATA[
{payload}
]]></file_editor:str_replace>
      <file_editor:undo_edit path="{f}"/>
    </_>
    """
    tasks = await ctml_shell_test(chan, ctml=ctml)
    for t in tasks:
        assert t.exception() is None, (t.caller_name(), t.exception())
    assert f.read_text() == "original\n"


# -- 7. FileEditorError raises ObserveError → aborts same-channel subsequent -


@pytest.mark.asyncio
async def test_error_aborts_sequence(tmp_path, editor):
    chan = new_file_editor_channel(editor)
    f = tmp_path / "will_exist.py"
    f.write_text("stub\n")

    # First: view a nonexistent file → ParameterInvalidError → raise_observe
    # Second: view the real file → should be aborted by the raise
    ctml = f"""
    <_>
      <file_editor:view path="{tmp_path}/does_not_exist.py"/>
      <file_editor:view path="{f}"/>
    </_>
    """
    tasks = await ctml_shell_test(chan, ctml=ctml)

    view_tasks = [t for t in tasks if t.caller_name().startswith("file_editor:view")]
    assert len(view_tasks) == 2
    # First view raised ObserveError; second view was aborted (never ran).
    assert view_tasks[0].exception() is not None
    assert view_tasks[1].cancelled() or view_tasks[1].exception() is not None


# -- 8. build_file_editor_channel with empty container → fallback ------------


@pytest.mark.asyncio
async def test_build_factory_fallback_to_default(tmp_path):
    container = Container()
    chan = build_file_editor_channel()(container)
    f = tmp_path / "hello.py"
    f.write_text("factory fallback works\n")

    tasks = await ctml_shell_test(
        chan,
        ctml=f'<file_editor:view path="{f}"/>',
    )
    assert tasks[0].exception() is None
    assert "factory fallback works" in tasks[0].result()


# -- 9. file_list (v2 read-side) ---------------------------------------------


@pytest.mark.asyncio
async def test_file_list_directory(tmp_path, chan):
    d = tmp_path / "tree"
    (d / "a.py").mkdir(parents=True)
    (d / "a.py" / "one.py").write_text("x\n")
    (d / "b.txt").write_text("hi\n")

    tasks = await ctml_shell_test(
        chan,
        ctml=f'<file_editor:file_list path="{d}"/>',
    )
    assert tasks[0].exception() is None, tasks[0].exception()
    result = tasks[0].result()
    assert "a.py" in result
    assert "b.txt" in result


# -- 10. glob (v2 read-side) -------------------------------------------------


@pytest.mark.asyncio
async def test_glob_recursive(tmp_path, chan):
    d = tmp_path / "tree"
    (d / "a.py").mkdir(parents=True)
    (d / "a.py" / "one.py").write_text("x\n")
    (d / "sub").mkdir()
    (d / "sub" / "two.py").write_text("y\n")

    tasks = await ctml_shell_test(
        chan,
        ctml=f'<file_editor:glob pattern="{d}/**/*.py"/>',
    )
    assert tasks[0].exception() is None, tasks[0].exception()
    result = tasks[0].result()
    assert "one.py" in result
    assert "two.py" in result


# -- 11. grep (v2 read-side) -------------------------------------------------


@pytest.mark.asyncio
async def test_grep_matching_lines(tmp_path, chan):
    f = tmp_path / "greptext.txt"
    f.write_text("alpha\nbeta one\nbeta two\n")

    tasks = await ctml_shell_test(
        chan,
        ctml=f'<file_editor:grep path="{f}" pattern="beta"/>',
    )
    assert tasks[0].exception() is None, tasks[0].exception()
    result = tasks[0].result()
    assert "2: beta one" in result
    assert "3: beta two" in result
