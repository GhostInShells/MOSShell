"""sqlite channel tests — 走 shell + interpreter.run 全链路.

覆盖: 生命周期 (open/query/tables/schema/sample/list/close)、context 常驻连接映射、
大结果集封顶 + 落盘、read_only、错误路径.
"""

import pytest

from ghoshell_moss.channels.sqlite_channel import new_sqlite_channel
from ghoshell_moss.core.ctml import new_ctml_shell
from ghoshell_moss.message import Message


async def _run(shell, logos: str) -> dict[str, str | None]:
    interpreter = await shell.interpreter()
    tasks = await interpreter.run(logos)
    return {cid: t.result(throw=False) for cid, t in tasks.items()}


def _joined(results: dict[str, str | None]) -> str:
    return "\n".join(str(v) for v in results.values() if v is not None)


@pytest.fixture
def shell():
    shell = new_ctml_shell()
    shell.main_channel.import_channels(new_sqlite_channel(name="sqlite"))
    return shell


# -- 生命周期 ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_flow(tmp_path, shell):
    db = tmp_path / "ghost.db"
    async with shell:
        r = await _run(shell, f'<sqlite:open db_path="{db}" name="mem"/>')
        assert "opened" in _joined(r)

        r = await _run(shell, "<sqlite:query name=\"mem\">CREATE TABLE users(id INTEGER PRIMARY KEY, name TEXT)</sqlite:query>")
        assert "OK" in _joined(r)

        r = await _run(shell, '<sqlite:tables name="mem"/>')
        assert "users" in _joined(r)

        r = await _run(shell, '<sqlite:schema name="mem" table="users"/>')
        assert "id: INTEGER PK" in _joined(r)

        r = await _run(shell, "<sqlite:query name=\"mem\">INSERT INTO users(name) VALUES('alice')</sqlite:query>")
        assert "OK" in _joined(r)

        r = await _run(shell, '<sqlite:query name="mem">SELECT * FROM users</sqlite:query>')
        assert "alice" in _joined(r)

        r = await _run(shell, '<sqlite:sample name="mem" table="users" limit="5"/>')
        assert "alice" in _joined(r)

        r = await _run(shell, '<sqlite:list/>')
        assert "mem" in _joined(r)

        r = await _run(shell, '<sqlite:close name="mem"/>')
        assert "closed" in _joined(r)

        assert db.exists()


# -- context 常驻连接映射 ----------------------------------------------------


@pytest.mark.asyncio
async def test_context_connection_map(tmp_path, shell):
    db = tmp_path / "ghost.db"
    async with shell:
        await _run(shell, f'<sqlite:open db_path="{db}" name="mem"/>')
        await shell.refresh_metas()
        meta = shell.channel_metas().get("sqlite")
        assert meta is not None
        context_text = "\n".join(
            Message.content_as_string(c) for m in meta.context for c in m.contents
        )
        assert "open connections" in context_text
        assert "mem" in context_text
        assert str(db) in context_text

        # 关闭后映射消失
        await _run(shell, '<sqlite:close name="mem"/>')
        await shell.refresh_metas()
        meta = shell.channel_metas().get("sqlite")
        assert meta is not None and meta.context == []


# -- 大结果集封顶 + 落盘 -----------------------------------------------------


@pytest.mark.asyncio
async def test_overflow_capped_and_dumped(tmp_path):
    db = tmp_path / "ghost.db"
    rd = tmp_path / "results"
    chan = new_sqlite_channel(name="sqlite", results_dir=str(rd), max_rows=3, max_chars=2000)
    shell = new_ctml_shell()
    shell.main_channel.import_channels(chan)
    async with shell:
        await _run(shell, f'<sqlite:open db_path="{db}" name="mem"/>')
        await _run(shell, '<sqlite:query name="mem">CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT)</sqlite:query>')
        for i in range(10):
            await _run(shell, f"<sqlite:query name=\"mem\">INSERT INTO t(v) VALUES('row{i}')</sqlite:query>")

        r = await _run(shell, '<sqlite:query name="mem">SELECT * FROM t</sqlite:query>')
        joined = _joined(r)
        assert "truncated: 10 rows total" in joined
        assert "full result:" in joined

        files = list(rd.glob("*.txt"))
        assert len(files) == 1
        assert "row9" in files[0].read_text()


# -- read_only ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_only(tmp_path, shell):
    db = tmp_path / "ro.db"
    async with shell:
        await _run(shell, f'<sqlite:open db_path="{db}" name="w"/>')
        await _run(shell, '<sqlite:query name="w">CREATE TABLE users(id INTEGER PRIMARY KEY, name TEXT)</sqlite:query>')
        await _run(shell, '<sqlite:close name="w"/>')

        r = await _run(shell, f'<sqlite:open db_path="{db}" name="r" read_only="True"/>')
        assert "opened" in _joined(r)

        r = await _run(shell, "<sqlite:query name=\"r\">INSERT INTO users(name) VALUES('bob')</sqlite:query>")
        assert "[sqlite error]" in _joined(r)


# -- 错误路径 ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_not_open(shell):
    async with shell:
        r = await _run(shell, '<sqlite:query name="ghost">SELECT 1</sqlite:query>')
        assert "not open" in _joined(r)


@pytest.mark.asyncio
async def test_sql_error(tmp_path, shell):
    db = tmp_path / "e.db"
    async with shell:
        await _run(shell, f'<sqlite:open db_path="{db}" name="mem"/>')
        r = await _run(shell, '<sqlite:query name="mem">SELECT * FROM no_such_table</sqlite:query>')
        assert "[sqlite error]" in _joined(r)


@pytest.mark.asyncio
async def test_open_missing_path(tmp_path, shell):
    bad = tmp_path / "no" / "dir" / "x.db"
    async with shell:
        r = await _run(shell, f'<sqlite:open db_path="{bad}" name="x"/>')
        assert "failed" in _joined(r)
