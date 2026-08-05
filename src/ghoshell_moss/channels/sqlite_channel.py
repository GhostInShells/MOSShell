"""sqlite3 数据库作为"文件资源协议" — 给模型 file editor 式的手段理解数据库 | 认知模块 | beta

单 channel 内部持有多个 sqlite 连接, SQL 只走 query.

设计沿革 (2026-08-05):
- 最初用 hub + virtual child 结构, 踩到两个坑:
  1. virtual channel 下一轮才生效, 同流 open→query 时序无法保证.
  2. 父命令 (open/close) 与子命令独立调度, close 可抢在子命令前移除子节点,
     pending 子命令卡死无报错.
  结论: 不用 virtual channel, 收窄为单 channel.
- __content__ 弃用: 自由文本路由对多库定位太绕. SQL 只走 query(name, sql).
- 连接映射 (name -> db_path) 经 context_messages 常驻 context, 列/数据按需取.
- 大结果集封顶: 内联只展示 head, 溢出可落盘 results_dir 返文件路径.

关键设计:
- 数据库即资源: 认知面 = open/close/list (生命周期) + tables/schema/sample (探索) + query (SQL).
- WAL + busy_timeout 保证跨进程 (ghost channel / d3 node) 共享同一 .db 时不互相阻塞.
- read_only 开关: 浏览别人的库时防止 ghost 损坏数据.
- 单 channel 内命令 FIFO, open→query→close 严格有序, 无时序竞争.

Example:
    from ghoshell_moss.channels.sqlite_channel import new_sqlite_channel
    main.import_channels(new_sqlite_channel(name="sqlite"))

    # CTML:
    #   <sqlite:open db_path="/data/ghost.db" name="mem"/>
    #   <sqlite:query name="mem">CREATE TABLE users(id INTEGER PRIMARY KEY, name TEXT)</sqlite:query>
    #   <sqlite:tables name="mem"/>
    #   <sqlite:schema name="mem" table="users"/>
    #   <sqlite:query name="mem">SELECT * FROM users</sqlite:query>
    #   <sqlite:close name="mem"/>
"""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ghoshell_moss.core.blueprint.channel_builder import MutableChannel, new_channel

__all__ = ["new_sqlite_channel"]

DEFAULT_MAX_ROWS = 100
DEFAULT_MAX_CHARS = 4000


# -- 连接持有者 -------------------------------------------------------------


@dataclass
class _SqliteConnection:
    """一个 sqlite 连接的生命周期状态 — 由 channel 命令闭包共享。"""

    db_path: str
    read_only: bool = False
    conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        path = Path(self.db_path).expanduser().resolve()
        mode = "ro" if self.read_only else "rwc"
        conn = sqlite3.connect(f"file:{path}?mode={mode}", uri=True, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        if not self.read_only:
            # 跨进程共享同一 .db 的关键: 多读单写不互斥.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        self.conn = conn

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    # -- 元信息 ------------------------------------------------------------

    def _table_names(self) -> list[str]:
        cur = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        return [r[0] for r in cur.fetchall()]

    def _view_names(self) -> list[str]:
        cur = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='view' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        return [r[0] for r in cur.fetchall()]

    def _describe_table(self, table: str) -> str:
        cols = self.conn.execute(f'PRAGMA table_info("{table}")').fetchall()
        parts = []
        for c in cols:
            col = f"{c['name']}: {c['type']}"
            if c["pk"]:
                col += " PK"
            parts.append(col)
        return f"{table}({', '.join(parts)})"

    def describe(self) -> str:
        lines = [f"DATABASE: {self.db_path}"]
        tables = self._table_names()
        views = self._view_names()
        if tables:
            lines.append("tables:")
            for t in tables:
                lines.append(f"  {self._describe_table(t)}")
        if views:
            lines.append("views:")
            for v in views:
                lines.append(f"  {v}")
        if not tables and not views:
            lines.append("(empty database — use query to CREATE TABLE)")
        return "\n".join(lines)

    # -- 查询 --------------------------------------------------------------

    def list_tables(self) -> str:
        lines = ["tables:"]
        for t in self._table_names():
            lines.append(f"  {t}")
        views = self._view_names()
        if views:
            lines.append("views:")
            for v in views:
                lines.append(f"  {v}")
        return "\n".join(lines)

    def describe_table(self, table: str) -> str:
        exists = table in self._table_names() or table in self._view_names()
        if not exists:
            return f"no such table or view: {table}"
        lines = [self._describe_table(table)]
        idxs = self.conn.execute(f'PRAGMA index_list("{table}")').fetchall()
        if idxs:
            lines.append("indexes:")
            for i in idxs:
                lines.append(f"  {i['name']}")
        return "\n".join(lines)

    def sample(self, table: str, limit: int = 10) -> str:
        if table not in self._table_names():
            return f"no such table: {table}"
        cur = self.conn.execute(f'SELECT * FROM "{table}" LIMIT ?', (limit,))
        return self._rows_to_text(cur)

    def execute(
            self,
            sql: str,
            *,
            max_rows: int,
            max_chars: int,
            results_dir_fn: Callable[[], str | None],
            prefix: str,
    ) -> str:
        sql = sql.strip().rstrip(";")
        if not sql:
            return ""
        try:
            cur = self.conn.execute(sql)
            if cur.description is None:
                self.conn.commit()
                count = cur.rowcount
                if count is not None and count >= 0:
                    return f"OK, {count} rows affected"
                return "OK"
            return self._format_result(
                cur, max_rows=max_rows, max_chars=max_chars,
                results_dir_fn=results_dir_fn, prefix=prefix,
            )
        except sqlite3.Error as e:
            return f"[sqlite error] {e}"

    def _format_result(
            self,
            cur: sqlite3.Cursor,
            *,
            max_rows: int,
            max_chars: int,
            results_dir_fn: Callable[[], str | None],
            prefix: str,
    ) -> str:
        columns = [d[0] for d in cur.description]
        rows = cur.fetchall()
        total = len(rows)
        head = rows[:max_rows]
        lines = [" | ".join(str(c) for c in columns)]
        for row in head:
            lines.append(" | ".join(str(row[c]) for c in columns))
        inline = "\n".join(lines)
        if len(inline) <= max_chars and total <= max_rows:
            return inline

        if len(inline) > max_chars:
            inline = inline[:max_chars] + "\n... (inline output truncated)"
        if total > max_rows:
            marker = f"... truncated: {total} rows total, showing {len(head)}"
        else:
            marker = f"... inline output truncated at {max_chars} chars"
        path = None
        try:
            results_dir = results_dir_fn()
        except Exception:
            results_dir = None
        if results_dir:
            path = self._dump_full(columns, rows, results_dir, prefix)
            if path:
                return f"{inline}\n{marker}\nfull result: {path}"
        return f"{inline}\n{marker}"

    def _dump_full(self, columns: list[str], rows: list, results_dir: str, prefix: str) -> str | None:
        try:
            dir_path = Path(results_dir)
            dir_path.mkdir(parents=True, exist_ok=True)
            path = dir_path / f"{prefix}_{uuid.uuid4().hex[:8]}.txt"
            lines = [" | ".join(str(c) for c in columns)]
            for row in rows:
                lines.append(" | ".join(str(row[c]) for c in columns))
            path.write_text("\n".join(lines), encoding="utf-8")
            return str(path)
        except OSError:
            return None

    def _rows_to_text(self, cur: sqlite3.Cursor) -> str:
        columns = [d[0] for d in cur.description] if cur.description else None
        rows = cur.fetchall()
        if columns is None:
            return ""
        lines = [" | ".join(str(c) for c in columns)]
        for row in rows:
            lines.append(" | ".join(str(row[c]) for c in columns))
        return "\n".join(lines)


# -- channel ----------------------------------------------------------------


def new_sqlite_channel(
    *,
    name: str = "sqlite",
    results_dir: str | None = None,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> MutableChannel:
    """单 channel: 持有多个 sqlite 连接, 以 name 寻址.

    - open/close/list 管理连接; tables/schema/sample 提供 file editor 式的探索;
      query(name, sql) 是唯一的 SQL 入口 (text__ 流式 body).
    - context_messages 常驻"open connections"映射 (name -> db_path).
    - query 内联结果封顶 (max_rows / max_chars); 溢出时若 results_dir 存在则落盘返路径.
      results_dir 未传入时, channel 自动分配一个临时目录并在关闭时删除.
    """
    connections: dict[str, _SqliteConnection] = {}
    results_dir_auto: str | None = None

    def _resolve_results_dir() -> str | None:
        nonlocal results_dir, results_dir_auto
        if results_dir is None:
            if results_dir_auto is None:
                results_dir_auto = tempfile.mkdtemp(prefix="sqlite_results_")
            results_dir = results_dir_auto
        return results_dir

    chan = new_channel(
        name=name,
        description="SQLite databases — open/close/list manage connections; "
                    "tables/schema/sample explore a db; query(name, sql) runs SQL.",
    )

    @chan.build.context_messages
    def _context() -> list[str]:
        if not connections:
            return []
        lines = ["open connections:"]
        for alias, c in connections.items():
            lines.append(f"  {alias} -> {c.db_path}")
        return ["\n".join(lines)]

    @chan.build.close
    async def _close() -> None:
        nonlocal results_dir_auto
        for holder in connections.values():
            holder.close()
        connections.clear()
        if results_dir_auto is not None:
            shutil.rmtree(results_dir_auto, ignore_errors=True)
            results_dir_auto = None

    @chan.build.command(name="open", blocking=True, always_observe=True)
    async def open_db(db_path: str, name: str = "", read_only: bool = False) -> str:
        """Open a sqlite database file as a named connection. Creates the file if missing."""
        alias = name or Path(db_path).stem
        if alias in connections:
            return f"[sqlite] {alias} already open"
        holder = _SqliteConnection(db_path=db_path, read_only=read_only)
        try:
            holder.connect()
        except Exception as e:
            return f"[sqlite] open {db_path} failed: {e}"
        connections[alias] = holder
        return f"[sqlite] opened {db_path} as {alias}"

    @chan.build.command(name="close", blocking=True, always_observe=False)
    async def close_db(name: str) -> str:
        """Close a named connection."""
        holder = connections.pop(name, None)
        if holder is None:
            return f"[sqlite] close {name}: not open"
        holder.close()
        return f"[sqlite] closed {name}"

    @chan.build.command(name="list", blocking=False, always_observe=True)
    async def list_db() -> str:
        """List open connections."""
        if not connections:
            return "[sqlite] no open databases"
        return "\n".join(f"  {alias} -> {c.db_path}" for alias, c in connections.items())

    @chan.build.command(name="query", blocking=True, always_observe=True)
    async def query(name: str, text__: str = "") -> str:
        """Run a single SQL statement against a named connection. DML auto-commits. Inline result is capped."""
        holder = connections.get(name)
        if holder is None:
            return f"[sqlite] query {name}: not open"
        return holder.execute(
            text__,
            max_rows=max_rows,
            max_chars=max_chars,
            results_dir_fn=_resolve_results_dir,
            prefix=name,
        )

    @chan.build.command(name="tables", blocking=False, always_observe=True)
    async def tables(name: str) -> str:
        """List all tables and views in a named connection."""
        holder = connections.get(name)
        if holder is None:
            return f"[sqlite] tables {name}: not open"
        return holder.list_tables()

    @chan.build.command(name="schema", blocking=False, always_observe=True)
    async def schema(name: str, table: str = "") -> str:
        """Show columns of a table (or the full db intro when table is empty)."""
        holder = connections.get(name)
        if holder is None:
            return f"[sqlite] schema {name}: not open"
        return holder.describe_table(table) if table else holder.describe()

    @chan.build.command(name="sample", blocking=False, always_observe=True)
    async def sample(name: str, table: str, limit: int = 10) -> str:
        """Sample rows from a table (up to `limit`)."""
        holder = connections.get(name)
        if holder is None:
            return f"[sqlite] sample {name}: not open"
        return holder.sample(table, limit)

    return chan
