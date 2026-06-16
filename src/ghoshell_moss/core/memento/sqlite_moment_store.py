"""SQLite-backed MomentStore — Moment 全局池.

设计:
- WAL 模式, check_same_thread=False, 多进程多线程共存.
- 表结构刻意简单: id (PK) + owner (索引列, 诊断用) + created (写入时间) + body (json).
- 序列化时丢弃 perspectives 与 hint (动态字段, 持久化无意义), 保留 compacted_perspectives.
- 不做 TTL / 清理 — moment 永存, 与 features 哲学一致.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Iterable

from ghoshell_moss.core.memento.abc import Moment, MomentStore


class SqliteMomentStore(MomentStore):
    """跨进程/跨线程安全的 Moment 池. 参考 SqliteCache 实现模式."""

    def __init__(self, db_path: str | Path) -> None:
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=3000")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS moments ("
            "  id TEXT PRIMARY KEY,"
            "  owner TEXT NOT NULL,"
            "  created REAL NOT NULL,"
            "  body TEXT NOT NULL"
            ")"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_moments_owner ON moments(owner)"
        )
        self._conn.commit()

    def put(self, moment: Moment, *, owner: str) -> None:
        body = moment.to_json(exclude_perspectives=True, exclude_hint=True)
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO moments(id, owner, created, body) VALUES(?, ?, ?, ?)",
                (moment.id, owner, time.time(), body),
            )

    def get(self, moment_id: str) -> Moment | None:
        row = self._conn.execute(
            "SELECT body FROM moments WHERE id = ?", (moment_id,)
        ).fetchone()
        if row is None:
            return None
        return Moment.model_validate_json(row[0])

    def get_many(self, moment_ids: Iterable[str]) -> dict[str, Moment]:
        ids = list(moment_ids)
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        rows = self._conn.execute(
            f"SELECT id, body FROM moments WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        return {row[0]: Moment.model_validate_json(row[1]) for row in rows}

    def close(self) -> None:
        self._conn.close()
