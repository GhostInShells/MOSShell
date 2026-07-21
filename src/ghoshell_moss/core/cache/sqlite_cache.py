from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Optional

from ghoshell_moss.contracts.cache import Cache


class SqliteCache(Cache):
    """
    基于 sqlite3 的跨进程 Cache 实现.

    所有进程各自 connect 同一个 .db 文件, WAL 模式保证并发安全.
    """

    def __init__(self, db_path: str | Path) -> None:
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=3000")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS strings ("
            "  key TEXT PRIMARY KEY,"
            "  val TEXT NOT NULL,"
            "  expired_at REAL DEFAULT 0"
            ")"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS hash_map ("
            "  key TEXT NOT NULL,"
            "  member TEXT NOT NULL,"
            "  value TEXT NOT NULL,"
            "  PRIMARY KEY (key, member)"
            ")"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS locks ("
            "  key TEXT PRIMARY KEY,"
            "  expired_at REAL DEFAULT 0"
            ")"
        )
        self._conn.commit()

    # ---- lock ----

    def lock(self, key: str, overdue: int = 0) -> bool:
        now = time.time()
        expired_at = now + overdue if overdue > 0 else 0

        with self._conn:
            self._conn.execute(
                "DELETE FROM locks WHERE key = ? AND expired_at > 0 AND expired_at < ?",
                (key, now),
            )
            try:
                self._conn.execute(
                    "INSERT INTO locks(key, expired_at) VALUES(?, ?)",
                    (key, expired_at),
                )
                return True
            except sqlite3.IntegrityError:
                return False

    def unlock(self, key: str) -> bool:
        with self._conn:
            cursor = self._conn.execute("DELETE FROM locks WHERE key = ?", (key,))
            return cursor.rowcount > 0

    # ---- KV ----

    def set(self, key: str, val: str, exp: int = 0) -> bool:
        expired_at = time.time() + exp if exp > 0 else 0
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO strings(key, val, expired_at) VALUES(?, ?, ?)",
                (key, val, expired_at),
            )
        return True

    def get(self, key: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT val, expired_at FROM strings WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        val, expired_at = row
        if expired_at > 0 and expired_at < time.time():
            self._conn.execute("DELETE FROM strings WHERE key = ?", (key,))
            self._conn.commit()
            return None
        return val

    def expire(self, key: str, exp: int) -> bool:
        expired_at = time.time() + exp if exp > 0 else 0
        with self._conn:
            cursor = self._conn.execute(
                "UPDATE strings SET expired_at = ? WHERE key = ?",
                (expired_at, key),
            )
            return cursor.rowcount > 0

    # ---- hash map ----

    def set_member(self, key: str, member: str, value: str) -> bool:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO hash_map(key, member, value) VALUES(?, ?, ?)",
                (key, member, value),
            )
        return True

    def get_member(self, key: str, member: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT value FROM hash_map WHERE key = ? AND member = ?",
            (key, member),
        ).fetchone()
        return row[0] if row is not None else None

    def remove_member(self, key: str, *member: str) -> int:
        with self._conn:
            cursor = self._conn.execute(
                "DELETE FROM hash_map WHERE key = ? AND member IN ({})".format(
                    ",".join("?" * len(member))
                ),
                (key, *member),
            )
            return cursor.rowcount

    # ---- remove ----

    def remove(self, *keys: str) -> int:
        count = 0
        with self._conn:
            for key in keys:
                c1 = self._conn.execute("DELETE FROM strings WHERE key = ?", (key,))
                c2 = self._conn.execute("DELETE FROM hash_map WHERE key = ?", (key,))
                self._conn.execute("DELETE FROM locks WHERE key = ?", (key,))
                if c1.rowcount > 0 or c2.rowcount > 0:
                    count += 1
        return count
