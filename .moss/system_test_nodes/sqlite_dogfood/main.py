"""SQLite dogfood node — 提供 sqlite channel 供 MCP dogfood.

Start:  moss nodes run .moss/system_test_nodes/sqlite_dogfood
Debug:  python main.py                        # ad-hoc launch (from_proc identity)
"""

import sqlite3
from pathlib import Path

from ghoshell_moss.channels.sqlite_channel import new_sqlite_channel
from ghoshell_moss.core.blueprint.matrix import Matrix

_DEMO_DB = Path(__file__).parent / "runtime" / "dogfood.db"
_RESULTS_DIR = Path(__file__).parent / "runtime" / "results"


def _seed(db_path: Path) -> None:
    """建一张 demo 表并插两行, 供 dogfood 直接查询."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, name TEXT, joined TEXT)")
    conn.execute("INSERT OR IGNORE INTO users VALUES (1, 'alice', '2026-08-01')")
    conn.execute("INSERT OR IGNORE INTO users VALUES (2, 'bob', '2026-08-02')")
    conn.execute("CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, kind TEXT, ts TEXT)")
    conn.execute("INSERT OR IGNORE INTO events VALUES (1, 'login', '2026-08-03')")
    conn.commit()
    conn.close()


async def main(matrix: Matrix):
    _seed(_DEMO_DB)
    channel = new_sqlite_channel(
        name="sqlite",
        results_dir=str(_RESULTS_DIR),
    )
    matrix.logger.info("[sqlite_dogfood] seeded %s, providing sqlite channel", _DEMO_DB)
    await matrix.provide_channel(channel)  # blocks until membrane closes


if __name__ == "__main__":
    Matrix.discover().run(main)
