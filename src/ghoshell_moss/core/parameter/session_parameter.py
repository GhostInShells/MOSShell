"""
SessionParameterStore — cross-process ParameterStore backed by Session.

SQLite is the ground truth.  Each process caches values in a local dict;
Zenoh stream carries invalidation signals (key, version) to keep caches
coherent across processes.

Read path:  pure dict lookup, zero IO.
Write path: SQLite CAS → update local dict → Zenoh pub invalidation.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from ghoshell_moss.core.blueprint.parameter import (
    Parameter,
    ParameterModel,
    ParameterStore,
    T_PARAM,
    VersionConflict,
)
from ghoshell_moss.core.blueprint.session import (
    Session,
    Sample,
)

__all__ = ["SessionParameterStore"]

_INVALIDATION_KEY = "parameters/invalidations"


class _Entry:
    __slots__ = ("value", "version")

    def __init__(self, value: ParameterModel, version: int) -> None:
        self.value = value
        self.version = version


class SessionParameter(Parameter[T_PARAM]):
    """Typed handle — delegates to the store's cache + SQLite."""

    def __init__(
        self,
        store: SessionParameterStore,
        key: str,
        model_type: type[T_PARAM],
    ) -> None:
        self._store = store
        self._key = key
        self._model_type = model_type

    @property
    def key(self) -> str:
        return self._key

    def get(self) -> T_PARAM:
        return self._store._cached_get(self._key, self._model_type)

    def set(self, value: T_PARAM, *, version: int | None = None) -> int:
        return self._store._write_and_notify(self._key, value, version)

    def version(self) -> int:
        return self._store._cached_version(self._key)

    def remove(self) -> bool:
        return self._store._remove(self._key)


class SessionParameterStore(ParameterStore):
    """
    Cross-process ParameterStore backed by Session's SQLite + Zenoh stream.

    Constructed from a Session — derives db path from tmp_storage,
    scope from session_scope, and uses session's sub/pub_stream_delta
    for invalidation signals.

    Usage::

        store = SessionParameterStore(session)
        param = store.declare(GhostPersona)
        cfg  = param.get()   # dict lookup — no IO
    """

    def __init__(self, session: Session, *, busy_timeout: int = 3000) -> None:
        db_path = Path(session.tmp_storage.abspath()) / "parameter.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self._session = session
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute(f"PRAGMA busy_timeout={busy_timeout}")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS parameters ("
            "  key        TEXT PRIMARY KEY,"
            "  value_json TEXT NOT NULL,"
            "  version    INTEGER NOT NULL DEFAULT 1,"
            "  updated_at REAL    NOT NULL"
            ")"
        )
        self._conn.commit()

        # In-process cache — the "高频读" hot path.
        # Populated lazily by declare(); never stores raw dicts.
        self._lock = threading.Lock()
        self._cache: dict[str, _Entry] = {}
        self._model_types: dict[str, type[ParameterModel]] = {}

        # Subscribe to cross-process invalidation signals
        self._stop_invalidation = session.sub_stream(
            _INVALIDATION_KEY,
            self._on_invalidation,
        )

    # -- ParameterStore ---------------------------------------

    def declare(
        self,
        model_type: type[T_PARAM],
        *,
        key: str | None = None,
    ) -> Parameter[T_PARAM]:
        resolved = key or model_type.param_name()
        with self._lock:
            self._model_types[resolved] = model_type
            if resolved not in self._cache:
                # Lazy-load from SQLite — first declare() in this process
                row = self._conn.execute(
                    "SELECT value_json, version FROM parameters WHERE key = ?",
                    (resolved,),
                ).fetchone()
                if row is not None:
                    self._cache[resolved] = _Entry(
                        value=model_type.model_validate_json(row[0]),
                        version=row[1],
                    )
                else:
                    self._cache[resolved] = _Entry(
                        value=model_type.param_default(),
                        version=0,
                    )
        return SessionParameter(self, resolved, model_type)

    def declared(self) -> list[str]:
        with self._lock:
            return sorted(self._cache.keys())

    # -- internal: read (pure dict) ---------------------------

    def _cached_get(
        self, key: str, model_type: type[T_PARAM],
    ) -> T_PARAM:
        with self._lock:
            entry = self._cache.get(key)
        if entry is None:
            return model_type.param_default()
        return entry.value  # type: ignore[return-value]

    def _cached_version(self, key: str) -> int:
        with self._lock:
            entry = self._cache.get(key)
        return entry.version if entry else 0

    # -- internal: write (SQLite + cache + notify) ------------

    def _write_and_notify(
        self,
        key: str,
        value: ParameterModel,
        version: int | None,
    ) -> int:
        now = time.time()
        js = value.model_dump_json()

        with self._lock:
            with self._conn:
                if version is not None:
                    new_version = self._cas_write(js, version, now, key)
                else:
                    new_version = self._force_write(js, now, key)

            # Update local cache
            model_type = self._model_types.get(key, type(value))
            self._cache[key] = _Entry(
                value=model_type.model_validate_json(js),
                version=new_version,
            )

        # Notify other processes (outside lock — fire-and-forget)
        self._pub_invalidation(key, new_version)
        return new_version

    def _cas_write(
        self, js: str, version: int, now: float, key: str,
    ) -> int:
        cur = self._conn.execute(
            "UPDATE parameters SET value_json = ?, version = version + 1, "
            "updated_at = ? WHERE key = ? AND version = ?",
            (js, now, key, version),
        )
        if cur.rowcount > 0:
            return version + 1

        row = self._conn.execute(
            "SELECT version FROM parameters WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            if version == 0:
                return self._insert_first(js, now, key)
            raise VersionConflict(key, version, 0)
        raise VersionConflict(key, version, row[0])

    def _force_write(self, js: str, now: float, key: str) -> int:
        self._conn.execute(
            "INSERT INTO parameters(key, value_json, version, updated_at) "
            "VALUES(?, ?, 1, ?) "
            "ON CONFLICT(key) DO UPDATE SET "
            "value_json = excluded.value_json, "
            "version = parameters.version + 1, "
            "updated_at = excluded.updated_at",
            (key, js, now),
        )
        row = self._conn.execute(
            "SELECT version FROM parameters WHERE key = ?", (key,)
        ).fetchone()
        return row[0]

    def _insert_first(self, js: str, now: float, key: str) -> int:
        self._conn.execute(
            "INSERT INTO parameters(key, value_json, version, updated_at) "
            "VALUES(?, ?, 1, ?)",
            (key, js, now),
        )
        return 1

    def _remove(self, key: str) -> bool:
        with self._conn:
            cur = self._conn.execute(
                "DELETE FROM parameters WHERE key = ?", (key,)
            )
            existed = cur.rowcount > 0
        with self._lock:
            self._cache.pop(key, None)
        if existed:
            self._pub_invalidation(key, -1)
        return existed

    # -- invalidation (zenoh pub/sub) -------------------------

    def _pub_invalidation(self, key: str, version: int) -> None:
        payload = json.dumps({"key": key, "version": version})
        try:
            self._session.pub_stream_delta(
                _INVALIDATION_KEY,
                payload.encode(),
            )
        except Exception:
            # invalidation is best-effort — if zenoh is down
            # the cache is still locally consistent
            pass

    def _on_invalidation(self, sample: Sample) -> None:
        try:
            data = json.loads(sample.payload)
            key = data["key"]
            remote_version = data["version"]
        except (json.JSONDecodeError, KeyError):
            return

        with self._lock:
            entry = self._cache.get(key)
            if entry is not None and remote_version <= entry.version:
                return  # already at or ahead of this version

        # Re-read from SQLite and update cache
        row = self._conn.execute(
            "SELECT value_json, version FROM parameters WHERE key = ?", (key,)
        ).fetchone()

        with self._lock:
            model_type = self._model_types.get(key)
            if row is None:
                if model_type is not None:
                    self._cache[key] = _Entry(
                        value=model_type.param_default(),
                        version=0,
                    )
                else:
                    self._cache.pop(key, None)
            elif model_type is not None:
                self._cache[key] = _Entry(
                    value=model_type.model_validate_json(row[0]),
                    version=row[1],
                )

    # -- cleanup -----------------------------------------------

    def close(self) -> None:
        """Release zenoh subscription and SQLite connection."""
        try:
            self._stop_invalidation()
        except Exception:
            pass
        try:
            self._conn.close()
        except Exception:
            pass
