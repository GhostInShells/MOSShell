from __future__ import annotations

import json
import multiprocessing
import tempfile
import time
from functools import partial
from pathlib import Path
from typing import Callable

import pytest
from pydantic import BaseModel

from ghoshell_moss.contracts.workspace import LocalStorage
from ghoshell_moss.core.blueprint.parameter import (
    ParameterModel,
    ParameterStore,
    VersionConflict,
)
from ghoshell_moss.core.blueprint.session import Sample
from ghoshell_moss.core.parameter import SessionParameterStore


# -- test models ------------------------------------------------


class GhostPersona(ParameterModel):
    name: str = "Echo"
    temperature: float = 0.7

    @classmethod
    def param_name(cls) -> str:
        return "ghost_persona"

    @classmethod
    def param_default(cls) -> "GhostPersona":
        return cls()


class RobotConfig(ParameterModel):
    joint_count: int = 6
    frame_rate: int = 30

    @classmethod
    def param_name(cls) -> str:
        return "robot_config"

    @classmethod
    def param_default(cls) -> "RobotConfig":
        return cls()


# -- mock session -----------------------------------------------


class _MockSession:
    """Minimal Session for testing — no Zenoh, synchronous invalidation."""

    def __init__(self, storage: LocalStorage) -> None:
        self._storage = storage
        self._callbacks: dict[str, Callable[[Sample], None]] = {}

    @property
    def tmp_storage(self) -> LocalStorage:
        return self._storage

    @property
    def session_scope(self) -> str:
        return "test"

    def sub_stream(
        self, relative_key: str, callback: Callable[[Sample], None],
    ) -> Callable[[], None]:
        self._callbacks[relative_key] = callback

        def _stop() -> None:
            self._callbacks.pop(relative_key, None)
        return _stop

    def pub_stream_delta(self, relative_key: str, delta: bytes) -> None:
        cb = self._callbacks.get(relative_key)
        if cb is not None:
            cb(Sample(relative_key=relative_key, payload=delta))


# -- fixtures ---------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> SessionParameterStore:
    storage = LocalStorage(tmp_path / "session_tmp")
    session = _MockSession(storage)
    return SessionParameterStore(session)


def _make_store(db_dir: str, **kw) -> SessionParameterStore:
    storage = LocalStorage(Path(db_dir))
    session = _MockSession(storage)
    return SessionParameterStore(session, **kw)


# -- single-process tests ---------------------------------------


class TestSessionParameterStore:
    """Single-process — cache + CAS + invalidation coherence."""

    def test_get_returns_default_on_miss(self, store: SessionParameterStore):
        param = store.declare(GhostPersona)
        v = param.get()
        assert isinstance(v, GhostPersona)
        assert v.name == "Echo"
        assert v.temperature == 0.7

    def test_set_then_get(self, store: SessionParameterStore):
        param = store.declare(GhostPersona)
        new_val = GhostPersona(name="Nova", temperature=0.9)
        param.set(new_val)
        v = param.get()
        assert v.name == "Nova"
        assert v.temperature == 0.9

    def test_force_write_increments_version(self, store: SessionParameterStore):
        param = store.declare(GhostPersona)
        v1 = param.set(GhostPersona(name="A"))
        v2 = param.set(GhostPersona(name="B"))
        assert v2 == v1 + 1

    def test_cas_succeeds_with_correct_version(self, store: SessionParameterStore):
        param = store.declare(GhostPersona)
        param.set(GhostPersona(name="v1"))
        v = param.version()
        new_v = param.set(GhostPersona(name="v2"), version=v)
        assert new_v == v + 1
        assert param.get().name == "v2"

    def test_cas_fails_with_wrong_version(self, store: SessionParameterStore):
        param = store.declare(GhostPersona)
        param.set(GhostPersona(name="v1"))
        with pytest.raises(VersionConflict):
            param.set(GhostPersona(name="v2"), version=99)

    def test_cas_with_version_0_creates_first(self, store: SessionParameterStore):
        param = store.declare(GhostPersona)
        # never set — version should be 0
        assert param.version() == 0
        new_v = param.set(GhostPersona(name="first"), version=0)
        assert new_v == 1
        assert param.get().name == "first"

    def test_cas_with_nonzero_version_on_missing_key_raises(self, store: SessionParameterStore):
        param = store.declare(GhostPersona)
        with pytest.raises(VersionConflict) as exc:
            param.set(GhostPersona(name="x"), version=5)
        assert exc.value.actual == 0

    def test_version_returns_0_for_undeclared_default(self, store: SessionParameterStore):
        param = store.declare(GhostPersona, key="never_set")
        assert param.version() == 0

    def test_remove(self, store: SessionParameterStore):
        param = store.declare(GhostPersona)
        param.set(GhostPersona(name="gone"))
        assert param.remove() is True
        assert param.remove() is False
        assert param.version() == 0

    def test_declare_idempotent(self, store: SessionParameterStore):
        a = store.declare(GhostPersona)
        b = store.declare(GhostPersona)
        a.set(GhostPersona(name="shared"))
        assert b.get().name == "shared"

    def test_declared_lists_keys(self, store: SessionParameterStore):
        store.declare(GhostPersona)
        store.declare(RobotConfig)
        store.declare(GhostPersona, key="alt_persona")
        keys = store.declared()
        assert "ghost_persona" in keys
        assert "robot_config" in keys
        assert "alt_persona" in keys

    def test_key_override(self, store: SessionParameterStore):
        default = store.declare(GhostPersona)
        custom = store.declare(GhostPersona, key="custom_persona")
        default.set(GhostPersona(name="default"))
        custom.set(GhostPersona(name="custom"))
        assert default.get().name == "default"
        assert custom.get().name == "custom"

    # -- invalidation coherence (synchronous mock) ---------------

    def test_invalidation_keeps_cache_coherent(self, store: SessionParameterStore):
        """Write through one handle, another handle sees the update via invalidation."""
        a = store.declare(GhostPersona)
        b = store.declare(GhostPersona)

        a.set(GhostPersona(name="via_a"))
        # b's cache should be updated by the mock's synchronous pub_stream_delta
        assert b.get().name == "via_a"

    def test_persistence_across_store_instances(self, tmp_path: Path):
        """Re-opening the store (same db) should load persisted values."""
        db_dir = tmp_path / "persist_test"
        db_dir.mkdir()

        store1 = _make_store(str(db_dir))
        p1 = store1.declare(GhostPersona)
        p1.set(GhostPersona(name="persisted"))

        # New store instance (simulating process restart or another module)
        store2 = _make_store(str(db_dir))
        p2 = store2.declare(GhostPersona)
        assert p2.get().name == "persisted"
        assert p2.version() > 0


# -- cross-process tests ----------------------------------------


class TestCrossProcessParameter:
    """SQLite-level correctness across processes (no Zenoh — cache is local)."""

    @staticmethod
    def _proc_writer(
        db_dir: str,
        ready: multiprocessing.Event,
        queue: multiprocessing.Queue,
    ) -> None:
        store = _make_store(db_dir)
        param = store.declare(GhostPersona)
        v = param.set(GhostPersona(name="cross_proc"))
        ready.set()  # signal: write committed
        queue.put(v)

    @staticmethod
    def _proc_reader(
        db_dir: str,
        ready: multiprocessing.Event,
        queue: multiprocessing.Queue,
    ) -> None:
        ready.wait()  # wait for writer to finish init + write
        store = _make_store(db_dir)
        param = store.declare(GhostPersona)
        queue.put((param.get().name, param.version()))

    def test_write_visible_in_another_process(self, tmp_path: Path):
        db_dir = tmp_path / "cross_proc"
        db_dir.mkdir()

        ctx = multiprocessing.get_context("spawn")
        ready = ctx.Event()
        queue = ctx.Queue()

        writer = ctx.Process(
            target=self._proc_writer, args=(str(db_dir), ready, queue),
        )
        reader = ctx.Process(
            target=self._proc_reader, args=(str(db_dir), ready, queue),
        )

        writer.start()
        reader.start()
        writer.join(timeout=30)
        reader.join(timeout=30)

        assert writer.exitcode == 0, f"writer crashed with exitcode {writer.exitcode}"
        assert reader.exitcode == 0, f"reader crashed with exitcode {reader.exitcode}"

        results = [queue.get() for _ in range(2)]
        versions = [r for r in results if isinstance(r, int)]
        assert len(versions) == 1
        assert versions[0] > 0
        names_versions = [r for r in results if isinstance(r, tuple)]
        assert len(names_versions) == 1
        assert names_versions[0][0] == "cross_proc"
        assert names_versions[0][1] > 0

    @staticmethod
    def _proc_cas_winner(
        db_dir: str,
        ready: multiprocessing.Event,
        queue: multiprocessing.Queue,
    ) -> None:
        store = _make_store(db_dir)
        param = store.declare(GhostPersona)
        param.set(GhostPersona(name="v0"))
        ready.set()  # signal: DB initialized, loser can proceed
        time.sleep(0.05)

        v = param.version()
        try:
            new_v = param.set(GhostPersona(name="winner"), version=v)
            queue.put(("win", new_v))
        except VersionConflict:
            queue.put("lose")

    @staticmethod
    def _proc_cas_loser(
        db_dir: str,
        ready: multiprocessing.Event,
        queue: multiprocessing.Queue,
    ) -> None:
        ready.wait()  # wait for winner to init DB
        time.sleep(0.02)
        store = _make_store(db_dir)
        param = store.declare(GhostPersona)
        v = param.version()
        try:
            new_v = param.set(GhostPersona(name="loser"), version=v)
            queue.put(("win", new_v))
        except VersionConflict:
            queue.put("lose")

    def test_cas_cross_process_only_one_wins(self, tmp_path: Path):
        db_dir = tmp_path / "cross_cas"
        db_dir.mkdir()

        ctx = multiprocessing.get_context("spawn")
        ready = ctx.Event()
        queue = ctx.Queue()

        a = ctx.Process(
            target=self._proc_cas_winner, args=(str(db_dir), ready, queue),
        )
        b = ctx.Process(
            target=self._proc_cas_loser, args=(str(db_dir), ready, queue),
        )

        a.start()
        b.start()
        a.join(timeout=30)
        b.join(timeout=30)

        assert a.exitcode == 0, f"winner crashed with exitcode {a.exitcode}"
        assert b.exitcode == 0, f"loser crashed with exitcode {b.exitcode}"

        results = []
        while not queue.empty():
            results.append(queue.get_nowait())

        wins = [r for r in results if isinstance(r, tuple) and r[0] == "win"]
        loses = [r for r in results if r == "lose"]
        assert len(wins) == 1, f"expected 1 win, got {wins}"
        assert len(loses) == 1, f"expected 1 lose, got {loses}"
