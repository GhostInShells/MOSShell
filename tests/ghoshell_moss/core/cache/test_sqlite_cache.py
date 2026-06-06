from __future__ import annotations

import multiprocessing
import tempfile
import time
from pathlib import Path

import pytest

from ghoshell_moss.core.cache import SqliteCache


class TestSqliteCache:
    """单进程内功能测试."""

    @pytest.fixture
    def cache(self) -> SqliteCache:
        with tempfile.TemporaryDirectory() as tmp:
            yield SqliteCache(Path(tmp) / "cache.db")

    # ---- KV ----

    def test_set_get(self, cache: SqliteCache):
        cache.set("a", "1")
        assert cache.get("a") == "1"

    def test_get_missing_returns_none(self, cache: SqliteCache):
        assert cache.get("nonexistent") is None

    def test_set_overwrites(self, cache: SqliteCache):
        cache.set("a", "first")
        cache.set("a", "second")
        assert cache.get("a") == "second"

    # ---- TTL ----

    def test_get_expired_returns_none(self, cache: SqliteCache):
        cache.set("a", "1", exp=0.001)  # 1ms, effectively already expired after sleep
        time.sleep(0.01)
        assert cache.get("a") is None

    def test_get_unexpired_returns_val(self, cache: SqliteCache):
        cache.set("a", "1", exp=60)
        assert cache.get("a") == "1"

    def test_exp_0_means_no_expiry(self, cache: SqliteCache):
        cache.set("a", "1", exp=0)
        assert cache.get("a") == "1"

    def test_expire_updates_ttl(self, cache: SqliteCache):
        cache.set("a", "1", exp=60)
        cache.expire("a", exp=0.001)
        time.sleep(0.01)
        assert cache.get("a") is None

    def test_expire_missing_key(self, cache: SqliteCache):
        assert cache.expire("no", 60) is False

    # ---- remove ----

    def test_remove_deletes_key(self, cache: SqliteCache):
        cache.set("a", "1")
        assert cache.remove("a") == 1
        assert cache.get("a") is None

    def test_remove_missing_returns_zero(self, cache: SqliteCache):
        assert cache.remove("no") == 0

    def test_remove_multiple(self, cache: SqliteCache):
        cache.set("a", "1")
        cache.set("b", "2")
        assert cache.remove("a", "b", "c") == 2

    # ---- hash map ----

    def test_set_member_get_member(self, cache: SqliteCache):
        cache.set_member("h", "m1", "v1")
        assert cache.get_member("h", "m1") == "v1"

    def test_get_member_missing(self, cache: SqliteCache):
        assert cache.get_member("h", "no") is None

    def test_set_member_overwrites(self, cache: SqliteCache):
        cache.set_member("h", "m1", "v1")
        cache.set_member("h", "m1", "v2")
        assert cache.get_member("h", "m1") == "v2"

    def test_remove_member(self, cache: SqliteCache):
        cache.set_member("h", "m1", "v1")
        cache.set_member("h", "m2", "v2")
        assert cache.remove_member("h", "m1", "m3") == 1
        assert cache.get_member("h", "m1") is None
        assert cache.get_member("h", "m2") == "v2"

    def test_remove_cleans_hash_map(self, cache: SqliteCache):
        cache.set_member("h", "m", "v")
        assert cache.remove("h") == 1
        assert cache.get_member("h", "m") is None

    # ---- lock ----

    def test_lock_unlock(self, cache: SqliteCache):
        assert cache.lock("lk") is True
        assert cache.lock("lk") is False
        cache.unlock("lk")
        assert cache.lock("lk") is True

    def test_unlock_missing(self, cache: SqliteCache):
        assert cache.unlock("no") is False

    def test_lock_overdue_0_means_no_expiry(self, cache: SqliteCache):
        cache.lock("lk", overdue=0)
        assert cache.lock("lk") is False

    def test_lock_expires(self, cache: SqliteCache):
        cache.lock("lk", overdue=0.001)
        time.sleep(0.01)
        assert cache.lock("lk") is True

    def test_remove_cleans_lock(self, cache: SqliteCache):
        cache.lock("lk")
        cache.remove("lk")
        assert cache.lock("lk") is True

    # ---- locked context manager ----

    def test_locked_acquires_and_releases(self, cache: SqliteCache):
        with cache.locked("ctx_lock"):
            assert cache.lock("ctx_lock") is False
        assert cache.lock("ctx_lock") is True

    def test_locked_raises_when_already_held(self, cache: SqliteCache):
        cache.lock("ctx_lock")
        with pytest.raises(RuntimeError, match="Failed to acquire lock"):
            with cache.locked("ctx_lock"):
                pass

    def test_locked_releases_on_exception(self, cache: SqliteCache):
        try:
            with cache.locked("ctx_lock"):
                raise ValueError("inner error")
        except ValueError:
            pass
        assert cache.lock("ctx_lock") is True


class TestSqliteCacheCrossProcess:
    """跨进程仲裁测试: 两个进程共享同一个 .db 文件."""

    @staticmethod
    def _proc_a_acquire_lock(db_path: str, result_queue: multiprocessing.Queue):
        cache = SqliteCache(db_path)
        ok = cache.lock("cross_lock")
        result_queue.put(("a_acquired", ok))
        time.sleep(0.1)
        cache.unlock("cross_lock")

    @staticmethod
    def _proc_b_try_lock(db_path: str, result_queue: multiprocessing.Queue):
        time.sleep(0.03)  # 等 A 先拿锁
        cache = SqliteCache(db_path)
        ok = cache.lock("cross_lock")
        result_queue.put(("b_acquired", ok))

    def test_cross_process_lock_arbitration(self, tmp_path: Path):
        db_path = tmp_path / "cross_cache.db"

        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()

        a = ctx.Process(target=self._proc_a_acquire_lock, args=(str(db_path), queue))
        b = ctx.Process(target=self._proc_b_try_lock, args=(str(db_path), queue))

        a.start()
        b.start()
        a.join()
        b.join()

        results = {}
        while not queue.empty():
            k, v = queue.get_nowait()
            results[k] = v

        assert results["a_acquired"] is True, "A should acquire the lock"
        assert results["b_acquired"] is False, "B should NOT acquire while A holds it"

    @staticmethod
    def _proc_writer(db_path: str, result_queue: multiprocessing.Queue):
        cache = SqliteCache(db_path)
        cache.set("shared", "from_writer")
        result_queue.put(True)

    @staticmethod
    def _proc_reader(db_path: str, result_queue: multiprocessing.Queue):
        time.sleep(0.05)  # wait for writer
        cache = SqliteCache(db_path)
        val = cache.get("shared")
        result_queue.put(val)

    def test_cross_process_read_after_write(self, tmp_path: Path):
        db_path = tmp_path / "cross_cache.db"

        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()

        writer = ctx.Process(target=self._proc_writer, args=(str(db_path), queue))
        reader = ctx.Process(target=self._proc_reader, args=(str(db_path), queue))

        writer.start()
        reader.start()
        writer.join()
        reader.join()

        results = [queue.get_nowait() for _ in range(2)]
        assert True in results
        assert "from_writer" in results
