"""ZenohLivenessListener 单元测试 — 全量查询 / 事件订阅 / reconcile / 回调."""

import asyncio
import logging

import pytest

from ghoshell_moss.depends import depend_matrix

depend_matrix()
import zenoh

from ghoshell_moss.tools.zenoh_helper import ZenohLivenessListener
from ghoshell_moss.message import unique_id


@pytest.fixture
def scope():
    return unique_id()


@pytest.fixture
def session():
    s = zenoh.open(zenoh.Config())
    yield s
    if not s.is_closed():
        s.close()


@pytest.fixture
def logger():
    return logging.getLogger("test_liveness")


def _liveness_prefix(scope: str) -> str:
    return f"MOSS/{scope}/test_liveness"


# ══════════════════════════════════════════════════════════════════
# 1. Full query
# ══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_get_liveness_keys_returns_online_tokens(session, scope, logger):
    prefix = _liveness_prefix(scope)
    listener = ZenohLivenessListener(
        liveness_prefix=prefix,
        session=session,
        logger=logger,
    )

    token_a = session.liveliness().declare_token(f"{prefix}/host/default")
    token_b = session.liveliness().declare_token(f"{prefix}/worker/camera")

    try:
        keys = await listener.get_liveness_keys_async()
        assert "host/default" in keys
        assert "worker/camera" in keys
    finally:
        token_a.undeclare()
        token_b.undeclare()


@pytest.mark.asyncio
async def test_get_liveness_keys_returns_empty_when_none(session, scope, logger):
    prefix = _liveness_prefix(scope)
    listener = ZenohLivenessListener(
        liveness_prefix=prefix,
        session=session,
        logger=logger,
    )
    keys = await listener.get_liveness_keys_async()
    assert keys == []


# ══════════════════════════════════════════════════════════════════
# 2. Initial full query on __aenter__
# ══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_enter_seeds_live_keys_from_full_query(session, scope, logger):
    prefix = _liveness_prefix(scope)
    listener = ZenohLivenessListener(
        liveness_prefix=prefix,
        session=session,
        logger=logger,
    )

    token = session.liveliness().declare_token(f"{prefix}/host/default")

    try:
        async with listener:
            keys = listener.live_keys
            assert "host/default" in keys
    finally:
        token.undeclare()


# ══════════════════════════════════════════════════════════════════
# 3. Event-driven update (PUT / DELETE)
# ══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_subscriber_detects_new_token(session, scope, logger):
    prefix = _liveness_prefix(scope)
    listener = ZenohLivenessListener(
        liveness_prefix=prefix,
        session=session,
        logger=logger,
    )

    async with listener:
        token = session.liveliness().declare_token(f"{prefix}/worker/new_one")
        try:
            await asyncio.sleep(0.3)
            assert "worker/new_one" in listener.live_keys
        finally:
            token.undeclare()


@pytest.mark.asyncio
async def test_subscriber_detects_delete(session, scope, logger):
    prefix = _liveness_prefix(scope)
    listener = ZenohLivenessListener(
        liveness_prefix=prefix,
        session=session,
        logger=logger,
    )

    token = session.liveliness().declare_token(f"{prefix}/worker/temp")

    async with listener:
        await asyncio.sleep(0.3)
        assert "worker/temp" in listener.live_keys

        token.undeclare()
        await asyncio.sleep(0.3)
        assert "worker/temp" not in listener.live_keys


# ══════════════════════════════════════════════════════════════════
# 4. on_online / on_offline callbacks
# ══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_on_online_callback_fires(session, scope, logger):
    prefix = _liveness_prefix(scope)
    online_calls: list[str] = []
    offline_calls: list[str] = []

    listener = ZenohLivenessListener(
        liveness_prefix=prefix,
        session=session,
        logger=logger,
        on_online=lambda k: online_calls.append(k),
        on_offline=lambda k: offline_calls.append(k),
    )

    token = session.liveliness().declare_token(f"{prefix}/worker/cb_test")
    _token_undeclared = False

    try:
        async with listener:
            await asyncio.sleep(0.3)
            assert "worker/cb_test" in online_calls

            token.undeclare()
            _token_undeclared = True
            await asyncio.sleep(0.3)
            assert "worker/cb_test" in offline_calls
    finally:
        if not _token_undeclared:
            try:
                token.undeclare()
            except RuntimeError:
                pass


@pytest.mark.asyncio
async def test_callbacks_optional_no_crash(session, scope, logger):
    prefix = _liveness_prefix(scope)
    listener = ZenohLivenessListener(
        liveness_prefix=prefix,
        session=session,
        logger=logger,
    )

    token = session.liveliness().declare_token(f"{prefix}/worker/no_cb")
    _undeclared = False

    try:
        async with listener:
            await asyncio.sleep(0.3)
            assert "worker/no_cb" in listener.live_keys
            token.undeclare()
            _undeclared = True
            await asyncio.sleep(0.3)
            assert "worker/no_cb" not in listener.live_keys
    finally:
        if not _undeclared:
            try:
                token.undeclare()
            except RuntimeError:
                pass


# ══════════════════════════════════════════════════════════════════
# 5. Reconcile loop
# ══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_reconcile_adds_missed_keys(session, scope, logger):
    """reconcile 发现 subscriber 漏掉的 key 时补回缓存并触发 on_online。"""
    prefix = _liveness_prefix(scope)
    reconcile_calls: list[str] = []

    listener = ZenohLivenessListener(
        liveness_prefix=prefix,
        session=session,
        logger=logger,
        on_online=lambda k: reconcile_calls.append(k),
        reconcile_interval=0.5,
    )

    token = session.liveliness().declare_token(f"{prefix}/worker/reconcile_add")

    try:
        async with listener:
            # 手动从缓存中移除，模拟 subscriber 漏了事件
            listener._live_keys.discard("worker/reconcile_add")

            # 等待 reconcile loop 触发
            await asyncio.sleep(1.0)
            assert "worker/reconcile_add" in listener.live_keys
            assert "worker/reconcile_add" in reconcile_calls
    finally:
        token.undeclare()


@pytest.mark.asyncio
async def test_reconcile_removes_stale_keys(session, scope, logger):
    """reconcile 发现缓存中有但实际已下线的 key 时移除并触发 on_offline。"""
    prefix = _liveness_prefix(scope)
    offline_calls: list[str] = []

    listener = ZenohLivenessListener(
        liveness_prefix=prefix,
        session=session,
        logger=logger,
        on_offline=lambda k: offline_calls.append(k),
        reconcile_interval=0.5,
    )

    async with listener:
        # 手动注入一个不在线的 key
        listener._live_keys.add("worker/ghost_key")

        await asyncio.sleep(1.0)
        assert "worker/ghost_key" not in listener.live_keys
        assert "worker/ghost_key" in offline_calls


# ══════════════════════════════════════════════════════════════════
# 6. Lifecycle
# ══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_exit_clears_cache(session, scope, logger):
    prefix = _liveness_prefix(scope)
    listener = ZenohLivenessListener(
        liveness_prefix=prefix,
        session=session,
        logger=logger,
    )

    token = session.liveliness().declare_token(f"{prefix}/host/default")

    try:
        async with listener:
            await asyncio.sleep(0.3)
            assert len(listener.live_keys) > 0

        # 退出后缓存被清空
        assert listener.live_keys == []
    finally:
        token.undeclare()


@pytest.mark.asyncio
async def test_enter_twice_is_idempotent(session, scope, logger):
    prefix = _liveness_prefix(scope)
    listener = ZenohLivenessListener(
        liveness_prefix=prefix,
        session=session,
        logger=logger,
    )

    async with listener:
        async with listener:
            # 重复进入不应报错
            pass


# ══════════════════════════════════════════════════════════════════
# 7. Multi-token
# ══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_multiple_tokens_independent(session, scope, logger):
    prefix = _liveness_prefix(scope)
    listener = ZenohLivenessListener(
        liveness_prefix=prefix,
        session=session,
        logger=logger,
    )

    token_a = session.liveliness().declare_token(f"{prefix}/a")
    token_b = session.liveliness().declare_token(f"{prefix}/b")
    _a_undeclared = False

    try:
        async with listener:
            await asyncio.sleep(0.3)
            assert "a" in listener.live_keys
            assert "b" in listener.live_keys

            token_a.undeclare()
            _a_undeclared = True
            await asyncio.sleep(0.3)
            assert "a" not in listener.live_keys
            assert "b" in listener.live_keys
    finally:
        if not _a_undeclared:
            try:
                token_a.undeclare()
            except Exception:
                pass
        try:
            token_b.undeclare()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════
# 8. key extraction correctness
# ══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_keys_with_slashes_preserved(session, scope, logger):
    """bridge address 含多层 / 时完整保留。"""
    prefix = _liveness_prefix(scope)
    listener = ZenohLivenessListener(
        liveness_prefix=prefix,
        session=session,
        logger=logger,
    )

    bridge = "host/default/01KVG93A5P3ZABCDEFGH"
    token = session.liveliness().declare_token(f"{prefix}/{bridge}")

    try:
        async with listener:
            await asyncio.sleep(0.3)
            assert bridge in listener.live_keys
    finally:
        token.undeclare()