"""SafeMode 离体单测 — 覆盖 TUI/拦截双面契约, 不接 ghost_runtime.

测试纪律 (对齐 test_pause_controller.py):
    - 不 mock/monkeypatch, 只测公开 API 效果.
    - 只留能揭露语义 bug 的用例; 不测 setter 记住状态这类无信息量细节.
"""

import threading

import pytest
from ghoshell_moss.core.blueprint.host import Verdict
from ghoshell_moss.host.safe_mode import SafeModeImpl


# ── enabled 语义: 开关只影响下一轮, 不动在途 ──────────────────


def test_set_enabled_returns_change_flag():
    """set_enabled 返回 True=变更, False=幂等 — 供 TUI 判断是否需要 invalidate."""
    sm = SafeModeImpl()
    assert sm.set_enabled(True) is True
    assert sm.set_enabled(True) is False
    assert sm.set_enabled(False) is True


def test_disable_does_not_resolve_in_flight_pending():
    """关开关不结算已挂 pending — 决策 2: 开关只影响下一轮判定, 不动在途."""
    sm = SafeModeImpl()
    sm.set_enabled(True)
    fut = sm.submit("hi")
    sm.set_enabled(False)

    assert not fut.done()
    assert sm.pending() is not None


# ── submit → pending 快照 ────────────────────────────────────


def test_submit_writes_pending_with_logos():
    sm = SafeModeImpl()
    fut = sm.submit("hello world")
    p = sm.pending()
    assert p is not None
    assert p['logos'] == "hello world"
    assert p['uuid']  # 非空
    assert not fut.done()


def test_submit_while_pending_raises():
    """任意时刻至多一个 pending — articulate loop 串行不该并发 submit."""
    sm = SafeModeImpl()
    sm.submit("first")
    with pytest.raises(RuntimeError):
        sm.submit("second")


# ── approve / reject 结算 ────────────────────────────────────


def test_approve_resolves_future_and_clears_pending():
    sm = SafeModeImpl()
    fut = sm.submit("logos")
    uuid = sm.pending()['uuid']

    assert sm.approve(uuid) is True
    assert fut.result(timeout=0.1) == Verdict(kind='approved')
    assert sm.pending() is None


def test_approve_with_note_carries_message():
    """approve(uuid, note='...') 把 note 装入 Verdict.message, 供拦截点 observe 用."""
    sm = SafeModeImpl()
    fut = sm.submit("logos")
    uuid = sm.pending()['uuid']

    assert sm.approve(uuid, note="looks good but tone down") is True
    assert fut.result(timeout=0.1) == Verdict(kind='approved', message="looks good but tone down")


def test_reject_carries_reason():
    sm = SafeModeImpl()
    fut = sm.submit("logos")
    uuid = sm.pending()['uuid']

    assert sm.reject(uuid, "too risky") is True
    assert fut.result(timeout=0.1) == Verdict(kind='rejected', message="too risky")
    assert sm.pending() is None


# ── stale uuid 静默 no-op (决策 8) ────────────────────────────


def test_stale_uuid_approve_is_noop():
    """错帧 uuid 不结算; pending 保留等正确 uuid."""
    sm = SafeModeImpl()
    fut = sm.submit("logos")

    assert sm.approve("bogus-uuid") is False
    assert not fut.done()
    assert sm.pending() is not None


def test_approve_after_resolved_is_noop():
    """结算过的 uuid 再次 approve 无效, 不影响下一轮."""
    sm = SafeModeImpl()
    fut = sm.submit("logos")
    uuid = sm.pending()['uuid']

    sm.approve(uuid)
    assert sm.approve(uuid) is False
    assert fut.result(timeout=0.1) == Verdict(kind='approved')


def test_approve_with_no_pending_is_noop():
    sm = SafeModeImpl()
    assert sm.approve("any") is False


# ── cancel_current: 拦截点 finally 用 (决策 4) ────────────────


def test_cancel_current_resolves_as_cancelled():
    sm = SafeModeImpl()
    fut = sm.submit("logos")

    assert sm.cancel_current() is True
    assert fut.result(timeout=0.1) == Verdict(kind='cancelled')
    assert sm.pending() is None


def test_cancel_current_without_pending_is_noop():
    """幂等: finally 里可能已被 approve/reject 结算过."""
    sm = SafeModeImpl()
    assert sm.cancel_current() is False


# ── 并发裁决安全 (uuid 竞态) ────────────────────────────────


def test_concurrent_approve_only_one_wins():
    """两个线程同时用同一 uuid approve — 只有一个 True, Future 只被 set 一次."""
    sm = SafeModeImpl()
    sm.submit("logos")
    uuid = sm.pending()['uuid']

    barrier = threading.Barrier(2)
    results: list[bool] = []
    lock = threading.Lock()

    def _race():
        barrier.wait()
        r = sm.approve(uuid)
        with lock:
            results.append(r)

    ts = [threading.Thread(target=_race) for _ in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    assert results.count(True) == 1
    assert results.count(False) == 1


# ── on_pending_changed 回调 ─────────────────────────────────


def test_callback_fires_on_submit_and_resolve():
    """TUI toolbar 依赖此回调刷新: submit / approve 都要触发 → 两次 invalidate."""
    sm = SafeModeImpl()
    calls: list[None] = []
    sm.on_pending_changed(lambda: calls.append(None))

    sm.submit("logos")
    uuid = sm.pending()['uuid']
    sm.approve(uuid)

    assert len(calls) == 2
