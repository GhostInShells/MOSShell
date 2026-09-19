"""DshRuntime 行为证据 — 代码驱动 (编译/注入/await) + 后台 task + 生命周期 cancel.

run_code 是 Compiler 编译 module 沙盒 + 提取 ``async def run(surface)`` + 注入
现场 surface + await 的最小面; run_code_bg 挂 task, 完成经 on_background_done 外发;
close 先 cancel 后台 task 再逐层关 sessions/connection. 测试用 fake connection /
fake surface, 不碰网络.
"""

import asyncio

import pytest

from ghoshell_moss.deepseek_harness.runtime import DshRuntime


class _FakeSurface:
    async def hello(self) -> str:
        return "hi-from-surface"


class _FakeSession:
    def __init__(self, session_id: str = "") -> None:
        self.session_id = session_id
        self.entered = False
        self.closed = False

    async def __aenter__(self) -> "_FakeSession":
        self.entered = True
        return self

    async def __aexit__(self, *exc) -> None:
        self.closed = True

    async def close(self) -> None:
        self.closed = True


class _FakeConnection:
    def __init__(self) -> None:
        self.entered = False
        self.closed = False
        self.sessions: dict[str, _FakeSession] = {}

    async def __aenter__(self) -> "_FakeConnection":
        self.entered = True
        return self

    async def __aexit__(self, *exc) -> None:
        self.closed = True

    def create_session(self, session_id: str) -> _FakeSession:
        session = _FakeSession(session_id)
        self.sessions[session_id] = session
        return session


def _runtime(on_background_done=None) -> DshRuntime:
    return DshRuntime(_FakeConnection(), on_background_done=on_background_done)


# ---- run_code ---- #


@pytest.mark.asyncio
async def test_run_code_injects_surface_and_returns_result():
    rt = _runtime()
    result = await rt.run_code(
        _FakeSurface(),
        "async def run(surface):\n    return await surface.hello()",
    )
    assert "hi-from-surface" in result


@pytest.mark.asyncio
async def test_run_code_captures_stdout():
    rt = _runtime()
    result = await rt.run_code(
        _FakeSurface(),
        "async def run(surface):\n    print('hello stdout')",
    )
    assert "hello stdout" in result


@pytest.mark.asyncio
async def test_run_code_compile_error():
    rt = _runtime()
    result = await rt.run_code(_FakeSurface(), "async def run(surface:\n    pass")
    assert "COMPILE ERROR" in result


@pytest.mark.asyncio
async def test_run_code_missing_run():
    rt = _runtime()
    result = await rt.run_code(_FakeSurface(), "x = 1")
    assert "COMPILE ERROR" in result
    assert "run" in result


@pytest.mark.asyncio
async def test_run_code_sync_run_rejected():
    rt = _runtime()
    result = await rt.run_code(_FakeSurface(), "def run(surface):\n    return 1")
    assert "must be `async def`" in result


@pytest.mark.asyncio
async def test_run_code_run_error_reports_model_traceback():
    rt = _runtime()

    async def _boom(surface):
        raise ValueError("kaboom")

    result = await rt.run_code(_boom, "async def run(surface):\n    raise ValueError('kaboom')")
    assert "RUN ERROR" in result
    assert "ValueError" in result


# ---- run_code_bg ---- #


@pytest.mark.asyncio
async def test_run_code_bg_queues_and_calls_back_with_next():
    done: list[tuple[str, bool]] = []
    event = asyncio.Event()

    def cb(result: str, next_flag: bool) -> None:
        done.append((result, next_flag))
        event.set()

    rt = _runtime(on_background_done=cb)
    rid = rt.run_code_bg(_FakeSurface(), "async def run(surface):\n    return 'bg'", next=True)
    assert "queued" in rid
    await asyncio.wait_for(event.wait(), 1)
    assert len(done) == 1
    assert "bg" in done[0][0]
    assert done[0][1] is True


# ---- lifecycle ---- #


@pytest.mark.asyncio
async def test_close_enters_then_closes_connection():
    conn = _FakeConnection()
    rt = DshRuntime(conn)
    await rt.__aenter__()
    assert conn.entered
    await rt.close()
    assert conn.closed
    # 幂等.
    await rt.close()


@pytest.mark.asyncio
async def test_close_cancels_background_task():
    called: list[str] = []
    rt = _runtime(on_background_done=lambda result, next_flag: called.append(result))
    rt.run_code_bg(_FakeSurface(), "async def run(surface):\n    await asyncio.sleep(10)")
    await asyncio.sleep(0)  # 让 task 进入 sleep.
    await rt.close()
    assert called == []  # 被 cancel, 完成回调永不触发.


# ---- sessions ---- #


@pytest.mark.asyncio
async def test_open_session_enters_and_close_session_closes():
    conn = _FakeConnection()
    rt = DshRuntime(conn)
    surface = await rt.open_session("s1")
    assert surface is not None
    assert conn.sessions["s1"].entered  # session 被 enter (起消费 task).
    assert rt.session_ids() == ["s1"]
    # 重复 open 返回同一 surface.
    assert await rt.open_session("s1") is surface
    await rt.close_session("s1")
    assert conn.sessions["s1"].closed
    assert rt.session_ids() == []


@pytest.mark.asyncio
async def test_concurrent_run_code_stdout_isolated():
    """并发 run_code 各用自己的 print buffer, 不串线 (bug 2)."""
    rt = _runtime()

    async def _run(tag: str) -> str:
        return await rt.run_code(
            _FakeSurface(),
            f"async def run(surface):\n    print('{tag}')\n    await asyncio.sleep(0.02)\n    print('{tag}-2')",
        )

    r1, r2 = await asyncio.gather(_run("A"), _run("B"))
    assert "A" in r1 and "A-2" in r1 and "B" not in r1
    assert "B" in r2 and "B-2" in r2 and "A" not in r2


@pytest.mark.asyncio
async def test_close_session_cancels_inflight_background_task():
    """close_session 会 cancel 该 session 的在飞后台 task (bug 3)."""
    called: list[str] = []
    rt = DshRuntime(_FakeConnection(), on_background_done=lambda result, next_flag: called.append(result))
    surface = await rt.open_session("s1")
    rt.run_code_bg(surface, "async def run(session):\n    await asyncio.sleep(10)")
    await asyncio.sleep(0)  # 让 task 进入 sleep.
    await rt.close_session("s1")
    await asyncio.sleep(0.05)
    assert called == []  # 被 cancel, 完成回调永不触发.
