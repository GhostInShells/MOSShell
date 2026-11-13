"""Shared stand-ins for the Dolores ghost tests.

Three sections, all purely mechanical:

1. **constructors** — ``Dolores`` / ``DoloresMeta`` built with every dependency defaulted,
   plus the patch loader that lets a ``!!js`` tag pass through as a plain scalar.
2. **the dsh event vocabulary** — the two shapes a session event takes on the wire. They are
   module functions, not members of one test class, so a second file can drive the same
   surface without importing a test.
3. **the ``DoloresRun`` harness** — fakes for the runtime's dependencies
   (session / ego / thinking / facade). ``DoloresRun`` talks to each of them through a narrow
   duck-typed face, so a few small classes are enough to drive its lifecycle without a dsh.
"""

import asyncio
import logging
from types import SimpleNamespace

import pytest
import yaml

class _PatchLoader(yaml.SafeLoader):
    """dsh patches may carry `!!js` expressions; reading them is not this test's business."""


_PatchLoader.add_multi_constructor(
    "tag:yaml.org,2002:js", lambda loader, suffix, node: loader.construct_scalar(node)
)


def _load_patch(text: str) -> list[dict]:
    doc = yaml.load(text, Loader=_PatchLoader) or []
    return [item for item in doc if isinstance(item, dict)]


def _dolores_meta(**kwargs):
    from ghoshell_moss.ghosts.dolores._meta import DoloresMeta

    return DoloresMeta(**kwargs)


def _dolores(meta=None, *, home=None, session=None, matrix=None, shell=None, base_instruction=None):
    from ghoshell_moss.ghosts.dolores._runtime import Dolores

    return Dolores(
        meta=meta or _dolores_meta(),
        home=home,
        session=session,
        matrix=matrix,
        shell=shell,
        base_instruction=base_instruction,
    )


class _OfflineConnection:
    """A connection that exists but can reach nothing — the shape a read-only test needs.

    Not ``None``: the runtime always holds a connection object (the dsh launcher), so a test
    that passes ``None`` builds a state production never builds. That mismatch is what once
    left a note task dying with ``AttributeError`` in the background, invisible until GC.
    """

    def __init__(self):
        self.attached: list[str] = []

    def attach_session(self, session_id: str, logger=None, *, follow: bool = True):
        self.attached.append(session_id)
        return _OfflineSession()


class _OfflineSession:
    """One handle whose only call fails at once — the sidecar's paved "unavailable" path."""

    async def chat_off_record(self, messages, *, anchor=None, **kwargs):
        raise RuntimeError("offline connection in tests")

    async def close(self) -> None:
        return None


# --- dsh event vocabulary ----------------------------------------------------
# One frame of a run is a stream of these. The constructor lives here rather than on a test class
# because more than one file drives the run surface; a test importing another test is a coupling
# that only shows up when the files move apart.

def dsh_event(event_type: str, data: dict, seq: int = 1):
    from ghoshell_moss.deepseek_harness.v0_2_0 import SessionEvent, SessionEventMeta

    return SessionEvent(meta=SessionEventMeta(type=event_type, seq=seq), data=data)


def dsh_tool_call(name: str, arguments: str = "{}", call_id: str = "call_x"):
    from ghoshell_moss.deepseek_harness.v0_2_0 import ToolCallEvent

    return ToolCallEvent(callId=call_id, name=name, arguments=arguments)


class FakeRunSession:
    """DoloresRun 的事件源 fake — on_session_event 注册/解绑 + 手动 emit."""

    def __init__(self):
        self.handlers: list = []

    def on_session_event(self, event_type, callback):
        self.handlers.append((event_type, callback))

        def dispose():
            self.handlers.remove((event_type, callback))

        return dispose

    async def emit(self, event):
        for _, cb in list(self.handlers):
            await cb(event)


class FakeRunEgo:
    """DoloresRun 的 ego fake — Duck-typed (session/enter_thinking/exit_thinking)."""

    def __init__(self, session):
        self.session = session
        # production ego exposes .logger (run 的 __aenter__ 用它记进入日志).
        self.logger = logging.getLogger("pytest.dolores")
        self.enter_calls = 0
        self.exit_calls = 0
        self.enter_error: Exception | None = None
        self.run_exits: list[bool] = []
        # 竞速赢了之后那一帧必须真的落地 —— 记 id 就够断言.
        self.pushed_moments: list[str] = []

    async def push_moment(self, moment, moment_id):
        self.pushed_moments.append(moment_id)

    async def probe_pending_moment_and_steer(self):
        # settle 2a 探针 fake: 默认无未消费 pending moment (settle 落进 2b). 用例按需覆盖.
        self.probe_calls = getattr(self, "probe_calls", 0) + 1
        return False

    def next_thinking_seq(self):
        self.enter_seq = getattr(self, 'enter_seq', 0) + 1
        return self.enter_seq

    async def enter_thinking(self, thinking, thinking_seq):
        self.enter_calls += 1
        if self.enter_error is not None:
            raise self.enter_error

    async def exit_thinking(self, thinking_seq):
        self.exit_calls += 1

    def on_run_exit(self, valid: bool):
        self.run_exits.append(valid)

    def declare_effort(self, effort):
        self.default_thinking_effort = effort
        return f"declared depth {effort}"


class FakeArticulator:
    """articulator 的 fake — send 累积 logos, 生命周期空操作; interpret_error 可注入."""

    def __init__(self, log: list | None = None):
        self.sent: list[str] = []
        self.interpret_error: Exception | None = None
        self._log = log
        # 生命周期痕迹: 展开是否发生, 是回归测试的核心判据 (见 test_ctml_append_opens_lifecycle).
        self.entered = 0
        self.exited = 0
        # 退出时收到的 (exc_type, exc_val, exc_tb) —— 判"是 commit 收尾还是异常退出",
        # 这两条在 kernel 里换来完全不同的结果 (见 TestMalformedInterpretStreamSettlement).
        self.exit_args: tuple | None = None

    async def __aenter__(self):
        self.entered += 1
        return self

    async def __aexit__(self, *args):
        self.exited += 1
        self.exit_args = args
        return None

    async def send(self, delta: str):
        self.sent.append(delta)

    async def wait_compiled(self, raise_interpret_error: bool = False):
        if raise_interpret_error and self.interpret_error is not None:
            raise self.interpret_error

    async def wait_observed(self, raise_interpret_error: bool = False):
        if raise_interpret_error and self.interpret_error is not None:
            raise self.interpret_error
        if self._log is not None:
            self._log.append("wait_observed")

    async def wait_action_done(self):
        if self._log is not None:
            self._log.append("wait_action_done")


class _FakeEpoch:
    index = 3


class _FakeObserver:
    """镜像生产的 Observer —— ``add_echoes`` 在**观测者**上, 不在 Thinking 上.

    这不是风格问题: 替身曾经在 Thinking 上发明了一个生产不存在的 ``add_echoes``,
    于是单测全绿而生产的 moss_reasoning 抛 AttributeError. 见 surface 对照测试.
    """

    epoch = _FakeEpoch()

    def __init__(self, log: list | None = None):
        self._log = log
        self._pending: list = []
        self._injected_callbacks: set = set()

    def add_echoes(self, result: list, need_observe: bool = False):
        if self._log is not None:
            self._log.append(f"observer.add_echoes:{need_observe}")

    def need_observe(self) -> bool:
        return False

    # Observer 契约的另外三件. 替身要长成真 observer 的形状 —— 否则单测全绿、
    # 生产 AttributeError (这个类的 docstring 记的就是那次事故).
    def has_pending(self) -> bool:
        return bool(self._pending)

    def drain_percepts(self) -> list:
        drained, self._pending = self._pending, []
        return drained

    def observe(self):
        """签发一帧 —— 生产里 observe 是唯一生成 Moment 的地方 (收轮工具的竞速/push 都走它)."""
        from ghoshell_moss.core.blueprint.moment import Moment

        self.observed_count = getattr(self, "observed_count", 0) + 1
        return Moment(id=f"fake-observed-{self.observed_count}", index=7 + self.observed_count)

    def on_percepts_injected(self, callback):
        self._injected_callbacks.add(callback)

        def _disposer():
            self._injected_callbacks.discard(callback)

        return _disposer


class FakeRunThinking:
    def __init__(self, log: list | None = None):
        from ghoshell_moss.core.blueprint.moment import Moment

        self.abort_reasons: list = []
        self.articulators: list[FakeArticulator] = []
        self._log = log
        # run 的进入日志读 attention/moment —— 与生产 Thinking 同名.
        # escalate: 每个 ghost event 的提权落点 (见 _escalate_on_ghost_event), 跑到这里记账.
        self.attention = SimpleNamespace(id="fake-attention", escalate=self._escalate)
        self.moment = Moment(id="fake-moment", index=7)

    def _escalate(self):
        if self._log is not None:
            self._log.append("escalate")

    def abort(self, reason):
        self.abort_reasons.append(reason)
        if self._log is not None:
            self._log.append("abort")

    def articulator(self, replan=False, wait_action_done=False) -> FakeArticulator:
        art = FakeArticulator(self._log)
        self.articulators.append(art)
        return art

    # interpret 的尾巴经这一步: 等 action 停 (含解释器关闭与轨迹落盘) + 签发最新 moment.
    async def wait_actions_done(self):
        if self._log is not None:
            self._log.append("wait_actions_done")

    def observe(self):
        from ghoshell_moss.core.blueprint.moment import Moment

        self.observed = Moment(id="fake-moment", index=7)
        return self.observed

    @property
    def observer(self):
        return _FakeObserver(self._log)


class _GatedThinking(FakeRunThinking):
    """``wait_actions_done`` 一直挂着, 直到测试放行 —— 用来量"等身体"和"人插话"的竞速.

    ``reached`` 在进入等待时就置位, 测试据此确定"接收轮工具已经停在竞速点上了"再插话;
    放行 ``gate`` 就是"身体跑完了".
    """

    def __init__(self):
        super().__init__()
        self.reached = asyncio.Event()
        self.gate = asyncio.Event()

    async def wait_actions_done(self):
        self.reached.set()
        await self.gate.wait()


class _NoopShellFacade:
    """最小 facade —— 只满足 handler 对 ``facade.shell.refresh_metas`` 的依赖.

    生产里 facade 在 thinking 之前就已接线; 测试里用它把"不关心暖层"的用例保持在无关状态.
    """

    class _Shell:
        async def refresh_metas(self, timeout=None, stale_time=None):
            pass

    def __init__(self):
        self.shell = _NoopShellFacade._Shell()


def fake_tool_call(call_id: str = "call_00_test"):
    """_dispatch_tool_result 只用 callId, 其余字段走默认."""
    from ghoshell_moss.deepseek_harness.v0_2_0 import ToolCallEvent

    return ToolCallEvent(callId=call_id)


class FakeDispatchEgo(FakeRunEgo):
    """_dispatch_tool_result 的 ego fake — 记录 RPC 调用, 可注入失败."""

    def __init__(self, session=None, *, raise_on_rpc: Exception | None = None):
        super().__init__(session)
        self.rpc_calls: list[tuple] = []
        self.rpc_turns: list = []
        self.raise_on_rpc = raise_on_rpc

    async def rpc_tool_result(self, call_id, result, moment=None, cancel=False, turn=None):
        if self.raise_on_rpc is not None:
            raise self.raise_on_rpc
        self.rpc_calls.append((call_id, result, moment, cancel))
        self.rpc_turns.append(turn)

    @property
    def model_route(self):
        """(provider, model) 的真实路由 — 生产里读最后一次 request/header, 这里给固定值.

        有了它, ``_push_effort_officially`` 才不会退到 ``session.model_selection()`` 的兜底
        (那条兜底会读 catalog 默认值, 生产里是被刻意避开的路径).
        """
        return ("deepseek-official", "deepseek-flash")

    def moment_payload(self, moment, moment_id):
        # 两条投递路共用同一种协议 — fake 也按同一形状给 (context + inputs + moment_id).
        return {
            "context": [{"type": "text", "text": f"moment:{moment_id}"}],
            "inputs": [],
            "moment_id": moment_id,
        }
