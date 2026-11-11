"""Nothing fails in the dark.

Answers: wherever an exception is swallowed, a call goes unclaimed, or a stream is opened and
nobody closes it — is there a log line? A degradation is allowed; a *silent* one is not.
"""

import asyncio
import contextlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from ._support import FakeDispatchEgo, FakeRunEgo, FakeRunSession, FakeRunThinking, _NoopShellFacade, dsh_tool_call

class _RecordingHandler(logging.Handler):
    """收 'moss' 下任意 logger 的记录 —— 生产的 logger 名由调用方传入, 传播不可假设."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def messages(self, level: int = logging.NOTSET) -> list[str]:
        return [r.getMessage() for r in self.records if r.levelno >= level]

    def has_traceback(self, level: int = logging.WARNING) -> bool:
        return any(r.exc_info is not None for r in self.records if r.levelno >= level)


@contextlib.contextmanager


def captured_logs(logger: logging.Logger):
    """挂一个 handler 到给定 logger 上收记录; 不改 propagate, 不动 root."""
    handler = _RecordingHandler()
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)


class _NamedLogger(logging.Logger):
    """显式命名的 logger —— 用来断言"日志落在了调用方给的那个 logger 上"."""


class TestSilentFailureSurfaces:
    """静默失败面: 被吞掉的异常 / 无人认领的调用 / 打开的边界没人关 —— 每一条都必须留痕.

    这一组测试针对的不是行为正确性, 而是**可观测性**: 每条断言都问"出错时日志里有没有东西".
    上一轮的教训正是反过来的 —— 行为错了但日志是空的, 于是整晚的推断都建在错误前提上.
    """

    @staticmethod
    def _run(session=None, ego=None, thinking=None, facade=None, logger=None):
        from ghoshell_moss.ghosts.dolores._run import DoloresRun

        session = session or FakeRunSession()
        return DoloresRun(
            ego=ego or FakeRunEgo(session),
            thinking=thinking or FakeRunThinking(),
            thinking_event=asyncio.Event(),
            facade=facade if facade is not None else _NoopShellFacade(),
            logger=logger,
        )

    @pytest.mark.asyncio
    async def test_unknown_moss_tool_call_is_logged_not_silent(self):
        """名字是我们的 (moss*), 却没有 handler 认领 = 这次调用永远等不到回话. 日志必须说话."""
        run = self._run()
        with captured_logs(run._logger) as logs:
            await run._handle_tool_use_event(
                dsh_tool_call("moss_mystery", "{}", "call_unknown"), {},
            )
        assert any("matched no dolores tool" in m and "call_unknown" in m
                   for m in logs.messages(logging.WARNING)), logs.messages()

    @pytest.mark.asyncio
    async def test_foreign_tool_calls_are_ignored_silently(self):
        """host 内环工具 (bash / read / edit / subagent ...) 由 dsh 自己结算 —— 我们连痕都不该留.

        2026-10-04 D04: 早先它们被当成"等我们回答的 tool"逐条 warning, 一次排查刷出几百行噪音,
        把真正的 D39 信号 (我们自己的 tool 没 handler) 埋掉了.
        """
        run = self._run()
        with captured_logs(run._logger) as logs:
            for name in ("bash", "read", "edit", "subagent"):
                await run._handle_tool_use_event(
                    dsh_tool_call(name, "{}", f"call_{name}"), {},
                )
        assert logs.messages(logging.WARNING) == [], logs.messages()

    @pytest.mark.asyncio
    async def test_tool_arguments_parse_failure_is_logged(self):
        """参数解析失败: 模型看得到 error 字段, 但日志里原先一个字都没有."""
        from ghoshell_moss.ghosts.dolores._tools import InterpretToolCall

        event = dsh_tool_call("moss_interpret", "{not json", "call_bad_args")
        with captured_logs(logging.getLogger("moss")) as logs:
            result = await InterpretToolCall.run_tool(event, lambda call: None)
        assert result is not None and result.error == "invalid tool parameter"
        assert any("failed to parse" in m and "moss_interpret" in m
                   for m in logs.messages(logging.WARNING)), logs.messages()

    @pytest.mark.asyncio
    async def test_run_uses_the_logger_handed_down(self):
        """run 不重建 logger: 记录必须落在调用方传进来的那个 logger 上.

        每层再调一次 get_moss_logger() 会把整棵 ghost 的日志从当前 node 的语境里摘出去.
        """
        handed = _NamedLogger("moss.test.handed_down")
        handed.propagate = False
        run = self._run(logger=handed)
        with captured_logs(handed) as logs:
            await run._handle_tool_use_event(
                dsh_tool_call("moss_nope", "{}", "call_x"), {},
            )
        assert logs.messages(logging.WARNING), "记录没有落在传下来的 logger 上"

    @pytest.mark.asyncio
    async def test_leftover_ctml_stream_is_closed_and_logged(self):
        """delta 开了 articulator 而 tool/call 从未到达: 边界必须被关上, 且必须留痕."""
        run = self._run()
        streams: dict = {}
        chunk = SimpleNamespace(type="tool-call-delta", id="call_orphan", name="moss_interpret",
                                argumentsDelta='{"ctml": "<noop/>"}')
        await run._handle_ctml_delta(chunk, streams)
        assert streams, "delta 应已开出流"
        articulator = streams["call_orphan"].articulator
        with captured_logs(run._logger) as logs:
            await run._close_leftover_streams(streams)
        assert articulator.exited == 1, "残留流的 articulator 边界没有被关"
        assert streams == {}
        assert any("never settled" in m and "call_orphan" in m
                   for m in logs.messages(logging.INFO)), logs.messages()

    @pytest.mark.asyncio
    async def test_enter_failure_is_logged_with_traceback(self):
        """enter 失败会把整轮 abort —— 一路都在抛, 但原先没有一处写日志."""
        session = FakeRunSession()
        ego = FakeRunEgo(session)
        ego.enter_error = RuntimeError("plugin refused enter")
        run = self._run(session=session, ego=ego)
        thinking = run._thinking
        with captured_logs(run._logger) as logs:
            async with run:
                await asyncio.sleep(0)
        assert run._enter_error is not None
        assert any("thinking/enter failed" in m for m in logs.messages(logging.ERROR)), logs.messages()
        assert logs.has_traceback(logging.ERROR)
        assert thinking.abort_reasons, "abort 仍应发生"

    @pytest.mark.asyncio
    async def test_corrupt_arguments_stream_is_logged(self):
        """参数流损坏: 解释器根本没见到这段 CTML, 所以 shell 的 ERROR 不会出现 —— 只能在这里留痕."""
        from ghoshell_moss.ghosts.dolores._tools import ToolCallResult

        run = self._run()
        streams: dict = {}
        chunk = SimpleNamespace(type="tool-call-delta", id="call_broken", name="moss_interpret",
                                argumentsDelta='{"ctml": "\\q"}')
        await run._handle_ctml_delta(chunk, streams)
        assert streams["call_broken"].stream.failed
        call_event = dsh_tool_call("moss_interpret", "{}", "call_broken")
        with captured_logs(run._logger) as logs:
            result = await run._handle_interpret(
                SimpleNamespace(tool_call_event=call_event, ctml=""), streams,
            )
        assert isinstance(result, ToolCallResult) and result.result == "ctml parse error"
        assert any("corrupt" in m and "call_broken" in m
                   for m in logs.messages(logging.WARNING)), logs.messages()


class TestMalformedInterpretStreamSettlement:
    """参数解析失败的 ``moss_interpret`` 必须当场了结它由 delta 开出来的 stream.

    这场戏只有几个角色, 一句一句看就懂了.

    ============ 角色 ============

    模型     一个词一个词地说出它要执行的 CTML; 说完了还要正式交一份"菜单"
             (tool/call 的 arguments)。
    delta    模型吐出的每个词 (tool-call-delta)。**MOSS 收到第一个词就把灶点着了。**
    菜单     模型最后交上来的完整 arguments。只有它能解析, 这一单才算真正受理。
    灶       这一单的执行位 (articulator + 它在 mindflow 里排的那个 action)。

    ============ 顺序 (就是线上 17:20:34 的顺序) ============

    1. 模型吐第一个词            → ``_handle_ctml_delta``      → 灶点着了
    2. 模型把 CTML 吐完          → ``_handle_ctml_delta``      → 菜已经在锅里
    3. 模型交上来的菜单是坏的     → ``_handle_tool_use_event``  → 只回一句"参数不对", 然后就结束了

    **第 3 步曾是一个 bug**: 菜单坏了可以不接这一单, 但**灶必须关** —— 不关, 这个执行位就一直占着。
    当时的代码不关: ``_handle_interpret`` 里的 ``streams.pop`` / ``__aexit__`` 只在 handler 被调用时
    才发生, 而参数解析失败这一支根本没走到 handler (2026-10-13 线上整帧冻结的触发点)。
    现在由 ``DoloresRun._void_stream`` 在 ``_handle_tool_use_event`` 里当场作废。

    这里立的是**两条不变量**: (1) 必须关, 且只能关一次; (2) 必须走"异常退出"那一支 ——
    见本文件末尾对 ``exit_args`` 的断言。

    ============ 不关灶会怎样, 不在这里断言 ============

    "灶一直被占着 → 下一位客人永远轮不到 → 整帧冻结" 是 **kernel 的鲁棒性**问题, 不由本文件断言。
    那条在 ``tests/ghoshell_moss/default/core/mindflow/test_abandoned_action.py``: 用生产装线
    (真三循环) 复现, 不含任何替身 —— 两层各自一条, 互不替代。

    ============ 为什么"坏菜单"还能把灶点着 ============

    因为切词的人不需要菜谱的句号。``_ctml_stream.py`` 的注释写得很直白:
    "The closing quote needs no prediction"。所以 ``{"ctml": "<noop/>`` 这种
    "开了头、没收尾"的坏串, 前半段照样能把 ``<noop/>`` 解出来下锅 —— 这就是线上那条的形状
    (moss.log 里 pydantic 报的正是 ``EOF while parsing a string``)。
    """

    #: 线上那条坏菜单的最小等价物: 开了头, 永远不收尾。
    _BROKEN_ARGS = '{"ctml": "<noop/>'
    #: 同一个字符串的逐词分片 —— 模型是先吐词、最后才交菜单的。
    _BROKEN_DELTAS = ('{"ctml": "', '<noop/>')

    @staticmethod
    def _run(thinking):
        from ghoshell_moss.ghosts.dolores._run import DoloresRun

        session = FakeRunSession()
        return DoloresRun(
            ego=FakeDispatchEgo(session),
            thinking=thinking,
            thinking_event=asyncio.Event(),
            facade=_NoopShellFacade(),
        )

    @staticmethod
    def _delta(call_id: str, arguments_delta: str):
        return SimpleNamespace(type="tool-call-delta", id=call_id, name="moss_interpret",
                               argumentsDelta=arguments_delta)

    @classmethod
    async def _stream_arguments(cls, run, call_id: str, pieces: tuple, streams: dict) -> None:
        """按碎片顺序喂一条 arguments 的 argumentsDelta 流 (dsh 就是逐 token 推的)."""
        for piece in pieces:
            await run._handle_ctml_delta(cls._delta(call_id, piece), streams)

    @pytest.mark.asyncio
    async def test_malformed_interpret_settles_its_delta_stream(self):
        """只问第 3 步: 菜单坏了、这一单不接了, 灶关了没有?"""
        run = self._run(FakeRunThinking())
        streams: dict = {}          # 场上"还开着的灶"

        # 模型吐词 (第 1、2 步): 第一个词就该点着灶, 后面的词是菜。
        await self._stream_arguments(run, "call_bad", self._BROKEN_DELTAS, streams)
        stream = streams["call_bad"]
        assert stream.articulator.entered == 1, "第一个词到了, 灶就该点着"
        assert stream.stream.matched, (
            "词本身没问题: 它按 {\"ctml\": \"… 前缀匹配、正常解码 —— 坏的只是最后交上来的那份菜单"
        )
        assert not stream.stream.failed, "参数流没有损坏 (损坏是另一条分支: stream.failed)"
        assert stream.stream.value == "<noop/>", "菜已经在锅里了"

        # 模型交菜单 (第 3 步): 同一份内容拼出来的串是坏 JSON, 所以 MOSS 只回一句"参数不对"。
        await run._handle_tool_use_event(
            dsh_tool_call("moss_interpret", self._BROKEN_ARGS, "call_bad"), streams,
        )

        # 这一单可以不接, 但灶必须关。
        assert "call_bad" not in streams, "灶还留在场上: 没有任何人会再来关它"
        assert stream.articulator.exited == 1, "灶没关 (articulator 的边界没有收尾)"

        # 收尾的**方式**也是契约的一部分: 必须走"异常退出"那一支, 不能 commit。
        #   - commit (exc_val is None) 会让解释器执行这份半截 CTML —— 而它的参数刚被拒;
        #   - 但如果传 ActionExitedException / StatementExitedException, kernel 走的是
        #     "attention 已经 abort" 那一支, 什么都不做 —— 灶照样占着, 等于没修。
        # 只有普通异常能落 action 的 stop event、放行 action loop, 又不惊动整帧。
        # 机制的生产验证在 tests/ghoshell_moss/default/core/mindflow/test_abandoned_action.py。
        from ghoshell_moss.core.blueprint.mindflow import StatementExitedException

        exit_args = stream.articulator.exit_args
        assert exit_args is not None, "边界没被收尾"
        assert exit_args[1] is not None, "commit 收尾会执行这份参数刚被拒的半截 CTML"
        assert not isinstance(exit_args[1], StatementExitedException), (
            "StatementExitedException 那一支假定 attention 已 abort, 在这里什么都不做 —— "
            "action 不会被放行, 卡死照旧"
        )


class TestEgoSilentFailureSurfaces:
    """ego 侧同一纪律: 降级路径与"阈值说要落锚但什么都没落"都必须留痕."""

    @pytest.mark.asyncio
    async def test_commit_without_manager_is_logged(self):
        from ghoshell_moss.ghosts.dolores._ego import DoloresEgo, DoloresEgoConfig, DoloresEgoContext

        ego = DoloresEgo(
            launcher=None,
            ctx=DoloresEgoContext(
                project_home=Path("/tmp"), project_name="t", name="t", mode="t",
                instruction="", facade=None,
            ),
            config=DoloresEgoConfig(),
            memento_manager=None,
        )
        with captured_logs(ego._logger) as logs:
            assert ego._commit() is None
        assert any("nothing anchored" in m for m in logs.messages(logging.WARNING)), logs.messages()

    @pytest.mark.asyncio
    async def test_missing_usage_warns_once(self):
        """usage 一直缺失 = 自动落锚整体静默失效. 报一次, 不逐帧刷屏."""
        from ghoshell_moss.deepseek_harness.v0_2_0 import SessionEvent, SessionEventMeta

        from ghoshell_moss.ghosts.dolores._ego import DoloresEgo, DoloresEgoConfig, DoloresEgoContext

        ego = DoloresEgo(
            launcher=None,
            ctx=DoloresEgoContext(
                project_home=Path("/tmp"), project_name="t", name="t", mode="t",
                instruction="", facade=None,
            ),
            config=DoloresEgoConfig(),
            memento_manager=object(),  # 只要非 None: 走到 usage 判定即可
        )
        event = SessionEvent(
            meta=SessionEventMeta(type="assistant/message", seq=1),
            data={"turn": 1, "step": 1, "message": {"role": "assistant"}},
        )
        with captured_logs(ego._logger) as logs:
            await ego._on_assistant_message(event)
            await ego._on_assistant_message(event)
        warned = [m for m in logs.messages(logging.WARNING) if "no usage" in m]
        assert len(warned) == 1, f"应只报一次, 实得 {len(warned)}"

    @pytest.mark.asyncio
    async def test_self_wake_without_broadcast_warns_once(self):
        """没接广播 = 自醒整体哑掉. 一次告警把它从"静默失效"变成"看得见的失效"."""
        from ghoshell_moss.ghosts.dolores._ego import DoloresEgo, DoloresEgoConfig, DoloresEgoContext

        ego = DoloresEgo(
            launcher=None,
            ctx=DoloresEgoContext(
                project_home=Path("/tmp"), project_name="t", name="t", mode="t",
                instruction="", facade=None,
            ),
            config=DoloresEgoConfig(),
        )
        with captured_logs(ego._logger) as logs:
            ego._emit_self_wake()
            ego._emit_self_wake()
        warned = [m for m in logs.messages(logging.WARNING) if "self-wake dropped" in m]
        assert len(warned) == 1, f"应只报一次, 实得 {len(warned)}"
