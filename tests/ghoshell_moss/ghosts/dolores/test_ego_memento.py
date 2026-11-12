"""The memory backend behind the ego — anchors, sideside notes, and the read surface.

Answers: what a span *is* once it is committed — half-open ``(start_turn, end_turn]``,
what fills a missing note, where compaction may safely cut, and what the read/chat face
hands back. All of it runs outside dsh, on a fake connection.
"""

import asyncio
import logging
from datetime import timedelta
from pathlib import Path

import pytest

class TestEgoMementoSidecar:
    """Note task — one off-record round per commit, with the mindfork container.

    Three outcomes, three memento states:
    - success → message (ready);
    - content failure (empty text) → terminal error note (broken, never retried);
    - model unavailable → leave empty (named notice is the record); no second writer.
    """

    class _Session:
        def __init__(self, conn, session_id):
            self.conn = conn
            self.session_id = session_id

        async def chat_off_record(self, messages, *, anchor=None, **kwargs):
            self.conn.chats.append((self.session_id, messages, anchor, kwargs))
            if self.conn.entered is not None:
                self.conn.entered.set()
            self.conn.attempts += 1
            if self.conn.hang:
                await asyncio.Event().wait()
            if self.conn.error is not None and self.conn.attempts <= self.conn.fail_until:
                raise self.conn.error
            if isinstance(self.conn.response, Exception):
                raise self.conn.response
            from ghoshell_dsh.blueprint.agent import ChatReply
            return ChatReply.model_validate(self.conn.response)

        async def close(self):
            self.conn.closed += 1

    class _FakeConnection:
        def __init__(self, response, *, hang=False, error=None, fail_until=0):
            self.response = response
            self.hang = hang
            self.error = error
            self.fail_until = fail_until
            self.attempts = 0
            self.chats: list[tuple[str, list, object, dict]] = []
            self.closed = 0
            self.entered = asyncio.Event() if hang else None
            self.follows: list[tuple[str, bool]] = []

        def attach_session(self, session_id, logger=None, *, follow=True):
            self.follows.append((session_id, follow))
            return TestEgoMementoSidecar._Session(self, session_id)

    @staticmethod
    def _manager(tmp_path: Path, connection, config=None):
        from ghoshell_moss.ghosts.dolores._ego_memento import EgoMementoConfig, EgoMementoManager
        from ghoshell_moss.memento import new_local_memento

        memento = new_local_memento(tmp_path / "owner")
        memento.create_branch("main")
        manager = EgoMementoManager(
            connection=connection, memento=memento, config=config or EgoMementoConfig()
        )
        return manager, memento

    @staticmethod
    def _state(manager, commit_id: str) -> str:
        return manager.bypass[commit_id].state.value

    @staticmethod
    def _note_text(conn) -> str:
        return conn.chats[0][1][0][0].text

    @pytest.mark.asyncio
    async def test_sidecar_writes_note(self, tmp_path: Path):
        conn = self._FakeConnection({"text": "title\nbody"})
        manager, memento = self._manager(tmp_path, conn)
        anchor = manager.commit(session_id="s1", start_turn=0, end_turn=1)

        manager.schedule_note(anchor.id)
        await manager.drain_bypass()

        note = memento.get_branch("main").notes()[anchor.id]
        assert note.message == "title\nbody"
        assert note.error == ""
        session_id, _, sent_anchor, kwargs = conn.chats[0]
        assert session_id == "s1"
        assert sent_anchor.turn == 1 and sent_anchor.seq is None
        # No `readonly`: emptying the tools menu starts a new prefix family and loses the cache.
        assert "readonly" not in kwargs
        assert kwargs["llm"]["reasoningEffort"] == "low"
        assert kwargs["max_tokens"] > 0
        text = self._note_text(conn)
        assert '<mindfork task="memento.note">' in text
        assert "concurrent 應身 of yourself above" in text
        assert "Your final output is the note itself" in text
        assert self._state(manager, anchor.id) == "ready"

    @pytest.mark.asyncio
    async def test_note_states_the_configured_body_budget(self, tmp_path: Path):
        """The body limit in the request comes from config, not from a literal in the prompt."""
        from ghoshell_moss.ghosts.dolores._ego_memento import EgoMementoConfig

        conn = self._FakeConnection({"text": "title\nbody"})
        manager, _ = self._manager(
            tmp_path, conn, config=EgoMementoConfig(note_body_tokens=1234)
        )
        anchor = manager.commit(session_id="s1", start_turn=0, end_turn=1)
        manager.schedule_note(anchor.id)
        await manager.drain_bypass()

        assert "1234 tokens" in self._note_text(conn)

    @pytest.mark.asyncio
    async def test_note_reports_its_token_usage(self, tmp_path: Path, caplog):
        """An off-record call never enters the session log — this INFO line is its accounting."""
        conn = self._FakeConnection({
            "text": "x",
            "usage": {"inputTokens": 12, "outputTokens": 3, "cacheReadTokens": 4000},
        })
        manager, _ = self._manager(tmp_path, conn)
        anchor = manager.commit(session_id="s1", start_turn=0, end_turn=1)

        with caplog.at_level(logging.INFO, logger="moss"):
            manager.schedule_note(anchor.id)
            await manager.drain_bypass()

        lines = [r.getMessage() for r in caplog.records if "emanation usage" in r.getMessage()]
        assert lines and "kind=note" in lines[0]
        assert "cache_read=4000" in lines[0]

    @pytest.mark.asyncio
    async def test_note_borrows_the_context_without_following_the_session(self, tmp_path: Path):
        """旁路只用一次性把手.

        这个 ref 常常就是 ego 自己的会话, 而 ego 早已 follow 着它. 再开一条同 streamId 的
        ``session/follow``, 宿主判重复并关掉整条 mux (1008) —— 局部功能不该有能力杀死全局传输.
        """
        conn = self._FakeConnection({"text": "x"})
        manager, _ = self._manager(tmp_path, conn)
        anchor = manager.commit(session_id="s1", start_turn=0, end_turn=1)

        manager.schedule_note(anchor.id)
        await manager.drain_bypass()

        assert conn.follows == [("s1", False)]

    @pytest.mark.asyncio
    async def test_note_message_first_commit_covers_all(self, tmp_path: Path):
        conn = self._FakeConnection({"text": "x"})
        manager, _ = self._manager(tmp_path, conn)
        first = manager.commit(session_id="s1", start_turn=0, end_turn=1)

        manager.schedule_note(first.id)
        await manager.drain_bypass()

        assert "This is the first commit" in self._note_text(conn)

    @pytest.mark.asyncio
    async def test_note_message_continues_the_trajectory(self, tmp_path: Path):
        conn = self._FakeConnection({"text": "x"})
        manager, memento = self._manager(tmp_path, conn)
        branch = memento.get_branch("main")
        first = manager.commit(session_id="s1", start_turn=0, end_turn=1)
        branch.note(first.id, "first-note")

        second = manager.commit(session_id="s1", start_turn=1, end_turn=2)
        manager.schedule_note(second.id)
        await manager.drain_bypass()

        message = self._note_text(conn)
        first_coord = branch.get_commit(first.seq).coord
        assert f"continues right after {first_coord}" in message
        assert "first-note" in message

    @pytest.mark.asyncio
    async def test_model_unavailable_leaves_note_empty(self, tmp_path: Path):
        from ghoshell_moss.ghosts.dolores._ego_memento import EgoMementoConfig

        conn = self._FakeConnection(RuntimeError("dsh down"))
        manager, memento = self._manager(
            tmp_path, conn, config=EgoMementoConfig(note_retry_timeout=0),
        )
        anchor = manager.commit(session_id="s1", start_turn=0, end_turn=1)

        manager.schedule_note(anchor.id)
        await manager.drain_bypass()

        assert memento.get_branch("main").notes() == {}
        assert self._state(manager, anchor.id) == "failed"

    @pytest.mark.asyncio
    async def test_sidecar_empty_text_writes_terminal_error_note(self, tmp_path: Path):
        conn = self._FakeConnection({"text": ""})
        manager, memento = self._manager(tmp_path, conn)
        anchor = manager.commit(session_id="s1", start_turn=0, end_turn=1)

        manager.schedule_note(anchor.id)
        await manager.drain_bypass()

        note = memento.get_branch("main").notes()[anchor.id]
        assert note.message == "" and note.error != ""

    @pytest.mark.asyncio
    async def test_retry_recovers_when_model_returns(self, tmp_path: Path):
        from ghoshell_moss.ghosts.dolores._ego_memento import EgoMementoConfig

        conn = self._FakeConnection(
            {"text": "recovered"}, error=RuntimeError("transient"), fail_until=2,
        )
        manager, memento = self._manager(
            tmp_path, conn,
            config=EgoMementoConfig(note_retry_timeout=1.0, note_backoff_base=0.01),
        )
        anchor = manager.commit(session_id="s1", start_turn=0, end_turn=1)

        manager.schedule_note(anchor.id)
        await manager.drain_bypass()

        assert memento.get_branch("main").notes()[anchor.id].message == "recovered"
        assert conn.attempts == 3

    @pytest.mark.asyncio
    async def test_same_commit_dispatched_once(self, tmp_path: Path):
        conn = self._FakeConnection({"text": "x"})
        manager, _ = self._manager(tmp_path, conn)
        anchor = manager.commit(session_id="s1", start_turn=0, end_turn=1)

        manager.schedule_note(anchor.id)
        manager.schedule_note(anchor.id)
        await manager.drain_bypass()

        assert len(conn.chats) == 1

    @pytest.mark.asyncio
    async def test_exit_drops_inflight_note_run(self, tmp_path: Path):
        conn = self._FakeConnection({"text": "x"}, hang=True)
        manager, memento = self._manager(tmp_path, conn)
        anchor = manager.commit(session_id="s1", start_turn=0, end_turn=1)

        async with manager:
            manager.schedule_note(anchor.id)
            await conn.entered.wait()

        assert manager.bypass[anchor.id].task.cancelled()
        assert memento.get_branch("main").notes() == {}


class TestMementoReadSurface:
    """branch / commit 读面 + read / chat 两个动作 —— 全部在 ego 外, 不触 dsh (fake connection)."""

    class _Connection:
        def __init__(self, response):
            self.response = response
            self.calls: list[tuple[str, dict]] = []
            self.chats: list[tuple[str, list, object]] = []
            self.follows: list[tuple[str, bool]] = []
            self.closed = 0

        async def call(self, path, payload=None, *, timeout=None):
            self.calls.append((path, payload))
            if isinstance(self.response, Exception):
                raise self.response
            return self.response

        def attach_session(self, session_id, logger=None, *, follow=True):
            self.follows.append((session_id, follow))
            return self._Session(self, session_id)

        class _Session:
            """``chat_off_record`` 的 fake: 记录消息与锚点, 回一份 ChatReply."""

            def __init__(self, conn, session_id):
                self.conn = conn
                self.session_id = session_id

            async def chat_off_record(self, messages, *, anchor=None, **kwargs):
                self.conn.chats.append((self.session_id, messages, anchor))
                if isinstance(self.conn.response, Exception):
                    raise self.conn.response
                from ghoshell_dsh.blueprint.agent import ChatReply

                return ChatReply.model_validate(self.conn.response)

            async def close(self):
                self.conn.closed += 1

    @staticmethod
    def _manager(tmp_path: Path, connection=None):
        from ghoshell_moss.memento import new_local_memento

        from ghoshell_moss.ghosts.dolores._ego_memento import EgoMementoConfig, EgoMementoManager

        memento = new_local_memento(tmp_path / "owner")
        memento.create_branch("main")
        return EgoMementoManager(
            connection=connection, memento=memento, config=EgoMementoConfig()
        ), memento

    @staticmethod
    def _coord(memento, seq: int) -> str:
        return memento.get_branch("main").get_commit(seq).coord

    def test_list_branches_reports_index_and_latest(self, tmp_path: Path):
        manager, memento = self._manager(tmp_path)
        side = memento.create_branch("side")
        anchor = manager.commit(session_id="s1", start_turn=0, end_turn=4)
        memento.get_branch("main").note(anchor.id, "标题\n正文")

        infos = {info.name: info for info in manager.list_branches()}

        assert infos["main"].commits_total == 1
        assert infos["main"].latest_title == "标题"
        assert infos["main"].latest_coord == self._coord(memento, anchor.seq)
        assert infos["side"].index == side.index
        assert infos["side"].commits_total == 0
        assert infos["side"].latest_coord == ""

    def test_list_commits_applies_time_window(self, tmp_path: Path):
        manager, memento = self._manager(tmp_path)
        branch = memento.get_branch("main")
        first = manager.commit(session_id="s1", start_turn=0, end_turn=1)
        second = manager.commit(session_id="s1", start_turn=1, end_turn=2)
        branch.note(first.id, "one\nbody-1")
        branch.note(second.id, "two\nbody-2")

        assert [d.title for d in manager.list_commits()] == ["one", "two"]
        assert [d.body for d in manager.list_commits()] == ["body-1", "body-2"]

        before_all = branch.commits()[0].created - timedelta(days=1)
        assert manager.list_commits(until_date=before_all) == []

    def test_view_message_reads_another_branch(self, tmp_path: Path):
        manager, memento = self._manager(tmp_path)
        anchor = manager.commit(session_id="s1", start_turn=0, end_turn=1)
        memento.get_branch("main").note(anchor.id, "main-note")
        side = memento.create_branch("side")
        side_anchor = side.commit(metatype="session", metadata={})
        side.note(side_anchor.id, "side-note")

        assert "main-note" in manager.view_message().to_content_string()
        assert "side-note" in manager.view_message("side").to_content_string()

    def test_view_message_folds_the_past_instead_of_rendering_it(self, tmp_path: Path):
        """The budget is the contract: n commits in detail, everything older folded.

        This used to concatenate ``history + latest`` and draw every commit in
        full, so a long branch shipped its whole life into memory while the
        budget only existed in the signature.
        """
        manager, memento = self._manager(tmp_path)
        branch = memento.get_branch("main")
        for i in range(30):
            anchor = manager.commit(session_id="s1", start_turn=i, end_turn=i + 1)
            branch.note(anchor.id, f"note {i}\n" + ("body " * 200))

        text = manager.view_message().to_content_string()

        assert text.count("<commit ") == 10
        assert text.count("<earlier ") == 1
        assert 'count="20"' in text
        # one folded entry, not twenty: the far side is a range plus a few titles
        assert len(text) < 12000

    def test_view_message_carries_node_path_when_present(self, tmp_path: Path):
        manager, memento = self._manager(tmp_path)
        branch = memento.get_branch("main")
        anchor = manager.commit(session_id="s1", start_turn=0, end_turn=1)
        branch.note(anchor.id, "main-note")

        assert "memento=" not in manager.view_message().to_content_string()

        node = branch.ensure_memento(anchor.seq)

        assert str(node) in manager.view_message().to_content_string()

    @pytest.mark.asyncio
    async def test_read_renders_the_route_events(self, tmp_path: Path):
        events = [
            {"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1}},
            {
                "type": "user/message", "seq": 1, "time": 1,
                "data": {"role": "user", "source": {"kind": "user"},
                         "content": [{"type": "text", "text": "为什么"}]},
            },
            {
                "type": "assistant/message", "seq": 2, "time": 1,
                "data": {"message": {"role": "assistant", "source": {"kind": "model"},
                                     "content": [{"type": "text", "text": "因为"}]}},
            },
            {"type": "turn/end", "seq": 3, "time": 1, "data": {"turn": 1}},
        ]
        conn = self._Connection({"events": events})
        manager, memento = self._manager(tmp_path, conn)
        anchor = manager.commit(session_id="s1", start_turn=0, end_turn=1)

        text = await manager.read(self._coord(memento, anchor.seq))

        assert text == "  > 为什么\n  ~ 因为"  # render_transcript 默认两级缩进
        assert conn.calls[0][1]["ref"]["session_id"] == "s1"

    @pytest.mark.asyncio
    async def test_read_raises_on_unreadable_source(self, tmp_path: Path):
        """源存档读不出来时显式抛 SourceUnreadableError —— 不回落成空串, 也不带出磁盘路径."""
        from ghoshell_moss.ghosts.dolores._ego_memento import SourceUnreadableError

        conn = self._Connection(RuntimeError(
            'dsh RPC /moss-api/ghost/dolores/read failed (400): '
            'SessionPersistenceCorruptionError: session "s1": stored log is corrupt: '
            'SessionFormatError: system/message requires a protected first surface head '
            "(raw log: /Users/someone/.dsh/sessions/x/session.v4.jsonl.zstd)"
        ))
        manager, memento = self._manager(tmp_path, conn)
        anchor = manager.commit(session_id="s1", start_turn=0, end_turn=1)
        coord = self._coord(memento, anchor.seq)

        with pytest.raises(SourceUnreadableError) as caught:
            await manager.read(coord)

        message = str(caught.value)
        assert coord in message and "s1" in message
        assert "no compatible read path" in message
        assert "/Users/" not in message  # (raw log: …) 尾巴被剥掉, 路径不进上下文

    @pytest.mark.asyncio
    async def test_bypass_skips_unreadable_source_without_marking_broken(self, tmp_path: Path):
        """格式不兼容 = 读不了, 不是旁路失败 —— 写空 note 防重试, 但不标 broken."""
        conn = self._Connection(RuntimeError(
            'dsh RPC /moss-api/ghost/dolores/bypass/run failed (400): '
            'SessionFormatUnsupportedMigrationError: Session migration from v3 to v4 refuses '
            'the transformed artifact: system/message requires a protected first surface head'
        ))
        manager, memento = self._manager(tmp_path, conn)
        anchor = manager.commit(session_id="s1", start_turn=0, end_turn=1)

        manager.schedule_note(anchor.id)
        await manager.drain_bypass()

        note = memento.get_branch("main").notes()[anchor.id]
        assert note.message == "" and note.error == ""
        assert manager.resolve(self._coord(memento, anchor.seq)).is_broken is False

    @pytest.mark.asyncio
    async def test_chat_frames_the_context_not_the_commit(self, tmp_path: Path):
        conn = self._Connection({"text": "reply"})
        manager, memento = self._manager(tmp_path, conn)
        anchor = manager.commit(session_id="s1", start_turn=0, end_turn=1)
        memento.get_branch("main").note(anchor.id, "note")
        question = "当时为什么这么定?"

        text = await manager.chat(self._coord(memento, anchor.seq), question)

        assert text == "reply"
        # 走 chat off record: 一条未发布的 detached session, 不碰旁路路由.
        assert conn.calls == []
        # 一次性把手: 目标会话可能正被 ego follow 着, 这里不再开第二条 follow 流.
        assert conn.follows == [("s1", False)]
        session_id, messages, sent_anchor = conn.chats[0]
        assert session_id == "s1"
        # one mindfork container: instruction + question ride together.
        assert len(messages) == 1
        container = messages[0][0].text
        assert '<mindfork task="memento.chat">' in container
        assert "concurrent 應身 of yourself above" in container
        assert "this message is your task" in container
        assert "outside your view" in container
        assert question in container
        # 锚点: 只有 turn 时用它定位, seq 缺省.
        assert sent_anchor.turn == 1 and sent_anchor.seq is None
        assert conn.closed == 1

    @pytest.mark.asyncio
    async def test_chat_refuses_broken_commit(self, tmp_path: Path):
        conn = self._Connection({"message": "reply"})
        manager, memento = self._manager(tmp_path, conn)
        anchor = manager.commit(session_id="s1", start_turn=0, end_turn=1)
        memento.get_branch("main").note(anchor.id, "占位", error="fatal")
        from ghoshell_moss.ghosts.dolores._ego_memento import BrokenCommitError

        with pytest.raises(BrokenCommitError):
            await manager.chat(self._coord(memento, anchor.seq), "问题")

        assert conn.calls == []  # 坏 commit 连旁路都不发
        assert conn.chats == []  # 也不开 off-record 会话


class TestMementoCommitSpan:
    """锚点区间规则 —— 半开 ``(start_turn, end_turn]``, 空区间不落锚点."""

    @staticmethod
    def _manager(tmp_path: Path):
        from ghoshell_moss.memento import new_local_memento

        from ghoshell_moss.ghosts.dolores._ego_memento import EgoMementoConfig, EgoMementoManager

        memento = new_local_memento(tmp_path / "owner")
        memento.create_branch("main")
        return EgoMementoManager(
            connection=None, memento=memento, config=EgoMementoConfig()
        ), memento

    def test_records_half_open_span(self, tmp_path: Path):
        manager, memento = self._manager(tmp_path)

        anchor = manager.commit(session_id="s1", start_turn=1, end_turn=2)

        ref = memento.get_branch("main").commits()[0].metadata["ref"]
        assert anchor is not None
        assert (ref["start_turn"], ref["end_turn"]) == (1, 2)

    def test_empty_span_is_not_committed(self, tmp_path: Path):
        manager, memento = self._manager(tmp_path)

        assert manager.commit(session_id="s1", start_turn=2, end_turn=2) is None
        assert memento.get_branch("main").commits() == []


class TestEgoMementoCompactCut:
    """SCS (安全 compact 位点) = tip commit 的 start_turn = 倒数第二个 commit 的 end_turn."""

    @staticmethod
    def _manager(tmp_path: Path):
        from ghoshell_moss.ghosts.dolores._ego_memento import EgoMementoConfig, EgoMementoManager
        from ghoshell_moss.memento import new_local_memento

        memento = new_local_memento(tmp_path / "owner")
        memento.create_branch("main")
        return EgoMementoManager(connection=None, memento=memento, config=EgoMementoConfig()), memento

    def test_none_without_commits(self, tmp_path: Path):
        manager, _ = self._manager(tmp_path)
        assert manager.compact_cut_turn() is None

    def test_none_with_a_single_commit(self, tmp_path: Path):
        # 只有一个 commit = 没有 n-1, 无可压 (tip.start_turn == 0).
        manager, _ = self._manager(tmp_path)
        manager.commit(session_id="s1", start_turn=0, end_turn=3)
        assert manager.compact_cut_turn() is None

    def test_returns_second_newest_end_turn(self, tmp_path: Path):
        # (0,3] (3,7] (7,9] → tip 是 (7,9], 其 start_turn=7 = 倒数第二个 commit 的 end_turn.
        manager, _ = self._manager(tmp_path)
        manager.commit(session_id="s1", start_turn=0, end_turn=3)
        manager.commit(session_id="s1", start_turn=3, end_turn=7)
        manager.commit(session_id="s1", start_turn=7, end_turn=9)
        assert manager.compact_cut_turn() == 7


# ── DoloresRun — thinking 交易 run 对象 (public + 可测) ─────────────
