"""The memory backend behind the ego — anchors, sideside notes, and the read surface.

Answers: what a span *is* once it is committed — half-open ``(start_turn, end_turn]``,
what fills a missing note, where compaction may safely cut, and what the read/chat face
hands back. All of it runs outside dsh, on a fake connection.
"""

import asyncio
from datetime import timedelta
from pathlib import Path

import pytest

class TestEgoMementoSidecar:
    """旁路 note 生产与补漏 — run 路由的 fake.

    旁路一轮有三种结局, 落在 memento 上是三种不同的状态:
    - 成功 → 写 message (ready);
    - 失败 (传输异常 / 空文本) → 写**终态 error note** (broken, 永不重试);
    - 取消 (关停) → **留空**, 由下次开机的 ``backfill()`` 捡起来.
    """

    class _FakeConnection:
        def __init__(self, response):
            self.response = response
            self.calls: list[tuple[str, dict]] = []

        async def call(self, path, payload=None, *, timeout=None):
            self.calls.append((path, payload))
            if isinstance(self.response, Exception):
                raise self.response
            return self.response

    class _StuckConnection:
        """永不返回 —— 只被取消 (关停路径)."""

        def __init__(self):
            self.entered = asyncio.Event()

        async def call(self, path, payload=None, *, timeout=None):
            self.entered.set()
            await asyncio.Event().wait()

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
        """旁路状态 (取字符串值, 免去在测试里摊开内部枚举)."""
        return manager.bypass[commit_id].state.value

    @pytest.mark.asyncio
    async def test_sidecar_writes_note(self, tmp_path: Path):
        conn = self._FakeConnection({"message": "title\nbody"})
        manager, memento = self._manager(tmp_path, conn)
        anchor = manager.commit(session_id="s1", start_turn=0, end_turn=1)

        manager.schedule_note(anchor.id)
        await manager.drain_bypass()

        note = memento.get_branch("main").notes()[anchor.id]
        assert note.message == "title\nbody"
        assert note.error == ""
        assert conn.calls[0][0].endswith("/bypass/run")
        assert conn.calls[0][1]["ref"]["session_id"] == "s1"
        # 旁路约束显式进载荷: 关思考 + 输出硬 cap (不再靠 plugin 身份判定间接降级).
        assert conn.calls[0][1]["reasoning_effort"] == "off"
        assert conn.calls[0][1]["max_tokens"] > 0
        assert self._state(manager, anchor.id) == "ready"

    @pytest.mark.asyncio
    async def test_note_prompt_first_commit_covers_all(self, tmp_path: Path):
        """首条 commit: 无前驱, 声明覆盖完整上下文."""
        conn = self._FakeConnection({"message": "x"})
        manager, _ = self._manager(tmp_path, conn)
        first = manager.commit(session_id="s1", start_turn=0, end_turn=1)

        manager.schedule_note(first.id)
        await manager.drain_bypass()

        assert "This is the first commit" in conn.calls[0][1]["prompt"]

    @pytest.mark.asyncio
    async def test_note_prompt_continues_the_trajectory(self, tmp_path: Path):
        """后续 commit: 注入前驱坐标 + 前文 message (只调度目标 commit, 不覆盖前驱 note)."""
        conn = self._FakeConnection({"message": "x"})
        manager, memento = self._manager(tmp_path, conn)
        branch = memento.get_branch("main")
        first = manager.commit(session_id="s1", start_turn=0, end_turn=1)
        branch.note(first.id, "first-note")

        second = manager.commit(session_id="s1", start_turn=1, end_turn=2)
        manager.schedule_note(second.id)
        await manager.drain_bypass()

        prompt = conn.calls[0][1]["prompt"]
        first_coord = branch.get_commit(first.seq).coord
        assert f"continues right after {first_coord}" in prompt
        assert "first-note" in prompt

    @pytest.mark.asyncio
    async def test_sidecar_failure_writes_terminal_error_note(self, tmp_path: Path):
        """失败 → 终态 error note (不是留空). 留空会被补漏当成"还没生产"而每次开机重跑."""
        conn = self._FakeConnection(RuntimeError("dsh down"))
        manager, memento = self._manager(tmp_path, conn)
        anchor = manager.commit(session_id="s1", start_turn=0, end_turn=1)

        manager.schedule_note(anchor.id)
        await manager.drain_bypass()

        note = memento.get_branch("main").notes()[anchor.id]
        assert note.message == ""
        assert "dsh down" in note.error  # 终态标记 + 失败原因
        assert self._state(manager, anchor.id) == "failed"
        # 读侧据此认它是 broken (chat 不可用, view 会折叠).
        assert memento.get_branch("main").get_commit(anchor.seq).is_broken

    @pytest.mark.asyncio
    async def test_sidecar_empty_text_writes_terminal_error_note(self, tmp_path: Path):
        """空文本同样算失败 —— 写 error note, 不留空."""
        conn = self._FakeConnection({"message": ""})
        manager, memento = self._manager(tmp_path, conn)
        anchor = manager.commit(session_id="s1", start_turn=0, end_turn=1)

        manager.schedule_note(anchor.id)
        await manager.drain_bypass()

        note = memento.get_branch("main").notes()[anchor.id]
        assert note.message == "" and note.error != ""

    @pytest.mark.asyncio
    async def test_same_commit_dispatched_once(self, tmp_path: Path):
        conn = self._FakeConnection({"message": "x"})
        manager, _ = self._manager(tmp_path, conn)
        anchor = manager.commit(session_id="s1", start_turn=0, end_turn=1)

        manager.schedule_note(anchor.id)
        manager.schedule_note(anchor.id)
        await manager.drain_bypass()

        assert len([call for call in conn.calls if call[0].endswith("/bypass/run")]) == 1

    @pytest.mark.asyncio
    async def test_exit_drops_inflight_note_run(self, tmp_path: Path):
        conn = self._StuckConnection()
        manager, memento = self._manager(tmp_path, conn)
        anchor = manager.commit(session_id="s1", start_turn=0, end_turn=1)

        async with manager:
            manager.schedule_note(anchor.id)
            await conn.entered.wait()  # 旁路确实跑起来了, 再走关停

        assert manager.bypass[anchor.id].task.cancelled()  # 不阻塞关停
        assert memento.get_branch("main").notes() == {}  # 空 note 留空 (内容仍可 read)

    # ── 启动补漏 (backfill) ──────────────────────────────────────────

    @staticmethod
    def _restart(tmp_path: Path, connection, config=None):
        """模拟重启: 重新打开同一个 memento store + 全新 manager (空的旁路治理态)."""
        from ghoshell_moss.ghosts.dolores._ego_memento import EgoMementoConfig, EgoMementoManager
        from ghoshell_moss.memento import new_local_memento

        memento = new_local_memento(tmp_path / "owner")
        manager = EgoMementoManager(
            connection=connection, memento=memento, config=config or EgoMementoConfig()
        )
        return manager, memento

    @pytest.mark.asyncio
    async def test_backfill_dispatches_notes_for_commits_missing_them(self, tmp_path: Path):
        conn = self._FakeConnection({"message": "title\nbody"})
        manager, memento = self._manager(tmp_path, conn)
        first = manager.commit(session_id="s1", start_turn=0, end_turn=1)
        second = manager.commit(session_id="s1", start_turn=1, end_turn=2)

        assert manager.backfill() == 2
        await manager.drain_bypass()

        notes = memento.get_branch("main").notes()
        assert notes[first.id].message == "title\nbody"
        assert notes[second.id].message == "title\nbody"

    @pytest.mark.asyncio
    async def test_backfill_returns_before_the_runs_finish(self, tmp_path: Path):
        """非阻塞: 只派发, 立刻返回 —— 不拖慢 boot."""
        conn = self._FakeConnection({"message": "x"})
        manager, memento = self._manager(tmp_path, conn)
        anchor = manager.commit(session_id="s1", start_turn=0, end_turn=1)

        assert manager.backfill() == 1
        assert self._state(manager, anchor.id) == "running"  # 在飞, 没被 await
        assert memento.get_branch("main").notes() == {}  # 返回时还没落

        await manager.drain_bypass()
        assert memento.get_branch("main").notes()[anchor.id].message == "x"

    def test_backfill_skips_commits_that_already_have_notes(self, tmp_path: Path):
        """一条 Note 无论是 ready 还是 terminal error 都算"有" —— 都不重补."""
        conn = self._FakeConnection({"message": "x"})
        manager, memento = self._manager(tmp_path, conn)
        branch = memento.get_branch("main")
        ready = manager.commit(session_id="s1", start_turn=0, end_turn=1)
        broken = manager.commit(session_id="s1", start_turn=1, end_turn=2)
        branch.note(ready.id, "already")
        branch.note(broken.id, message="", error="fatal")

        assert manager.backfill() == 0
        assert conn.calls == []
        assert branch.notes()[broken.id].error == "fatal"  # 失败标记不被覆盖

    def test_backfill_skips_commits_without_a_cut_ref(self, tmp_path: Path):
        """非本 manager 落的 commit (无 ref) 补不了 —— 静默跳过, 不报错也不派发."""
        conn = self._FakeConnection({"message": "x"})
        manager, memento = self._manager(tmp_path, conn)
        memento.get_branch("main").commit(message="", metatype="foreign", metadata={})

        assert manager.backfill() == 0
        assert conn.calls == []

    @pytest.mark.asyncio
    async def test_backfill_respects_the_tail_bound(self, tmp_path: Path):
        from ghoshell_moss.ghosts.dolores._ego_memento import EgoMementoConfig

        conn = self._FakeConnection({"message": "x"})
        manager, _ = self._manager(tmp_path, conn, config=EgoMementoConfig(resume_tail=2))
        for i in range(4):
            manager.commit(session_id="s1", start_turn=i, end_turn=i + 1)

        assert manager.backfill() == 2
        await manager.drain_bypass()
        assert {call[1]["ref"]["end_turn"] for call in conn.calls} == {3, 4}

    def test_backfill_disabled_when_tail_is_zero(self, tmp_path: Path):
        from ghoshell_moss.ghosts.dolores._ego_memento import EgoMementoConfig

        conn = self._FakeConnection({"message": "x"})
        manager, _ = self._manager(tmp_path, conn, config=EgoMementoConfig(resume_tail=0))
        manager.commit(session_id="s1", start_turn=0, end_turn=1)

        assert manager.backfill() == 0
        assert conn.calls == []

    @pytest.mark.asyncio
    async def test_backfill_after_restart_does_not_retry_failed_commits(self, tmp_path: Path):
        """失败是终态: 重启后补漏不再碰它 —— 否则一个恒失败的 commit 每次开机都白跑一轮."""
        conn = self._FakeConnection(RuntimeError("dsh down"))
        manager, memento = self._manager(tmp_path, conn)
        anchor = manager.commit(session_id="s1", start_turn=0, end_turn=1)
        manager.schedule_note(anchor.id)
        await manager.drain_bypass()
        assert memento.get_branch("main").notes()[anchor.id].error != ""

        conn2 = self._FakeConnection({"message": "x"})
        restarted, _ = self._restart(tmp_path, conn2)

        assert restarted.backfill() == 0
        assert conn2.calls == []

    @pytest.mark.asyncio
    async def test_backfill_recovers_notes_dropped_at_shutdown(self, tmp_path: Path):
        """关停时被取消的旁路留空 → 下次开机补上. 这是补漏存在的理由."""
        conn = self._StuckConnection()
        manager, memento = self._manager(tmp_path, conn)
        anchor = manager.commit(session_id="s1", start_turn=0, end_turn=1)
        async with manager:
            manager.schedule_note(anchor.id)
            await conn.entered.wait()

        assert memento.get_branch("main").notes() == {}  # 取消 ≠ 失败: 不写 error, 留空

        conn2 = self._FakeConnection({"message": "recovered"})
        restarted, _ = self._restart(tmp_path, conn2)

        assert restarted.backfill() == 1
        await restarted.drain_bypass()
        assert memento.get_branch("main").notes()[anchor.id].message == "recovered"


class TestMementoReadSurface:
    """branch / commit 读面 + read / chat 两个动作 —— 全部在 ego 外, 不触 dsh (fake connection)."""

    class _Connection:
        def __init__(self, response):
            self.response = response
            self.calls: list[tuple[str, dict]] = []
            self.chats: list[tuple[str, list, object]] = []
            self.closed = 0

        async def call(self, path, payload=None, *, timeout=None):
            self.calls.append((path, payload))
            if isinstance(self.response, Exception):
                raise self.response
            return self.response

        def create_session(self, session_id, logger=None):
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
        assert manager.backfill() == 0  # 空 note 也算"有", 不再重试

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
        session_id, messages, sent_anchor = conn.chats[0]
        assert session_id == "s1"
        # 两条消息: 处境说明在前, 问题在后 —— 消息怎么分由 Python 侧决定.
        assert len(messages) == 2
        assert "talking to the context that commit belongs to" in messages[0][0].text
        assert "outside your view" in messages[0][0].text
        assert messages[1][0].text == question
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
