"""The memory backend behind the ego — anchors, sideside notes, and the read surface.

Answers: what a span *is* once it is committed — half-open ``(start_turn, end_turn]``,
what fills a missing note, where compaction may safely cut, and what the read/chat face
hands back. All of it runs outside dsh, on a fake connection.
"""

import asyncio
import logging
import re
from datetime import timedelta
from pathlib import Path

import pytest

from ._support import _OfflineConnection, _dolores_meta

class TestEgoMementoSidecar:
    """Note task — one off-record round per commit, with the emanation-reminder container.

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

        memento = new_local_memento(tmp_path / "owner", writable=True)
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
        assert '<emanation-reminder task="memento.note"' in text
        assert "concurrent Emanation of yourself above" in text
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

        assert 'budget="1234"' in self._note_text(conn)

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

        memento = new_local_memento(tmp_path / "owner", writable=True)
        memento.create_branch("main")
        return EgoMementoManager(
            connection=connection or _OfflineConnection(), memento=memento, config=EgoMementoConfig()
        ), memento

    @staticmethod
    def _coord(memento, seq: int) -> str:
        return memento.get_branch("main").get_commit(seq).coord

    def test_list_branches_is_owner_data_only(self, tmp_path: Path):
        manager, memento = self._manager(tmp_path)
        side = memento.create_branch("side")
        manager.commit(session_id="s1", start_turn=0, end_turn=4)

        entries = {ref.name: index for index, ref in manager.list_branches()}

        assert entries == {"main": 1, "side": side.index}
        # A listing must not need a branch's trajectory: make those reads explode if taken.
        branch = memento.get_branch("main")

        def boom(*args, **kwargs):
            raise AssertionError("branch trajectory read")

        branch.commits = boom
        branch.notes = boom
        assert manager.list_branches()

    @pytest.mark.asyncio
    async def test_digest_render_keeps_the_folded_face(self, tmp_path: Path):
        manager, memento = self._manager(tmp_path)
        branch = memento.get_branch("main")
        for i in range(5):
            branch.commit(message=f"c{i}")

        text = (await manager.aview_message(n=2)).to_content_string()

        assert '<earlier count="3" errors="0" first="1-1" last="1-3"' in text
        assert "1-1  c0" in text
        assert "1-3  c2" in text
        assert 'seq="1-4"' in text and 'seq="1-5"' in text

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

    @pytest.mark.asyncio
    async def test_view_message_reads_another_branch(self, tmp_path: Path):
        manager, memento = self._manager(tmp_path)
        anchor = manager.commit(session_id="s1", start_turn=0, end_turn=1)
        memento.get_branch("main").note(anchor.id, "main-note")
        side = memento.create_branch("side")
        side_anchor = side.commit(metatype="session", metadata={})
        side.note(side_anchor.id, "side-note")

        assert "main-note" in (await manager.aview_message()).to_content_string()
        assert "side-note" in (await manager.aview_message("side")).to_content_string()

    @pytest.mark.asyncio
    async def test_view_message_folds_the_past_instead_of_rendering_it(self, tmp_path: Path):
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

        text = (await manager.aview_message()).to_content_string()

        # 6 详展 + 24 折叠; the folded count is over raw commits, and its title budget is 20.
        assert text.count("<commit ") == 6
        assert text.count("<earlier ") == 1
        assert 'count="24"' in text
        assert 'titles="20"' in text
        # every folded title carries a visible marker, so a collapsed newline cannot merge lines
        assert "\n· 1-" in text
        # one folded entry, not twenty-four: the far side is a range plus a few titles
        assert len(text) < 12000

    @pytest.mark.asyncio
    async def test_digest_counts_raw_commits_and_reports_broken_ones(self, tmp_path: Path):
        """count is over commits, not rendered entries — a broken run still consumes its range.

        Two consecutive broken notes collapse to one entry so they cannot flood the view; the
        count and the error count are what keep that fold from silently losing two commits.
        """
        manager, memento = self._manager(tmp_path)
        branch = memento.get_branch("main")
        ids = []
        for i in range(12):
            anchor = manager.commit(session_id="s1", start_turn=i, end_turn=i + 1)
            ids.append(anchor.id)
            branch.note(anchor.id, f"note {i}")
        branch.note(ids[0], "", error="run failed: timeout")
        branch.note(ids[1], "", error="run failed: timeout")

        text = (await manager.aview_message()).to_content_string()

        assert 'count="6"' in text, "history is six raw commits"
        assert 'errors="2"' in text, "the collapsed run is still two broken commits"
        assert 'titles="5"' in text, "two broken entries collapse to one title line"

    @pytest.mark.asyncio
    async def test_commit_note_body_is_escaped_before_it_enters_the_surface(self, tmp_path: Path):
        """A note is text, not structure: its angle brackets must not become tags."""
        manager, memento = self._manager(tmp_path)
        branch = memento.get_branch("main")
        anchor = manager.commit(session_id="s1", start_turn=0, end_turn=1)
        branch.note(anchor.id, "shape <memento> & <commit seq=...>")

        text = (await manager.aview_message()).to_content_string()

        assert "&lt;memento&gt;" in text
        assert "&amp;" in text
        assert "<memento root=" in text, "the container itself is still markup"

    @pytest.mark.asyncio
    async def test_view_points_at_memento_md_with_a_path_relative_to_the_container_root(self, tmp_path: Path):
        """Existence is ``MEMENTO.md``; the view gives its path, not a boolean.

        The container states the root once, every anchor's path is relative to it — so a
        commit with a memento reads as a file you can open, and no anchor carries an absolute
        path.
        """
        manager, memento = self._manager(tmp_path)
        branch = memento.get_branch("main")
        anchor = manager.commit(session_id="s1", start_turn=0, end_turn=1)
        branch.note(anchor.id, "main-note")

        assert 'memento="' not in (await manager.aview_message()).to_content_string()

        index = branch.ensure_memento(anchor.seq)
        text = (await manager.aview_message()).to_content_string()

        assert f'root="{memento.root}"' in text
        assert f'memento="{index.relative_to(memento.root)}"' in text
        # The anchor's own line never carries the absolute path.
        assert str(index) not in text

    @pytest.mark.asyncio
    async def test_view_message_groups_the_window_by_day(self, tmp_path: Path):
        """The window is a day's work; the date is paid once, the clock travels per commit.

        A bare ``D13`` cannot say which month it belongs to — this branch already spans
        three of them — so the day carries the absolute date and the zone; each commit
        carries only the clock, on the body's current clock.
        """
        manager, memento = self._manager(tmp_path)
        branch = memento.get_branch("main")
        for i in range(3):
            anchor = manager.commit(session_id="s1", start_turn=i, end_turn=i + 1)
            branch.note(anchor.id, f"note {i}")

        created = [c.created for c in branch.commits()]
        local = created[0].astimezone()
        text = (await manager.aview_message()).to_content_string()

        assert text.count("<day ") == 1
        assert f'<day date="{local.strftime("%Y-%m-%d")}' in text
        assert re.search(r'<day date="\d{4}-\d{2}-\d{2}[+-]\d', text)
        # the clock is the body's clock, minute precision, no zone repeated per line
        assert re.search(r'<commit seq="1-\d+" at="\d{2}:\d{2}"', text)
        assert 'created="' not in text

    @pytest.mark.asyncio
    async def test_view_message_stops_before_a_coordinate(self, tmp_path: Path):
        """A snapshot rendered for compaction must end where the surface takes over.

        The cut commit's span stays verbatim above the recap; rendering it into the snapshot
        as well would put two versions of the same span in one context.
        """
        manager, memento = self._manager(tmp_path)
        branch = memento.get_branch("main")
        coords = []
        for i in range(3):
            anchor = manager.commit(session_id="s1", start_turn=i, end_turn=i + 1)
            branch.note(anchor.id, f"note {i}")
            coords.append(manager.coord(anchor))

        text = (await manager.aview_message(until=coords[2])).to_content_string()

        # exclusive: the cut commit and everything after it are the surface's job
        assert coords[0] in text and coords[1] in text
        assert coords[2] not in text
        assert 'total="2"' in text

    @pytest.mark.asyncio
    async def test_view_message_does_not_claim_an_owner(self, tmp_path: Path):
        """owner was a directory name, not an identity; the container no longer states one."""
        manager, memento = self._manager(tmp_path)
        manager.commit(session_id="s1", start_turn=0, end_turn=1)

        assert 'owner="' not in (await manager.aview_message()).to_content_string()

    @pytest.mark.asyncio
    async def test_view_message_points_at_the_forked_parent(self, tmp_path: Path):
        """The branch opening reserves the parent: one pointer, never the parent's history."""
        manager, memento = self._manager(tmp_path)
        branch = memento.get_branch("main")
        anchor = manager.commit(session_id="s1", start_turn=0, end_turn=1)
        branch.note(anchor.id, "main-note")
        branch.fork("child")

        text = (await manager.aview_message("child")).to_content_string()

        assert f'<fork from="main" index="{branch.index}"' in text
        assert f'coord="{branch.index}-{anchor.seq}"' in text
        assert "<commit" not in text  # the parent's commits stay at the parent

    @pytest.mark.asyncio
    async def test_view_message_shows_what_arrived_from_another_branch(self, tmp_path: Path):
        """A confluence is the only cross-branch write, so it must be visible where it landed.

        Only the title travels (the contract copies nothing else); the body stays at the
        origin and is one ``read`` away.
        """
        from ghoshell_moss.memento.abcd import BranchAddr

        manager, memento = self._manager(tmp_path)
        anchor = manager.commit(session_id="s1", start_turn=0, end_turn=1)
        main = memento.get_branch("main")
        side = main.fork("side")
        main.note(anchor.id, "main-note", recipients=[
            BranchAddr(branch_id=side.meta().branch_id, branch_name="side", branch_index=side.index),
        ])
        landed = side.commit(metatype="session", metadata={})
        side.note(landed.id, "side-note")

        text = (await manager.aview_message("side")).to_content_string()

        assert '<from branch="main"' in text
        assert f'coord="{main.index}-{anchor.seq}"' in text
        assert "main-note" in text

    @pytest.mark.asyncio
    async def test_view_message_shows_a_branch_it_spawned(self, tmp_path: Path):
        """The outgoing half of the same exchange: a child is absorbed by the next commit."""
        manager, memento = self._manager(tmp_path)
        branch = memento.get_branch("main")
        anchor = manager.commit(session_id="s1", start_turn=0, end_turn=1)
        branch.note(anchor.id, "main-note")
        branch.fork("child")
        later = manager.commit(session_id="s1", start_turn=1, end_turn=2)
        branch.note(later.id, "later-note")

        text = (await manager.aview_message()).to_content_string()

        assert '<to branch="child"' in text

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
        # one emanation-reminder container: instruction + question ride together.
        assert len(messages) == 1
        container = messages[0][0].text
        assert '<emanation-reminder task="memento.chat">' in container
        assert "concurrent Emanation of yourself above" in container
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

        memento = new_local_memento(tmp_path / "owner", writable=True)
        memento.create_branch("main")
        return EgoMementoManager(
            connection=_OfflineConnection(), memento=memento, config=EgoMementoConfig()
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

        memento = new_local_memento(tmp_path / "owner", writable=True)
        memento.create_branch("main")
        return EgoMementoManager(connection=_OfflineConnection(), memento=memento, config=EgoMementoConfig()), memento

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


class TestBranchGround:
    """The line's state face — nested in memory, planted from a seed, absent-safe."""

    @staticmethod
    def _manager(tmp_path: Path, *, seed_ground: Path | None = None):
        from ghoshell_moss.memento import new_local_memento

        from ghoshell_moss.ghosts.dolores._ego_memento import EgoMementoConfig, EgoMementoManager

        memento = new_local_memento(tmp_path / "owner", writable=True)
        memento.create_branch("main")
        manager = EgoMementoManager(
            connection=_OfflineConnection(),
            memento=memento,
            config=EgoMementoConfig(),
            seed_ground=seed_ground,
        )
        return manager, memento

    @staticmethod
    def _seed(tmp_path: Path) -> Path:
        seed = tmp_path / "seed" / "branch_ground"
        (seed / "skills").mkdir(parents=True)
        (seed / "GROUND.md").write_text("# seed\n\nSEED-TEXT", encoding="utf-8")
        (seed / "skills" / "GROUND.md").write_text("# skills\n", encoding="utf-8")
        return seed

    @pytest.mark.asyncio
    async def test_absent_ground_renders_the_seed_as_default(self, tmp_path: Path):
        manager, _ = self._manager(tmp_path, seed_ground=self._seed(tmp_path))

        text = (await manager.aview_message()).to_content_string()

        assert '<branch name="main"' in text
        assert 'default="true"' in text
        assert "SEED-TEXT" in text

    @pytest.mark.asyncio
    async def test_a_written_ground_renders_itself_not_the_seed(self, tmp_path: Path):
        manager, memento = self._manager(tmp_path, seed_ground=self._seed(tmp_path))
        memento.get_branch("main").path.joinpath("GROUND.md").write_text(
            "# main\n\nMY-OWN-FIELD", encoding="utf-8"
        )

        text = (await manager.aview_message()).to_content_string()

        assert "MY-OWN-FIELD" in text
        assert "SEED-TEXT" not in text
        assert 'default="true"' not in text
        # One field wears one container: the renderer's <ground> is not wrapped a second time.
        assert text.count("<ground ") == 1

    @pytest.mark.asyncio
    async def test_a_new_line_is_planted_with_the_seed(self, tmp_path: Path):
        manager, memento = self._manager(tmp_path, seed_ground=self._seed(tmp_path))

        async with manager:
            branch = manager.fork_branch("idea", "a direction")

        planted = memento.root / "branches" / branch.ref.branch_id
        assert (planted / "GROUND.md").read_text(encoding="utf-8").startswith("# seed")
        assert (planted / "skills" / "GROUND.md").is_file()

    @pytest.mark.asyncio
    async def test_the_shipped_seed_fits_its_own_body_on_a_deep_path(self, tmp_path: Path):
        """The template a new line is planted with must deliver its body, not a marker.

        The path is deliberately deep: the head (absolute path + name + description) counts
        against the same budget, so a 1200-char seed truncated its own body at this depth.
        """
        seed = _dolores_meta().stubs_dir() / ".dolores" / "seeds" / "branch_ground"
        deep = tmp_path / ("d" * 120) / ("e" * 120)
        deep.mkdir(parents=True)
        manager, _ = self._manager(deep, seed_ground=seed)

        async with manager:
            manager.fork_branch("idea", "a direction")

        text = (await manager.aview_message("idea")).to_content_string()

        assert "# Branch ground" in text
        assert 'reason="budget"' not in text

    @pytest.mark.asyncio
    async def test_fork_hint_carries_the_origin_title_and_parent_ground(self, tmp_path: Path):
        manager, memento = self._manager(tmp_path, seed_ground=self._seed(tmp_path))
        main = memento.get_branch("main")
        anchor = manager.commit(session_id="s1", start_turn=0, end_turn=1)
        main.note(anchor.id, "the split\nwhy it split")
        main.path.joinpath("GROUND.md").write_text("# main\n\nmain's field", encoding="utf-8")

        async with manager:
            manager.fork_branch("idea", "a direction")

        text = (await manager.aview_message("idea")).to_content_string()

        assert 'title="the split"' in text
        assert 'ground="1"' in text
