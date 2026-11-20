"""``memento`` — the channel the ghost uses as its memory organ.

Answers: what the model can do through it (read a span, commit an anchor, list branches),
and what it refuses to do when there is no backend or the coordinate is bad.
"""

import asyncio
from datetime import timedelta
from pathlib import Path

import pytest

from ._support import _OfflineConnection


class TestMementoChannel:
    """memento channel — ghost 的记忆器官: 读面透传 + 落锚写面 + 坏 commit 失败 + notice 列 branch."""

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

        def attach_session(self, session_id, logger=None, *, follow=True):
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
    def _manager(tmp_path: Path, connection=None, mementos=None):
        from ghoshell_moss.memento import new_local_memento

        from ghoshell_moss.ghosts.dolores._ego_memento import EgoMementoConfig, EgoMementoManager

        memento = new_local_memento(tmp_path / "owner", writable=True)
        memento.create_branch("main")
        return EgoMementoManager(
            connection=connection or _OfflineConnection(),
            memento=memento,
            config=EgoMementoConfig(),
            mementos=mementos,
        ), memento

    @staticmethod
    def _coord(memento, seq: int) -> str:
        return memento.get_branch("main").get_commit(seq).coord

    @staticmethod
    def _committer(manager, spans=((0, 1),)):
        """假 ego —— 每次调用封一段, 段用完后返回"没有新东西"的回执.

        与 ego 同形: 真实 ego 在 turn/end 落地, 这里在调用时落地 (通道只验透传); 落锚之后
        **照常**排旁路, 让开与否归 ``schedule_note`` 自己判 —— 约定落在那一个判据上, 不落在
        调用方的小心翼翼上.
        """
        pending = list(spans)
        calls: list[tuple[str, bool]] = []

        def commit_anchor(note: str, replan: bool = False) -> str:
            calls.append((note, replan))
            if not pending:
                return "[memento] nothing new since the last anchor — not committed"
            start, end = pending.pop(0)
            anchor = manager.commit(
                session_id="s1", start_turn=start, end_turn=end, message=note
            )
            if anchor is None:
                return "[memento] nothing new since the last anchor — not committed"
            manager.schedule_note(anchor.id)
            return "[memento] commit queued — it takes effect at the end of this turn."

        commit_anchor.calls = calls
        return commit_anchor

    def _channel(self, tmp_path: Path, connection=None, *, with_commit: bool = False):
        from ghoshell_moss.ghosts.dolores.memento_channel import build_memento_channel

        manager, memento = self._manager(tmp_path, connection)
        chan = build_memento_channel(
            manager,
            commit_anchor=self._committer(manager) if with_commit else None,
        )
        return chan, manager, memento

    @pytest.mark.asyncio
    async def test_read_surface_is_exposed(self, tmp_path: Path):
        chan, _, _ = self._channel(tmp_path)
        async with chan.bootstrap() as runtime:
            for name in ("view", "read", "history", "chat", "memento_dir", "fork", "archive", "branches"):
                assert runtime.get_command(name) is not None

    @pytest.mark.asyncio
    async def test_memento_dir_returns_the_address_without_creating_it(self, tmp_path: Path):
        """An address is not a write: the directory appears only when its writer creates it."""
        chan, manager, memento = self._channel(tmp_path)
        anchor = manager.commit(session_id="s1", start_turn=0, end_turn=1)
        async with chan.bootstrap() as runtime:
            path = await runtime.execute_command(
                "memento_dir", args=(self._coord(memento, anchor.seq),)
            )
        node_dir = (
            memento.get_branch("main").get_commit(anchor.seq).memento_path(memento.root).parent
        )
        assert path == str(node_dir)
        assert not node_dir.exists()

    @pytest.mark.asyncio
    async def test_memento_dir_on_unknown_coord_returns_hint(self, tmp_path: Path):
        chan, _, _ = self._channel(tmp_path)
        async with chan.bootstrap() as runtime:
            result = await runtime.execute_command("memento_dir", args=("1-99",))
        assert result == "[memento] no commit at `1-99`"

    @pytest.mark.asyncio
    async def test_read_returns_the_transcript(self, tmp_path: Path):
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
        chan, manager, memento = self._channel(tmp_path, self._Connection({"events": events}))
        anchor = manager.commit(session_id="s1", start_turn=0, end_turn=1)
        async with chan.bootstrap() as runtime:
            text = await runtime.execute_command(
                "read", args=(self._coord(memento, anchor.seq),)
            )
        assert text == "  > 为什么\n  ~ 因为"

    @pytest.mark.asyncio
    async def test_chat_on_broken_commit_returns_hint(self, tmp_path: Path):
        chan, manager, memento = self._channel(tmp_path, self._Connection({"message": "reply"}))
        anchor = manager.commit(session_id="s1", start_turn=0, end_turn=1)
        memento.get_branch("main").note(anchor.id, "", error="bypass failed")
        async with chan.bootstrap() as runtime:
            result = await runtime.execute_command(
                "chat", args=(self._coord(memento, anchor.seq), "why?")
            )
        assert "no context to talk to" in result
        assert "read it instead" in result

    @pytest.mark.asyncio
    async def test_chat_returns_a_receipt_and_delivers_the_answer_later(self, tmp_path, monkeypatch):
        """异步化: 命令立刻回执, 答案作为一条 mind 分类的投递回来 —— 不再占住这一轮."""
        from ghoshell_moss.core.blueprint.channel_builder import CommandUtil

        conn = self._Connection({"text": "the answer"})
        chan, manager, memento = self._channel(tmp_path, conn)
        anchor = manager.commit(session_id="s1", start_turn=0, end_turn=1)
        sent = []
        monkeypatch.setattr(CommandUtil, "send_signal", lambda signal: sent.append(signal))
        async with chan.bootstrap() as runtime:
            receipt = await runtime.execute_command(
                "chat", args=(self._coord(memento, anchor.seq), "why?")
            )
            for _ in range(100):
                if sent:
                    break
                await asyncio.sleep(0.01)
        assert receipt.startswith("[memento] asking")
        assert len(sent) == 1
        # 投递走 mind/delivery; role=mind / mode=next 由 MindDeliveryNucleus 在 impulse 上设置.
        assert sent[0].name == "mind/delivery"
        assert sent[0].messages[0].to_content_string() == "the answer"

    @pytest.mark.asyncio
    async def test_chat_notice_says_none_when_no_chat_is_running(self, tmp_path: Path):
        """The fragment is always present: idle is a readable state, not an absent one."""
        chan, _, _ = self._channel(tmp_path)
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            meta = runtime.self_meta()
        assert meta.named_notices.get("chat") == "none"

    @pytest.mark.asyncio
    async def test_chat_notice_counts_running_tasks_and_returns_to_none(self, tmp_path, monkeypatch):
        """In flight it says how many; when the answer lands the count returns to zero."""
        from ghoshell_moss.core.blueprint.channel_builder import CommandUtil

        entered = asyncio.Event()
        release = asyncio.Event()

        class _SlowSession(self._Connection._Session):
            async def chat_off_record(self, messages, *, anchor=None, **kwargs):
                entered.set()
                await release.wait()
                return await super().chat_off_record(messages, anchor=anchor, **kwargs)

        class _SlowConnection(self._Connection):
            def attach_session(self, session_id, logger=None, *, follow=True):
                return _SlowSession(self, session_id)

        conn = _SlowConnection({"text": "the answer"})
        chan, manager, memento = self._channel(tmp_path, conn)
        anchor = manager.commit(session_id="s1", start_turn=0, end_turn=1)
        sent = []
        monkeypatch.setattr(CommandUtil, "send_signal", lambda signal: sent.append(signal))
        async with chan.bootstrap() as runtime:
            receipt = await runtime.execute_command(
                "chat", args=(self._coord(memento, anchor.seq), "why?")
            )
            await asyncio.wait_for(entered.wait(), 2)
            await runtime.refresh_metas()
            running = runtime.self_meta().named_notices.get("chat")
            release.set()
            for _ in range(100):
                if sent:
                    break
                await asyncio.sleep(0.01)
            await runtime.refresh_metas()
            idle = runtime.self_meta().named_notices.get("chat")
        assert receipt.startswith("[memento] asking")
        assert running == "1 running"
        assert idle == "none"
        assert len(sent) == 1

    @pytest.mark.asyncio
    async def test_chat_failure_is_delivered_instead_of_disappearing(self, tmp_path, monkeypatch):
        """A failed chat comes back as content; the notice returns to none either way."""
        from ghoshell_moss.core.blueprint.channel_builder import CommandUtil

        conn = self._Connection(RuntimeError("connection lost"))
        chan, manager, memento = self._channel(tmp_path, conn)
        anchor = manager.commit(session_id="s1", start_turn=0, end_turn=1)
        sent = []
        monkeypatch.setattr(CommandUtil, "send_signal", lambda signal: sent.append(signal))
        async with chan.bootstrap() as runtime:
            receipt = await runtime.execute_command(
                "chat", args=(self._coord(memento, anchor.seq), "why?")
            )
            for _ in range(100):
                if sent:
                    break
                await asyncio.sleep(0.01)
            await runtime.refresh_metas()
            idle = runtime.self_meta().named_notices.get("chat")
        assert receipt.startswith("[memento] asking")
        assert len(sent) == 1
        text = sent[0].messages[0].to_content_string()
        assert "failed" in text
        assert "connection lost" in text
        assert idle == "none"

    @pytest.mark.asyncio
    async def test_read_on_unknown_coord_returns_hint(self, tmp_path: Path):
        chan, _, _ = self._channel(tmp_path)
        async with chan.bootstrap() as runtime:
            result = await runtime.execute_command("read", args=("1-99",))
        assert result == "[memento] no commit at `1-99`"

    @pytest.mark.asyncio
    async def test_notice_counts_only_the_current_branch(self, tmp_path: Path):
        chan, manager, memento = self._channel(tmp_path)
        memento.create_branch("side")
        manager.commit(session_id="s1", start_turn=0, end_turn=1)
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            meta = runtime.self_meta()
        assert "main #1" in meta.notice
        assert "commits=1" in meta.notice
        assert "side #2" in meta.notice
        side_line = next(line for line in meta.notice.splitlines() if "side" in line)
        assert "commits=" not in side_line

    @pytest.mark.asyncio
    async def test_checkout_is_not_exposed_without_a_backend(self, tmp_path: Path):
        """No transition backend, no command — the same rule as `commit`."""
        chan, _, _ = self._channel(tmp_path)
        async with chan.bootstrap() as runtime:
            assert runtime.get_command("checkout") is None

    @pytest.mark.asyncio
    async def test_checkout_passes_name_title_and_body_through(self, tmp_path: Path):
        from ghoshell_moss.ghosts.dolores.memento_channel import build_memento_channel

        manager, _ = self._manager(tmp_path)
        seen: dict = {}

        def hook(name: str, title: str, note: str) -> str:
            seen.update(name=name, title=title, note=note)
            return "[memento] checkout queued"

        chan = build_memento_channel(manager, checkout_hook=hook)
        async with chan.bootstrap() as runtime:
            receipt = await runtime.execute_command(
                "checkout", kwargs={"name": "idea", "title": "why", "text__": "the body"}
            )
        assert receipt == "[memento] checkout queued"
        assert seen == {"name": "idea", "title": "why", "note": "the body"}

    @pytest.mark.asyncio
    async def test_commit_is_not_exposed_without_a_backend(self, tmp_path: Path):
        """没有落锚后端就不挂命令 —— 挂一条只会答"做不了"的命令是在骗模型."""
        chan, _, _ = self._channel(tmp_path)
        async with chan.bootstrap() as runtime:
            assert runtime.get_command("commit") is None

    @pytest.mark.asyncio
    async def test_own_note_makes_the_ghost_the_author(self, tmp_path: Path):
        conn = self._Connection({"message": "the sidecar would have said this"})
        chan, manager, memento = self._channel(tmp_path, conn, with_commit=True)
        async with chan.bootstrap() as runtime:
            result = await runtime.execute_command(
                "commit", args=("D26 the fix holds\nverified after restart",)
            )
        assert result == "[memento] commit queued — it takes effect at the end of this turn."
        branch = memento.get_branch("main")
        note = branch.notes()[branch.commits()[0].id]
        assert note.message == "D26 the fix holds\nverified after restart"
        # 自己写了 note = 有作者了, 旁路整条让开 (不是"排了但被跳过").
        assert manager.bypass == {}
        assert conn.calls == []

    @pytest.mark.asyncio
    async def test_empty_note_hands_the_summary_to_the_sidecar(self, tmp_path: Path):
        conn = self._Connection({"text": "auto note"})
        chan, manager, memento = self._channel(tmp_path, conn, with_commit=True)
        async with chan.bootstrap() as runtime:
            result = await runtime.execute_command("commit", args=("",))
        assert result == "[memento] commit queued — it takes effect at the end of this turn."
        await manager.drain_bypass()
        branch = memento.get_branch("main")
        assert branch.notes()[branch.commits()[0].id].message == "auto note"

    @pytest.mark.asyncio
    async def test_commit_on_an_empty_span_returns_a_hint(self, tmp_path: Path):
        chan, _, _ = self._channel(tmp_path, with_commit=True)
        async with chan.bootstrap() as runtime:
            first = await runtime.execute_command("commit", args=("one",))
            second = await runtime.execute_command("commit", args=("two",))
        assert first == "[memento] commit queued — it takes effect at the end of this turn."
        assert second == "[memento] nothing new since the last anchor — not committed"

    @pytest.mark.asyncio
    async def test_an_op_commit_composes_the_op_line(self, tmp_path: Path):
        """op 是参数, 不是从散文里解析出来的: 首行进 headline, 其余进 body."""
        from ghoshell_moss.ghosts.dolores.memento_channel import build_memento_channel

        manager, memento = self._manager(tmp_path)
        committer = self._committer(manager)
        chan = build_memento_channel(manager, commit_anchor=committer)
        async with chan.bootstrap() as runtime:
            result = await runtime.execute_command(
                "commit", args=("the next frame\n\nopen: what still has no answer", "replan")
            )

        assert result == "[memento] commit queued — it takes effect at the end of this turn."
        branch = memento.get_branch("main")
        assert branch.notes()[branch.commits()[0].id].message == (
            "replan: the next frame\n\nopen: what still has no answer"
        )
        assert committer.calls == [(
            "replan: the next frame\n\nopen: what still has no answer", True,
        )]

    @pytest.mark.asyncio
    async def test_an_op_commit_without_a_note_is_refused(self, tmp_path: Path):
        """op 是对 note 的一个声明; 让 summarizer 填内容等于伪造这个声明."""
        from ghoshell_moss.ghosts.dolores.memento_channel import build_memento_channel

        manager, memento = self._manager(tmp_path)
        committer = self._committer(manager)
        chan = build_memento_channel(manager, commit_anchor=committer)
        async with chan.bootstrap() as runtime:
            result = await runtime.execute_command("commit", args=("  ", "replan"))

        assert "needs a note" in result
        assert committer.calls == []
        assert memento.get_branch("main").commits() == []

    @pytest.mark.asyncio
    async def test_an_invalid_op_is_refused_before_the_anchor(self, tmp_path: Path):
        from ghoshell_moss.ghosts.dolores.memento_channel import build_memento_channel

        manager, memento = self._manager(tmp_path)
        committer = self._committer(manager)
        chan = build_memento_channel(manager, commit_anchor=committer)
        async with chan.bootstrap() as runtime:
            result = await runtime.execute_command("commit", args=("a note", "not an op"))

        assert "refused" in result
        assert committer.calls == []
        assert memento.get_branch("main").commits() == []

    @pytest.mark.asyncio
    async def test_fork_opens_a_line_from_the_current_position(self, tmp_path: Path):
        chan, manager, memento = self._channel(tmp_path)
        manager.commit(session_id="s1", start_turn=0, end_turn=1)
        async with chan.bootstrap() as runtime:
            receipt = await runtime.execute_command("fork", args=("idea", "a direction"))
        assert "forked `idea` #2" in receipt
        child = memento.get_branch("idea")
        assert child.meta().fork_from.commit_id == memento.get_branch("main").commits()[-1].id

    @pytest.mark.asyncio
    async def test_fork_refuses_a_coordinate_from_another_line(self, tmp_path: Path):
        chan, manager, memento = self._channel(tmp_path)
        manager.commit(session_id="s1", start_turn=0, end_turn=1)
        other = memento.create_branch("other")
        other.commit(message="elsewhere")
        coord = other.get_commit(1).coord
        async with chan.bootstrap() as runtime:
            receipt = await runtime.execute_command("fork", args=("idea", "", coord))
        assert "can only be forked" in receipt

    @pytest.mark.asyncio
    async def test_archive_refuses_the_main_line(self, tmp_path: Path):
        chan, _, _ = self._channel(tmp_path)
        async with chan.bootstrap() as runtime:
            receipt = await runtime.execute_command("archive", args=("main",))
        assert "main line" in receipt

    @pytest.mark.asyncio
    async def test_archive_closes_the_subtree_deepest_first(self, tmp_path: Path):
        chan, manager, memento = self._channel(tmp_path)
        manager.commit(session_id="s1", start_turn=0, end_turn=1)
        async with chan.bootstrap() as runtime:
            await runtime.execute_command("fork", args=("idea", "a direction"))
        memento.get_branch("idea").fork("leaf")
        async with chan.bootstrap() as runtime:
            receipt = await runtime.execute_command("archive", args=("idea",))
        assert "archived 2 line(s)" in receipt
        assert receipt.index("leaf") < receipt.index("idea")
        assert memento.get_branch("idea") is None

    @pytest.mark.asyncio
    async def test_branches_pages_the_roster_and_marks_archived(self, tmp_path: Path):
        chan, manager, memento = self._channel(tmp_path)
        manager.commit(session_id="s1", start_turn=0, end_turn=1)
        async with chan.bootstrap() as runtime:
            await runtime.execute_command("fork", args=("idea", "a direction"))
        memento.get_branch("idea").fork("leaf")
        memento.archive_branch("leaf")
        async with chan.bootstrap() as runtime:
            first = await runtime.execute_command("branches", args=(0, 2))
            second = await runtime.execute_command("branches", args=(2, 2))
        assert "branches 1-2 of 3" in first
        assert "you are here" in first
        assert "branches 3-3 of 3" in second
        assert "(archived)" in second
        assert "leaf" in second

    @pytest.mark.asyncio
    async def test_view_scrolls_back_before_a_coordinate(self, tmp_path: Path):
        """游标是排他的: 给一个坐标, 看到的是它**之前**那一窗 —— 坐标本身已经在你脚下."""
        chan, manager, memento = self._channel(tmp_path)
        for turn in range(5):
            manager.commit(
                session_id="s1", start_turn=turn, end_turn=turn + 1, message=f"step {turn}"
            )
        async with chan.bootstrap() as runtime:
            window = await runtime.execute_command(
                "view", kwargs={"limit": 2, "coord": self._coord(memento, 5)}
            )
        assert "step 3" in window
        assert "step 2" in window
        assert "step 4" not in window                  # 游标那一条不重复出现在"它的过去"里
        assert '<commit seq="1-4"' in window           # 窗口退到游标的前一条为止
        assert '<commit seq="1-2"' not in window       # 更早的只留在 <earlier> 摘要里, 不进详情

    @pytest.mark.asyncio
    async def test_view_cursor_only_scrolls_its_own_line(self, tmp_path: Path):
        """坐标属于别的线时明确拒绝 —— 悄悄整条渲染等于用错了坐标还以为看到了历史."""
        chan, manager, memento = self._channel(tmp_path)
        manager.commit(session_id="s1", start_turn=0, end_turn=1)
        other = memento.create_branch("other")
        other.commit(message="elsewhere")
        async with chan.bootstrap() as runtime:
            receipt = await runtime.execute_command(
                "view", kwargs={"coord": other.get_commit(1).coord}
            )
        assert "only scrolls the line it belongs to" in receipt

    @pytest.mark.asyncio
    async def test_view_cursor_on_a_missing_coordinate_says_so(self, tmp_path: Path):
        chan, _, _ = self._channel(tmp_path)
        async with chan.bootstrap() as runtime:
            receipt = await runtime.execute_command("view", kwargs={"coord": "1-999"})
        assert receipt == "[memento] no commit at `1-999`"

    @pytest.mark.asyncio
    async def test_history_reads_a_bare_timestamp_on_the_body_clock(self, tmp_path: Path):
        """不带时区的时间按本机时钟解释 —— 与渲染出来的日期同一套口径, 不是 UTC."""
        chan, manager, memento = self._channel(tmp_path)
        anchor = manager.commit(session_id="s1", start_turn=0, end_turn=1, message="late night")
        created = memento.get_branch("main").get_commit(anchor.seq).created.astimezone()
        start = (created - timedelta(minutes=1)).replace(tzinfo=None).isoformat()
        end = (created + timedelta(minutes=1)).replace(tzinfo=None).isoformat()
        async with chan.bootstrap() as runtime:
            window = await runtime.execute_command("history", kwargs={"from_date": start, "until_date": end})
        assert "late night" in window


class TestMementoOwnerScope:
    """A read may name another owner; an unnamed line stays that owner's `main`."""

    def test_a_named_owner_resolves_against_its_own_branches(self, tmp_path):
        from ghoshell_moss.ghosts.dolores.mementos import FsDoloresMementos

        mementos = FsDoloresMementos(tmp_path / ".memento")
        other = mementos.get("dreamers/journal", or_create=True)
        other.create_branch("journal")

        manager, _own = TestMementoChannel._manager(tmp_path, mementos=mementos)

        assert [ref.name for _, ref in manager.list_branches(owner="dreamers/journal")] == ["journal"]
        assert [ref.name for _, ref in manager.list_branches()] == ["main"]
        total, page = manager.branch_catalog(owner="dreamers/journal")
        assert total == 1
        assert page[0][1].name == "journal"

    def test_naming_an_owner_without_a_registry_is_an_error(self, tmp_path):
        manager, _own = TestMementoChannel._manager(tmp_path)

        with pytest.raises(LookupError):
            manager.list_branches(owner="ghost")
