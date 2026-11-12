"""``memento`` — the channel the ghost uses as its memory organ.

Answers: what the model can do through it (read a span, commit an anchor, list branches),
and what it refuses to do when there is no backend or the coordinate is bad.
"""

import asyncio
from pathlib import Path

import pytest

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

    @staticmethod
    def _committer(manager, spans=((0, 1),)):
        """假 ego —— 每次调用封一段, 段用完后返回"没有新东西"的回执.

        与 ego 同形: 真实 ego 在 turn/end 落地, 这里在调用时落地 (通道只验透传); 落锚之后
        **照常**排旁路, 让开与否归 ``schedule_note`` 自己判 —— 约定落在那一个判据上, 不落在
        调用方的小心翼翼上.
        """
        pending = list(spans)

        def commit_anchor(note: str) -> str:
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

        return commit_anchor

    def _channel(self, tmp_path: Path, connection=None, *, with_commit: bool = False):
        from ghoshell_moss.ghosts.dolores.memento_channel import build_memento_channel

        manager, memento = self._manager(tmp_path, connection)
        chan = build_memento_channel(
            manager,
            storage_root=tmp_path / "owner",
            commit_anchor=self._committer(manager) if with_commit else None,
        )
        return chan, manager, memento

    @pytest.mark.asyncio
    async def test_read_surface_is_exposed(self, tmp_path: Path):
        chan, _, _ = self._channel(tmp_path)
        async with chan.bootstrap() as runtime:
            for name in ("view", "read", "history", "chat", "node"):
                assert runtime.get_command(name) is not None

    @pytest.mark.asyncio
    async def test_node_is_created_on_first_use(self, tmp_path: Path):
        chan, manager, memento = self._channel(tmp_path)
        anchor = manager.commit(session_id="s1", start_turn=0, end_turn=1)
        async with chan.bootstrap() as runtime:
            listing = await runtime.execute_command(
                "node", args=(self._coord(memento, anchor.seq),)
            )
        node_dir = (
            memento.get_branch("main").get_commit(anchor.seq).memento_path(memento.root).parent
        )
        assert listing.startswith(f"Directory: {node_dir}")
        assert "MEMENTO.md" in listing
        assert (node_dir / "MEMENTO.md").exists()

    @pytest.mark.asyncio
    async def test_node_lists_what_was_left_there_without_touching_it(self, tmp_path: Path):
        """节点是写者的地盘 —— memento 只列, 不重写."""
        chan, manager, memento = self._channel(tmp_path)
        anchor = manager.commit(session_id="s1", start_turn=0, end_turn=1)
        coord = self._coord(memento, anchor.seq)
        node_dir = (
            memento.get_branch("main").get_commit(anchor.seq).memento_path(memento.root).parent
        )
        async with chan.bootstrap() as runtime:
            await runtime.execute_command("node", args=(coord,))
            (node_dir / "souvenir.md").write_text("for my future self", encoding="utf-8")
            listing = await runtime.execute_command("node", args=(coord,))
        assert "souvenir.md" in listing
        assert "MEMENTO.md" in listing
        assert (node_dir / "souvenir.md").read_text(encoding="utf-8") == "for my future self"

    @pytest.mark.asyncio
    async def test_node_on_unknown_coord_returns_hint(self, tmp_path: Path):
        chan, _, _ = self._channel(tmp_path)
        async with chan.bootstrap() as runtime:
            result = await runtime.execute_command("node", args=("1-99",))
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
        # 投递走 Dolores 自己的 ego signal, 而不是全局的 input signal.
        assert sent[0].name == "dolores/ego"
        assert sent[0].role == "mind"
        assert sent[0].messages[0].to_content_string() == "the answer"

    @pytest.mark.asyncio
    async def test_read_on_unknown_coord_returns_hint(self, tmp_path: Path):
        chan, _, _ = self._channel(tmp_path)
        async with chan.bootstrap() as runtime:
            result = await runtime.execute_command("read", args=("1-99",))
        assert result == "[memento] no commit at `1-99`"

    @pytest.mark.asyncio
    async def test_instruction_names_trajectory_and_storage(self, tmp_path: Path):
        chan, manager, memento = self._channel(tmp_path)
        anchor = manager.commit(session_id="s1", start_turn=0, end_turn=1)
        memento.get_branch("main").note(anchor.id, "note")
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            meta = runtime.self_meta()
        assert "memento trajectory" in meta.instruction
        assert str((tmp_path / "owner").resolve()) in meta.instruction
        assert "main" in meta.notice
        assert "commits=1" in meta.notice

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
