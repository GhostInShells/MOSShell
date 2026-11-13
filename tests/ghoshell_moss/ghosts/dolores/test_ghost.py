"""The ``Dolores`` object itself — construction, instruction, memory, composition.

Answers: what a Dolores *is* before it starts thinking. The ego is elsewhere
(``test_ego.py``); this file stops at the shell the ego runs on.
"""

import asyncio
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghoshell_moss.core.blueprint.ghost import Ghost
from ghoshell_moss.core.blueprint.moment import Echoes
from ghoshell_moss.core.session.mock_session import MockSession

from ._support import _dolores, _dolores_meta, _OfflineConnection

class TestDolores:
    def test_is_ghost_abc(self):
        assert isinstance(_dolores(), Ghost)

    def test_lifecycle_no_error(self):
        ghost = _dolores()

        async def run():
            async with ghost:
                pass

        asyncio.run(run())

    def test_dsh_launcher_raises_before_start(self):
        ghost = _dolores()
        with pytest.raises(RuntimeError):
            ghost.dsh_launcher

    def test_no_dsh_launch_without_matrix(self):
        ghost = _dolores(home=None, session=MockSession())

        async def run():
            async with ghost:
                pass

        asyncio.run(run())
        # 无 matrix → 不拉起 dsh, launcher 保持 None.
        assert ghost._dsh_launcher is None
        with pytest.raises(RuntimeError):
            ghost.dsh_launcher


class TestDoloresArticulate:
    def test_yields_placeholder_logos(self):
        """无 ego (matrix=None) → think 只产出占位 logos, 不 crash."""
        ghost = _dolores()

        async def collect():
            return [delta async for delta in ghost.think(None)]

        assert asyncio.run(collect()) == [""]


class TestDoloresInstruction:
    def test_system_prompt_derives_two_meta_segments(self):
        """system_prompt = 原型元信息 + 身份描述, 从结构化 meta 派生, 无 baseline 时不含 baseline."""
        meta = _dolores_meta()
        text = _dolores(meta=meta).system_prompt()
        assert "prototype: Dolores" in text
        assert f"version: {meta.VERSION}" in text
        assert "name: dolores" in text
        assert "description:" in text

    @pytest.mark.asyncio
    async def test_ground_instruction_none_without_home(self):
        """无 home → 无 root ground, ground_instruction 返回 None."""
        ghost = _dolores()
        async with ghost:
            assert await ghost.ground_instruction() is None

    @pytest.mark.asyncio
    async def test_ground_instruction_renders_held_ground(self, tmp_path: Path):
        """home 存在 → stubs 同步落 GROUND.md, __aenter__ 打开 root ground, 渲染非空."""
        ghost = _dolores(home=tmp_path)
        async with ghost:
            text = await ghost.ground_instruction()
            assert text is not None
            assert text.strip() != ""


class TestDoloresMemoriesFromHome:
    """Dolores.memories — ground 渲染为第一条存在主义记忆."""

    @pytest.mark.asyncio
    async def test_memories_empty_without_home(self):
        """无 home → 无 ground 渲染, memories 返回空."""
        ghost = _dolores()
        async with ghost:
            assert ghost.memories() == []

    @pytest.mark.asyncio
    async def test_ground_is_content_for_the_instruction_not_the_memory(self, tmp_path: Path):
        """home 存在 → ground 渲染进 instruction; memory 里不再重复一份."""
        ghost = _dolores(home=tmp_path)
        async with ghost:
            assert ghost.memories() == []
            prompt = ghost.system_prompt()
            assert "<!-- ground: ghost home -->" in prompt
            assert ghost._ground_text and ghost._ground_text.strip() in prompt


class TestDoloresMemories:
    @pytest.mark.asyncio
    async def test_returns_single_memory_container(self, tmp_path: Path):
        from ghoshell_moss.memento import new_local_memento

        from ghoshell_moss.ghosts.dolores._ego_memento import EgoMementoConfig, EgoMementoManager

        memento = new_local_memento(tmp_path / "owner")
        memento.create_branch("main")
        manager = EgoMementoManager(connection=_OfflineConnection(), memento=memento, config=EgoMementoConfig())
        anchor = manager.commit(session_id="s1", start_turn=0, end_turn=1)
        memento.get_branch("main").note(anchor.id, "note")

        ghost = _dolores()
        ghost._memento_manager = manager
        # 渲染是异步的: refresh 写缓存, memories() 只读.
        await ghost.refresh_memory()

        memories = ghost.memories()

        assert len(memories) == 1
        text = memories[0].to_content_string()
        assert "<memory" in text
        # 快照要带"它成形于哪一刻" —— 绝对锚, UTC, 秒精度
        assert re.search(r'<memory instanced_at="\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z"', text)
        assert '<memento branch="main"' in text and "note" in text

    def test_returns_empty_when_nothing_to_remember(self):
        ghost = _dolores()
        assert ghost.memories() == []


class TestBuildChannel:
    @pytest.mark.asyncio
    async def test_mounts_frame_channel_when_frame_root_given(self, tmp_path: Path):
        from ghoshell_moss.ground import DefaultGroundSet

        from ghoshell_moss.ghosts.dolores.channel import build_dolores_channel

        (tmp_path / "GROUND.md").write_text("---\nname: g\n---\n# G\n")
        gs = DefaultGroundSet(workspace_root=tmp_path)
        frame_root = tmp_path / "frames"
        frame_root.mkdir()
        chan = build_dolores_channel(groundset=gs, frame_root=frame_root
        )
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            names = {m.name for m in runtime.metas().values()}
            assert "frame" in names
            assert "ground" in names

    @pytest.mark.asyncio
    async def test_no_frame_channel_without_frame_root(self, tmp_path: Path):
        from ghoshell_moss.ground import DefaultGroundSet

        from ghoshell_moss.ghosts.dolores.channel import build_dolores_channel

        (tmp_path / "GROUND.md").write_text("---\nname: g\n---\n# G\n")
        gs = DefaultGroundSet(workspace_root=tmp_path)
        chan = build_dolores_channel(groundset=gs)
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            names = {m.name for m in runtime.metas().values()}
            assert "frame" not in names
            assert "ground" in names

    @pytest.mark.asyncio
    async def test_model_organ_mounts_and_its_identity_rides_the_ghost_notice(self, tmp_path: Path):
        """ghost.model 挂上来, 当前模型走 ghost 根的 named notice (变才重发)."""
        from ghoshell_moss.ground import DefaultGroundSet

        from ghoshell_moss.ghosts.dolores.channel import build_dolores_channel
        from ghoshell_moss.ghosts.dolores.model_channel import ModelOrgan

        (tmp_path / "GROUND.md").write_text("---\nname: g\n---\n# G\n")
        gs = DefaultGroundSet(workspace_root=tmp_path)
        organ = ModelOrgan(
            read_catalog=self._empty_catalog,
            read_identity=lambda: ("deepseek-official", "deepseek-flash", "low"),
            declare=lambda provider, model, effort: f"declared {provider}/{model}",
        )
        chan = build_dolores_channel(groundset=gs, model=organ)
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            metas = {m.name: m for m in runtime.metas().values()}
        assert "model" in metas
        ghost_meta = metas["ghost"]
        assert ghost_meta.named_notices["model"] == "deepseek-official/deepseek-flash · thinking low"

    @pytest.mark.asyncio
    async def test_no_model_organ_means_no_model_channel(self, tmp_path: Path):
        """没有后端就不挂这个器官 —— 挂一个只会答"做不了"的命令是在骗模型."""
        from ghoshell_moss.ground import DefaultGroundSet

        from ghoshell_moss.ghosts.dolores.channel import build_dolores_channel

        (tmp_path / "GROUND.md").write_text("---\nname: g\n---\n# G\n")
        gs = DefaultGroundSet(workspace_root=tmp_path)
        chan = build_dolores_channel(groundset=gs)
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            metas = {m.name: m for m in runtime.metas().values()}
        assert "model" not in metas
        assert "model" not in metas["ghost"].named_notices

    @staticmethod
    async def _empty_catalog():
        from ghoshell_moss.deepseek_harness.v0_2_0 import ModelCatalog

        return ModelCatalog()

    def test_catalog_renders_models_with_their_own_depth_ladder(self):
        """目录文本按 provider 分组, 每个模型带自己的档位表 —— 换模型前先看见它能想多深."""
        from ghoshell_moss.deepseek_harness.v0_2_0 import (
            ModelCatalog,
            ModelCatalogModel,
            ModelProviderGroup,
            ModelReasoning,
            ModelReasoningEffort,
        )

        from ghoshell_moss.ghosts.dolores.model_channel import _catalog_text

        catalog = ModelCatalog(
            groups=[
                ModelProviderGroup(
                    id="deepseek-official",
                    name="DeepSeek",
                    models=[
                        ModelCatalogModel(
                            id="deepseek-flash",
                            name="DeepSeek-V41-Flash",
                            reasoning=ModelReasoning(
                                efforts=[ModelReasoningEffort(id="off"), ModelReasoningEffort(id="max")],
                                defaultEffort="off",
                            ),
                        )
                    ],
                )
            ],
            routableProviders=["deepseek-official"],
        )
        text = _catalog_text(catalog)
        assert "deepseek-official" in text
        assert "deepseek-flash" in text
        assert "off/max" in text
        assert "default off" in text

    @pytest.mark.asyncio
    async def test_init_frame_lands_in_notice(self, tmp_path: Path):
        from ghoshell_moss.channels.frame_channel import Frame
        from ghoshell_moss.ground import DefaultGroundSet

        from ghoshell_moss.ghosts.dolores.channel import build_dolores_channel

        (tmp_path / "GROUND.md").write_text("---\nname: g\n---\n# G\n")
        gs = DefaultGroundSet(workspace_root=tmp_path)
        frame_root = tmp_path / "frames"
        frame_root.mkdir()
        seed = Frame(label="orient", questions=["Where am I?"])
        chan = build_dolores_channel(groundset=gs, frame_root=frame_root, init_frame=seed
        )
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            frame_meta = next(m for m in runtime.metas().values() if m.name == "frame")
        assert "Where am I?" in frame_meta.named_notices["orient"]


class TestDoloresMomentPayload:
    """DoloresEgo 的 moment 映射 — context (inject) + inputs (steer) 两条 message."""

    def _ego(self):
        from ghoshell_moss.ghosts.dolores._ego import DoloresEgo, DoloresEgoContext

        return DoloresEgo(
            launcher=None,
            ctx=DoloresEgoContext(
                project_home=Path("."),
                project_name="pytest",
                name="dolores",
                mode="pytest",
                instruction="i",
                facade=None,
            ),
        )

    def test_moment_payload_splits_context_and_inputs(self):
        from ghoshell_moss.core.blueprint.moment import Echoes, Moment
        from ghoshell_moss.message import Message

        moment = Moment(
            previous=Echoes(messages=[Message.new().with_content("echo")]),
            inputs={"test": [Message.new().with_content("percept")]},
            hint="hint text",
            command_logos="cmd!",
        )
        payload = self._ego()._moment_payload(moment, "0-1")
        assert payload["moment_id"] == "0-1"
        context_text = "".join(c["text"] for c in payload["context"] if c.get("type") == "text")
        inputs_text = "".join(c["text"] for c in payload["inputs"] if c.get("type") == "text")
        # context: echoes + executing (cmd!), 排除 percept/hint.
        assert "echo" in context_text and "cmd!" in context_text
        assert "percept" not in context_text and "hint text" not in context_text
        # inputs: percept 平铺 + hint (hint 排后).
        assert "percept" in inputs_text and "hint text" in inputs_text
        assert inputs_text.index("percept") < inputs_text.index("hint text")

    def test_moment_payload_empty_when_no_context_or_inputs(self):
        from ghoshell_moss.core.blueprint.moment import Moment

        payload = self._ego()._moment_payload(Moment(), "0-0")
        assert payload["context"] == []
        assert payload["inputs"] == []

    def test_moment_payload_carries_both_slots(self):
        """一条 moment 投递必须同时带 context 与 inputs —— 少一半就是一次静默丢字.

        病根: ``observe()`` 把挂起的 percepts 抽干塞进它产出的 moment, 而投递走哪条路由别人决定.
        所以任何一条路只要结构上不带 inputs, 就会把已消费的文字蒸发 (D13 的 buffered 语音)。
        """
        from ghoshell_moss.core.blueprint.moment import Echoes, Moment
        from ghoshell_moss.message import Message

        moment = Moment(
            previous=Echoes(messages=[Message.new().with_content("echo")]),
            inputs={"test": [Message.new().with_content("percept")]},
        )
        payload = self._ego().moment_payload(moment, "0-1")
        assert payload["moment_id"] == "0-1"
        context_text = "".join(c["text"] for c in payload["context"] if c.get("type") == "text")
        assert "echo" in context_text
        inputs_text = "".join(c["text"] for c in payload["inputs"] if c.get("type") == "text")
        assert "percept" in inputs_text

    def test_moment_payload_empty_when_no_content(self):
        from ghoshell_moss.core.blueprint.moment import Moment

        payload = self._ego().moment_payload(Moment(), "0-0")
        assert payload["context"] == []
        assert payload["inputs"] == []


class TestDoloresEpochPayload:
    """DoloresEgo 的 epoch 槽位 — <cognition_epoch> 容器 (recap + baseline), epoch 变更时才返回."""

    def _ego(self):
        from ghoshell_moss.ghosts.dolores._ego import DoloresEgo, DoloresEgoContext

        return DoloresEgo(
            launcher=None,
            ctx=DoloresEgoContext(
                project_home=Path("."),
                project_name="pytest",
                name="dolores",
                mode="pytest",
                instruction="i",
                facade=None,
            ),
        )

    def _thinking(self, epoch):
        class _Observer:
            @property
            def epoch(self):
                return epoch

        class _Thinking:
            def __init__(self, observer):
                self.observer = observer

        return _Thinking(_Observer())

    def test_epoch_payload_renders_epoch_container(self):
        from ghoshell_moss.core.blueprint.moment import Epoch
        from ghoshell_moss.message import Message

        epoch = Epoch(
            id="e1",
            index=1,
            recap=[Message.new(tag="summary").with_content("past")],
            baseline={"facade": "channel tree"},
        )
        payload = self._ego()._epoch_payload(self._thinking(epoch))
        assert payload is not None
        text = "".join(c["text"] for c in payload if c.get("type") == "text")
        assert "<cognition_epoch" in text
        assert 'index="1"' in text
        assert "<recap>" in text
        assert "past" in text
        assert "<baseline>" in text
        assert "<facade>" in text
        assert "channel tree" in text

    def test_epoch_payload_none_when_epoch_unchanged(self):
        from ghoshell_moss.core.blueprint.moment import Epoch

        ego = self._ego()
        epoch = Epoch(id="e1", index=1, recap=[], baseline={})
        assert ego._epoch_payload(self._thinking(epoch)) is None  # 空 epoch → 不注入
        # 已记录的 epoch 再次进入 → None (不变更).
        epoch2 = Epoch(id="e1", index=1, recap=[], baseline={})
        assert ego._epoch_payload(self._thinking(epoch2)) is None


class TestDoloresEpoch:
    """epoch 周期 — 首次 thinking 开 epoch; compact 后开新 epoch 并交付 memory recap."""

    @staticmethod
    def _ghost():
        ghost = _dolores()
        moments = _FakeMoments()
        ghost._mindflow = SimpleNamespace(moments=moments)
        return ghost, moments

    @staticmethod
    def _memory_message():
        from ghoshell_moss.message import Message

        return Message.new(tag="memory").with_content("ground")

    @pytest.mark.asyncio
    async def test_first_epoch_has_no_memory_recap(self):
        ghost, moments = self._ghost()
        ghost._ego = _FakeThinkEgo()

        assert [delta async for delta in ghost.think(None)] == ["x"]

        assert moments.epochs == 1
        assert moments.recap_funcs == {}

    @pytest.mark.asyncio
    async def test_compact_opens_epoch_with_memory_recap(self):
        ghost, moments = self._ghost()
        ghost._ego = _FakeThinkEgo(memory=[self._memory_message()])

        assert [delta async for delta in ghost.think(None)] == ["x"]

        assert ghost._ego.compact_calls == 1
        assert moments.epochs == 1
        assert "memory" in moments.recap_funcs
        assert "<memory>" in moments.epoch_recaps[0][0].to_content_string()

    @pytest.mark.asyncio
    async def test_cancelled_compact_is_completed_on_the_next_frame(self):
        """上次 think 在 compact RPC 窗口被取消 → 本帧取走遗留 memory 补开 epoch."""
        ghost, moments = self._ghost()
        ghost._ego = _FakeThinkEgo(leftover=[self._memory_message()])

        assert [delta async for delta in ghost.think(None)] == ["x"]

        assert ghost._ego.take_calls == 1
        assert "memory" in moments.recap_funcs
        assert "<memory>" in moments.epoch_recaps[0][0].to_content_string()


class _FakeMoments:
    """moments 的录制替身 — epoch 次数 / recap 注册 / recap 产物."""

    def __init__(self):
        self.epochs = 0
        self.recap_funcs: dict = {}
        self.epoch_recaps: list[list] = []

    def with_epoch_recap(self, key, recap_func):
        self.recap_funcs[key] = recap_func

    def new_epoch(self, recap):
        self.epochs += 1
        self.epoch_recaps.append([msg for func in self.recap_funcs.values() for msg in func()])


class _FakeRunContext:
    def __init__(self, logos: list[str]):
        self._logos = logos

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def logos(self):
        for delta in self._logos:
            yield delta


class _FakeThinkEgo:
    def __init__(self, *, memory=None, leftover=None):
        self._memory = memory
        self._leftover = leftover
        self.compact_calls = 0
        self.take_calls = 0

    async def compact_if_pending(self):
        self.compact_calls += 1
        return self._memory

    def take_compact_memory(self):
        self.take_calls += 1
        memory, self._leftover = self._leftover, None
        return memory

    def run_thinking(self, thinking):
        return _FakeRunContext(["x"])
