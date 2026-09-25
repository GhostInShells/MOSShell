"""Dolores Ghost 原型测试 — 骨架 + stubs 同步阶段.

覆盖:
- GhostMeta/Ghost ABC 契约
- stubs 同步三种路径: init / override / noop
- 构造无副作用 (写盘收敛到 __aenter__)
- session.output 提示在 init/override 时发出, noop 不发声
"""

import asyncio
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from ghoshell_container import Container

from ghoshell_moss.core.blueprint.ghost import Ghost, GhostMeta
from ghoshell_moss.core.blueprint.moment import Echoes
from ghoshell_moss.core.session.mock_session import MockSession


def _dolores_meta(**kwargs):
    from ._meta import DoloresMeta

    return DoloresMeta(**kwargs)


def _dolores(meta=None, *, home=None, session=None, matrix=None, shell=None, base_instruction=None):
    from ._runtime import Dolores

    return Dolores(
        meta=meta or _dolores_meta(),
        home=home,
        session=session,
        matrix=matrix,
        shell=shell,
        base_instruction=base_instruction,
    )


class TestDoloresMeta:
    def test_defaults(self):
        from .nucleus import DoloresEgoNucleusMeta

        meta = _dolores_meta()
        assert meta.name() == "dolores"
        assert meta.prototype() == "Dolores"
        # Dolores 默认挂载 ego 自醒 nucleus (self-wake 通道).
        metas = meta.nuclei_metas()
        assert len(metas) == 1
        assert isinstance(metas[0], DoloresEgoNucleusMeta)

    def test_nuclei_metas_fully_replaced_when_passed(self):
        """显式传 nuclei_metas 时完全替换默认, 不叠加."""
        from ghoshell_moss.core.blueprint.mindflow import NucleusMeta
        from .nucleus import DoloresEgoNucleusMeta

        custom = DoloresEgoNucleusMeta()
        meta = _dolores_meta(nuclei_metas=[custom])
        assert meta.nuclei_metas() == [custom]

    def test_is_ghost_meta_abc(self):
        assert isinstance(_dolores_meta(), GhostMeta)

    def test_version_constant(self):
        assert isinstance(_dolores_meta().VERSION, str)
        assert _dolores_meta().VERSION

    def test_stubs_dir_contains_ground(self):
        stubs = _dolores_meta().stubs_dir()
        assert stubs.is_dir()
        assert (stubs / "GROUND.md").exists()

    def test_factory_returns_dolores(self):
        from ._runtime import Dolores

        meta = _dolores_meta()
        ghost = meta.factory(Container())
        assert isinstance(ghost, Dolores)
        assert isinstance(ghost, Ghost)
        assert ghost.meta is meta
        # 空 container 无 GhostWorkspace/Session/Matrix → 均为 None, 无副作用.
        assert ghost._home is None
        assert ghost._session is None
        assert ghost._matrix is None
        assert ghost._dsh_launcher is None

    def test_dsh_stubs_and_plugin_stub(self):
        dsh_stubs = _dolores_meta().dsh_stubs_dir()
        assert dsh_stubs.is_dir()
        assert (dsh_stubs / "profiles/web/package.json").exists()
        # plugin 源在独立 stub, 创建时复制进 dsh profile.
        assert _dolores_meta().dsh_plugin_stub().is_file()
        # ego preset 源在独立 dir, 创建时复制进 .agent-presets.
        assert _dolores_meta().dsh_preset_dir().is_dir()
        assert (_dolores_meta().dsh_preset_dir() / "dolores-ego/agent.cordis.yml").is_file()


class TestStubsSync:
    def test_construct_has_no_side_effect(self, tmp_path: Path):
        _dolores(home=tmp_path)
        assert not (tmp_path / ".dolores.yml").exists()
        assert not (tmp_path / "GROUND.md").exists()

    def test_init_creates_home_and_marker(self, tmp_path: Path):
        session = MockSession()
        ghost = _dolores(home=tmp_path, session=session)

        async def run():
            async with ghost:
                pass

        asyncio.run(run())

        assert (tmp_path / "GROUND.md").exists()
        config = yaml.safe_load((tmp_path / ".dolores.yml").read_text())
        assert config["version"] == ghost._meta.VERSION
        # dirs 物化 + dsh_stubs 同步.
        assert (tmp_path / ".dsh").is_dir()
        assert (tmp_path / "skills").is_dir()
        assert (tmp_path / ".dsh/profiles/web/moss-dolores-ghost-plugin.ts").exists()
        # ego preset 复制进 .agent-presets.
        assert (tmp_path / ".dsh/.agent-presets/dolores-ego/agent.cordis.yml").exists()
        # startup 目录 (开机启动文档) 随 stubs 同步.
        assert (tmp_path / "startup" / "default.startup.yml").exists()
        assert (tmp_path / "startup" / "GROUND.md").exists()
        assert len(session.outputs) == 1
        assert session.outputs[0].role == "system"

    def test_init_materializes_missing_home(self, tmp_path: Path):
        """home 目录本身不存在时也要能 init.

        删掉 ghost home 目录重启是重建实例的正常路径 (比逐个删文件干净), 此时
        .dolores.yml 缺失应当直接由 stub 种下 —— 但它的父目录得先被建出来, 否则
        copy2 直接 FileNotFoundError.
        """
        home = tmp_path / "ghost_home"   # 刻意不创建
        session = MockSession()
        ghost = _dolores(home=home, session=session)

        async def run():
            async with ghost:
                pass

        asyncio.run(run())

        assert home.is_dir()
        config = yaml.safe_load((home / ".dolores.yml").read_text())
        assert config["version"] == ghost._meta.VERSION
        assert (home / "GROUND.md").exists()

    def test_load_startup_fallback_and_parse(self, tmp_path: Path):
        # matrix=None 时 mode_name 为空 → 回退 default.startup.yml.
        (tmp_path / "startup").mkdir(parents=True)
        (tmp_path / "startup" / "default.startup.yml").write_text(
            'command: "<say>hi</say>"\ninstruction: "预热"\n', encoding="utf-8"
        )
        ghost = _dolores(home=tmp_path)

        assert ghost._resolve_startup_doc() == tmp_path / "startup" / "default.startup.yml"
        doc = ghost._load_startup()
        assert doc is not None
        assert doc.command == "<say>hi</say>"
        assert doc.instruction == "预热"
        assert doc.frame is None

    def test_load_startup_parses_frame_block(self, tmp_path: Path):
        (tmp_path / "startup").mkdir(parents=True)
        (tmp_path / "startup" / "default.startup.yml").write_text(
            "instruction: hi\n"
            "frame:\n"
            "  label: orient\n"
            "  description: reconstruct\n"
            "  questions:\n"
            "    - Where am I?\n"
            "    - Who is here?\n",
            encoding="utf-8",
        )
        ghost = _dolores(home=tmp_path)
        doc = ghost._load_startup()
        assert doc is not None
        assert doc.frame is not None
        assert doc.frame.label == "orient"
        assert doc.frame.questions == ["Where am I?", "Who is here?"]

    def test_default_stub_parses_into_startup_doc(self):
        """The shipped default.startup.yml must validate — it is what every new ghost boots on."""
        from ._meta import DoloresMeta
        from ._startup import StartupDoc

        stub = DoloresMeta().stubs_dir() / "startup" / "default.startup.yml"
        assert stub.is_file()
        data = yaml.safe_load(stub.read_text(encoding="utf-8"))
        doc = StartupDoc.model_validate(data)
        # The shipped stub carries a seed orientation frame — regression-catch if we ever
        # remove it silently.
        assert doc.frame is not None
        assert doc.frame.questions

    def test_override_on_version_mismatch(self, tmp_path: Path):
        (tmp_path / ".dolores.yml").write_text("version: dev_0\n")
        (tmp_path / "stale.txt").write_text("dynamic data")
        session = MockSession()
        ghost = _dolores(home=tmp_path, session=session)

        async def run():
            async with ghost:
                pass

        asyncio.run(run())

        assert (tmp_path / "GROUND.md").exists()
        config = yaml.safe_load((tmp_path / ".dolores.yml").read_text())
        assert config["version"] == ghost._meta.VERSION
        # 动态数据文件不被 stubs 覆盖触碰.
        assert (tmp_path / "stale.txt").read_text() == "dynamic data"
        assert len(session.outputs) == 1
        assert "override" in session.outputs[0].messages_string()

    def test_override_preserves_ghost_authored_ground(self, tmp_path: Path):
        """版本重建不得覆盖 ghost 自治内容: 已存在的 identity 保留, 缺失的骨架文件照常补种."""
        (tmp_path / "existence").mkdir(parents=True)
        (tmp_path / "existence" / "identity.md").write_text("ghost 自己提炼的 identity", encoding="utf-8")
        (tmp_path / ".dolores.yml").write_text("version: dev_0\n")
        session = MockSession()
        ghost = _dolores(home=tmp_path, session=session)

        async def run():
            async with ghost:
                pass

        asyncio.run(run())

        assert (tmp_path / "existence" / "identity.md").read_text() == "ghost 自己提炼的 identity"
        # 缺失的骨架文件 (GROUND.md) 仍 seed-once 补种.
        assert (tmp_path / "GROUND.md").exists()

    def test_override_preserves_dolores_yml_user_fields(self, tmp_path: Path):
        """配置是读后改写: 只动 version, 用户字段 (memento.force_tokens) 保留."""
        (tmp_path / ".dolores.yml").write_text(
            "version: dev_0\nmemento:\n  force_tokens: 12345\n", encoding="utf-8"
        )
        ghost = _dolores(home=tmp_path)

        async def run():
            async with ghost:
                pass

        asyncio.run(run())

        config = yaml.safe_load((tmp_path / ".dolores.yml").read_text())
        assert config["version"] == ghost._meta.VERSION
        assert config["memento"]["force_tokens"] == 12345

    def test_noop_when_version_matches(self, tmp_path: Path):
        (tmp_path / ".dolores.yml").write_text(f"version: {_dolores_meta().VERSION}\n")
        (tmp_path / "GROUND.md").write_text("already here")
        session = MockSession()
        ghost = _dolores(home=tmp_path, session=session)

        async def run():
            async with ghost:
                pass

        asyncio.run(run())

        # 版本一致 → 不覆盖, 不发声.
        assert (tmp_path / "GROUND.md").read_text() == "already here"
        assert session.outputs == []

    def test_home_none_is_noop(self):
        session = MockSession()
        ghost = _dolores(home=None, session=session)

        async def run():
            async with ghost:
                pass

        asyncio.run(run())
        assert session.outputs == []


class TestDoloresDefaultModel:
    """dsh 的默认模型归 ghost home 所有: 启动时压进 .dsh/settings.yaml 的 agent-default-model.

    这一层不做 UI 的对手 —— 它是权威. 无它, 网页 Models 面挑一个纯文本模型就会让 dsh 在
    出站前把每张图投影成文本占位, ghost 于是"看不见"图.
    """

    @staticmethod
    def _boot(home: Path) -> None:
        ghost = _dolores(home=home, session=MockSession())

        async def run():
            async with ghost:
                pass

        asyncio.run(run())

    @staticmethod
    def _settings(home: Path) -> dict:
        return yaml.safe_load((home / ".dsh" / "settings.yaml").read_text(encoding="utf-8"))

    def test_creates_settings_with_vision_capable_default(self, tmp_path: Path, monkeypatch):
        # 没有 settings.yaml 的首次启动: 落一份, 默认模型必须是有视觉的 id.
        monkeypatch.delenv("DOLORES_DEFAULT_MODEL", raising=False)
        monkeypatch.delenv("DOLORES_DEFAULT_MODEL_PROVIDER", raising=False)

        self._boot(tmp_path)

        section = self._settings(tmp_path)["agent-default-model"]
        assert section == {"provider": "deepseek-official", "model": "deepseek-flash"}

    def test_overwrites_a_text_only_model(self, tmp_path: Path, monkeypatch):
        # 网页 Models 面挑了纯文本模型 (deepseek-v4-flash) → 启动时必须被换回有视觉的默认.
        monkeypatch.delenv("DOLORES_DEFAULT_MODEL", raising=False)
        monkeypatch.delenv("DOLORES_DEFAULT_MODEL_PROVIDER", raising=False)
        (tmp_path / ".dsh").mkdir()
        (tmp_path / ".dsh" / "settings.yaml").write_text(
            "agent-default-model:\n"
            "  provider: deepseek-official\n"
            "  model: deepseek-v4-flash\n"
            "  reasoningEffort: off\n",
            encoding="utf-8",
        )

        self._boot(tmp_path)

        # 文本级 patch: 换掉 model, 其余叶子 (含 `reasoningEffort: off`) 逐字保留.
        text = (tmp_path / ".dsh" / "settings.yaml").read_text(encoding="utf-8")
        assert "model: deepseek-flash\n" in text
        assert "model: deepseek-v4-flash\n" not in text
        assert "reasoningEffort: off\n" in text

    def test_preserves_other_settings_sections(self, tmp_path: Path, monkeypatch):
        monkeypatch.delenv("DOLORES_DEFAULT_MODEL", raising=False)
        monkeypatch.delenv("DOLORES_DEFAULT_MODEL_PROVIDER", raising=False)
        (tmp_path / ".dsh").mkdir()
        (tmp_path / ".dsh" / "settings.yaml").write_text(
            "ui-onboarding:\n  welcomeNoticeVersion: 2026-08-13.1\n",
            encoding="utf-8",
        )

        self._boot(tmp_path)

        settings = self._settings(tmp_path)
        assert settings["ui-onboarding"]["welcomeNoticeVersion"] == "2026-08-13.1"
        assert settings["agent-default-model"]["model"] == "deepseek-flash"

    def test_env_selects_the_model(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("DOLORES_DEFAULT_MODEL", "deepseek-v4-flash-vision-exp")
        monkeypatch.setenv("DOLORES_DEFAULT_MODEL_PROVIDER", "deepseek-official")

        self._boot(tmp_path)

        assert self._settings(tmp_path)["agent-default-model"]["model"] == "deepseek-v4-flash-vision-exp"


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


class TestDoloresEgoObserveContinuation:
    """DoloresEgo.needs_observe — observe 续帧标记, plugin 据此在 inputs 为空时也开一轮."""

    @staticmethod
    def _thinking(previous=None):
        return SimpleNamespace(moment=SimpleNamespace(previous=previous))

    def test_previous_requires_observe(self):
        from ghoshell_moss.ghosts.dolores._ego import DoloresEgo

        assert DoloresEgo.needs_observe(None, self._thinking(Echoes(need_observe=True))) is True

    def test_previous_without_observe(self):
        from ghoshell_moss.ghosts.dolores._ego import DoloresEgo

        assert DoloresEgo.needs_observe(None, self._thinking(Echoes(need_observe=False))) is False

    def test_no_previous_is_not_a_continuation(self):
        from ghoshell_moss.ghosts.dolores._ego import DoloresEgo

        assert DoloresEgo.needs_observe(None, self._thinking(None)) is False


class TestDoloresEgoReasoningEffort:
    """enter_thinking 的 reasoning_effort 一次性解析 — 帧 effort 优先, 否则用 ego 一次性声明, 消费即置空."""

    @staticmethod
    def _ego(launcher):
        from ._ego import DoloresEgo, DoloresEgoContext

        return DoloresEgo(
            launcher=launcher,
            ctx=DoloresEgoContext(
                project_home=Path("."),
                project_name="pytest",
                name="dolores",
                mode="pytest",
                instruction="i",
                facade=None,
            ),
        )

    @staticmethod
    def _thinking(effort):
        from ghoshell_moss.core.blueprint.moment import Moment

        return SimpleNamespace(
            moment=Moment(index=0),
            observer=SimpleNamespace(
                epoch=SimpleNamespace(id="0", index=0, recap=[], baseline={}),
            ),
            effort=lambda: effort,
        )

    class _Launcher:
        def __init__(self):
            self.calls = []

        async def call(self, path, payload=None, *, timeout=None):
            self.calls.append((path, payload))
            return {}

    @pytest.mark.asyncio
    async def test_frame_effort_wins_over_default(self):
        """帧自带明确 effort (非 '') → 一次性声明不施加, 帧 effort 直通 reasoning_effort."""
        launcher = self._Launcher()
        ego = self._ego(launcher)
        ego.default_thinking_effort = "high"

        await ego.enter_thinking(self._thinking("low"))

        payload = launcher.calls[0][1]
        assert payload["effort"] == "low"
        assert payload["reasoning_effort"] == "low"
        assert ego.default_thinking_effort is None  # 每轮消费完即置空

    @pytest.mark.asyncio
    async def test_default_effort_redefines_when_frame_is_default(self):
        """帧 effort == '' → 用 ego 的一次性声明重定义 reasoning_effort, 消费即置空."""
        launcher = self._Launcher()
        ego = self._ego(launcher)
        ego.default_thinking_effort = "high"

        await ego.enter_thinking(self._thinking(""))

        payload = launcher.calls[0][1]
        assert payload["effort"] == ""
        assert payload["reasoning_effort"] == "high"
        assert ego.default_thinking_effort is None

    @pytest.mark.asyncio
    async def test_no_default_means_no_override(self):
        """default 已消费 (None) → reasoning_effort 为 None, 不覆盖 dsh/UI 持有的强度.

        ego 首帧消费掉初始 "off" 后 default 恒 None; 之后 enter 不再带 reasoning_effort,
        强度归 dsh/UI. 此处显式置 None 模拟"已消费"的状态.
        """
        launcher = self._Launcher()
        ego = self._ego(launcher)
        ego.default_thinking_effort = None

        await ego.enter_thinking(self._thinking(""))

        payload = launcher.calls[0][1]
        assert payload["reasoning_effort"] is None


class TestDoloresMemories:
    """Dolores.memories — ground 渲染为第一条存在主义记忆."""

    @pytest.mark.asyncio
    async def test_memories_empty_without_home(self):
        """无 home → 无 ground 渲染, memories 返回空."""
        ghost = _dolores()
        async with ghost:
            assert ghost.memories() == []

    @pytest.mark.asyncio
    async def test_memories_returns_ground_as_first(self, tmp_path: Path):
        """home 存在 → ground 渲染文本包成 ground tag 的记忆, 是唯一一条."""
        ghost = _dolores(home=tmp_path)
        async with ghost:
            memories = ghost.memories()
            assert len(memories) == 1
            text = memories[0].to_content_string()
            assert "<ground>" in text and "</ground>" in text


class TestDoloresEgoNucleus:
    """DoloresEgoNucleus — BACKGROUND 挑战包 (发完丢), attended 加工成 INFO 运行包."""

    @pytest.mark.asyncio
    async def test_signal_produces_background_empty_impulse(self):
        from ghoshell_moss.core.blueprint.mindflow import Priority

        from .nucleus import DoloresEgoNucleus, new_dolores_ego_signal

        nucleus = DoloresEgoNucleus()
        impulses = []
        async with nucleus:
            nucleus.with_bus(lambda signal: None, impulses.append)
            nucleus.add_signal(new_dolores_ego_signal())

        assert len(impulses) == 1
        imp = impulses[0]
        assert imp.priority == Priority.BACKGROUND
        assert imp.messages == []
        # 默认 mode (空) = 正常仲裁, 非 silent buffer.
        assert imp.mode == ""

    @pytest.mark.asyncio
    async def test_startup_kind_carries_command_and_instruction(self):
        from .nucleus import DoloresEgoNucleus, new_dolores_ego_signal

        nucleus = DoloresEgoNucleus()
        impulses = []
        async with nucleus:
            nucleus.with_bus(lambda signal: None, impulses.append)
            nucleus.add_signal(new_dolores_ego_signal(
                kind="startup", command="<say>hi</say>", instruction="预热"
            ))

        assert len(impulses) == 1
        imp = impulses[0]
        assert imp.logos == "<say>hi</say>"
        assert len(imp.messages) == 1
        body = imp.messages[0].to_content_string()
        assert "<startup>" in body
        assert "预热" in body

    @pytest.mark.asyncio
    async def test_attended_rewrites_to_info(self):
        from ghoshell_moss.core.blueprint.mindflow import Impulse, Priority

        from .nucleus import DoloresEgoNucleus

        nucleus = DoloresEgoNucleus()
        async with nucleus:
            challenge = Impulse(source="dolores_ego_nucleus", priority=Priority.BACKGROUND)
            rewritten = nucleus.attended(challenge)

        assert rewritten is not None
        assert rewritten.priority == Priority.INFO
        assert rewritten.messages == []

    @pytest.mark.asyncio
    async def test_ignores_foreign_signal(self):
        from ghoshell_moss.core.blueprint.mindflow import Signal

        from .nucleus import DoloresEgoNucleus

        nucleus = DoloresEgoNucleus()
        impulses = []
        async with nucleus:
            nucleus.with_bus(lambda signal: None, impulses.append)
            nucleus.add_signal(Signal(name="some/other"))

        assert impulses == []

    @pytest.mark.asyncio
    async def test_meta_factory_builds_nucleus(self):
        from ghoshell_container import Container

        from .nucleus import DoloresEgoNucleus, DoloresEgoNucleusMeta

        nucleus = DoloresEgoNucleusMeta().factory(Container())
        assert isinstance(nucleus, DoloresEgoNucleus)
        assert nucleus.name() == "dolores_ego_nucleus"


class TestDoloresEgoSelfWake:
    """DoloresEgo 的 self-wake gate — articulate flag 决定 turn/start 是否自醒."""

    def _ego(self):
        from ._ego import DoloresEgo, DoloresEgoContext

        # gate 路径不触 dsh / 不读 ghost 反参 — ctx/launcher 传最小 dummy 即可.
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

    @pytest.mark.asyncio
    async def test_turn_start_self_wakes_when_idle(self):
        ego = self._ego()
        emitted = []
        ego.bind_signal_broadcast(emitted.append)
        await ego._on_session_activity(None)  # type: ignore[arg-type]  # gate 不读 event

        assert len(emitted) == 1
        assert emitted[0].name == "dolores/ego"

    @pytest.mark.asyncio
    async def test_turn_start_suppressed_when_articulating(self):
        ego = self._ego()
        emitted = []
        ego.bind_signal_broadcast(emitted.append)
        ego._thinking_event.set()  # 模拟 run 交易进行中 (is_thinking 只读, 直接置 event)
        await ego._on_session_activity(None)  # type: ignore[arg-type]

        assert emitted == []


class TestDoloresEgoCommit:
    """ego 的锚点 commit — 验证期 force_tokens=0 (每 turn commit) 下检查 memento 锚点数据."""

    @staticmethod
    def _turn_end(turn: int):
        from ghoshell_moss.deepseek_harness.types.session_events import SessionEvent

        return SessionEvent.from_dict({"type": "turn/end", "seq": turn, "data": {"turn": turn}})

    @staticmethod
    def _set_up(tmp_path: Path, *, force_tokens: int = 0, warn_tokens: int = 50_000):
        from ghoshell_moss.ghosts.dolores._ego import DoloresEgo, DoloresEgoContext
        from ghoshell_moss.ghosts.dolores._ego_memento import EgoMementoConfig, EgoMementoManager
        from ghoshell_moss.memento import new_local_memento

        memento = new_local_memento(tmp_path / "owner")
        memento.create_branch("main")
        manager = EgoMementoManager(
            connection=None, memento=memento,
            config=EgoMementoConfig(force_tokens=force_tokens, warn_tokens=warn_tokens),
        )
        ego = DoloresEgo(
            launcher=None,  # commit 路径不触 dsh
            ctx=DoloresEgoContext(
                project_home=Path("."),
                project_name="pytest",
                name="dolores",
                mode="pytest",
                instruction="i",
                facade=None,
            ),
            memento_manager=manager,
        )
        ego._ego_session_id = "s1"
        return ego, memento

    @pytest.mark.asyncio
    async def test_each_turn_commit_span_is_chained(self, tmp_path: Path):
        ego, memento = self._set_up(tmp_path)
        for turn in (1, 2, 3):
            await ego._on_turn_end(self._turn_end(turn))

        metas = [c.metadata for c in memento.get_branch("main").commits()]

        # 每 turn 一锚点, 区间半开平铺: (0,1] (1,2] (2,3] —— 下界逐字抄上一个的 end_turn.
        assert [m["ref"]["start_turn"] for m in metas] == [0, 1, 2]
        assert [m["ref"]["end_turn"] for m in metas] == [1, 2, 3]
        assert [m["prev_turn"] for m in metas] == [0, 1, 2]
        assert {m["ref"]["session_id"] for m in metas} == {"s1"}

    @pytest.mark.asyncio
    async def test_exit_seals_turns_left_unratified(self, tmp_path: Path):
        """封尾的本职: 阈值没到、有已完成却没被追认的 turn → 退出时补一个锚点."""
        ego, memento = self._set_up(tmp_path, force_tokens=10 ** 9)
        await ego._on_turn_end(self._turn_end(1))
        assert memento.get_branch("main").commits() == []  # 阈值远未到 → 不落锚点

        await ego.__aexit__(None, None, None)

        commits = memento.get_branch("main").commits()
        assert len(commits) == 1
        ref = commits[0].metadata["ref"]
        assert (ref["start_turn"], ref["end_turn"]) == (0, 1)

    @pytest.mark.asyncio
    async def test_exit_adds_nothing_when_already_ratified(self, tmp_path: Path):
        """区间为空 (上一锚点就落在同一个 turn) → 不落空锚点, 也不白跑一次旁路."""
        ego, memento = self._set_up(tmp_path)
        await ego._on_turn_end(self._turn_end(1))
        await ego._on_turn_end(self._turn_end(2))
        before = len(memento.get_branch("main").commits())
        notices = len(ego._notices)

        await ego.__aexit__(None, None, None)

        assert len(memento.get_branch("main").commits()) == before
        assert len(ego._notices) == notices  # 也没有多出"已生成 commit"提醒

    @pytest.mark.asyncio
    async def test_abnormal_exit_does_not_commit(self, tmp_path: Path):
        ego, memento = self._set_up(tmp_path)
        await ego._on_turn_end(self._turn_end(1))
        committed = len(memento.get_branch("main").commits())

        await ego.__aexit__(RuntimeError, RuntimeError("boom"), None)

        # 异常退出不管: 锚点数不变.
        assert len(memento.get_branch("main").commits()) == committed

    @staticmethod
    def _assistant_message(turn: int, *, input_tokens: int, cache_read: int = 0):
        from ghoshell_moss.deepseek_harness.types.session_events import SessionEvent

        return SessionEvent.from_dict({
            "type": "assistant/message", "seq": turn * 10,
            "data": {
                "turn": turn, "step": 1, "message": {"role": "assistant"},
                "usage": {"inputTokens": input_tokens, "cacheReadTokens": cache_read},
            },
        })

    @pytest.mark.asyncio
    async def test_usage_drives_warn_notice_once_per_window(self, tmp_path: Path):
        """assistant/message 的 usage 刷新窗口 → 达 K 排一次 warn notice (每窗口一次)."""
        ego, memento = self._set_up(tmp_path, force_tokens=10 ** 9, warn_tokens=100)
        await ego._on_assistant_message(self._assistant_message(1, input_tokens=80, cache_read=70))
        assert ego._window_size == 150
        await ego._on_turn_end(self._turn_end(1))

        assert memento.get_branch("main").commits() == []  # 未到 T, 不落锚点
        warns = [n for n in ego._notices if n.meta.attributes.get("kind") == "warn"]
        assert len(warns) == 1

        # 同窗口再涨也只提醒一次.
        await ego._on_assistant_message(self._assistant_message(2, input_tokens=200))
        await ego._on_turn_end(self._turn_end(2))
        warns = [n for n in ego._notices if n.meta.attributes.get("kind") == "warn"]
        assert len(warns) == 1

    @pytest.mark.asyncio
    async def test_usage_drives_force_commit_and_resets_window(self, tmp_path: Path):
        """达 T 强制落锚点; 提交后窗口基准重置, 增量不足则不再触发."""
        ego, memento = self._set_up(tmp_path, force_tokens=200, warn_tokens=100)
        await ego._on_assistant_message(self._assistant_message(1, input_tokens=250))
        await ego._on_turn_end(self._turn_end(1))
        commits = memento.get_branch("main").commits()
        assert len(commits) == 1
        assert commits[0].metadata["ref"]["end_turn"] == 1

        # 基准已重置到 250: 窗口 300 → 增量 50, 不触发.
        await ego._on_assistant_message(self._assistant_message(2, input_tokens=300))
        await ego._on_turn_end(self._turn_end(2))
        assert len(memento.get_branch("main").commits()) == 1

        # 窗口 500 → 增量 250 ≥ T, 再次强制, 区间接续 (1,3].
        await ego._on_assistant_message(self._assistant_message(3, input_tokens=500))
        await ego._on_turn_end(self._turn_end(3))
        commits = memento.get_branch("main").commits()
        assert len(commits) == 2
        assert (commits[1].metadata["ref"]["start_turn"], commits[1].metadata["ref"]["end_turn"]) == (1, 3)

    @staticmethod
    def _request_header(turn: int, *, provider: str, model: str, effort: str = ""):
        from ghoshell_moss.deepseek_harness.types.session_events import SessionEvent

        config = {"provider": provider, "model": model}
        if effort:
            config["reasoningEffort"] = effort
        return SessionEvent.from_dict({
            "type": "request/header", "seq": turn * 10,
            "data": {
                "header": {"config": config},
                "reason": "change",
            },
        })

    @pytest.mark.asyncio
    async def test_first_request_header_queues_identity_notice(self, tmp_path: Path):
        """首次观测必发: 思考档决定交互礼仪, 模型开局就得知道自己在哪一档."""
        ego, _ = self._set_up(tmp_path)
        assert ego._notices == []

        await ego._on_request_header(self._request_header(1, provider="deepseek", model="m", effort="off"))

        notices = [n for n in ego._notices if n.meta.tag == "model_notice"]
        assert len(notices) == 1
        text = notices[0].to_content_string()
        assert "deepseek" in text and "m" in text
        assert "off" in text
        assert "directly" in text  # off 的礼仪提示

    @pytest.mark.asyncio
    async def test_unchanged_header_queues_nothing(self, tmp_path: Path):
        """同一身份重复上报不重复排队."""
        ego, _ = self._set_up(tmp_path)
        await ego._on_request_header(self._request_header(1, provider="deepseek", model="m", effort="high"))
        await ego._on_request_header(self._request_header(2, provider="deepseek", model="m", effort="high"))

        assert len([n for n in ego._notices if n.meta.tag == "model_notice"]) == 1

    @pytest.mark.asyncio
    async def test_effort_change_queues_notice_naming_previous(self, tmp_path: Path):
        """思考档切换 → 再排一条, 并点明从哪一档变来 (礼仪随之切换)."""
        ego, _ = self._set_up(tmp_path)
        await ego._on_request_header(self._request_header(1, provider="deepseek", model="m", effort="high"))
        await ego._on_request_header(self._request_header(2, provider="deepseek", model="m", effort="max"))

        notices = [n for n in ego._notices if n.meta.tag == "model_notice"]
        assert len(notices) == 2
        text = notices[1].to_content_string()
        assert "high" in text and "max" in text
        assert "changed" in text

    @pytest.mark.asyncio
    async def test_identity_notice_drains_into_enter_payload(self, tmp_path: Path):
        """notice 经每帧 enter 排空 — 模型在下一轮思考里看到自己的身份."""
        ego, _ = self._set_up(tmp_path)
        await ego._on_request_header(self._request_header(1, provider="deepseek", model="m", effort="low"))

        drained = ego._drain_notices()

        assert len(drained) == 1
        assert "deepseek" in drained[0]
        assert ego._notices == []  # 排空后不残留


class TestEgoMementoSidecar:
    """旁路 note 生产 — run 路由的 fake: 回 message 写 note; 异常/空 → 终态留空."""

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
    def _manager(tmp_path: Path, connection):
        from ._ego_memento import EgoMementoConfig, EgoMementoManager
        from ghoshell_moss.memento import new_local_memento

        memento = new_local_memento(tmp_path / "owner")
        memento.create_branch("main")
        manager = EgoMementoManager(connection=connection, memento=memento, config=EgoMementoConfig())
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
    async def test_sidecar_failure_leaves_empty_note(self, tmp_path: Path):
        conn = self._FakeConnection(RuntimeError("dsh down"))
        manager, memento = self._manager(tmp_path, conn)
        anchor = manager.commit(session_id="s1", start_turn=0, end_turn=1)

        manager.schedule_note(anchor.id)
        await manager.drain_bypass()

        assert memento.get_branch("main").notes() == {}  # 留空, 不自动重试
        assert self._state(manager, anchor.id) == "failed"

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


class TestMementoReadSurface:
    """branch / commit 读面 + read / chat 两个动作 —— 全部在 ego 外, 不触 dsh (fake connection)."""

    class _Connection:
        def __init__(self, response):
            self.response = response
            self.calls: list[tuple[str, dict]] = []

        async def call(self, path, payload=None, *, timeout=None):
            self.calls.append((path, payload))
            if isinstance(self.response, Exception):
                raise self.response
            return self.response

    @staticmethod
    def _manager(tmp_path: Path, connection=None):
        from ghoshell_moss.memento import new_local_memento

        from ._ego_memento import EgoMementoConfig, EgoMementoManager

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
    async def test_chat_frames_the_context_not_the_commit(self, tmp_path: Path):
        conn = self._Connection({"message": "reply"})
        manager, memento = self._manager(tmp_path, conn)
        anchor = manager.commit(session_id="s1", start_turn=0, end_turn=1)
        memento.get_branch("main").note(anchor.id, "note")
        question = "当时为什么这么定?"

        text = await manager.chat(self._coord(memento, anchor.seq), question)

        assert text == "reply"
        path, payload = conn.calls[0]
        assert path.endswith("/bypass/run")
        assert question in payload["prompt"]
        # 要求 #7: 说清对话对象是上下文 (那段对话历史), 而不是 commit 本身.
        assert "talking to the context that commit belongs to" in payload["prompt"]
        assert "outside your view" in payload["prompt"]

    @pytest.mark.asyncio
    async def test_chat_refuses_broken_commit(self, tmp_path: Path):
        conn = self._Connection({"message": "reply"})
        manager, memento = self._manager(tmp_path, conn)
        anchor = manager.commit(session_id="s1", start_turn=0, end_turn=1)
        memento.get_branch("main").note(anchor.id, "占位", error="fatal")
        from ._ego_memento import BrokenCommitError

        with pytest.raises(BrokenCommitError):
            await manager.chat(self._coord(memento, anchor.seq), "问题")

        assert conn.calls == []  # 坏 commit 连旁路都不发


class TestMementoCommitSpan:
    """锚点区间规则 —— 半开 ``(start_turn, end_turn]``, 空区间不落锚点."""

    @staticmethod
    def _manager(tmp_path: Path):
        from ghoshell_moss.memento import new_local_memento

        from ._ego_memento import EgoMementoConfig, EgoMementoManager

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


class TestEgoMementoResumeRef:
    """ego 重建切点的选择 — 纯逻辑 (真 memento), 只认摘要已就绪的 commit."""

    @staticmethod
    def _manager(tmp_path: Path):
        from ghoshell_moss.ghosts.dolores._ego_memento import EgoMementoConfig, EgoMementoManager
        from ghoshell_moss.memento import new_local_memento

        memento = new_local_memento(tmp_path / "owner")
        memento.create_branch("main")
        return EgoMementoManager(connection=None, memento=memento, config=EgoMementoConfig()), memento

    def test_none_without_commits(self, tmp_path: Path):
        manager, _ = self._manager(tmp_path)
        assert manager.resume_ref() is None

    def test_none_when_no_note_ready(self, tmp_path: Path):
        manager, _ = self._manager(tmp_path)
        manager.commit(session_id="s1", start_turn=0, end_turn=3)
        assert manager.resume_ref() is None

    def test_none_when_only_note_is_empty(self, tmp_path: Path):
        manager, memento = self._manager(tmp_path)
        anchor = manager.commit(session_id="s1", start_turn=0, end_turn=3)
        memento.get_branch("main").note(anchor.id, "")

        assert manager.resume_ref() is None

    def test_falls_back_to_last_commit_with_ready_note(self, tmp_path: Path):
        manager, memento = self._manager(tmp_path)
        branch = memento.get_branch("main")
        first = manager.commit(session_id="s1", start_turn=0, end_turn=3)
        second = manager.commit(session_id="s1", start_turn=3, end_turn=7)
        branch.note(first.id, "first")
        branch.note(second.id, "second")
        # 尾巴上还有一个摘要没产出来的 commit: 切点回退到已就绪的那个, 它之后的原文当 raw 尾巴带过去.
        manager.commit(session_id="s1", start_turn=7, end_turn=9)

        ref = manager.resume_ref()

        assert ref is not None
        assert (ref.session_id, ref.start_turn, ref.end_turn) == ("s1", 3, 7)


# ── DoloresRun — thinking 交易 run 对象 (public + 可测) ─────────────


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
        self.enter_calls = 0
        self.exit_calls = 0
        self.enter_error: Exception | None = None

    async def enter_thinking(self, thinking):
        self.enter_calls += 1
        if self.enter_error is not None:
            raise self.enter_error

    async def exit_thinking(self):
        self.exit_calls += 1


class FakeArticulator:
    """articulator 的 fake — send 累积 logos, 生命周期空操作; interpret_error 可注入."""

    def __init__(self, log: list | None = None):
        self.sent: list[str] = []
        self.interpret_error: Exception | None = None
        self._log = log
        # 生命周期痕迹: 展开是否发生, 是回归测试的核心判据 (见 test_ctml_append_opens_lifecycle).
        self.entered = 0
        self.exited = 0

    async def __aenter__(self):
        self.entered += 1
        return self

    async def __aexit__(self, *args):
        self.exited += 1
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
    epoch = _FakeEpoch()


class FakeRunThinking:
    def __init__(self, log: list | None = None):
        self.abort_reasons: list = []
        self.articulators: list[FakeArticulator] = []
        self._log = log

    def abort(self, reason):
        self.abort_reasons.append(reason)
        if self._log is not None:
            self._log.append("abort")

    def articulator(self, replan=False, wait_action_done=False) -> FakeArticulator:
        art = FakeArticulator(self._log)
        self.articulators.append(art)
        return art

    def add_echoes(self, *messages, observe=False):
        if self._log is not None:
            self._log.append(f"add_echoes:{observe}")

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
        return _FakeObserver()


def fake_tool_call(call_id: str = "call_00_test"):
    """_dispatch_tool_result 只用 callId, 其余字段走默认."""
    from ghoshell_moss.deepseek_harness.types.session_events import ToolCallEvent

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

    def moment_context_parts(self, moment, moment_id):
        return [{"type": "text", "text": f"moment:{moment_id}"}]


class TestDoloresRun:
    """DoloresRun 生命周期 + 事件消费 — public 类, 轻量 fake 即可验证."""

    def _run(self, session=None, ego=None, thinking=None):
        from ._run import DoloresRun

        session = session or FakeRunSession()
        return DoloresRun(
            ego=ego or FakeRunEgo(session),
            thinking=thinking or FakeRunThinking(),
            thinking_event=asyncio.Event(),
            facade=None,
        )

    @staticmethod
    def _event(event_type: str, data: dict, seq: int = 1):
        from ghoshell_moss.deepseek_harness.types.session_events import SessionEvent, SessionEventMeta

        return SessionEvent(meta=SessionEventMeta(type=event_type, seq=seq), data=data)

    @classmethod
    def _text_chunk(cls, text: str, seq: int = 1):
        return cls._event(
            "assistant/chunk",
            {"turn": 1, "step": 1, "chunk": {"type": "text-delta", "text": text}},
            seq=seq,
        )

    @staticmethod
    def _tool_call(name: str, arguments: str = "{}", call_id: str = "call_x"):
        from ghoshell_moss.deepseek_harness.types.session_events import ToolCallEvent

        return ToolCallEvent(callId=call_id, name=name, arguments=arguments)

    @pytest.mark.asyncio
    async def test_channel_facade_recursive_lists_and_single_reads(self):
        """moss_channel_facade: recursive 列前缀下 channel, 非 recursive 读单个完整面."""
        from ._run import DoloresRun

        class _Meta:
            def __init__(self, description):
                self.description = description

        class _Facade:
            def channel_metas(self, available_only=True):
                return {"ghost.frame": _Meta("the frame"), "ghost.voice": _Meta("the voice")}

            def get_channel_full_facade(self, path):
                return f"<{path}>full surface</{path}>" if path == "ghost.frame" else ""

        session = FakeRunSession()
        ego = FakeDispatchEgo(session)
        run = DoloresRun(
            ego=ego, thinking=FakeRunThinking(), thinking_event=asyncio.Event(), facade=_Facade(),
        )
        await run._handle_tool_use_event(
            self._tool_call("moss_channel_facade", '{"channel_path": "ghost"}', call_id="call_rec"), {},
        )
        await run._handle_tool_use_event(
            self._tool_call("moss_channel_facade", '{"channel_path": "ghost.frame", "recursive": false}', call_id="call_one"), {},
        )

        results = {call_id: result for call_id, result, *_ in ego.rpc_calls}
        assert "ghost.frame" in results["call_rec"]
        assert "the voice" in results["call_rec"]
        assert "full surface" in results["call_one"]

    @pytest.mark.asyncio
    async def test_channel_facade_reports_unknown_path(self):
        """未知 path (非 recursive) 报错不抛 (工具结果回给模型, 不烧整轮)."""
        from ._run import DoloresRun

        class _Facade:
            def channel_metas(self, available_only=True):
                return {}

            def get_channel_full_facade(self, path):
                return ""

        session = FakeRunSession()
        ego = FakeDispatchEgo(session)
        run = DoloresRun(
            ego=ego, thinking=FakeRunThinking(), thinking_event=asyncio.Event(), facade=_Facade(),
        )
        await run._handle_tool_use_event(
            self._tool_call("moss_channel_facade", '{"channel_path": "nope", "recursive": false}'), {},
        )
        assert "no such channel" in ego.rpc_calls[0][1]

    @pytest.mark.asyncio
    async def test_shell_status_reads_facade(self):
        """moss_shell_status 读 shell 状态描述, 不产生 moment."""
        from ._run import DoloresRun

        class _Status:
            def description(self):
                return "idle, 2 actions running"

        class _Facade:
            def status(self):
                return _Status()

        session = FakeRunSession()
        ego = FakeDispatchEgo(session)
        run = DoloresRun(
            ego=ego, thinking=FakeRunThinking(), thinking_event=asyncio.Event(), facade=_Facade(),
        )
        await run._handle_tool_use_event(self._tool_call("moss_shell_status", call_id="call_s"), {})
        assert ego.rpc_calls == [("call_s", "idle, 2 actions running", None, False)]

    @pytest.mark.asyncio
    async def test_observe_observes_moment(self):
        """moss_observe 等动作完 + observe 最新 moment (返回 moment_ref 并注入)."""
        from ghoshell_moss.core.blueprint.moment import Moment

        from ._run import DoloresRun

        class _Epoch:
            index = 3

        class _Observer:
            epoch = _Epoch()

        class _Shell:
            async def refresh_metas(self, timeout=None, stale_time=None):
                pass

        class _Facade:
            def __init__(self):
                self.shell = _Shell()

        class _Thinking(FakeRunThinking):
            async def wait_actions_done(self):
                pass

            def observe(self):
                return Moment(id="m", index=7)

            @property
            def observer(self):
                return _Observer()

        session = FakeRunSession()
        ego = FakeDispatchEgo(session)
        thinking = _Thinking()
        run = DoloresRun(
            ego=ego, thinking=thinking, thinking_event=asyncio.Event(), facade=_Facade(),
        )
        await run._handle_tool_use_event(self._tool_call("moss_observe", call_id="call_w"), {})
        assert ego.rpc_calls == [
            ("call_w", {"moment_ref": "3-7"}, [{"type": "text", "text": "moment:3-7"}], False),
        ]

    @pytest.mark.asyncio
    async def test_observe_interrupt_replans_then_observes(self):
        """interrupt=true → 先空 replan (kind='clear') 中止当前行动, 等它落地, 再 wait + observe."""
        from ghoshell_moss.core.blueprint.moment import Moment

        from ._run import DoloresRun

        class _Epoch:
            index = 3

        class _Observer:
            epoch = _Epoch()

        class _Shell:
            async def refresh_metas(self, timeout=None, stale_time=None):
                pass

        class _Facade:
            def __init__(self):
                self.shell = _Shell()

        class _Thinking(FakeRunThinking):
            def __init__(self):
                super().__init__()
                self.replans: list[bool] = []

            def articulator(self, replan=False, wait_action_done=False):
                self.replans.append(replan)
                return super().articulator(replan=replan, wait_action_done=wait_action_done)

            async def wait_actions_done(self):
                pass

            def observe(self):
                return Moment(id="m", index=9)

            @property
            def observer(self):
                return _Observer()

        session = FakeRunSession()
        ego = FakeDispatchEgo(session)
        thinking = _Thinking()
        run = DoloresRun(
            ego=ego, thinking=thinking, thinking_event=asyncio.Event(), facade=_Facade(),
        )
        await run._handle_tool_use_event(
            self._tool_call("moss_observe", '{"interrupt": true}', call_id="call_i"), {},
        )

        assert thinking.replans == [True]
        assert thinking.articulators[0].sent == []  # 空 replan, 没发内容
        assert ego.rpc_calls[0][1] == {"moment_ref": "3-9"}

    @pytest.mark.asyncio
    async def test_aenter_binds_listener_and_aexit_cleans_up(self):
        session = FakeRunSession()
        ego = FakeRunEgo(session)
        run = self._run(session=session, ego=ego)
        async with run:
            assert run._thinking_event.is_set()  # 交易中 (run aenter 置位)
            assert len(session.handlers) == 1  # catch-all 监听已绑
            await asyncio.sleep(0)  # 让出 loop, enter task 跑
            assert ego.enter_calls == 1
        assert not run._thinking_event.is_set()  # 交易结束 (run aexit 复位)
        assert ego.exit_calls == 1
        assert len(session.handlers) == 0  # 解绑

    @pytest.mark.asyncio
    async def test_logos_skips_text_and_stops_on_turn_end(self):
        """plain text 跳过 (不解析为 CTML), turn/end 让 logos() 自止 (无需消费方 break)."""
        session = FakeRunSession()
        ego = FakeRunEgo(session)
        thinking = FakeRunThinking()
        run = self._run(session=session, ego=ego, thinking=thinking)
        async with run:
            await session.emit(self._text_chunk("plain text nobody reads", seq=1))
            await session.emit(self._event("turn/end", {"turn": 1}, seq=2))
            collected = []
            async for delta in run.logos():
                collected.append(delta)
        assert collected == []  # plain text 不产出 logos
        assert thinking.articulators == []  # 无 articulator 被创建
        assert thinking.abort_reasons == []  # completed → 不打断

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "reason",
        [
            {"kind": "aborted", "reason": {"kind": "user"}},
            {"kind": "error"},
            {"kind": "max-tokens"},
            {"kind": "interrupted"},
            {"kind": "completed"},
        ],
    )
    async def test_turn_end_never_aborts_thinking(self, reason):
        """turn/end 只收线本帧, 绝不冒泡 abort — 无论 reason.kind 是什么.

        冒泡 (Thinking.abort → attention.abort) 会连 attention 一起杀掉, 使 need_observe 驱动的
        回声帧循环 `while not attention.is_aborted() and need_observe()` 永远起不来 —— interpret
        error 自愈帧 / moss_reasoning 续帧全丢. 本帧的退出走自然路径: logos() 在此返回 → run
        收线 → articulate 返回 → 帧自己结束; run 退出时另 cancel dsh, 双向对齐.
        """
        session = FakeRunSession()
        ego = FakeRunEgo(session)
        thinking = FakeRunThinking()
        run = self._run(session=session, ego=ego, thinking=thinking)
        async with run:
            await session.emit(self._text_chunk("<say>hi</say>", seq=1))
            await session.emit(self._event("turn/end", {"turn": 1, "reason": reason}, seq=2))
            async for _ in run.logos():
                pass
        assert thinking.abort_reasons == []

    @pytest.mark.asyncio
    async def test_enter_error_propagates_and_aborts(self):
        session = FakeRunSession()
        ego = FakeRunEgo(session)
        thinking = FakeRunThinking()
        ego.enter_error = RuntimeError("enter boom")
        run = self._run(session=session, ego=ego, thinking=thinking)
        with pytest.raises(RuntimeError, match="enter boom"):
            async with run:
                async for _ in run.logos():
                    pass
        assert thinking.abort_reasons  # enter 异常 → thinking.abort

    @pytest.mark.asyncio
    async def test_consumer_exception_aborts_thinking(self):
        session = FakeRunSession()
        ego = FakeRunEgo(session)
        thinking = FakeRunThinking()
        run = self._run(session=session, ego=ego, thinking=thinking)
        with pytest.raises(RuntimeError, match="consumer boom"):
            async with run:
                raise RuntimeError("consumer boom")
        assert thinking.abort_reasons

    @pytest.mark.asyncio
    async def test_dispatch_tool_result_passes_call_id_result_and_moment_parts(self):
        """正常路径: callId + result + moment parts 原样交给 RPC."""
        from ghoshell_moss.core.blueprint.moment import Moment

        from ._tools import ToolCallResult

        ego = FakeDispatchEgo()
        run = self._run(ego=ego)
        result = ToolCallResult(
            call=fake_tool_call("call_ok"),
            result={"moment_ref": "1-7"},
            moment=Moment(id="1-7"),
        )
        await run._dispatch_tool_result(result)
        assert ego.rpc_calls == [("call_ok", {"moment_ref": "1-7"}, [{"type": "text", "text": "moment:1-7"}], False)]

    @pytest.mark.asyncio
    async def test_dispatch_tool_result_absorbs_rpc_failure(self):
        """迟到回话不再致命: RPC 失败被吸收, 不冒泡 (旧行为会打穿整轮 logos 流)."""
        from ._tools import ToolCallResult

        ego = FakeDispatchEgo(raise_on_rpc=RuntimeError("no pending tool call for call_late"))
        run = self._run(ego=ego)
        result = ToolCallResult(call=fake_tool_call("call_late"), result={"moment_ref": "1-8"})
        await run._dispatch_tool_result(result)  # 不抛
        assert ego.rpc_calls == []

    @pytest.mark.asyncio
    async def test_dispatch_tool_result_without_moment_sends_empty_parts(self):
        """interleaved_ctml 这类不携带 moment 的 tool: moment parts 为 None."""
        from ._tools import ToolCallResult

        ego = FakeDispatchEgo()
        run = self._run(ego=ego)
        result = ToolCallResult(call=fake_tool_call("call_ctml"), result="ok")
        await run._dispatch_tool_result(result)
        assert ego.rpc_calls == [("call_ctml", "ok", None, False)]

    @pytest.mark.asyncio
    async def test_dispatch_tool_result_carries_the_call_turn(self):
        """结果协议带上 tool 被调用时的 turn —— 插件据此把"早到结果"绑到正确的轮次, 并按轮清理.

        插件侧 /tool-result 可能先于 dsh 派发该 tool 的 execute 到达 (MOSS 走 mux 事件流更快);
        结果里的 turn 让插件在 execute 尚未注册时也能按轮归属, 而不是悬空.
        """
        from ghoshell_moss.deepseek_harness.types.session_events import ToolCallEvent

        from ._tools import ToolCallResult

        ego = FakeDispatchEgo()
        run = self._run(ego=ego)
        call = ToolCallEvent(callId="call_turn", turn=7)
        result = ToolCallResult(call=call, result="ok")
        await run._dispatch_tool_result(result)
        assert ego.rpc_calls == [("call_turn", "ok", None, False)]
        assert ego.rpc_turns == [7]

    @pytest.mark.asyncio
    async def test_wait_next_result_carries_cancel(self):
        """moss_wait_next 用结果协议里的 cancel flag 收线 — 不是 final answer, 也不用 abort."""
        from ._run import DoloresRun

        class _Thinking:
            def __init__(self):
                self.actions_done = 0

            async def wait_actions_done(self):
                self.actions_done += 1

        thinking = _Thinking()
        session = FakeRunSession()
        ego = FakeDispatchEgo(session)
        run = DoloresRun(
            ego=ego, thinking=thinking, thinking_event=asyncio.Event(), facade=None,
        )
        await run._handle_tool_use_event(self._tool_call("moss_wait_next", call_id="call_w"), {})

        assert thinking.actions_done == 1
        assert ego.rpc_calls == [("call_w", "yielded", None, True)]

    # ── ctml 流式接线 ───────────────────────────────────────────────

    @staticmethod
    def _ctml_delta(call_id, *, name=None, arguments_delta=None, seq=1):
        """assistant/chunk 的 tool-call-delta — 名字只挂在首个 delta 上."""
        chunk = {"type": "tool-call-delta", "id": call_id}
        if name is not None:
            chunk["name"] = name
        if arguments_delta is not None:
            chunk["argumentsDelta"] = arguments_delta
        return TestDoloresRun._event("assistant/chunk", {"turn": 1, "step": 1, "chunk": chunk}, seq=seq)

    @staticmethod
    def _tool_call_event(name, arguments, call_id, seq=1):
        return TestDoloresRun._event(
            "tool/call",
            {"turn": 1, "step": 1, "callId": call_id, "name": name, "arguments": arguments},
            seq=seq,
        )

    @staticmethod
    def _turn_end(seq=1):
        return TestDoloresRun._event("turn/end", {"turn": 1, "reason": {"kind": "completed"}}, seq=seq)

    @pytest.mark.asyncio
    async def test_interpret_delta_streams_into_articulator_then_returns_moment(self):
        """流式 ctml: delta 逐片进 articulator, tool/call 收线 → 等 observed → 签发并注入 moment."""
        session = FakeRunSession()
        ego = FakeDispatchEgo(session)
        thinking = FakeRunThinking()
        run = self._run(session=session, ego=ego, thinking=thinking)
        async with run:
            await session.emit(self._ctml_delta("c1", name="moss_interpret", arguments_delta='{"ctml":"<say>', seq=1))
            await session.emit(self._ctml_delta("c1", arguments_delta='hi</say>"}', seq=2))
            await session.emit(self._tool_call_event("moss_interpret", '{"ctml":"<say>hi</say>"}', "c1", seq=3))
            await session.emit(self._turn_end(seq=4))
            async for _ in run.logos():
                pass

        assert len(thinking.articulators) == 1
        assert "".join(thinking.articulators[0].sent) == "<say>hi</say>"
        assert ego.rpc_calls == [
            ("c1", {"moment_ref": "3-7"}, [{"type": "text", "text": "moment:3-7"}], False),
        ]

    @pytest.mark.asyncio
    async def test_interpret_error_returns_cancel(self):
        """interpret error → 一行 ctml syntax error + cancel (细节留给下一轮 echoes)."""
        from ghoshell_moss.core.concepts.errors import InterpretError

        class _ErrorThinking(FakeRunThinking):
            def articulator(self, replan=False, wait_action_done=False):
                art = super().articulator(replan=replan, wait_action_done=wait_action_done)
                art.interpret_error = InterpretError("bad ctml")
                return art

        session = FakeRunSession()
        ego = FakeDispatchEgo(session)
        thinking = _ErrorThinking()
        run = self._run(session=session, ego=ego, thinking=thinking)
        async with run:
            await session.emit(self._ctml_delta("c1", name="moss_interpret", arguments_delta='{"ctml":"<bad>"}', seq=1))
            await session.emit(self._tool_call_event("moss_interpret", '{"ctml":"<bad>"}', "c1", seq=2))
            await session.emit(self._turn_end(seq=3))
            async for _ in run.logos():
                pass

        assert ego.rpc_calls == [("c1", "ctml syntax error", None, True)]

    @pytest.mark.asyncio
    async def test_interpret_without_deltas_sends_whole_argument(self):
        """无 delta (形状不符/整段回退): tool/call 一次性发完整 ctml."""
        session = FakeRunSession()
        ego = FakeDispatchEgo(session)
        thinking = FakeRunThinking()
        run = self._run(session=session, ego=ego, thinking=thinking)
        async with run:
            await session.emit(self._tool_call_event("moss_interpret", '{"ctml":"<say>hi</say>"}', "c1", seq=1))
            await session.emit(self._turn_end(seq=2))
            async for _ in run.logos():
                pass

        assert "".join(thinking.articulators[0].sent) == "<say>hi</say>"
        assert ego.rpc_calls == [
            ("c1", {"moment_ref": "3-7"}, [{"type": "text", "text": "moment:3-7"}], False),
        ]

    @pytest.mark.asyncio
    async def test_interpret_opens_and_closes_the_articulator_lifecycle(self):
        """回归: interpret 必须展开 articulator 的 async-with 生命周期 (__aenter__/__aexit__ 各一次).

        旧实现在 send + wait_compiled 后直接 return ToolCallResult, 从不退出边界 —— articulator 停在
        _commit 之后, 收尾永不发生, 插件侧于是把 pending tool 判成 settled, 结果回执被丢弃
        ("tool-result RPC dropped ... already settled plugin-side")。本用例锁住"边界被展开"。
        """
        session = FakeRunSession()
        ego = FakeDispatchEgo(session)
        thinking = FakeRunThinking()
        run = self._run(session=session, ego=ego, thinking=thinking)
        async with run:
            await session.emit(self._ctml_delta("c1", name="moss_interpret", arguments_delta='{"ctml":"<say>', seq=1))
            await session.emit(self._ctml_delta("c1", arguments_delta='hi</say>"}', seq=2))
            await session.emit(self._tool_call_event("moss_interpret", '{"ctml":"<say>hi</say>"}', "c1", seq=3))
            await session.emit(self._turn_end(seq=4))
            async for _ in run.logos():
                pass

        art = thinking.articulators[0]
        assert art.entered == 1, "articulator 必须被展开一次"
        assert art.exited == 1, "articulator 必须被收尾一次"
        assert "".join(art.sent) == "<say>hi</say>"

    @pytest.mark.asyncio
    async def test_interpret_waits_for_observed_not_actions_done(self):
        """interpret 只等 observed 命令落地, 不等动作全部跑完 —— 非 observe 命令跨帧继续跑.

        这是第五轮的关键语义转向: 等什么由 CTML 里的 @observe 声明, 而非"所有动作跑完".
        """
        session = FakeRunSession()
        ego = FakeDispatchEgo(session)
        log: list = []
        thinking = FakeRunThinking(log)
        run = self._run(session=session, ego=ego, thinking=thinking)
        async with run:
            await session.emit(self._ctml_delta("c1", name="moss_interpret", arguments_delta='{"ctml":"<say>', seq=1))
            await session.emit(self._ctml_delta("c1", arguments_delta='hi</say>"}', seq=2))
            await session.emit(self._tool_call_event("moss_interpret", '{"ctml":"<say>hi</say>"}', "c1", seq=3))
            await session.emit(self._turn_end(seq=4))
            async for _ in run.logos():
                pass

        assert log.count("wait_observed") == 1, "interpret 等 observed 命令落地"
        assert log.count("wait_action_done") == 0, "interpret 不再等动作全部跑完"
        assert log.count("wait_actions_done") == 0, "interpret 不再等 action 停 (跨帧继续)"
        assert ego.rpc_calls == [
            ("c1", {"moment_ref": "3-7"}, [{"type": "text", "text": "moment:3-7"}], False),
        ]

    @pytest.mark.asyncio
    async def test_react_streams_and_cuts_turn(self):
        """moss_react: 共享流式解析, 只等 compiled 就 cancel 本回合, 不签发 moment."""
        session = FakeRunSession()
        ego = FakeDispatchEgo(session)
        thinking = FakeRunThinking()
        run = self._run(session=session, ego=ego, thinking=thinking)
        async with run:
            await session.emit(self._ctml_delta("c2", name="moss_react", arguments_delta='{"ctml":"<say>', seq=1))
            await session.emit(self._ctml_delta("c2", arguments_delta='hi</say>"}', seq=2))
            await session.emit(self._tool_call_event("moss_react", '{"ctml":"<say>hi</say>"}', "c2", seq=3))
            await session.emit(self._turn_end(seq=4))
            async for _ in run.logos():
                pass

        art = thinking.articulators[0]
        assert art.entered == 1, "articulator 必须被展开一次"
        assert art.exited == 1, "articulator 必须被收尾一次"
        assert "".join(art.sent) == "<say>hi</say>"
        assert ego.rpc_calls == [("c2", "reacted", None, True)]

    @pytest.mark.asyncio
    async def test_reasoning_sets_one_shot_effort_and_cancels(self):
        """moss_reasoning: 设 ego 的一次性 effort + add_echoes(observe=True) 驱动下一帧 + cancel 中断."""
        from ._run import DoloresRun

        session = FakeRunSession()
        ego = FakeDispatchEgo(session)
        log: list = []
        thinking = FakeRunThinking(log)
        run = DoloresRun(ego=ego, thinking=thinking, thinking_event=asyncio.Event(), facade=None)
        await run._handle_tool_use_event(self._tool_call("moss_reasoning", '{"effort": "high"}', call_id="c_r"), {})

        assert ego.default_thinking_effort == "high"
        assert log == ["add_echoes:True"], "reasoning 必须 mark need_observe 驱动下一帧"
        assert ego.rpc_calls == [("c_r", {"effort": "high"}, None, True)]

    @pytest.mark.asyncio
    async def test_interpret_stream_parse_error_returns_cancel(self):
        """流式解析出错 (非法转义) → 回 "ctml parse error" + cancel (不 fallback)."""
        session = FakeRunSession()
        ego = FakeDispatchEgo(session)
        thinking = FakeRunThinking()
        run = self._run(session=session, ego=ego, thinking=thinking)
        async with run:
            await session.emit(self._ctml_delta("c1", name="moss_interpret", arguments_delta='{"ctml":"<say>\\x', seq=1))
            await session.emit(self._tool_call_event("moss_interpret", '{"ctml":"<say>"}', "c1", seq=2))
            await session.emit(self._turn_end(seq=3))
            async for _ in run.logos():
                pass

        assert ego.rpc_calls == [("c1", "ctml parse error", None, True)]


class TestDoloresMomentPayload:
    """DoloresEgo 的 moment 映射 — context (inject) + inputs (steer) 两条 message."""

    def _ego(self):
        from ._ego import DoloresEgo, DoloresEgoContext

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
            percepts={"test": [Message.new().with_content("percept")]},
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

    def test_moment_context_parts_contains_context_only(self):
        from ghoshell_moss.core.blueprint.moment import Echoes, Moment
        from ghoshell_moss.message import Message

        moment = Moment(
            previous=Echoes(messages=[Message.new().with_content("echo")]),
            percepts={"test": [Message.new().with_content("percept")]},
        )
        parts = self._ego().moment_context_parts(moment, "0-1")
        text = "".join(c["text"] for c in parts if c.get("type") == "text")
        assert "echo" in text
        assert "percept" not in text

    def test_moment_context_parts_empty_when_no_content(self):
        from ghoshell_moss.core.blueprint.moment import Moment

        assert self._ego().moment_context_parts(Moment(), "0-0") == []


class TestDoloresEpochPayload:
    """DoloresEgo 的 epoch 槽位 — <cognition_epoch> 容器 (recap + baseline), epoch 变更时才返回."""

    def _ego(self):
        from ._ego import DoloresEgo, DoloresEgoContext

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


class TestMementoChannel:
    """memento channel — ghost 的记忆器官: 读面透传 + 坏 commit 失败 + notice 列 branch."""

    class _Connection:
        def __init__(self, response):
            self.response = response
            self.calls: list[tuple[str, dict]] = []

        async def call(self, path, payload=None, *, timeout=None):
            self.calls.append((path, payload))
            if isinstance(self.response, Exception):
                raise self.response
            return self.response

    @staticmethod
    def _manager(tmp_path: Path, connection=None):
        from ghoshell_moss.memento import new_local_memento

        from ._ego_memento import EgoMementoConfig, EgoMementoManager

        memento = new_local_memento(tmp_path / "owner")
        memento.create_branch("main")
        return EgoMementoManager(
            connection=connection, memento=memento, config=EgoMementoConfig()
        ), memento

    @staticmethod
    def _coord(memento, seq: int) -> str:
        return memento.get_branch("main").get_commit(seq).coord

    def _channel(self, tmp_path: Path, connection=None):
        from .memento_channel import build_memento_channel

        manager, memento = self._manager(tmp_path, connection)
        chan = build_memento_channel(manager, storage_root=tmp_path / "owner")
        return chan, manager, memento

    @pytest.mark.asyncio
    async def test_read_surface_is_exposed(self, tmp_path: Path):
        chan, _, _ = self._channel(tmp_path)
        async with chan.bootstrap() as runtime:
            for name in ("view", "read", "history", "chat"):
                assert runtime.get_command(name) is not None

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


class TestDoloresMemories:
    def test_returns_single_memory_container(self, tmp_path: Path):
        from ghoshell_moss.memento import new_local_memento

        from ._ego_memento import EgoMementoConfig, EgoMementoManager

        memento = new_local_memento(tmp_path / "owner")
        memento.create_branch("main")
        manager = EgoMementoManager(connection=None, memento=memento, config=EgoMementoConfig())
        anchor = manager.commit(session_id="s1", start_turn=0, end_turn=1)
        memento.get_branch("main").note(anchor.id, "note")

        ghost = _dolores()
        ghost._ground_text = "GROUND"
        ghost._memento_manager = manager

        memories = ghost.memories()

        assert len(memories) == 1
        text = memories[0].to_content_string()
        assert "<memory" in text
        assert "<ground" in text and "GROUND" in text
        assert "<branch" in text and "note" in text

    def test_returns_empty_when_nothing_to_remember(self):
        ghost = _dolores()
        assert ghost.memories() == []


class TestBuildChannel:
    @pytest.mark.asyncio
    async def test_mounts_frame_channel_when_frame_root_given(self, tmp_path: Path):
        from ghoshell_moss.ground import DefaultGroundSet

        from .channel import build_dolores_channel

        (tmp_path / "GROUND.md").write_text("---\nname: g\n---\n# G\n")
        gs = DefaultGroundSet(workspace_root=tmp_path)
        frame_root = tmp_path / "frames"
        frame_root.mkdir()
        chan = build_dolores_channel(
            groundset=gs, workspace_root=tmp_path, frame_root=frame_root
        )
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            names = {m.name for m in runtime.metas().values()}
            assert "frame" in names
            assert "ground" in names

    @pytest.mark.asyncio
    async def test_no_frame_channel_without_frame_root(self, tmp_path: Path):
        from ghoshell_moss.ground import DefaultGroundSet

        from .channel import build_dolores_channel

        (tmp_path / "GROUND.md").write_text("---\nname: g\n---\n# G\n")
        gs = DefaultGroundSet(workspace_root=tmp_path)
        chan = build_dolores_channel(groundset=gs, workspace_root=tmp_path)
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            names = {m.name for m in runtime.metas().values()}
            assert "frame" not in names
            assert "ground" in names

    @pytest.mark.asyncio
    async def test_init_frame_lands_in_notice(self, tmp_path: Path):
        from ghoshell_moss.channels.frame_channel import Frame
        from ghoshell_moss.ground import DefaultGroundSet

        from .channel import build_dolores_channel

        (tmp_path / "GROUND.md").write_text("---\nname: g\n---\n# G\n")
        gs = DefaultGroundSet(workspace_root=tmp_path)
        frame_root = tmp_path / "frames"
        frame_root.mkdir()
        seed = Frame(label="orient", questions=["Where am I?"])
        chan = build_dolores_channel(
            groundset=gs, workspace_root=tmp_path, frame_root=frame_root, init_frame=seed
        )
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            frame_meta = next(m for m in runtime.metas().values() if m.name == "frame")
        assert "Where am I?" in frame_meta.named_notices["orient"]
