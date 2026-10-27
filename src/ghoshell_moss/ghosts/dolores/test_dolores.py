"""Dolores Ghost 原型测试 — 骨架 + stubs 同步阶段.

覆盖:
- GhostMeta/Ghost ABC 契约
- stubs 同步三种路径: init / override / noop
- 构造无副作用 (写盘收敛到 __aenter__)
- session.output 提示在 init/override 时发出, noop 不发声
"""

import asyncio
import contextlib
import logging
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

    def test_load_startup_parses_thinking_effort(self, tmp_path: Path):
        """boot 思考档声明可读; 手写未加引号的 off (YAML 1.1 bool) 不能把整份 startup 打掉."""
        (tmp_path / "startup").mkdir(parents=True)
        (tmp_path / "startup" / "default.startup.yml").write_text(
            'instruction: hi\ndefault_thinking_effort: off\n', encoding="utf-8"
        )
        ghost = _dolores(home=tmp_path)
        doc = ghost._load_startup()
        assert doc is not None
        assert doc.instruction == "hi", "整份文档必须存活"
        assert doc.default_thinking_effort == "off"

    def test_load_startup_thinking_effort_defaults_empty(self, tmp_path: Path):
        """缺省 = 不表态 (不覆盖 dsh/UI 持有的档位)."""
        (tmp_path / "startup").mkdir(parents=True)
        (tmp_path / "startup" / "default.startup.yml").write_text("instruction: hi\n", encoding="utf-8")
        ghost = _dolores(home=tmp_path)
        doc = ghost._load_startup()
        assert doc is not None
        assert doc.default_thinking_effort == ""

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
    """``agent-default-model`` 是 **seed-once**: 首次启动若段不存在 → 从 DoloresMeta.default_model
    写入; 之后段归 ghost home 所有 (dsh web Models 面可改), 启动不再覆盖. meta 若无 default_model →
    完全不碰 settings.yaml, 让 dsh 用自己的默认.
    """

    @staticmethod
    def _boot(home: Path, meta=None) -> None:
        ghost = _dolores(meta=meta, home=home, session=MockSession())

        async def run():
            async with ghost:
                pass

        asyncio.run(run())

    @staticmethod
    def _settings(home: Path) -> dict:
        return yaml.safe_load((home / ".dsh" / "settings.yaml").read_text(encoding="utf-8"))

    def test_seeds_section_from_meta_when_absent(self, tmp_path: Path):
        meta = _dolores_meta(default_model=("anthropic", "claude-opus-4-7"))
        self._boot(tmp_path, meta=meta)
        section = self._settings(tmp_path)["agent-default-model"]
        assert section == {"provider": "anthropic", "model": "claude-opus-4-7"}

    def test_does_not_overwrite_existing_section(self, tmp_path: Path):
        # UI 已经写过段 (哪怕是纯文本模型) → ghost 不再抢它.
        (tmp_path / ".dsh").mkdir()
        (tmp_path / ".dsh" / "settings.yaml").write_text(
            "agent-default-model:\n"
            "  provider: deepseek-official\n"
            "  model: deepseek-v4-flash\n"
            "  reasoningEffort: off\n",
            encoding="utf-8",
        )
        meta = _dolores_meta(default_model=("anthropic", "claude-opus-4-7"))
        self._boot(tmp_path, meta=meta)
        text = (tmp_path / ".dsh" / "settings.yaml").read_text(encoding="utf-8")
        assert "model: deepseek-v4-flash\n" in text
        assert "claude-opus-4-7" not in text

    def test_no_default_model_meta_leaves_settings_untouched(self, tmp_path: Path):
        self._boot(tmp_path)  # meta.default_model is None by default
        text = (tmp_path / ".dsh" / "settings.yaml").read_text(encoding="utf-8") \
            if (tmp_path / ".dsh" / "settings.yaml").exists() else ""
        assert "agent-default-model" not in text


class TestDoloresProviders:
    """``meta.providers`` → ``profiles/web/cordis.patch.yml`` 的 ``id: llm-pi-ai`` 条目, seed-once.
    base_url_env 在 seed 时从 ``os.environ`` 解析为真值写入 (文件在 gitignored 的 ghost home 内),
    api_key_env 以名字落下, 运行时由 dsh 解析.
    """

    @staticmethod
    def _patch_path(home: Path) -> Path:
        return home / ".dsh" / "profiles" / "web" / "cordis.patch.yml"

    @staticmethod
    def _boot(home: Path, meta) -> None:
        ghost = _dolores(meta=meta, home=home, session=MockSession())

        async def run():
            async with ghost:
                pass

        asyncio.run(run())

    def test_seeds_pi_ai_entry_with_resolved_base_url(self, tmp_path: Path, monkeypatch):
        from ._meta import DoloresProviderSpec

        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://proxy.example.com/anthropic")
        meta = _dolores_meta(providers=[
            DoloresProviderSpec(
                route="anthropic",
                api_key_env="ANTHROPIC_API_KEY",
                base_url_env="ANTHROPIC_BASE_URL",
            ),
        ])
        self._boot(tmp_path, meta=meta)

        docs = list(yaml.safe_load_all(self._patch_path(tmp_path).read_text(encoding="utf-8")))
        entries = next(d for d in docs if isinstance(d, list))
        pi_ai = next(e for e in entries if isinstance(e, dict) and e.get("id") == "llm-pi-ai")
        provider = pi_ai["providers"]["anthropic"]
        assert provider["apiKeyEnv"] == "ANTHROPIC_API_KEY"
        assert provider["baseURL"] == "https://proxy.example.com/anthropic"

    def test_seed_once_does_not_duplicate(self, tmp_path: Path, monkeypatch):
        from ._meta import DoloresProviderSpec

        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://a.example")
        meta = _dolores_meta(providers=[
            DoloresProviderSpec(route="anthropic", api_key_env="ANTHROPIC_API_KEY",
                                base_url_env="ANTHROPIC_BASE_URL"),
        ])
        self._boot(tmp_path, meta=meta)
        # Second boot with a different env URL — must not re-write.
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://b.example")
        self._boot(tmp_path, meta=meta)
        text = self._patch_path(tmp_path).read_text(encoding="utf-8")
        assert text.count("id: llm-pi-ai") == 1
        assert "https://a.example" in text
        assert "https://b.example" not in text

    def test_no_providers_meta_leaves_patch_as_stub(self, tmp_path: Path):
        self._boot(tmp_path, meta=_dolores_meta())
        text = self._patch_path(tmp_path).read_text(encoding="utf-8")
        assert "llm-pi-ai" not in text


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
    """enter_thinking 的思考档解析 — 单一 effort 字段; 声明只在被真正采用的那一帧消费."""

    @staticmethod
    def _ego(launcher, *, default_thinking_effort=""):
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
            default_thinking_effort=default_thinking_effort,
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
    async def test_frame_effort_wins_and_declaration_survives(self):
        """帧自带明确 effort → 直通; 未被采用的声明不消费 (留给下一个真正能带它的帧)."""
        launcher = self._Launcher()
        ego = self._ego(launcher)
        ego.default_thinking_effort = "high"

        await ego.enter_thinking(self._thinking("low"), 1)

        payload = launcher.calls[0][1]
        assert payload["effort"] == "low"
        assert "reasoning_effort" not in payload, "档位只有一个字段, 不再两路下发"
        assert ego.default_thinking_effort == "high"

    @pytest.mark.asyncio
    async def test_none_frame_does_not_eat_the_declaration(self):
        """'none' 帧不驱动 turn, 也不该把待采用的声明吃掉 —— 否则 moss_reasoning 会静默失效."""
        launcher = self._Launcher()
        ego = self._ego(launcher)
        ego.default_thinking_effort = "high"

        await ego.enter_thinking(self._thinking("none"), 1)

        assert launcher.calls[0][1]["effort"] == "none"
        assert ego.default_thinking_effort == "high"

    @pytest.mark.asyncio
    async def test_default_declaration_is_consumed_when_used(self):
        """帧 effort == '' → 采用 ego 的待定声明, 采用即消费 (只采用一次)."""
        launcher = self._Launcher()
        ego = self._ego(launcher)
        ego.default_thinking_effort = "high"

        await ego.enter_thinking(self._thinking(""), 1)

        payload = launcher.calls[0][1]
        assert payload["effort"] == "high"
        assert ego.default_thinking_effort is None

    @pytest.mark.asyncio
    async def test_no_declaration_means_empty_effort(self):
        """无声明 + 帧不表态 → effort 为空串: 不驱动档位变更, 档位归 dsh/UI."""
        launcher = self._Launcher()
        ego = self._ego(launcher)

        await ego.enter_thinking(self._thinking(""), 1)

        assert launcher.calls[0][1]["effort"] == ""
        assert ego.default_thinking_effort is None

    @pytest.mark.asyncio
    async def test_startup_declaration_seeds_the_ego(self):
        """startup 的 default_thinking_effort 落到 ego 的待定声明上 (boot 声明)."""
        launcher = self._Launcher()
        ego = self._ego(launcher, default_thinking_effort="low")

        assert ego.default_thinking_effort == "low"

        await ego.enter_thinking(self._thinking(""), 1)

        assert launcher.calls[0][1]["effort"] == "low"
        assert ego.default_thinking_effort is None

    @pytest.mark.asyncio
    async def test_empty_startup_declares_nothing(self):
        """startup 空值 = 不表态: ego 不带初值, 不覆盖 dsh/UI 持有的档位."""
        launcher = self._Launcher()
        ego = self._ego(launcher)

        assert ego.default_thinking_effort is None

    @pytest.mark.asyncio
    async def test_model_declaration_rides_the_next_frame_and_is_consumed_once(self):
        """模型声明与档位同一条路: 下一帧带出 provider/model, 采用即消费."""
        launcher = self._Launcher()
        ego = self._ego(launcher)

        ego.declare_model("deepseek-official", "deepseek-v4-pro")
        await ego.enter_thinking(self._thinking(""), 1)

        payload = launcher.calls[0][1]
        assert payload["provider"] == "deepseek-official"
        assert payload["model"] == "deepseek-v4-pro"
        assert ego.default_model is None

    @pytest.mark.asyncio
    async def test_no_model_declaration_sends_empty_strings(self):
        """无声明 = 不表态: 空串让插件保持当前模型 (不是把 provider 清零)."""
        launcher = self._Launcher()
        ego = self._ego(launcher)

        await ego.enter_thinking(self._thinking(""), 1)

        payload = launcher.calls[0][1]
        assert payload["provider"] == ""
        assert payload["model"] == ""

    @pytest.mark.asyncio
    async def test_declare_model_carries_the_effort_and_refuses_half_a_route(self):
        """换模型可以顺手带档位; provider/model 缺一不可 —— 半个 route 会写坏下一次请求."""
        launcher = self._Launcher()
        ego = self._ego(launcher)

        receipt = ego.declare_model("deepseek-official", "deepseek-v4-pro", "max")
        assert "deepseek-v4-pro" in receipt and "max" in receipt
        assert ego.default_model == ("deepseek-official", "deepseek-v4-pro")
        assert ego.default_thinking_effort == "max"

        with pytest.raises(ValueError):
            ego.declare_model("", "deepseek-v4-pro")

    def test_model_identity_is_none_until_a_request_is_observed(self):
        """温数据的读口: 还没跑过请求 = None, 不拿目录默认值冒充"我在跑的模型"."""
        launcher = self._Launcher()
        ego = self._ego(launcher)

        assert ego.model_identity is None
        assert ego.model_route is None

    def test_declare_model_updates_identity_immediately(self):
        """声明模型那一刻身份就变 —— notice 不等下一帧 header 才反映改动."""
        launcher = self._Launcher()
        ego = self._ego(launcher)

        ego.declare_model("deepseek-official", "deepseek-v4-pro", "high")

        assert ego.model_identity == ("deepseek-official", "deepseek-v4-pro", "high")

    def test_declare_effort_updates_identity_immediately(self):
        """只改档位那一刻身份也变 —— 模型保持, 档位立即切到新值."""
        launcher = self._Launcher()
        ego = self._ego(launcher)
        ego.declare_model("deepseek-official", "deepseek-flash", "low")

        ego.declare_effort("high")

        assert ego.model_identity == ("deepseek-official", "deepseek-flash", "high")

    def test_declare_model_queues_identity_notice_immediately(self):
        """声明即排 notice —— 下一帧 enter 就带上新身份, 不等 header 回传."""
        launcher = self._Launcher()
        ego = self._ego(launcher)

        ego.declare_model("deepseek-official", "deepseek-v4-pro", "high")

        notices = [n for n in ego._notices if n.meta.tag == "model_notice"]
        assert len(notices) == 1
        assert "deepseek-v4-pro" in notices[0].to_content_string()

    @pytest.mark.asyncio
    async def test_lost_enter_does_not_burn_the_epoch(self):
        """enter RPC 丢了不能打戳 —— 否则 recap (压缩后 memory 的唯一通路) 永不重发."""
        from ghoshell_moss.message import Message

        class _FlakyLauncher(self._Launcher):
            fail = False

            async def call(self, path, payload=None, *, timeout=None):
                if self.fail:
                    raise RuntimeError("enter lost")
                return await super().call(path, payload, timeout=timeout)

        launcher = _FlakyLauncher()
        ego = self._ego(launcher)
        thinking = self._thinking("")
        thinking.observer.epoch = SimpleNamespace(
            id="e1", index=1,
            recap=[Message.new(tag="memory").with_content("ground")],
            baseline={"facade": "<facade/>"},
        )

        launcher.fail = True
        with pytest.raises(RuntimeError):
            await ego.enter_thinking(thinking, 1)
        assert ego._moment_epoch is None

        launcher.fail = False
        await ego.enter_thinking(thinking, 2)
        assert launcher.calls[0][1]["epoch"] is not None
        assert ego._moment_epoch == "e1"

        # 送达后同 epoch 不再重复.
        await ego.enter_thinking(thinking, 3)
        assert launcher.calls[1][1]["epoch"] is None


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


class _FakeCompactManager:
    """Duck-typed EgoMementoManager — compact 只读阈值配置与 SCS."""

    def __init__(self, cut_turn=7, compact_tokens=0):
        self.config = SimpleNamespace(compact_tokens=compact_tokens)
        self._cut_turn = cut_turn

    def compact_cut_turn(self):
        return self._cut_turn


class _FakeCompactLauncher:
    def __init__(self, replaced: int = 1):
        self._replaced = replaced
        self.calls: list[tuple[str, dict]] = []

    async def call(self, path, payload):
        self.calls.append((path, payload))
        return {"replaced": self._replaced}


class _CancellingLauncher:
    """取消落在 compact RPC 中间 — 插件可能已经换了 surface."""

    async def call(self, path, payload):
        raise asyncio.CancelledError()


def _compact_ego(*, memories, memento_manager=None, launcher=None):
    from ._ego import DoloresEgo, DoloresEgoContext

    ego = DoloresEgo(
        launcher=launcher,
        ctx=DoloresEgoContext(
            project_home=Path("."),
            project_name="pytest",
            name="dolores",
            mode="pytest",
            instruction="i",
            facade=None,
        ),
        memories=memories,
        memento_manager=memento_manager,
    )
    ego._ego_session_id = "s1"
    return ego


class TestDoloresEgoCompact:
    """compact 的 payload 拆分 — memory 走 epoch recap, surface 只留说明."""

    @staticmethod
    def _memory_message():
        from ghoshell_moss.message import Message

        return Message.new(tag="memory").with_content("ground")

    @pytest.mark.asyncio
    async def test_payload_splits_memory_from_notice(self):
        ego = _compact_ego(memories=lambda: [self._memory_message()])

        memory, notice = ego._compact_memory_payload()

        assert "<memory>" in memory[0].to_content_string()
        assert "compacted" in notice
        assert "<memory>" not in notice

    @pytest.mark.asyncio
    async def test_compact_if_pending_returns_memory_and_sends_notice(self):
        launcher = _FakeCompactLauncher()
        ego = _compact_ego(
            memories=lambda: [self._memory_message()],
            memento_manager=_FakeCompactManager(cut_turn=7),
            launcher=launcher,
        )
        ego._pending_compact = True

        memory = await ego.compact_if_pending()

        assert memory is not None and "<memory>" in memory[0].to_content_string()
        path, payload = launcher.calls[0]
        assert path.endswith("/compact")
        assert payload["cut_turn"] == 7
        assert "<memory>" in payload["memory"][0]["text"]
        assert "compacted" in payload["notice"]
        assert ego._pending_compact is False
        # 成功返回即交付, 不留下跨帧暂存.
        assert ego.take_compact_memory() is None

    @pytest.mark.asyncio
    async def test_nothing_shadowed_is_not_a_compact(self):
        """plugin 返回 replaced=0 → 实际没压, 不开新 epoch."""
        ego = _compact_ego(
            memories=lambda: [self._memory_message()],
            memento_manager=_FakeCompactManager(cut_turn=7),
            launcher=_FakeCompactLauncher(replaced=0),
        )
        ego._pending_compact = True

        assert await ego.compact_if_pending() is None
        assert ego.take_compact_memory() is None

    @pytest.mark.asyncio
    async def test_compact_if_pending_noop_without_pending(self):
        ego = _compact_ego(
            memories=lambda: [],
            memento_manager=_FakeCompactManager(),
            launcher=_FakeCompactLauncher(),
        )

        assert await ego.compact_if_pending() is None

    @pytest.mark.asyncio
    async def test_cancelled_compact_leaves_memory_for_the_next_frame(self):
        """取消落在 RPC 窗口: 插件可能已换 surface —— memory 必须留给下一帧补开 epoch."""
        ego = _compact_ego(
            memories=lambda: [self._memory_message()],
            memento_manager=_FakeCompactManager(cut_turn=7),
            launcher=_CancellingLauncher(),
        )
        ego._pending_compact = True

        with pytest.raises(asyncio.CancelledError):
            await ego.compact_if_pending()

        memory = ego.take_compact_memory()
        assert memory is not None and "<memory>" in memory[0].to_content_string()
        assert ego.take_compact_memory() is None  # 取走即清


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
        from ._ego_memento import EgoMementoConfig, EgoMementoManager
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
        from ._ego_memento import EgoMementoConfig, EgoMementoManager
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
        from ._ego_memento import EgoMementoConfig

        conn = self._FakeConnection({"message": "x"})
        manager, _ = self._manager(tmp_path, conn, config=EgoMementoConfig(resume_tail=2))
        for i in range(4):
            manager.commit(session_id="s1", start_turn=i, end_turn=i + 1)

        assert manager.backfill() == 2
        await manager.drain_bypass()
        assert {call[1]["ref"]["end_turn"] for call in conn.calls} == {3, 4}

    def test_backfill_disabled_when_tail_is_zero(self, tmp_path: Path):
        from ._ego_memento import EgoMementoConfig

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
        from ._ego_memento import SourceUnreadableError

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


class TestDoloresRun:
    """DoloresRun 生命周期 + 事件消费 — public 类, 轻量 fake 即可验证."""

    def _run(self, session=None, ego=None, thinking=None, facade=None):
        from ._run import DoloresRun

        session = session or FakeRunSession()
        return DoloresRun(
            ego=ego or FakeRunEgo(session),
            thinking=thinking or FakeRunThinking(),
            thinking_event=asyncio.Event(),
            facade=facade or _NoopShellFacade(),
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
    async def test_closing_tool_races_the_percept_push(self):
        """收轮工具在飞时人插一句: 立旗 + 照推 —— 不再死押, 这一轮要接着走."""
        run = self._run()
        run._thinking_event.set()
        seen: dict = {}

        async def _fake_dispatch(event, streams):
            seen["in_flight"] = run._ctml_in_flight
            run._on_percepts_injected(["he said something mid-action"])
            seen["barge"] = run._barge_in_event is not None and run._barge_in_event.is_set()
            seen["push"] = run._buffered_push
            run._buffered_push.cancel()

        run._dispatch_tool_use_event = _fake_dispatch
        await run._handle_tool_use_event(
            self._tool_call("moss_react", '{"ctml": "<noop/>"}', call_id="call_react"), {},
        )

        assert seen["in_flight"] is True, "react 必须在立旗状态下执行"
        assert seen["barge"] is True, "人插话必须叫醒那个等待, 否则竞速永远没人赢"
        assert seen["push"] is not None, "竞速赢了就该照推 moment"
        assert run._ctml_in_flight is False, "工具返回后旗子必须放下"
        assert run._barge_in_event is None, "竞速信号只归这一次调用"

    @pytest.mark.asyncio
    async def test_wait_next_tool_call_arms_the_barge_in_race(self):
        """moss_wait_next 同样进竞速 (它也是在等身体), 只是赢下之后是"这轮接着走"而不是"收轮"."""
        run = self._run()
        run._thinking_event.set()
        seen: dict = {}

        async def _fake_dispatch(event, streams):
            run._on_percepts_injected(["he said something while I yield"])
            seen["barge"] = run._barge_in_event is not None and run._barge_in_event.is_set()
            seen["push"] = run._buffered_push
            run._buffered_push.cancel()

        run._dispatch_tool_use_event = _fake_dispatch
        await run._handle_tool_use_event(self._tool_call("moss_wait_next", "{}"), {})

        assert seen["barge"] is True
        assert seen["push"] is not None

    @pytest.mark.asyncio
    async def test_percepts_stay_in_the_buffer_once_the_turn_is_closing(self):
        """竞速已经判给身体 (旗子撤下) 之后落地的 percepts 只留 buffer —— 这一轮正在收尾."""
        run = self._run()
        run._thinking_event.set()
        run._ctml_in_flight = True
        run._barge_in_event = None
        seen: dict = {}

        run._on_percepts_injected(["late words"])
        seen["push"] = run._buffered_push

        assert seen["push"] is None, "定了收轮之后不许再推给一个正在收尾的 turn"

    @pytest.mark.asyncio
    async def test_percepts_during_settle_only_arm_the_signal(self):
        """settle 期间 (不安全) signal 只置位 signal_event, 不 drain —— 留给 settle 竞速决定.

        这是 safe flag 的核心: turn/end 后到 settle 结论之间, 若急着 drain, 竞速判 action_done
        退出时 signal 已经被抽干 (has_pending 假), mindflow 不签下一帧 → 两头落空. 所以不安全窗口
        只通知不 drain, 竞速判退出时 signal 仍留在 observer 兜底.
        """
        run = self._run()
        run._thinking_event.set()
        run._settling = True
        run._signal_event = asyncio.Event()

        run._on_percepts_injected(["他说了一句"])

        assert run._signal_event.is_set(), "settle 期间 signal 必须叫醒竞速"
        assert run._buffered_push is None, "settle 期间不 drain、不 push"
        assert run._pushed_this_run is False, "settle 期间不标记 push"

    @pytest.mark.asyncio
    async def test_settle_exits_when_actions_done_and_no_signal(self):
        """turn/end(normal) 后 settle: 无 push、无新 signal → action_done 赢 → 退出本 run."""
        run = self._run()
        run._thinking_event.set()
        assert await run._settle_after_turn_end() is False, "无 signal 时 settle 判退出"

    @pytest.mark.asyncio
    async def test_interpret_tool_call_does_not_hold_the_percept_push(self):
        """moss_interpret 不立旗: 它跑完这一轮还要继续, 人此刻说的话下一步就该读到."""
        run = self._run()
        run._thinking_event.set()
        seen: dict = {}

        async def _fake_dispatch(event, streams):
            seen["in_flight"] = run._ctml_in_flight
            run._on_percepts_injected(["he said something mid-action"])
            seen["push"] = run._buffered_push
            assert run._buffered_push is not None, "interpret 期间必须照常开推"
            run._buffered_push.cancel()

        run._dispatch_tool_use_event = _fake_dispatch
        await run._handle_tool_use_event(
            self._tool_call("moss_interpret", '{"ctml": "<noop/>"}'), {},
        )

        assert seen["in_flight"] is False, "interpret 不押推送"
        assert run._ctml_in_flight is False

    @pytest.mark.asyncio
    async def test_non_ctml_tool_does_not_hold_the_percept_push(self):
        """非 CTML 工具 (bash / 工具面读) 不立旗 — 跑 bash 时人说的话照旧推进来."""
        run = self._run()
        seen: dict = {}

        async def _fake_dispatch(event, streams):
            seen["in_flight"] = run._ctml_in_flight

        run._dispatch_tool_use_event = _fake_dispatch
        await run._handle_tool_use_event(self._tool_call("moss_channel_facade", '{}'), {})

        assert seen["in_flight"] is False

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
    async def test_ghost_event_escalates_the_attention_only_while_the_turn_runs(self):
        """DSH 每签发一个事件就提权一次 —— thinking 期间人开口才抢不动、只能排队折进来.

        反例 (D18 实测): 没有提权时, 只跑 tool call 的那几秒里强度按衰减曲线掉下去, 人一句
        INFO 就赢下仲裁 → preempt → 整轮 abort, 正在跑的 tool call 一起陪葬.
        """
        log: list = []
        run = self._run(thinking=FakeRunThinking(log))
        event = self._event("tool/start", {"turn": 1})

        run._thinking_event.set()
        await run._on_event(event)
        assert "escalate" in log

        # 收线之后不再提权 —— 否则一个已经结束的 attention 永远不衰减.
        log.clear()
        run._thinking_event.clear()
        await run._on_event(event)
        assert "escalate" not in log

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
    async def test_shell_clear_reaches_shell_directly(self):
        """moss_shell_clear 直接调 shell.clear(), 不经过 CTML 的 __main__ 队列.

        这是排队堵死时的逃生口: CTML 的 <clear/> 自己也是一条命令, 会和被堵的命令排在一起.
        它不该签发 moment, 也不该掐回合 —— 下一步要能看见清完之后的现场.
        """
        from ._run import DoloresRun

        calls = []

        class _Shell:
            async def clear(self):
                calls.append("clear")

        class _Facade:
            def __init__(self):
                self.shell = _Shell()

        session = FakeRunSession()
        ego = FakeDispatchEgo(session)
        run = DoloresRun(
            ego=ego, thinking=FakeRunThinking(), thinking_event=asyncio.Event(), facade=_Facade(),
        )
        await run._handle_tool_use_event(self._tool_call("moss_shell_clear", call_id="call_c"), {})

        assert calls == ["clear"], "shell.clear() 没被调用 —— 逃生口等于不存在"
        call_id, result, error, cancel = ego.rpc_calls[0]
        assert call_id == "call_c"
        assert error is None
        assert cancel is False, "清壳不该掐掉回合, 否则下一步看不到结果"
        assert "cleared" in result

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
            ("call_w", {"moment_ref": "3-7"}, {"context": [{"type": "text", "text": "moment:3-7"}], "inputs": [], "moment_id": "3-7"}, False),
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
        """plain text 跳过 (不解析为 CTML), turn/end 被转发成 GhostEvent 后 logos() 自止."""
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
        # plain text 不产出 logos; 流以 thinking 开场, turn/end 只收线、不再转发.
        assert [d.event for d in collected] == ["ghost/lifecycle/thinking"]
        assert thinking.articulators == []  # 无 articulator 被创建
        assert thinking.abort_reasons == []  # completed → 不打断

    @pytest.mark.asyncio
    async def test_logos_forwards_complete_dsh_events_only(self):
        """只转发完整事件 (turn/tool/assistant-message); assistant/chunk 逐 token 流被排除."""
        session = FakeRunSession()
        ego = FakeRunEgo(session)
        thinking = FakeRunThinking()
        run = self._run(session=session, ego=ego, thinking=thinking)
        async with run:
            await session.emit(self._event("turn/start", {"turn": 1}, seq=1))
            await session.emit(self._event("assistant/message", {"turn": 1, "step": 1, "message": {}}, seq=2))
            await session.emit(self._event("tool/call", {"callId": "c1", "name": "mystery_tool", "arguments": "{}"}, seq=3))
            await session.emit(self._event("tool/result", {"message": {}, "error": None}, seq=4))
            await session.emit(self._text_chunk("reasoning nobody sees", seq=5))  # 排除: 流
            await session.emit(self._event("turn/end", {"turn": 1}, seq=6))
            collected = []
            async for delta in run.logos():
                collected.append(delta)
        # lifecycle/thinking 开场; 之后只剩行动脊椎 —— tool/call. turn/*、assistant/message
        # 和 tool/result 还在 session log 里, 但不再往观测面复制一遍.
        assert [e.event for e in collected] == [
            "ghost/lifecycle/thinking",
            "dsh/tool/call",
        ]
        # 转发的 payload 是原始信封 (to_dict), data 原样在内.
        assert collected[1].payload["seq"] == 3
        # content 是给表面的那一行: 摘要由我们自己拼, 不指望 dsh 给.
        assert collected[0].content == "thinking"
        assert collected[1].content == "mystery_tool"

    @pytest.mark.asyncio
    async def test_tool_event_summary_prefers_the_callers_own_description(self):
        """观测面上那一行取调用方自己写的 description, 不是参数原文."""
        session = FakeRunSession()
        ego = FakeRunEgo(session)
        thinking = FakeRunThinking()
        run = self._run(session=session, ego=ego, thinking=thinking)
        async with run:
            await session.emit(self._event(
                "tool/call",
                {"callId": "c1", "name": "bash",
                 "arguments": '{"command": "git status", "description": "Show working tree status"}'},
                seq=1,
            ))
            await session.emit(self._event("turn/end", {"turn": 1}, seq=2))
            collected = []
            async for delta in run.logos():
                collected.append(delta)
        assert collected[1].content == "bash: Show working tree status"

    @pytest.mark.asyncio
    async def test_tool_event_summary_never_leaks_argument_text(self):
        """没有 description 就只报工具名 —— 参数里可能有文件正文/路径/口令, 不许上总线."""
        session = FakeRunSession()
        ego = FakeRunEgo(session)
        thinking = FakeRunThinking()
        run = self._run(session=session, ego=ego, thinking=thinking)
        async with run:
            await session.emit(self._event(
                "tool/call",
                {"callId": "c1", "name": "write",
                 "arguments": '{"file_path": "/tmp/secret.md", "content": "private text"}'},
                seq=1,
            ))
            await session.emit(self._event("turn/end", {"turn": 1}, seq=2))
            collected = []
            async for delta in run.logos():
                collected.append(delta)
        assert collected[1].content == "write"
        assert "private" not in collected[1].payload.get("content", "")
        assert "secret" not in str(collected[1].content)

    @pytest.mark.asyncio
    async def test_own_tool_calls_never_reach_the_surface(self):
        """moss_interpret / moss_react 是管道不是行动: 不上观测面."""
        session = FakeRunSession()
        ego = FakeRunEgo(session)
        thinking = FakeRunThinking()
        run = self._run(session=session, ego=ego, thinking=thinking)
        async with run:
            await session.emit(self._event(
                "tool/call",
                {"callId": "c1", "name": "moss_surface_probe",
                 "arguments": '{"ctml": "<say>hi</say>"}'},
                seq=1,
            ))
            await session.emit(self._event("turn/end", {"turn": 1}, seq=2))
            collected = []
            async for delta in run.logos():
                collected.append(delta)
        assert [getattr(d, "event", None) for d in collected] == ["ghost/lifecycle/thinking"]

    def test_is_own_tool_call_discriminates_by_namespace(self):
        """判据挂在 moss 命名空间上 —— 与 tool 分发用的是同一条前缀."""
        from ghoshell_moss.ghosts.dolores._run import _is_own_tool_call

        def call(name: str):
            return self._event(
                "tool/call",
                {"callId": "c1", "name": name, "arguments": "{}"},
            )

        assert _is_own_tool_call(call("moss_interpret")) is True
        assert _is_own_tool_call(call("moss_react")) is True
        assert _is_own_tool_call(call("bash")) is False
        assert _is_own_tool_call(call("read")) is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "reason, expected",
        [
            ({"kind": "completed"}, None),
            ({"kind": "max-tokens"}, None),
            ({"kind": "brand-new-kind"}, None),
            ({"kind": "aborted", "reason": {"kind": "hook", "reason": "moss cancel_turn"}}, None),
            ({"kind": "aborted", "reason": {"kind": "user"}}, "turn/end aborted (user)"),
            ({"kind": "interrupted"}, "turn/end interrupted"),
            ({"kind": "blocked", "reason": {"kind": "user"}}, "turn/end blocked (user)"),
            ({"kind": "error"}, "turn/end error"),
        ],
    )
    async def test_turn_end_classifies_before_ending_the_frame(self, reason, expected):
        """turn/end 分两类: 自然收线不动 attention, 中断类 abort 并带 reason.

        旧版这条叫 `test_turn_end_never_aborts_thinking`, 断言"无论 kind 都绝不 abort" —— 那是把一个
        方向的判据当成了通性: 它只对"我们自己让 plugin 切的 turn"成立 (cause=hook, 身体在返回前已
        drain). UI 点停与运行时故障若也走自然退出, 身体会继续跑 (语音照旧播完), 故障还会自续成环.
        现在判据挂在 cause 上 (不挂 turn 号), 见 ``_run._turn_end_abort_reason``.

        端到端判据 (语音是否真停) 不在单测里: 见 regressions/dolores-ghost-prototype TC-003.
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
        if expected is None:
            assert thinking.abort_reasons == []
        else:
            assert thinking.abort_reasons == [expected]

    @pytest.mark.asyncio
    async def test_turn_end_interrupt_lands_before_the_consumer_can_run(self):
        """中断在**入队前**落地: 消费被 tool handler 占住时, 人点停也要立刻生效.

        2026-10-04 D04 实测的根: ``_handle_react`` 在 logos() 里 await wait_actions_done() 等身体跑完,
        期间的 turn/end 只能排队 —— 人点的停 (aborted/user) 在队列里等了 41 秒, 语音自己播完才被消费.
        判据是入队时就 abort (Attention.abort 同步立即生效), 不等消费.
        """
        session = FakeRunSession()
        ego = FakeRunEgo(session)
        thinking = FakeRunThinking()
        run = self._run(session=session, ego=ego, thinking=thinking)
        async with run:
            await session.emit(self._text_chunk("<say>hi</say>", seq=1))
            await session.emit(
                self._event(
                    "turn/end",
                    {"turn": 1, "reason": {"kind": "aborted", "reason": {"kind": "user"}}},
                    seq=2,
                )
            )
            # 关键: 这里**不消费** logos() —— 消费正被占住, 模拟 D04 的真实时序.
        assert thinking.abort_reasons == ["turn/end aborted (user)"]

    @pytest.mark.asyncio
    async def test_our_own_hook_cancel_does_not_abort_on_enqueue(self):
        """正常收轮 (cause=hook) 不得因为入了队就 abort —— 否则每次 react 一开口就被自己清掉."""
        session = FakeRunSession()
        ego = FakeRunEgo(session)
        thinking = FakeRunThinking()
        run = self._run(session=session, ego=ego, thinking=thinking)
        async with run:
            await session.emit(self._text_chunk("<say>hi</say>", seq=1))
            await session.emit(
                self._event(
                    "turn/end",
                    {"turn": 1, "reason": {"kind": "aborted",
                                           "reason": {"kind": "hook", "reason": "moss cancel_turn"}}},
                    seq=2,
                )
            )
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
        assert ego.rpc_calls == [("call_ok", {"moment_ref": "1-7"}, {"context": [{"type": "text", "text": "moment:1-7"}], "inputs": [], "moment_id": "1-7"}, False)]

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
            ("c1", {"moment_ref": "3-7"}, {"context": [{"type": "text", "text": "moment:3-7"}], "inputs": [], "moment_id": "3-7"}, False),
        ]

    @pytest.mark.asyncio
    async def test_reply_first_runs_the_interpret_path_and_does_not_close_the_turn(self):
        """reply_first 是同一机制换个名字: 流式进 articulator, 签 moment, cancel 为假."""
        session = FakeRunSession()
        ego = FakeDispatchEgo(session)
        thinking = FakeRunThinking()
        run = self._run(session=session, ego=ego, thinking=thinking)
        async with run:
            await session.emit(self._ctml_delta("c1", name="moss_reply_first", arguments_delta='{"ctml":"<say>', seq=1))
            await session.emit(self._ctml_delta("c1", arguments_delta='先看一眼</say>"}', seq=2))
            await session.emit(self._tool_call_event("moss_reply_first", '{"ctml":"<say>先看一眼</say>"}', "c1", seq=3))
            await session.emit(self._turn_end(seq=4))
            async for _ in run.logos():
                pass

        assert "".join(thinking.articulators[0].sent) == "<say>先看一眼</say>"
        assert ego.rpc_calls == [
            ("c1", {"moment_ref": "3-7"}, {"context": [{"type": "text", "text": "moment:3-7"}], "inputs": [], "moment_id": "3-7"}, False),
        ], "与 interpret 同一条路: 带 moment 回来, cancel 为假 —— 这一轮还要继续"

    @pytest.mark.asyncio
    async def test_interpret_refreshes_metas_before_signing_the_moment(self):
        """签 moment 之前必须把暖层刷到"动作之后" —— 否则模型看到的还是动作前的世界 (晚一拍)."""
        from ._run import DoloresRun

        log: list[str] = []

        class _Shell:
            async def refresh_metas(self, timeout=None, stale_time=None):
                log.append("refresh_metas")

        class _Facade:
            def __init__(self):
                self.shell = _Shell()

        class _Thinking(FakeRunThinking):
            def observe(self):
                log.append("observe")
                return super().observe()

        session = FakeRunSession()
        ego = FakeDispatchEgo(session)
        run = DoloresRun(
            ego=ego,
            thinking=_Thinking(log=log),
            thinking_event=asyncio.Event(),
            facade=_Facade(),
        )
        async with run:
            await session.emit(self._ctml_delta("c1", name="moss_interpret", arguments_delta='{"ctml":"<say>', seq=1))
            await session.emit(self._ctml_delta("c1", arguments_delta='hi</say>"}', seq=2))
            await session.emit(self._tool_call_event("moss_interpret", '{"ctml":"<say>hi</say>"}', "c1", seq=3))
            await session.emit(self._turn_end(seq=4))
            async for _ in run.logos():
                pass

        assert "refresh_metas" in log and log[log.index("refresh_metas") + 1] == "observe"
        assert ego.rpc_calls == [
            ("c1", {"moment_ref": "3-7"}, {"context": [{"type": "text", "text": "moment:3-7"}], "inputs": [], "moment_id": "3-7"}, False),
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
            ("c1", {"moment_ref": "3-7"}, {"context": [{"type": "text", "text": "moment:3-7"}], "inputs": [], "moment_id": "3-7"}, False),
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
        # interpret 的 handler 不等 action 停; 这 1 次 wait_actions_done 来自 run 收尾的 settle 竞速
        # (主权在 body action: turn/end 后 run 仍要等 body 静默才退出).
        assert log.count("wait_actions_done") == 1, "interpret 的 handler 不等 action 停 (跨帧继续)"
        assert ego.rpc_calls == [
            ("c1", {"moment_ref": "3-7"}, {"context": [{"type": "text", "text": "moment:3-7"}], "inputs": [], "moment_id": "3-7"}, False),
        ]

    @pytest.mark.asyncio
    async def test_react_streams_and_cuts_turn(self):
        """moss_react: 共享流式解析, 等 compiled + action 停, 再 cancel 本回合, 不签发 moment."""
        session = FakeRunSession()
        ego = FakeDispatchEgo(session)
        log: list = []
        thinking = FakeRunThinking(log)
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
        # react 的竞速等 action 停 1 次; turn/end 后的 settle 竞速再等 1 次 (run 收口).
        assert log.count("wait_actions_done") == 2, "react 先等 action 停再 cancel, settle 再等一次收口"
        assert ego.rpc_calls == [("c2", "reacted", None, True)]

    @staticmethod
    async def _drain(run) -> None:
        async for _ in run.logos():
            pass

    @pytest.mark.asyncio
    async def test_react_yields_to_a_barge_in_and_keeps_the_turn(self):
        """准全双工: react 正等身体跑完时人插一句 —— 工具立刻返回, 不 cancel, 帧照推.

        判据 (他给的): 竞速赢了不是"结束这一轮", 是"这一轮接着往下走". 所以结果是 cancel=False,
        而且那一帧必须已经落地 (下一 step 才读得到). 身体的动作不取消, 跨帧照旧跑.
        """
        session = FakeRunSession()
        ego = FakeDispatchEgo(session)
        thinking = _GatedThinking()
        run = self._run(session=session, ego=ego, thinking=thinking)
        try:
            async with run:
                await session.emit(self._ctml_delta("c3", name="moss_react", arguments_delta='{"ctml":"<noop/>"}', seq=1))
                await session.emit(self._tool_call_event("moss_react", '{"ctml":"<noop/>"}', "c3", seq=2))
                drain = asyncio.create_task(self._drain(run))
                await asyncio.wait_for(thinking.reached.wait(), timeout=2)
                run._on_percepts_injected(["他说了一句"])
                await session.emit(self._turn_end(seq=3))
                thinking.gate.set()  # 放行 settle 竞速的 wait_actions_done, 让 run 收口
                await asyncio.wait_for(drain, timeout=2)
        finally:
            thinking.gate.set()

        call_id, result, moment, cancel = ego.rpc_calls[0]
        assert call_id == "c3"
        assert cancel is False, "竞速赢了就不许收轮 —— 这一轮要接着读那句新的话"
        assert "continues" in result
        assert ego.pushed_moments, "赢了竞速的那一帧必须落地, 否则下一 step 读不到它"

    @pytest.mark.asyncio
    async def test_wait_next_continues_the_turn_when_someone_speaks(self):
        """moss_wait_next: 交出这一轮之前人说话了 —— 交出的前提没了, 这一轮接着走."""
        session = FakeRunSession()
        ego = FakeDispatchEgo(session)
        thinking = _GatedThinking()
        run = self._run(session=session, ego=ego, thinking=thinking)
        try:
            async with run:
                await session.emit(self._tool_call_event("moss_wait_next", "{}", "c4", seq=1))
                drain = asyncio.create_task(self._drain(run))
                await asyncio.wait_for(thinking.reached.wait(), timeout=2)
                run._on_percepts_injected(["他抢先说了一句"])
                await session.emit(self._turn_end(seq=2))
                thinking.gate.set()  # 放行 settle 竞速的 wait_actions_done, 让 run 收口
                await asyncio.wait_for(drain, timeout=2)
        finally:
            thinking.gate.set()

        call_id, result, moment, cancel = ego.rpc_calls[0]
        assert call_id == "c4"
        assert cancel is False
        assert "continues" in result
        assert ego.pushed_moments

    @pytest.mark.asyncio
    async def test_reasoning_records_depth_and_cancels(self):
        """moss_reasoning: 入口只标记 + cancel 收轮; 等身体/推档位/补回声都在 run 收尾."""
        from ._run import DoloresRun

        session = FakeRunSession()
        ego = FakeDispatchEgo(session)
        log: list = []
        thinking = FakeRunThinking(log)
        run = DoloresRun(ego=ego, thinking=thinking, thinking_event=asyncio.Event(), facade=None)
        await run._handle_tool_use_event(self._tool_call("moss_reasoning", '{"effort": "high"}', call_id="c_r"), {})

        assert ego.default_thinking_effort == "high"
        assert run._need_switch_thinking_effort == "high", "入口只把要换的档位记成状态"
        assert log == [], "工具入口不执行副作用: 不等动作、不推档位、不提交回声"
        assert ego.rpc_calls == [("c_r", {"effort": "high"}, None, True)]

        # 副作用全部在 run 收尾落地, 顺序: 等身体 → 推档位 → 补回声 (都在 exit_thinking 之前).
        # 替身上没有 select_model, 于是走 _push_effort_officially 的"永不致命"退路.
        await run.__aexit__(None, None, None)
        assert log == ["wait_actions_done", "observer.add_echoes:True"], (
            "收尾必须先把身体等完, 再补 need_observe —— 少哪一步, 档位换了下一帧也没人签发"
        )
        assert run._need_switch_thinking_effort is None, "pending 收尾后必须清掉, 不留悬声明"

    def test_tool_surface_matches_production_thinking(self):
        """handlers 在 thinking 上用的每个 API, 生产类必须真的有.

        回归防线. 这块曾经栽在哪: FakeRunThinking 发明了一个生产 ``Thinking`` 上不存在的
        ``add_echoes``, 单测全绿, 而生产里 moss_reasoning 抛 AttributeError → 被
        ``run_tool`` 吞成 result=None (cancel 随之丢失) → 本轮永不结束, 模型只看到 null.
        教训不是"补一个断言", 而是: **替身不许发明生产没有的 API**.
        """
        from ghoshell_moss.core.blueprint.mindflow import Thinking

        for name in ("articulator", "wait_actions_done", "observe", "observer", "abort"):
            assert hasattr(Thinking, name), f"Thinking 上没有 {name} —— handler 会在运行期静默失败"
        assert not hasattr(Thinking, "add_echoes"), (
            "add_echoes 属于 observer (Moments), 不属于 Thinking —— 用错对象就是那个被吞掉的 AttributeError"
        )

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

    def test_moment_payload_carries_both_slots(self):
        """一条 moment 投递必须同时带 context 与 inputs —— 少一半就是一次静默丢字.

        病根: ``observe()`` 把挂起的 percepts 抽干塞进它产出的 moment, 而投递走哪条路由别人决定.
        所以任何一条路只要结构上不带 inputs, 就会把已消费的文字蒸发 (D13 的 buffered 语音)。
        """
        from ghoshell_moss.core.blueprint.moment import Echoes, Moment
        from ghoshell_moss.message import Message

        moment = Moment(
            previous=Echoes(messages=[Message.new().with_content("echo")]),
            percepts={"test": [Message.new().with_content("percept")]},
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
    """memento channel — ghost 的记忆器官: 读面透传 + 落锚写面 + 坏 commit 失败 + notice 列 branch."""

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

    @staticmethod
    def _committer(manager, spans=((0, 1),)):
        """假 ego —— 每次调用封一段, 段用完后返回 None (= 没有新东西可封).

        与 ego 同形: 落锚之后**照常**排旁路, 让开与否归 ``schedule_note`` 自己判 —— 约定
        落在那一个判据上, 不落在调用方的小心翼翼上.
        """
        pending = list(spans)

        def commit_anchor(note: str) -> str | None:
            if not pending:
                return None
            start, end = pending.pop(0)
            anchor = manager.commit(
                session_id="s1", start_turn=start, end_turn=end, message=note
            )
            if anchor is None:
                return None
            manager.schedule_note(anchor.id)
            return manager.coord(anchor)

        return commit_anchor

    def _channel(self, tmp_path: Path, connection=None, *, with_commit: bool = False):
        from .memento_channel import build_memento_channel

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
        assert result == "[memento] committed 1-1"
        branch = memento.get_branch("main")
        note = branch.notes()[branch.commits()[0].id]
        assert note.message == "D26 the fix holds\nverified after restart"
        # 自己写了 note = 有作者了, 旁路整条让开 (不是"排了但被跳过").
        assert manager.bypass == {}
        assert conn.calls == []

    @pytest.mark.asyncio
    async def test_empty_note_hands_the_summary_to_the_sidecar(self, tmp_path: Path):
        conn = self._Connection({"message": "auto note"})
        chan, manager, memento = self._channel(tmp_path, conn, with_commit=True)
        async with chan.bootstrap() as runtime:
            result = await runtime.execute_command("commit", args=("",))
        assert result == "[memento] committed 1-1"
        await manager.drain_bypass()
        branch = memento.get_branch("main")
        assert branch.notes()[branch.commits()[0].id].message == "auto note"

    @pytest.mark.asyncio
    async def test_commit_on_an_empty_span_returns_a_hint(self, tmp_path: Path):
        chan, _, _ = self._channel(tmp_path, with_commit=True)
        async with chan.bootstrap() as runtime:
            first = await runtime.execute_command("commit", args=("one",))
            second = await runtime.execute_command("commit", args=("two",))
        assert first == "[memento] committed 1-1"
        assert second == "[memento] nothing new since the last anchor — not committed"


class TestDoloresMemories:
    @pytest.mark.asyncio
    async def test_returns_single_memory_container(self, tmp_path: Path):
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
        # 渲染是异步的 (要跑 branch_view 槽位), 在 __aenter__ 里做一次, memories() 读缓存.
        ghost._memento_view = await manager.aview_message()

        memories = ghost.memories()

        assert len(memories) == 1
        text = memories[0].to_content_string()
        assert "<memory" in text
        assert "<ground" in text and "GROUND" in text
        assert "<branch" in text and "note" in text

    @pytest.mark.asyncio
    async def test_branch_ground_rides_into_boot_memory(self, tmp_path: Path):
        """D21 的现场: 计划写在 branch 目录里, 开机却只收到锚点摘要.

        判据 —— ``branch_view`` 是 branch view 生成的一部分, 所以它必须跟着 view 一起
        进 boot memory; 不是等谁想起来再去拼.
        """
        from ghoshell_moss.memento import new_local_memento

        from ._ego_memento import EgoMementoConfig, EgoMementoManager

        memento = new_local_memento(tmp_path / "owner")
        memento.create_branch("main")
        manager = EgoMementoManager(connection=None, memento=memento, config=EgoMementoConfig())
        anchor = manager.commit(session_id="s1", start_turn=0, end_turn=1)
        memento.get_branch("main").note(anchor.id, "note")

        branch = memento.get_branch("main")
        (branch.path / "plan.md").write_text("# the plan\n", encoding="utf-8")
        hooks = branch.path / "hooks"
        hooks.mkdir()
        (hooks / "branch_view.py").write_text(
            "import os, pathlib\n"
            "d = pathlib.Path(os.environ['MEMENTO_BRANCH_DIR'])\n"
            "print((d / 'plan.md').read_text())\n",
            encoding="utf-8",
        )

        ghost = _dolores()
        ghost._ground_text = "GROUND"
        ghost._memento_manager = manager
        ghost._memento_view = await manager.aview_message()

        text = ghost.memories()[0].to_content_string()
        assert "# the plan" in text, "branch 目录渲染的东西没进 boot memory"

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

        from .channel import build_dolores_channel

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

        from .channel import build_dolores_channel
        from .model_channel import ModelOrgan

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

        from .channel import build_dolores_channel

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
        from ghoshell_moss.deepseek_harness.types.sessions import ModelCatalog

        return ModelCatalog()

    def test_catalog_renders_models_with_their_own_depth_ladder(self):
        """目录文本按 provider 分组, 每个模型带自己的档位表 —— 换模型前先看见它能想多深."""
        from ghoshell_moss.deepseek_harness.types.sessions import (
            ModelCatalog,
            ModelCatalogModel,
            ModelProviderGroup,
            ModelReasoning,
            ModelReasoningEffort,
        )

        from .model_channel import _catalog_text

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

        from .channel import build_dolores_channel

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
        from ._run import DoloresRun

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
                TestDoloresRun._tool_call("moss_mystery", "{}", "call_unknown"), {},
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
                    TestDoloresRun._tool_call(name, "{}", f"call_{name}"), {},
                )
        assert logs.messages(logging.WARNING) == [], logs.messages()

    @pytest.mark.asyncio
    async def test_tool_arguments_parse_failure_is_logged(self):
        """参数解析失败: 模型看得到 error 字段, 但日志里原先一个字都没有."""
        from ._tools import InterpretToolCall

        event = TestDoloresRun._tool_call("moss_interpret", "{not json", "call_bad_args")
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
                TestDoloresRun._tool_call("moss_nope", "{}", "call_x"), {},
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
        from ._tools import ToolCallResult

        run = self._run()
        streams: dict = {}
        chunk = SimpleNamespace(type="tool-call-delta", id="call_broken", name="moss_interpret",
                                argumentsDelta='{"ctml": "\\q"}')
        await run._handle_ctml_delta(chunk, streams)
        assert streams["call_broken"].stream.failed
        call_event = TestDoloresRun._tool_call("moss_interpret", "{}", "call_broken")
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
        from ._run import DoloresRun

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
            TestDoloresRun._tool_call("moss_interpret", self._BROKEN_ARGS, "call_bad"), streams,
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
        from ._ego import DoloresEgo, DoloresEgoConfig, DoloresEgoContext

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
        from ghoshell_moss.deepseek_harness.types.session_events import SessionEvent, SessionEventMeta

        from ._ego import DoloresEgo, DoloresEgoConfig, DoloresEgoContext

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
        from ._ego import DoloresEgo, DoloresEgoConfig, DoloresEgoContext

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


class TestTurnEndClassification:
    """turn/end → 要不要显式退出 thinking 的判定表 (纯函数, 不依赖替身).

    这是 cancel 治理的判据面: 自然收线绝不动 attention, 中断类才 abort —— 唯一例外是 cause 属于我们自己
    的 hook cancel (moss_react / moss_wait_next / moss_reasoning 自己收的轮), 身体在返回结果前已 drain.
    2026-10-04 D04 的教训: 判据曾经挂在 turn 号上, 于是每一个 react 轮都被算成"自己人", 人点的
    aborted/user 被一起豁免, 41.3 秒语音全量播完. 端到端行为 (语音是否真停) 归
    features/regressions/dolores-ghost-prototype TC-003, 不在单测里.
    """

    @staticmethod
    def _end(kind: str, turn: int = 7, cause=None):
        from ghoshell_moss.deepseek_harness.types.session_events import TurnEnd, TurnEndReason
        return TurnEnd(turn=turn, reason=TurnEndReason(kind=kind, reason=cause))

    @staticmethod
    def _cause(kind: str, reason: str = ""):
        from ghoshell_moss.deepseek_harness.types.session_events import AgentCancelCause
        return AgentCancelCause(kind=kind, reason=reason or None)

    def test_natural_ends_do_not_touch_the_body(self):
        from ghoshell_moss.ghosts.dolores._run import _turn_end_abort_reason
        assert _turn_end_abort_reason(self._end("completed")) is None
        assert _turn_end_abort_reason(self._end("max-tokens")) is None

    def test_unknown_kind_is_treated_as_natural(self):
        from ghoshell_moss.ghosts.dolores._run import _turn_end_abort_reason
        assert _turn_end_abort_reason(self._end("something-new-from-dsh")) is None

    def test_interruption_kinds_abort_with_a_reason(self):
        from ghoshell_moss.ghosts.dolores._run import _turn_end_abort_reason
        assert _turn_end_abort_reason(self._end("aborted")) == "turn/end aborted"
        assert _turn_end_abort_reason(self._end("interrupted")) == "turn/end interrupted"
        assert _turn_end_abort_reason(self._end("blocked")) == "turn/end blocked"
        assert _turn_end_abort_reason(self._end("error")) == "turn/end error"

    def test_cause_rides_along_when_dsh_provides_one(self):
        from ghoshell_moss.ghosts.dolores._run import _turn_end_abort_reason
        assert _turn_end_abort_reason(
            self._end("aborted", cause=self._cause("user"))
        ) == "turn/end aborted (user)"
        assert _turn_end_abort_reason(
            self._end("blocked", cause=self._cause("user"))
        ) == "turn/end blocked (user)"

    def test_our_own_hook_cancel_is_not_an_interruption(self):
        """cause=hook: moss_react / moss_wait_next / moss_reasoning 自己收的轮 — 不得 abort."""
        from ghoshell_moss.ghosts.dolores._run import _turn_end_abort_reason
        assert _turn_end_abort_reason(
            self._end("aborted", cause=self._cause("hook", "moss cancel_turn"))
        ) is None
        assert _turn_end_abort_reason(self._end("blocked", cause=self._cause("hook"))) is None

    def test_error_wins_over_a_hook_cause(self):
        """故障优先: 即便这轮也被我们标过收轮, 运行时错误仍然要 abort (否则自续成环)."""
        from ghoshell_moss.ghosts.dolores._run import _turn_end_abort_reason
        assert _turn_end_abort_reason(
            self._end("error", cause=self._cause("hook"))
        ) == "turn/end error (hook)"
