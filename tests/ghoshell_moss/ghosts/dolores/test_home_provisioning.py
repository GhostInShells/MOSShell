"""What the ghost seeds into its home on boot.

Answers: given an empty (or half-written) ghost home, what does the runtime put there —
and what does it deliberately refuse to touch.

- ``TestProfilePatchMigration`` — the profile layer is the ghost's own; migration only
  touches the trace we planted, never the human's edits around it.
- ``TestStubsSync`` — the three sync paths (init / override / noop) over ``home/stubs``.
- ``TestDoloresDefaultModel`` / ``TestDoloresProviders`` — ``seed-once``: a missing section is
  written from the meta, an existing one is left alone.
"""

import asyncio
from pathlib import Path

import pytest
import yaml

from ghoshell_moss.core.session.mock_session import MockSession

from ._support import _dolores, _dolores_meta, _load_patch

class TestProfilePatchMigration:
    """The profile layer is the ghost's own; migration touches only the trace we planted."""

    LEGACY = (
        "# mount the ghost plugin\n"
        "- insert:\n"
        "    - id: dolores-ghost-plugin\n"
        "      name: './moss-dolores-ghost-plugin.ts'\n"
        "\n"
        "- id: ui-deliverables\n"
        "  disabled: true\n"
    )

    def test_the_legacy_entry_is_removed_and_the_rest_survives(self):
        from ghoshell_moss.ghosts.dolores._runtime import _strip_legacy_insert

        stripped = _strip_legacy_insert(self.LEGACY)
        assert stripped is not None
        assert "dolores-ghost-plugin" not in stripped
        assert "moss-dolores-ghost-plugin.ts" not in stripped
        assert "ui-deliverables" in stripped

    def test_an_item_the_ghost_rewrote_is_left_alone(self):
        from ghoshell_moss.ghosts.dolores._runtime import _strip_legacy_insert

        text = "- id: dolores-ghost-plugin\n  disabled: true\n"
        assert _strip_legacy_insert(text) is None

    def test_nothing_to_remove_returns_none(self):
        from ghoshell_moss.ghosts.dolores._runtime import _strip_legacy_insert

        assert _strip_legacy_insert("[]\n") is None

    def test_suppressions_replace_the_empty_template(self):
        from ghoshell_moss.ghosts.dolores._runtime import _seed_suppressions

        seeded = _seed_suppressions("# a fresh layer\n[]\n")
        assert yaml.safe_load(seeded) == [
            {"id": "ui-deliverables", "disabled": True},
            {"id": "file-reference-local", "disabled": True},
        ]

    def test_seeding_is_idempotent(self):
        from ghoshell_moss.ghosts.dolores._runtime import _seed_suppressions

        once = _seed_suppressions("")
        assert _seed_suppressions(once) == once


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
        # dirs materialized; the dsh home itself is not seeded without a matrix (no dsh to run).
        assert (tmp_path / ".dsh").is_dir()
        assert (tmp_path / "skills").is_dir()
        assert not (tmp_path / ".dsh" / "profiles").exists()
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
        from ghoshell_moss.ghosts.dolores._meta import DoloresMeta
        from ghoshell_moss.ghosts.dolores._startup import StartupDoc

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
    """``meta.providers`` -> the profile layer's ``id: llm-pi-ai`` entry, seed-once.

    The declaration names environment facts (``$ANTHROPIC_API_KEY``); the seed drops the
    marker. A credential becomes a bare reference name, an endpoint becomes a ``!!js``
    expression -- the home never holds either value.
    """

    @staticmethod
    def _patch_path(home: Path) -> Path:
        return home / ".dsh" / "profiles" / "web" / "cordis.patch.yml"

    @staticmethod
    def _boot(home: Path, meta) -> None:
        """Seed the provider block into the profile layer.

        The profile itself belongs to dsh (it provisions it at boot), so the unit surface is
        the seed: stand the profile dir up by hand and call it directly.
        """
        ghost = _dolores(meta=meta, home=home, session=MockSession())
        (home / ".dsh" / "profiles" / "web").mkdir(parents=True, exist_ok=True)
        ghost._sync_providers(home / ".dsh", "web")

    def test_the_provider_block_lands_under_config(self, tmp_path: Path):
        from ghoshell_moss.ghosts.dolores._meta import DoloresProviderSpec

        meta = _dolores_meta(providers=[
            DoloresProviderSpec(
                route="anthropic",
                api_key_env="$ANTHROPIC_API_KEY",
                base_url_env="$ANTHROPIC_BASE_URL",
            ),
        ])
        self._boot(tmp_path, meta=meta)

        entries = _load_patch(self._patch_path(tmp_path).read_text(encoding="utf-8"))
        pi_ai = next(e for e in entries if e.get("id") == "llm-pi-ai")
        # the loader hands a plugin entry.config and ignores anything else on the entry
        assert "providers" not in pi_ai
        provider = pi_ai["config"]["providers"]["anthropic"]
        assert provider["apiKeyEnv"] == "ANTHROPIC_API_KEY"

    def test_base_url_is_an_environment_expression(self, tmp_path: Path):
        from ghoshell_moss.ghosts.dolores._meta import DoloresProviderSpec

        meta = _dolores_meta(providers=[
            DoloresProviderSpec(route="anthropic", api_key_env="$ANTHROPIC_API_KEY",
                                base_url_env="$ANTHROPIC_BASE_URL"),
        ])
        self._boot(tmp_path, meta=meta)
        text = self._patch_path(tmp_path).read_text(encoding="utf-8")
        # the home carries the name, never this machine's endpoint
        assert "!!js" in text and "process.env.ANTHROPIC_BASE_URL" in text
        assert "proxy.example.com" not in text

    def test_seed_once_does_not_duplicate(self, tmp_path: Path, monkeypatch):
        from ghoshell_moss.ghosts.dolores._meta import DoloresProviderSpec

        meta = _dolores_meta(providers=[
            DoloresProviderSpec(route="anthropic", api_key_env="$ANTHROPIC_API_KEY",
                                base_url_env="$ANTHROPIC_BASE_URL"),
        ])
        self._boot(tmp_path, meta=meta)
        first = self._patch_path(tmp_path).read_text(encoding="utf-8")
        # Second boot with a different meta — must not re-write.
        self._boot(tmp_path, meta=meta)
        assert self._patch_path(tmp_path).read_text(encoding="utf-8") == first
        assert first.count("id: llm-pi-ai") == 1

    def test_a_declaration_without_the_marker_is_refused(self):
        from ghoshell_moss.ghosts.dolores._meta import DoloresProviderSpec
        from ghoshell_moss.ghosts.dolores._runtime import _env_name

        with pytest.raises(ValueError):
            _env_name("ANTHROPIC_BASE_URL")

    def test_no_providers_meta_writes_nothing(self, tmp_path: Path):
        self._boot(tmp_path, meta=_dolores_meta())
        assert not self._patch_path(tmp_path).exists()
