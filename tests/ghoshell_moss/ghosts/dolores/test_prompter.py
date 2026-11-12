"""Prompt registry contract tests — manifest, key/path bijection, versions, slots."""

from pathlib import Path

import pytest

from ghoshell_moss.contracts.workspace import LocalStorage
from ghoshell_moss.ghosts.dolores.configs import DoloresConfigStore
from ghoshell_moss.ghosts.dolores.mindset.prompter import (
    PromptError,
    PromptManager,
    PromptSpec,
    PromptVersionNotFoundError,
    SlotMismatchError,
    UnknownPromptError,
)


def _manager(tmp_path: Path, defaults: list[PromptSpec] | None = None) -> tuple[PromptManager, DoloresConfigStore]:
    store = DoloresConfigStore(
        LocalStorage(tmp_path / "configs"), mode_name="", materialize=True
    )
    manager = PromptManager(
        store,
        tmp_path / "prompts",
        defaults=defaults or [
            PromptSpec(key="memento.note", slots={"coord": "predecessor coordinate", "prior": "prior notes"}),
            PromptSpec(key="memento.chat"),
        ],
    )
    return manager, store


def test_manifest_is_written_and_key_path_is_bijective(tmp_path: Path):
    manager, _ = _manager(tmp_path)

    assert set(manager.spec("memento.note").slots) == {"coord", "prior"}
    # dotted key -> hyphenated directory; version is the file stem
    assert manager.path_of("memento.note", "v1") == (
        tmp_path / "prompts" / "memento-note" / "v1.md"
    )
    assert (tmp_path / "configs" / "prompts.yml").is_file()


def test_create_version_and_render(tmp_path: Path):
    manager, _ = _manager(tmp_path)

    prompt = manager.create_version("memento.note", "coord={{coord}}\nprior={{prior}}", "v1")

    assert prompt.version == "v1"
    assert manager.versions("memento.note") == ["v1"]
    assert manager.get("memento.note").content.startswith("coord=")
    assert manager.render("memento.note", coord="1-2", prior="...") == "coord=1-2\nprior=..."


def test_activate_and_reset(tmp_path: Path):
    manager, _ = _manager(tmp_path)
    manager.create_version("memento.note", "v1 {{coord}} {{prior}}", "v1")
    manager.create_version("memento.note", "v2 {{coord}} {{prior}}", "v2")

    assert manager.get("memento.note").version == "v1"  # the spec's default
    manager.activate("memento.note", "v2")
    assert manager.get("memento.note").version == "v2"
    assert manager.reset("memento.note") is True
    assert manager.get("memento.note").version == "v1"
    assert manager.reset("memento.note") is False


def test_versions_are_immutable(tmp_path: Path):
    manager, _ = _manager(tmp_path)
    manager.create_version("memento.note", "x", "v1")

    with pytest.raises(PromptError):
        manager.create_version("memento.note", "y", "v1")


def test_unknown_key_is_an_error(tmp_path: Path):
    manager, _ = _manager(tmp_path)

    with pytest.raises(UnknownPromptError):
        manager.get("memento.missing")
    with pytest.raises(UnknownPromptError):
        manager.create_version("memento.missing", "x", "v1")


def test_missing_version_and_slot_mismatch(tmp_path: Path):
    manager, _ = _manager(tmp_path)
    with pytest.raises(PromptVersionNotFoundError):
        manager.get("memento.note")

    manager.create_version("memento.note", "{{coord}}", "v1")
    with pytest.raises(SlotMismatchError):
        manager.render("memento.note", coord="1-2")  # `prior` is declared but missing

    manager.create_version("memento.note", "{{coord}} {{unknown}}", "v2")
    manager.activate("memento.note", "v2")
    with pytest.raises(SlotMismatchError):
        manager.render("memento.note", coord="1-2", prior="p")


@pytest.mark.parametrize(("key", "version"), [
    ("bad-key", "v1"),
    ("bad/key", "v1"),
    ("memento.note", "../v1"),
    ("memento.note", "a/b"),
])
def test_invalid_key_or_version_is_rejected(tmp_path: Path, key: str, version: str):
    manager, _ = _manager(tmp_path)

    with pytest.raises(ValueError):
        manager.path_of(key, version)


def test_register_persists_to_the_config(tmp_path: Path):
    manager, _ = _manager(tmp_path)
    manager.register(PromptSpec(key="domain.new"))
    with pytest.raises(PromptError):
        manager.register(PromptSpec(key="domain.new"))

    # a fresh store over the same directory reads the manifest from disk
    again = DoloresConfigStore(
        LocalStorage(tmp_path / "configs"), mode_name="", materialize=True
    )
    reloaded = PromptManager(again, tmp_path / "prompts")
    assert reloaded.spec("domain.new").key == "domain.new"


def test_seed_rewrites_the_default_version_in_place(tmp_path: Path):
    """The seed body is code-owned: a changed seed rewrites the same version file."""
    store = DoloresConfigStore(LocalStorage(tmp_path / "configs"), mode_name="", materialize=True)
    first = PromptManager(store, tmp_path / "prompts", defaults=[PromptSpec(key="memento.note", seed="first")])
    assert first.get("memento.note").content == "first"

    second = PromptManager(store, tmp_path / "prompts", defaults=[PromptSpec(key="memento.note", seed="second")])
    assert second.get("memento.note").version == "v1"
    assert second.get("memento.note").content == "second"


def test_local_version_is_isolated_from_the_seed(tmp_path: Path):
    """Version isolation is a local version plus an active pointer — the seed never touches it."""
    store = DoloresConfigStore(LocalStorage(tmp_path / "configs"), mode_name="", materialize=True)
    manager = PromptManager(store, tmp_path / "prompts", defaults=[PromptSpec(key="memento.note", seed="first")])
    manager.create_version("memento.note", "local", "v2")
    manager.activate("memento.note", "v2")

    again = PromptManager(store, tmp_path / "prompts", defaults=[PromptSpec(key="memento.note", seed="third")])
    assert again.get("memento.note").version == "v2"
    assert (tmp_path / "prompts" / "memento-note" / "v1.md").read_text(encoding="utf-8") == "third"
    assert (tmp_path / "prompts" / "memento-note" / "v2.md").read_text(encoding="utf-8") == "local"


def test_provider_resolves_from_the_registered_ghost_home(tmp_path: Path):
    """The store/manager roots come from the registered ghost home, not the environment."""
    from ghoshell_container import Container

    from ghoshell_moss.core.blueprint.ghost import GhostHome
    from ghoshell_moss.ghosts.dolores.configs import DoloresConfigStoreProvider
    from ghoshell_moss.ghosts.dolores.mindset.prompter import (
        DefaultPrompterKeys,
        PromptManagerProvider,
    )

    class _GhostHomeStub:
        def home(self) -> Path:
            return tmp_path

        def home_meta_dir(self) -> Path:
            return tmp_path / ".dolores"

    con = Container(name="prompt-manager-provider-test")
    con.register(DoloresConfigStoreProvider(), PromptManagerProvider())
    con.set(GhostHome, _GhostHomeStub())

    manager = con.force_fetch(PromptManager)

    assert manager.spec("memento.note").key == "memento.note"
    assert "Record the commit" in manager.get(DefaultPrompterKeys.MEMENTO_NOTE).content
    assert (tmp_path / ".dolores" / "configs" / "prompts.yml").is_file()
    assert manager.root == tmp_path / ".dolores" / "prompts"
