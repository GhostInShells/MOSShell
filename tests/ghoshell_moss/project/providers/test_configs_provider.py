"""EnvConfigStoreProvider — 从 Environment 装配 ConfigStore 时的层叠形态.

契约: env 有 ghost 身份时, store 叠一层 <ghost_home>/configs; 该层只做覆盖 ——
读没命中的递归到 workspace 层, 创建下沉到 workspace, 目录缺失不凭空造.
"""

import os
import subprocess
import sys
import textwrap

import pytest

from ghoshell_moss.contracts.configs import ConfigType, YamlConfigStore
from ghoshell_moss.contracts.workspace import LocalStorage
from ghoshell_moss.core.blueprint.environment import Environment
from ghoshell_moss.project.providers.configs_provider import EnvConfigStoreProvider


class TtsConfig(ConfigType):
    voice: str = "default"

    @classmethod
    def conf_name(cls) -> str:
        return "tts"


@pytest.fixture(autouse=True)
def _reset_env_singleton():
    """seal 会写进程级单例与 os.environ, 用完还原."""
    with Environment.fixture():
        yield


def _sealed_env(tmp_path, *, ghost: str) -> Environment:
    ws = tmp_path / "moss"
    ws.mkdir()
    project = tmp_path / "proj"
    project.mkdir()
    env = Environment(workspace=ws, project=project, mode="desktop", ghost=ghost)
    env.seal()
    return env


def _build(tmp_path, *, ghost: str) -> tuple[Environment, object]:
    env = _sealed_env(tmp_path, ghost=ghost)
    return env, EnvConfigStoreProvider().factory(None)


def test_ghost_layer_shadows_workspace(tmp_path):
    env, store = _build(tmp_path, ghost="dolores")
    YamlConfigStore(env.workspace.configs(), mode_name="desktop").save(
        TtsConfig(voice="workspace"),
    )
    assert store.get(TtsConfig).voice == "workspace"

    LocalStorage(env.ghost_home / "configs", create=False).put(
        "tts.yml", b"voice: dolores\n",
    )
    assert store.get(TtsConfig).voice == "dolores"


def test_ghost_layer_is_mode_aware(tmp_path):
    """ghost 层和 workspace 层跑同一套 mode → 通用解析."""
    env, store = _build(tmp_path, ghost="dolores")
    ghost_configs = LocalStorage(env.ghost_home / "configs", create=False)
    ghost_configs.put("tts.yml", b"voice: ghost-base\n")
    ghost_configs.put("tts.desktop.yml", b"voice: ghost-desktop\n")
    assert store.get(TtsConfig).voice == "ghost-desktop"


def test_create_seeds_workspace_not_ghost_home(tmp_path):
    env, store = _build(tmp_path, ghost="dolores")
    assert store.get_or_create(TtsConfig(voice="seeded")).voice == "seeded"
    assert env.workspace.configs().exists("tts.desktop.yml")
    assert not (env.ghost_home / "configs").exists()


def test_save_lands_in_ghost_layer(tmp_path):
    env, store = _build(tmp_path, ghost="dolores")
    store.save(TtsConfig(voice="mine"))
    assert (env.ghost_home / "configs" / "tts.desktop.yml").exists()
    assert not env.workspace.configs().exists("tts.desktop.yml")


def test_no_ghost_identity_is_single_layer(tmp_path):
    """无 ghost 身份 (MCP / moss-shell 这类第三方形态) 不该叠层."""
    env, store = _build(tmp_path, ghost="none")
    YamlConfigStore(env.workspace.configs(), mode_name="desktop").save(
        TtsConfig(voice="workspace"),
    )
    assert store.get(TtsConfig).voice == "workspace"
    assert not (env.ghost_home / "configs").exists()


def test_descendant_process_resolves_ghost_layer(tmp_path):
    """只带环境变量的后代进程也解析到 ghost 层 — node/cell 就是这个场景.

    断言的是行为 (这个进程属于 ghost dolores, 就该解析到它的覆盖), 不是机制
    (env 必须逐字继承): 将来谁为了让 env 更干净加一层过滤, 只要行为保住这条
    仍是绿的.
    """
    ws = tmp_path / "moss"
    (ws / "configs").mkdir(parents=True)
    (ws / "ghosts" / "dolores" / "configs").mkdir(parents=True)
    (tmp_path / "proj").mkdir()
    (ws / "configs" / "tts.desktop.yml").write_text("voice: workspace\n")
    (ws / "ghosts" / "dolores" / "configs" / "tts.yml").write_text("voice: ghost\n")

    child_env = {k: v for k, v in os.environ.items() if not k.startswith("MOSS_")}
    child_env.update({
        "MOSS_WORKSPACE": str(ws),
        "MOSS_PROJECT_DIR": str(tmp_path / "proj"),
        "MOSS_GHOST_NAME": "dolores",
        "MOSS_MODE_NAME": "desktop",
    })
    code = textwrap.dedent("""
        from ghoshell_moss.contracts.configs import ConfigType
        from ghoshell_moss.project.providers.configs_provider import EnvConfigStoreProvider

        class Tts(ConfigType):
            voice: str = 'x'

            @classmethod
            def conf_name(cls): return 'tts'

        print(EnvConfigStoreProvider().factory(None).get(Tts).voice)
    """)
    out = subprocess.run(
        [sys.executable, "-c", code], env=child_env,
        capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip() == "ghost"
