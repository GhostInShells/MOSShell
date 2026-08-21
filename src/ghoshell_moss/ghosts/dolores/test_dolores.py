"""Dolores Ghost 原型测试 — 骨架 + stubs 同步阶段.

覆盖:
- GhostMeta/Ghost ABC 契约
- stubs 同步三种路径: init / override / noop
- 构造无副作用 (写盘收敛到 __aenter__)
- session.output 提示在 init/override 时发出, noop 不发声
"""

import asyncio
from pathlib import Path

import pytest
import yaml
from ghoshell_container import Container

from ghoshell_moss.core.blueprint.ghost import Ghost, GhostMeta
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
        meta = _dolores_meta()
        assert meta.name() == "dolores"
        assert meta.prototype() == "Dolores"
        assert meta.nuclei_metas() == []

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
        assert (tmp_path / ".dsh/profiles/web/plugin.ts").exists()
        assert len(session.outputs) == 1
        assert session.outputs[0].role == "system"

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
        """无 trajectory (shell=None) → articulate 只产出占位 logos, 不 crash."""
        ghost = _dolores()

        async def collect():
            return [delta async for delta in ghost.articulate(None)]

        assert asyncio.run(collect()) == [""]


class TestDoloresTrajectory:
    def test_no_trajectory_without_shell(self):
        """无 shell → 不挂载 trajectory, articulate 静默占位."""
        ghost = _dolores()

        async def run():
            async with ghost:
                assert ghost._trajectory is None
                with pytest.raises(RuntimeError):
                    ghost.trajectory
                collected = [delta async for delta in ghost.articulate(None)]
                assert collected == [""]

        asyncio.run(run())

    @pytest.mark.asyncio
    async def test_mounts_trajectory_when_shell_running(self):
        """shell running → __aenter__ 挂载 trajectory, 句柄可访问且 is_running."""
        from ghoshell_moss.core.ctml.shell import new_ctml_shell

        shell = new_ctml_shell("dolores_traj_mount")
        ghost = _dolores(shell=shell)
        async with shell:
            async with ghost:
                assert ghost._trajectory is not None
                assert ghost.trajectory.is_running()

    @pytest.mark.asyncio
    async def test_skips_mount_when_shell_not_running(self):
        """shell 未启动 → 跳过挂载, trajectory 保持 None."""
        from ghoshell_moss.core.ctml.shell import new_ctml_shell

        shell = new_ctml_shell("dolores_traj_skip")
        ghost = _dolores(shell=shell)
        async with ghost:
            assert ghost._trajectory is None
            with pytest.raises(RuntimeError):
                ghost.trajectory

    @pytest.mark.asyncio
    async def test_articulate_emits_epoch_and_frame_to_output(self):
        """articulate 首帧写 epoch_start 全量 facade, 每轮写 trajectory frame → output."""
        from ghoshell_moss.core.ctml.shell import new_ctml_shell

        shell = new_ctml_shell("dolores_traj_output")
        session = MockSession()
        ghost = _dolores(shell=shell, session=session)
        async with shell:
            async with ghost:
                collected = [delta async for delta in ghost.articulate(None)]
        assert collected == [""]
        # output 面出现 trajectory 角色: epoch start (全量 facade) + frame (帧投影).
        traj_outputs = [o for o in session.outputs if o.role == "trajectory"]
        assert len(traj_outputs) >= 2
        assert "epoch start" in traj_outputs[0].log
        epoch_text = traj_outputs[0].messages_string()
        assert "<channel" in epoch_text
        frame_text = traj_outputs[1].messages_string()
        assert "<moss" in frame_text


class TestDoloresInstruction:
    def test_system_prompt_derives_two_meta_segments(self):
        """system_prompt = 原型元信息 + 身份描述, 从结构化 meta 派生, 无 baseline 时不含 baseline."""
        text = _dolores().system_prompt()
        assert "prototype: Dolores" in text
        assert "version: dev_1" in text
        assert "name: dolores" in text
        assert "description:" in text

    def test_system_prompt_prepends_base_instruction(self):
        """baseline (MossSystemPrompter.base_instruction) 在两段之前."""
        text = _dolores(base_instruction="BASELINE").system_prompt()
        assert text.startswith("BASELINE")
        assert "prototype: Dolores" in text
        assert "name: dolores" in text

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
