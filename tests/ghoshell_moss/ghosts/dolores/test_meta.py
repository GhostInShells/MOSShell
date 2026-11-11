"""``DoloresMeta`` / ``GhostMeta`` — the ghost's declared identity.

Answers: what must a ghost meta carry before the runtime may build anything from it.
"""

import json

import yaml

from ghoshell_container import Container
from ghoshell_moss.core.blueprint.ghost import Ghost, GhostMeta

from ._support import _dolores_meta

class TestDoloresMeta:
    def test_defaults(self):
        from ghoshell_moss.ghosts.dolores.nucleus import DoloresEgoNucleusMeta

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
        from ghoshell_moss.ghosts.dolores.nucleus import DoloresEgoNucleusMeta

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
        from ghoshell_moss.ghosts.dolores._runtime import Dolores

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

    def test_dsh_plugin_bundle_is_a_package(self):
        bundle = _dolores_meta().dsh_plugin_bundle()
        assert bundle.is_dir()
        manifest = json.loads((bundle / "package.json").read_text(encoding="utf-8"))
        assert manifest["dsh"]["bundle"]["patch"] == "./cordis.patch.yml"
        assert (bundle / "index.ts").is_file()
        # the patch must mount the package the manifest declares -- one fact, two files.
        patch = yaml.safe_load((bundle / "cordis.patch.yml").read_text(encoding="utf-8"))
        assert patch[0]["insert"][0]["name"] == manifest["name"]
