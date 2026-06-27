from typing import Iterable

from ghoshell_moss.contracts.resource import ResourceStorageMeta
from ghoshell_moss.core.blueprint.mindflow import NucleusMeta
from ghoshell_moss.core.blueprint.project import Manifest, ModeManifests, HOST_MODE_MANIFESTS_PACKAGE
from ghoshell_moss.core.blueprint.states_channel import PrimeChannel
from ghoshell_moss.project.manifests.impl import ScannedMatrixManifest
from ghoshell_moss.project.manifests.channels import search_main_channel
from ghoshell_moss.project.manifests.nuclei import search_nucleus_manifests
from ghoshell_moss.project.manifests.resources import search_resource_manifests

__all__ = ['ScannedModeManifests']


class ScannedModeManifests(ScannedMatrixManifest, ModeManifests):
    """基于 Python 包扫描的 ModeManifests 实现.

    继承 ScannedMatrixManifest 的所有 scanner，额外提供 channel, nuclei, resources.
    """

    def __init__(self, root_package: str = HOST_MODE_MANIFESTS_PACKAGE):
        super().__init__(root_package)

    def channel(self) -> Manifest[PrimeChannel]:
        return search_main_channel(f'{self._root}.channels')

    def nuclei(self) -> Iterable[Manifest[NucleusMeta]]:
        yield from search_nucleus_manifests(f'{self._root}.nuclei')

    def resources(self) -> Iterable[Manifest[ResourceStorageMeta]]:
        yield from search_resource_manifests(f'{self._root}.resources')
