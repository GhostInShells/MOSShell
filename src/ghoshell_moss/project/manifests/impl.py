from typing import Iterable

from ghoshell_container import Provider
from ghoshell_moss.contracts.configs import ConfigType
from ghoshell_moss.contracts.resource import ResourceStorageMeta
from ghoshell_moss.core.blueprint.mindflow import SignalSchema
from ghoshell_moss.core.blueprint.parameter import ParameterSchema
from ghoshell_moss.core.blueprint.project import Manifest, MatrixManifest
from ghoshell_moss.core.blueprint.environment import MATRIX_MANIFESTS_PACKAGE
from ghoshell_moss.core.concepts.topic import TopicSchema
from ghoshell_moss.project.manifests.providers import search_provider_manifests
from ghoshell_moss.project.manifests.configs import search_config_manifests
from ghoshell_moss.project.manifests.signals import search_signal_manifests
from ghoshell_moss.project.manifests.topics import search_topic_manifests
from ghoshell_moss.project.manifests.parameters import search_parameter_manifests
from ghoshell_moss.project.manifests.resources import search_resource_manifests

__all__ = ['ScannedMatrixManifest']


class ScannedMatrixManifest(MatrixManifest):
    """基于 Python 包扫描的 MatrixManifest 实现.

    每个方法委托给对应的 scanner, 从约定的子包路径发现.
    """

    def __init__(self, root_package: str = MATRIX_MANIFESTS_PACKAGE):
        self._root = root_package

    def root_package(self) -> str:
        return self._root

    def providers(self) -> Iterable[Manifest[Provider]]:
        yield from search_provider_manifests(f'{self._root}.providers')

    def configs(self) -> Iterable[Manifest[ConfigType]]:
        yield from search_config_manifests(f'{self._root}.configs')

    def topics(self) -> Iterable[Manifest[TopicSchema]]:
        yield from search_topic_manifests(f'{self._root}.topics')

    def signals(self) -> Iterable[Manifest[SignalSchema]]:
        yield from search_signal_manifests(f'{self._root}.signals')

    def parameters(self) -> Manifest[ParameterSchema]:
        return search_parameter_manifests(f'{self._root}.parameters')

    def resources(self) -> Iterable[Manifest[ResourceStorageMeta]]:
        yield from search_resource_manifests(f'{self._root}.resources')
