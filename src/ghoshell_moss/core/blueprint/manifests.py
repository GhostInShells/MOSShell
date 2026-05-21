from typing import Any
from abc import ABC, abstractmethod
from typing_extensions import Self
from dataclasses import dataclass

from ghoshell_moss.core.blueprint.mindflow import NucleusMeta
from ghoshell_moss.contracts.configs import ConfigType, ConfigSchema, ConfigStore
from ghoshell_moss.contracts.resource import ResourceInfo, ResourceItem, ResourceStorageMeta
from ghoshell_moss.core.concepts.topic import TopicSchema, TopicModel, TopicName
from ghoshell_moss.core.concepts.channel import Channel, ChannelName
from ghoshell_moss.core.concepts.command import Command
from ghoshell_common.helpers import generate_import_path, import_from_path
from ghoshell_container import Provider
from pathlib import Path
from pydantic import Field
import inspect

__all__ = [
    'TopicInfo',
    'ConfigInfo',
    'ProviderInfo',
    'CtmlVersionInfo',
    'ResourceStorageInfo',
    'ResourceStorageItem',
    'NucleusMetaInfo',
    'Manifests',
]


@dataclass(frozen=True)
class TopicInfo:
    """
    Topic info.
    """
    found: str  # 发现 topic 的 module name, 如 MOSS.manifests.topics
    file: str  # 发现 topic 的 module filename
    model: str  # topic 如果是通过 TopicModel 定义的, 此处是它的 import path.
    schema: TopicSchema  # topic schema.

    @classmethod
    def from_topic_type(
            cls,
            found: str,
            file: str,
            model: type[TopicModel] | TopicSchema,
            topic_name: str | None = None,
    ) -> Self:
        if isinstance(model, type) and issubclass(model, TopicModel):
            model_path = generate_import_path(model)
            schema = model.topic_schema(topic_name or None)
        elif isinstance(model, TopicSchema):
            model_path = ''
            schema = model
        else:
            raise TypeError(f"'{type(model)}' is not a topic model")

        return TopicInfo(found=found, file=file, schema=schema, model=model_path)

    @property
    def model_source(self) -> str:
        """source of topic model"""
        if self.model:
            model_type = import_from_path(self.model)
            return inspect.getsource(model_type)
        return ''

    @property
    def description(self) -> str:
        """topic description"""
        return self.schema.description

    @property
    def name(self) -> str:
        """topic name"""
        return self.schema.topic_name

    @property
    def type(self) -> str:
        """topic type"""
        return self.schema.topic_type

    @property
    def json_schema(self) -> dict[str, Any]:
        """topic JSON Schema"""
        return self.schema.json_schema


@dataclass
class ConfigInfo:
    """
    Configuration model information
    """
    found_import_path: str  # 发现 config 的 module name, 如 MOSS.manifests.topics
    found_at_file: str  # 发现 config 的 module filename
    config: ConfigType  # config 是一个实例, 一定要有默认值. 真实的值会被 config store 以 yaml 保存到目录里. 不过那是运行时配置.

    @property
    def schema(self) -> ConfigSchema:
        return self.config.to_config_schema()

    @property
    def name(self) -> str:
        return self.config.conf_name()

    @property
    def source(self) -> str:
        return inspect.getsource(type(self.config))

    @property
    def model_path(self) -> str:
        return generate_import_path(type(self.config))

    def file(self, store: ConfigStore) -> str:
        return store.get_config_path(self.config.conf_name())

    @property
    def description(self) -> str:
        return self.config.to_config_schema().description

    def default_value(self) -> dict[str, Any]:
        return self.config.model_dump()

    def dump_yaml(self) -> str:
        return self.config.to_yaml()


# 管理从环境中发现能力的逻辑.
@dataclass(frozen=True)
class ProviderInfo:
    """
    contract info of the provider.
    """
    found: str
    'the python module import path where found the contract provider, pattern foo.bar:attr'

    file: str
    'the python file absolute path where found the contract provider'

    provider: Provider

    @property
    def name(self) -> str:
        """python import path of the contract"""
        return generate_import_path(self.provider.contract())

    @property
    def aliases(self) -> list[str]:
        result = []
        for alias in self.provider.aliases():
            result.append(generate_import_path(alias))
        return result

    @property
    def docstring(self) -> str:
        """docstring  of the contract"""
        return inspect.getdoc(self.provider.contract()) or ''

    @property
    def provider_type(self) -> str:
        return generate_import_path(type(self.provider))

    @property
    def description(self) -> str:
        return self.docstring.split('\n')[0]

    @property
    def singleton(self) -> bool:
        return self.provider.singleton()

    @property
    def source(self) -> str:
        contract = self.provider.contract()

        # 1. 基础判断：如果是内置 C 函数/方法
        if inspect.isbuiltin(contract):
            return "# [MOSS] Native Builtin (C-level)"

        try:
            # 2. 尝试获取模块和源码路径
            module = inspect.getmodule(contract)
            # 如果模块没有 __file__ 属性，说明是 C 扩展或内置模块（如 sys, zenoh 核心等）
            if not getattr(module, "__file__", None):
                return f"# [MOSS] Non-Python Source (Module: {module.__name__ if module else 'Unknown'})"

            # 3. 尝试获取源码
            return inspect.getsource(contract)
        except (TypeError, OSError, ImportError):
            # TypeError: 对象不是类、函数等
            # OSError: 找不到源码文件（比如 zenoh.Session 这种编译后的 .so/.pyd 文件）
            return f"# [MOSS] Source unavailable (Compiled or Dynamic: {type(contract).__name__})"


@dataclass(frozen=True)
class CtmlVersionInfo:
    file: Path

    @property
    def version(self) -> str:
        if self.file.name.endswith('.md'):
            return self.file.name[:-3]
        return self.file.name


_CtmlVersion = str


class ResourceStorageInfo(ResourceInfo):
    """Meta describing a discovered ResourceStorageMeta."""

    host: str = Field(description="Package where this storage was discovered")
    path: str = Field(description="Storage identity: {storage_scheme}:{storage_host}")
    description: str = Field(default="", description="Description from ResourceStorageMeta")
    found_module: str = Field(default="", description="Python module where discovered")
    found_file: str = Field(default="", description="File path where discovered")
    storage_scheme: str = Field(default="", description="The scheme this storage provides")
    storage_host: str = Field(default="", description="The host this storage serves")

    @classmethod
    def scheme(cls) -> str:
        return "resource-storage"

    @classmethod
    def scheme_description(cls) -> str:
        return "ResourceStorageMeta discovered in MOSS manifests packages"


class ResourceStorageItem(ResourceItem[ResourceStorageInfo, ResourceStorageMeta], ABC):
    """Wraps a ResourceStorageMeta. get() instantiates via factory()."""

    @abstractmethod
    def get_sync(self) -> ResourceStorageMeta:
        pass

    @property
    @abstractmethod
    def info(self) -> ResourceStorageInfo:
        pass


@dataclass(frozen=True)
class NucleusMetaInfo:
    """Meta info describing a discovered NucleusFactory."""

    nucleus_meta: NucleusMeta

    found_module: str
    """Python {module}:{attr} where discovered"""

    found_file: str
    """Absolute file path where discovered"""

    @property
    def name(self) -> str:
        """NucleusFactory.name()"""
        return self.nucleus_meta.name()

    @property
    def description(self) -> str:
        """NucleusFactory.description()"""
        return self.nucleus_meta.description()

    @property
    def signal_names(self) -> list[str]:
        """Signal names declared by factory.signals()"""
        return [signal_meta.signal_name() for signal_meta in self.nucleus_meta.signals()]


class Manifests:
    """
    MOSS 在环境中发现的各种资源的声明.
    需要根据具体的场景去实现.
    """

    def channels(self) -> dict[ChannelName, Channel]:
        """
        从环境中发现的运行时的一级 Channel. 会自动注册到 Shell main channel
        通过 ghoshell_moss.core.concepts.channel.Channel 实例发现.
        """
        return {}

    def primitives(self) -> dict[str, Command]:
        """
        从环境中发现的运行时原语. 会自动注册到 shell main channel
        通过 ghoshell_moss.core.concepts.command.Command 实例发现.
        """
        return {}

    def configs(self) -> dict[str, ConfigInfo]:
        """
        环境中发现的配置实例. Runtime 启动时, 如果发现配置不存在, 会初始化它.
        通过 ghoshell_moss.contracts.ConfigType 实例发现.
        """
        return {}

    def ctml_versions(self) -> dict[_CtmlVersion, CtmlVersionInfo]:
        return {}

    def topics(self) -> dict[TopicName, TopicInfo]:
        """
        环境中发现的 topic 协议. 未来会用来约束可通讯的节点.
        通过 ghoshell_moss.core.concepts.topic.TopicModel | TopicSchema 发现.
        """
        return {}

    def providers(self) -> list[ProviderInfo]:
        """
        环境中发现的 IoC 容器依赖, 会自动注册到 IoC 容器中.
        通过 ghoshell_container.Provider  实例发现.
        """
        return []

    def resource_storage_manifests(self) -> list[ResourceStorageItem]:
        return []

    def nuclei(self) -> dict[str, NucleusMetaInfo]:
        """
        环境中发现的 NucleusFactory 声明。用于开发/测试基础设施。
        通过 ghoshell_moss.core.blueprint.mindflow.NucleusFactory 实例发现。
        """
        return {}

    def explain(self) -> str:
        """
        用自然语言自描述 manifest 的结构与含义。
        面向智能模型——CLI 调用它作为唯一真相入口。
        """
        return """
# MOSS Manifests — 环境能力声明体系

manifests 是 MOSS 环境中所有能力的自解释声明。Matrix 启动时自动扫描、
发现并注入 IoC 容器。声明不是配置文件，是 Python 实例。

## 声明类型

| 类型 | 职责 | 工作空间路径 | 发现方式 |
|------|------|-------------|---------|
| **providers** | IoC 依赖注入：声明"这个接口由这个工厂生产" | `MOSS.manifests.providers` | `isinstance(obj, Provider)`，以 `contract()` 的 import path 为键 |
| **channels** | 一级 Channel：Shell 主通道下直接可调用的能力单元 | `MOSS.manifests.channels` | `isinstance(obj, Channel)`，以 `Channel.name()` 为键 |
| **primitives** | Shell 原语：sleep、noop、observe、interrupt 等基础命令 | `MOSS.manifests.primitives` | `isinstance(obj, Command)`，以 `Command.name()` 为键 |
| **configs** | 配置模型：声明配置的 schema 和默认值 | `MOSS.manifests.configs` | `isinstance(obj, ConfigType)`，以 `ConfigType.conf_name()` 为键 |
| **topics** | 事件协议：约束可通讯的 topic 类型 | `MOSS.manifests.topics` | `isinstance(obj, TopicModel)` 或 `isinstance(obj, TopicSchema)`，以 topic_name 为键 |
| **resources** | 资源存储：声明可寻址的资源数据集 | `MOSS.manifests.resources` | `isinstance(obj, ResourceStorageMeta)`，以 `scheme://host/path` 为键 |
| **nuclei** | 感知核：Mindflow 输入信号源的声明 | `MOSS.manifests.nuclei` | `isinstance(obj, NucleusMeta)`，以 `NucleusMeta.name()` 为键 |
| **CTML versions** | CTML 提示词版本：环境可覆盖默认版本 | `ctml_versions/` 目录 | 扫描 `.md` 文件，以文件名为版本号 |

所有类型共享同一发现模式：`scan_package(约定路径)` → `isinstance` 过滤 → 按类型特定键聚合。

## 深入路径

- IoC 容器：`moss how-tos read get-moss-design/how-ioc-container-work-in-moss.md`
- Matrix 能力发现：`moss how-tos read get-moss-design/how-matrix-discovers-capabilities.md`
- 术语表：`moss how-tos read get-moss-design/glossary.md`
- 架构拓扑：`moss docs read architecture-topology.md`
""".strip()
