from abc import ABC, abstractmethod

from typing import Any, TYPE_CHECKING, Iterable, Union
from typing_extensions import Self
from dataclasses import dataclass

from ghoshell_moss.contracts.configs import ConfigType, ConfigSchema
from ghoshell_moss.core.concepts.topic import TopicSchema, TopicModel, TopicName
from ghoshell_moss.core.concepts.channel import Channel, ChannelName
from ghoshell_moss.core.concepts.command import Command, CommandFunc
from ghoshell_common.helpers import generate_import_path, import_from_path
from ghoshell_container import Provider
from .app import AppInfo
import inspect

__all__ = [
    'AppInfo',
    'TopicInfo',
    'ConfigInfo',
    'ContractInfo',
    'ConfigInfo',
    'Manifests',
]


@dataclass
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
    found: str  # 发现 config 的 module name, 如 MOSS.manifests.topics
    file: str  # 发现 config 的 module filename
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

    @property
    def description(self) -> str:
        return self.config.to_config_schema().description

    def default_value(self) -> dict[str, Any]:
        return self.config.model_dump()

    def dump_yaml(self) -> str:
        return self.config.to_yaml()


# 管理从环境中发现能力的逻辑.
@dataclass(frozen=True)
class ContractInfo:
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
        return inspect.getdoc(self.provider.contract())

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
        return inspect.getsource(self.provider.contract())


@dataclass
class ConfigInfo:
    """
    Configuration model information
    """
    found: str  # 发现 config 的 module name, 如 MOSS.manifests.topics
    file: str  # 发现 config 的 module filename
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

    @property
    def description(self) -> str:
        return self.config.to_config_schema().description

    def default_value(self) -> dict[str, Any]:
        return self.config.model_dump()

    def dump_yaml(self) -> str:
        return self.config.to_yaml()


class Manifests(ABC):
    """
    MOSS 在环境中发现的各种资源的声明.
    """

    @abstractmethod
    def apps(self) -> list[AppInfo]:
        pass

    @abstractmethod
    def channels(self) -> dict[ChannelName, Channel]:
        """
        声明运行时的一级 Channel.
        """
        pass

    @abstractmethod
    def primitives(self) -> list[Union[Command, CommandFunc]]:
        """
        运行时的原语.
        """
        pass

    @abstractmethod
    def configs(self) -> dict[str, ConfigInfo]:
        pass

    @abstractmethod
    def topics(self) -> dict[TopicName, TopicInfo]:
        pass

    @abstractmethod
    def contracts(self) -> list[ContractInfo]:
        pass
