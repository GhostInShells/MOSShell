import yaml
from abc import ABC, abstractmethod
from typing import TypeVar, Type, Optional, Union
from typing_extensions import Self
from pydantic import BaseModel
from ghoshell_common.helpers import generate_import_path
from ghoshell_common.helpers import yaml_pretty_dump
from ghoshell_container import IoCContainer, Provider
from .workspace import Storage, Workspace

__all__ = [
    'ConfigType', 'ConfigStore',
    'YamlConfigStore',
    'LocalConfigStore',
    'WorkspaceConfigProvider',
]


class ConfigType(BaseModel, ABC):
    """
    从 workspace 中获取配置文件, 基于 Pydantic Model 建模.
    实际存储则考虑由 ConfigStore 决定.
    """

    @classmethod
    @abstractmethod
    def conf_name(cls) -> str:
        """
        当前 Config 存储时对于 configs 目录的相对路径.
        """
        pass

    def to_yaml(self) -> str:
        from ghoshell_common.helpers import yaml_pretty_dump
        data = self.model_dump(exclude_none=True)
        return yaml_pretty_dump(data)

    @classmethod
    def from_yaml(cls, data: str) -> Self:
        dict_data = yaml.safe_load(data)
        return cls.model_validate(dict_data)


CONF_TYPE = TypeVar('CONF_TYPE', bound=ConfigType)


def get_conf(container: IoCContainer, conf_type: type[CONF_TYPE]) -> CONF_TYPE:
    """
    快捷函数.
    """
    store = container.force_fetch(ConfigStore)
    return store.get(conf_type)


def get_or_create_conf(container: IoCContainer, conf: CONF_TYPE) -> CONF_TYPE:
    store = container.force_fetch(ConfigStore)
    return store.get_or_create(conf)


def save_conf(container: IoCContainer, conf: ConfigType) -> None:
    store = container.force_fetch(ConfigStore)
    store.save(conf)


class ConfigStore(ABC):
    """
    存储所有 Config 对象的仓库.
    """

    @abstractmethod
    def get(self, conf_type: Type[CONF_TYPE]) -> CONF_TYPE:
        """
        从仓库中读取一个配置对象.
        :param conf_type: C 类型配置对象的类.
        :return: C 类型的实例.
        :exception: FileNotFoundError
        """
        pass

    @abstractmethod
    def get_or_create(self, conf: CONF_TYPE) -> CONF_TYPE:
        """
        如果配置对象不存在, 则创建一个.
        """
        pass

    @abstractmethod
    def save(self, conf: ConfigType) -> None:
        """
        保存一个 Config 对象.
        :param conf: the conf object
        """
        pass


class LocalConfigStore(ConfigStore, ABC):
    """
    基于 Storage 的配置仓库实现，增加了简单的内存缓存。
    """

    def __init__(self, storage: Storage):
        self._storage = storage
        # 内存缓存：Key 是配置类本身，Value 是已实例化的配置对象
        self._cache: dict[Type[ConfigType], ConfigType] = {}

    def _full_path(self, conf_type_or_obj: Union[Type[ConfigType], ConfigType]) -> str:
        """统一路径处理：自动补全 .yml 后缀"""
        name = conf_type_or_obj.conf_name()
        return f"{name}.yml"

    def get(self, conf_type: Type[CONF_TYPE]) -> CONF_TYPE:
        # 1. 优先命中缓存
        if conf_type in self._cache:
            return self._cache[conf_type]

        # 2. 缓存未命中，从 Storage 读取
        path = self._full_path(conf_type)
        content = self._storage.get(path)
        data = self._unmarshal(content)

        # 3. 实例化并存入缓存
        instance = conf_type.model_validate(data)
        self._cache[conf_type] = instance
        return instance

    def get_or_create(self, conf: CONF_TYPE) -> CONF_TYPE:
        conf_type = type(conf)
        path = self._full_path(conf_type)

        if not self._storage.exists(path):
            # 不存在则保存当前传入的默认对象
            self.save(conf)
            return conf

        # 存在则执行标准 get (会处理缓存逻辑)
        return self.get(conf_type)

    def save(self, conf: ConfigType) -> None:
        """保存配置并同步更新缓存"""
        conf_type = type(conf)
        data = conf.model_dump(exclude_none=True)
        marshaled = self._marshal(data, conf_type)

        path = self._full_path(conf_type)
        self._storage.put(path, marshaled)

        # 同步更新内存，确保后续 get 拿到的是刚保存的这个实例
        self._cache[conf_type] = conf

    def invalidate(self, conf_type: Optional[Type[ConfigType]] = None) -> None:
        """
        手动清理缓存的入口。
        如果传入具体类型则清理该类型，不传则清空全部。
        """
        if conf_type:
            self._cache.pop(conf_type, None)
        else:
            self._cache.clear()

    @abstractmethod
    def _unmarshal(self, data: bytes) -> dict:
        pass

    @abstractmethod
    def _marshal(self, data: dict, conf_type: type[ConfigType]) -> bytes:
        pass


class YamlConfigStore(LocalConfigStore):
    """
    A Configs(repository) based on Storage, no matter what the Storage is.
    """

    def _unmarshal(self, data: bytes) -> dict:
        result = yaml.safe_load(data)
        if isinstance(result, dict):
            return result
        raise ValueError(f"load invalid configs data")

    def _marshal(self, data: dict, conf_type: type[ConfigType]) -> bytes:
        content = yaml_pretty_dump(data)
        import_path = generate_import_path(conf_type)
        content = f"# dump from `{import_path}` \n" + content
        return content.encode('utf-8')


class WorkspaceConfigProvider(Provider[ConfigStore]):

    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> ConfigStore:
        ws = con.force_fetch(Workspace)
        storage = ws.configs()
        return YamlConfigStore(storage)
