import yaml
from abc import ABC, abstractmethod
from typing import TypeVar, Type, Optional
from pydantic import BaseModel
from ghoshell_common.helpers import generate_import_path
from ghoshell_common.helpers import yaml_pretty_dump
from os.path import join, abspath, exists

__all__ = [
    'ConfigType', 'ConfigStore',
    'YamlConfigStore',
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


CONF_TYPE = TypeVar('CONF_TYPE', bound=ConfigType)


class ConfigStore(ABC):
    """
    存储所有 Config 对象的仓库.
    """

    @abstractmethod
    def get(self, conf_type: Type[CONF_TYPE], relative_path: Optional[str] = None) -> CONF_TYPE:
        """
        从仓库中读取一个配置对象.
        :param conf_type: C 类型配置对象的类.
        :param relative_path: 默认不需要填. 如果读取路径不是 C 类型默认的, 才需要传入.
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
    def save(self, conf: ConfigType, relative_path: Optional[str] = None) -> None:
        """
        保存一个 Config 对象.
        :param conf: the conf object
        :param relative_path: if pass, override the conf_type default path.
        """
        pass


class BaseConfigStore(ConfigStore, ABC):
    """
    A Configs(repository) based on Storage, no matter what the Storage is.
    """

    def get(self, conf_type: Type[CONF_TYPE], real_name: Optional[str] = None) -> CONF_TYPE:
        relative_path = self._relative_path(real_name or conf_type.conf_name())
        content = self._get(relative_path)
        return conf_type.unmarshal(content)

    @staticmethod
    def _relative_path(config_name: str) -> str:
        return f"{config_name}.yml"

    @abstractmethod
    def _unmarshal(self, data: bytes) -> dict:
        pass

    def get_or_create(self, conf: CONF_TYPE) -> CONF_TYPE:
        path = conf.conf_name()
        if not self._exists(path):
            self._put(path, conf.marshal())
            return conf
        return self.get(type(conf))

    @abstractmethod
    def _get(self, relative_path: str) -> bytes:
        """
        get content from python
        :raise FileNotFoundError: if path does not exist.
        """
        pass

    @abstractmethod
    def _marshal(self, data: dict, conf_type: type[ConfigType]) -> bytes:
        pass

    @abstractmethod
    def _put(self, relative_path: str, content: bytes) -> None:
        """
        save content to path.
        """
        pass

    @abstractmethod
    def _exists(self, relative_path: str) -> bool:
        """
        check file exists
        """
        pass

    def save(self, conf: ConfigType, real_name: Optional[str] = None) -> None:
        data = conf.model_dump(exclude_none=True)
        marshaled = conf.marshal(data)
        relative_path = real_name or conf.conf_name()
        self._put(relative_path, marshaled)


class YamlConfigStore(BaseConfigStore):
    """
    A Configs(repository) based on Storage, no matter what the Storage is.
    """

    def __init__(self, configs_dir: str):
        self._configs_dir = abspath(configs_dir)

    def _unmarshal(self, data: bytes) -> dict:
        result = yaml.safe_load(data)
        if isinstance(result, dict):
            return result
        raise ValueError(f"load invalid configs data")

    def _marshal(self, data: dict, conf_type: type[ConfigType]) -> bytes:
        content = yaml_pretty_dump(data)
        import_path = generate_import_path(self._configs_dir)
        content = f"# dump from `{import_path}` \n" + content
        return content.encode('utf-8')

    def _get(self, relative_path: str) -> bytes:
        abs_path = abspath(join(self._configs_dir, relative_path))
        with open(abs_path, 'rb') as f:
            return f.read()

    def _put(self, relative_path: str, content: bytes) -> None:
        abs_path = abspath(join(self._configs_dir, relative_path))
        with open(abs_path, 'wb') as f:
            f.write(content)

    def _exists(self, relative_path: str) -> bool:
        abs_path = abspath(join(self._configs_dir, relative_path))
        return exists(abs_path)
