import yaml
from abc import ABC, abstractmethod
from typing import TypeVar, Type, Optional, Union, Any, ClassVar, Callable
from typing_extensions import Self
from pydantic import BaseModel, Field
from ghoshell_common.helpers import generate_import_path
from ghoshell_common.helpers import yaml_pretty_dump
from ghoshell_container import IoCContainer, Provider
from .workspace import Storage, Workspace
import os
import pathlib

__all__ = [
    'ConfigType', 'ConfigStore', 'ConfigSchema',
    'YamlConfigStore',
    'LocalConfigStore',
    'WorkspaceYamlConfigStoreProvider',
    'CONF_TYPE',
]


class ConfigSchema(BaseModel):
    name: str = Field(
        description="config name, determine config key in ConfigStore.",
    )
    description: str = Field(
        default='',
        description="config description.",
    )
    json_schema: dict[str, Any] = Field(
        description="config json schema.",
    )


class ConfigType(BaseModel, ABC):
    """
    从 workspace 中获取配置文件, 基于 Pydantic Model 建模.
    实际存储则考虑由 ConfigStore 决定.
    """
    RESOLVE_ENV_KEY: ClassVar[bool] = True

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

    def resolve(self, environ: dict[str, str] | None = None) -> Self:
        if not self.RESOLVE_ENV_KEY:
            return self
        data = self.model_dump()
        data = _resolve_config_data_from_env(data, environ=environ)
        return self.model_validate(data, strict=False)

    @classmethod
    def from_yaml(cls, data: str) -> Self:
        dict_data = yaml.safe_load(data)
        return cls.model_validate(dict_data)

    @classmethod
    def to_config_schema(cls) -> ConfigSchema:
        return ConfigSchema(
            name=cls.conf_name(),
            description=cls.__doc__ or '',
            json_schema=cls.model_json_schema(),
        )


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
    def set_config(self, conf: ConfigType, override: bool = False) -> None:
        """
        设置一个 config 实例, 可以选择是否覆盖原始文件.
        """
        pass

    @abstractmethod
    def get_config_path(self, config_name: str) -> str:
        """
        返回一个预期的配置地址.
        """
        pass

    @abstractmethod
    def save(self, conf: ConfigType) -> None:
        """
        保存一个 Config 对象.
        :param conf: the conf object
        """
        pass

    @abstractmethod
    def invalidate(self, conf_type_or_name: Optional[Type[ConfigType] | str] = None) -> None:
        """
        手动清理缓存的入口。
        如果传入具体类型则清理该类型，不传则清空全部。
        """
        pass


_ConfName = str


class LocalConfigStore(ConfigStore, ABC):
    """
    基于 Storage 的配置仓库实现，增加了简单的内存缓存。

    mode_name 非空时:
      - 读取: 优先 {name}.{mode}.yml，不存在则 fallback 到 {name}.yml
      - 写入: 始终写到 {name}.{mode}.yml
      - 缓存 key 始终是 conf_name (不含 mode 后缀)
    """

    def __init__(
            self,
            storage: Storage,
            environ: dict[str, str] | None = None,
            on_save: Callable[[str], None] | None = None,
            *,
            mode_name: str = '',
    ) -> None:
        self._storage = storage
        self._cache: dict[_ConfName, ConfigType] = {}
        self._environ = environ  # None means use os.environ at resolve time
        self._on_save = on_save
        self._mode_name = mode_name

    # -- path helpers -------------------------------------------------

    @classmethod
    def _make_config_filename(cls, config_name: str, mode_name: str = '') -> str:
        mode_suffix = f".{mode_name}" if mode_name else ''
        return f"{config_name}{mode_suffix}.yml"

    def _config_filename(self, conf_type_or_obj: Union[Type[ConfigType], ConfigType]) -> str:
        name = conf_type_or_obj.conf_name()
        return self._make_config_filename(name)

    def _resolve_write_filename(self, config_name: str) -> str:
        """写入目标文件名: mode 存在时写到 mode-specific 文件."""
        return self._make_config_filename(config_name, self._mode_name)

    def _resolve_read_path(self, config_name: str) -> pathlib.Path:
        """读取时 mode-first 查找: {name}.{mode}.yml → {name}.yml."""
        root = self._storage.abspath()
        if self._mode_name:
            mode_file = root / self._make_config_filename(config_name, self._mode_name)
            if mode_file.exists():
                return mode_file
        return root / self._make_config_filename(config_name)

    def get_config_path(self, config_name: str) -> str:
        """公开方法: 当前 mode 下的预期文件路径."""
        filename = self._resolve_write_filename(config_name)
        return str(self._storage.abspath().joinpath(filename).absolute())

    # -- core operations -----------------------------------------------

    def get(self, conf_type: Type[CONF_TYPE]) -> CONF_TYPE:
        conf_name = conf_type.conf_name()
        if conf_name in self._cache:
            return self._cache[conf_name]

        path = self._resolve_read_path(conf_name)
        if not path.exists():
            raise FileNotFoundError(
                f"Config file not found: {conf_type} "
                f"(expected {path})"
            )

        content = path.read_bytes()
        data = self._unmarshal(content)
        instance = conf_type(**data)
        resolved = instance.resolve(environ=self._environ)
        self._cache[conf_name] = resolved
        return resolved

    def set_config(self, conf: ConfigType, override: bool = False) -> None:
        conf_name = conf.conf_name()
        if override:
            self.save(conf)
        else:
            self._cache[conf_name] = conf.resolve(environ=self._environ)
            if self._on_save is not None:
                self._on_save(conf_name)

    def get_or_create(self, conf: CONF_TYPE) -> CONF_TYPE:
        conf_type = type(conf)
        conf_name = conf_type.conf_name()

        if conf_name in self._cache:
            return self._cache[conf_name]

        # mode-aware: 先检查 mode-specific 文件，再 fallback base
        read_path = self._resolve_read_path(conf_name)
        if read_path.exists():
            return self.get(conf_type)

        return self._save(conf)

    def _save(self, conf: ConfigType) -> ConfigType:
        """保存配置到磁盘并同步缓存."""
        conf_type = type(conf)
        conf_name = conf_type.conf_name()
        data = conf.model_dump(exclude_none=True)
        marshaled = self._marshal(data, conf_type)

        filename = self._resolve_write_filename(conf_name)
        self._storage.put(filename, marshaled)

        resolved = conf.resolve(environ=self._environ)
        self._cache[conf_name] = resolved
        if self._on_save is not None:
            self._on_save(conf_name)
        return resolved

    def save(self, conf: ConfigType) -> None:
        self._save(conf)

    def invalidate(self, conf_type_or_name: Optional[Type[ConfigType] | str] = None) -> None:
        """手动清理缓存。传类型/名称清理单项，不传清空全部。"""
        if conf_type_or_name is None:
            self._cache.clear()
            return
        elif isinstance(conf_type_or_name, str):
            conf_name = conf_type_or_name
        elif isinstance(conf_type_or_name, type) and issubclass(conf_type_or_name, ConfigType):
            conf_name = conf_type_or_name.conf_name()
        else:
            raise TypeError(f"{conf_type_or_name} is not a ConfigType")
        self._cache.pop(conf_name, None)

    @abstractmethod
    def _unmarshal(self, data: bytes) -> dict:
        pass

    @abstractmethod
    def _marshal(self, data: dict, conf_type: type[ConfigType]) -> bytes:
        pass


def _resolve_config_data_from_env(
        data: dict[str, Any],
        environ: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    recursively replace environment variables with their respective values.
    """
    if environ is None:
        environ = os.environ
    resolved_data = {}
    for key, value in data.items():
        if isinstance(value, dict):
            resolved_data[key] = _resolve_config_data_from_env(value, environ=environ)
        elif isinstance(value, list):
            resolved_data[key] = [
                _resolve_config_data_from_env(item, environ=environ)
                if isinstance(item, dict) else item
                for item in value
            ]
        elif isinstance(value, str) and value.startswith('$'):
            resolved_data[key] = environ.get(value[1:], value)
        else:
            resolved_data[key] = value
    return resolved_data


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


class WorkspaceYamlConfigStoreProvider(Provider[ConfigStore]):

    def __init__(
            self,
            *configs: ConfigType,
            on_save: Callable[[str], None] | None = None,
    ):
        self._configs = list(configs)
        self._on_save = on_save

    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> ConfigStore:
        ws = con.force_fetch(Workspace)
        storage = ws.configs()

        config_store = YamlConfigStore(storage, on_save=self._on_save)
        for config in self._configs:
            config_store.get_or_create(config)
        return config_store
