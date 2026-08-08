"""Configuration contract — typed config schemas, YAML stores, and IoC bootstrapping."""

import yaml
from abc import ABC, abstractmethod
from typing import TypeVar, Type, Optional, Any, ClassVar, Callable
from typing_extensions import Self
from pydantic import BaseModel, Field
from ghoshell_common.helpers import generate_import_path
from ghoshell_common.helpers import yaml_pretty_dump
from ghoshell_container import IoCContainer, Provider, Bootstrapper
from .workspace import Storage, Workspace
import os
import pathlib

__all__ = [
    'ConfigType', 'ConfigStore', 'ConfigSchema',
    'YamlConfigStore',
    'LocalConfigStore',
    'WorkspaceYamlConfigStoreProvider',
    'CONF_TYPE',
    'ConfigInstanceRegisterBootstrapper',
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
    default: dict[str, Any] | None = Field(
        default=None,
        description="config default value.",
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
        data = self.model_dump(exclude_none=True, mode='json')
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

    mode 感知: store 构造时可绑定默认 mode_name (通常由 Provider 从 Environment
    注入). 读写方法均支持显式 mode kwarg 覆盖默认 mode.
    """

    # mode 与 fallback 语义放 ABC 上而不是仅在 impl 层承担, 因为消费者需要
    # 显式覆盖能力 (示例: workspace 管理面读所有 mode 的配置视图).
    # 缓存策略由 impl 决定, 契约不承诺.

    @abstractmethod
    def get(
            self,
            conf_type: Type[CONF_TYPE],
            *,
            mode: str | None = None,
            fallback: bool = True,
    ) -> CONF_TYPE:
        """
        从仓库中读取一个配置对象.

        :param conf_type: 配置对象的类型.
        :param mode: 显式指定 mode. None 时用 store 默认 mode. 空串 '' 表示强制
                     读 base 视图 (不带 mode 后缀的文件).
        :param fallback: True 时 mode 文件不存在则回退到 base 文件; False 时严格
                         只读指定 mode, 缺就抛 FileNotFoundError.
        :return: 配置实例.
        :exception FileNotFoundError: 目标文件不存在 (fallback 也未命中).
        """
        pass

    @abstractmethod
    def get_or_create(
            self,
            conf: CONF_TYPE,
            *,
            mode: str | None = None,
            fallback: bool = True,
    ) -> CONF_TYPE:
        """
        存在则读, 不存在则以 conf 为默认值创建.

        :param conf: 目标类型的默认值实例. 文件不存在时以此写入磁盘.
        :param mode: 同 get.
        :param fallback: 同 get. 注意 fallback=True 且 base 文件存在时, 返回值
                         来自 base, 但后续 save 仍写 mode 文件 (mode-aware 写入).
        """
        pass

    @abstractmethod
    def set_config(
            self,
            conf: ConfigType,
            override: bool = False,
            *,
            mode: str | None = None,
    ) -> None:
        """
        设置 config 实例. override=True 时落盘, False 时仅进缓存.

        :param conf: 配置实例.
        :param override: True 走 save 语义, 落盘 + 缓存; False 只写缓存.
        :param mode: 显式指定 mode. None 时用 store 默认 mode.
        """
        pass

    @abstractmethod
    def get_config_path(self, config_name: str, *, mode: str | None = None) -> str:
        """
        返回给定 config_name 的预期存储路径.

        :param mode: 显式指定 mode. None 用 store 默认.
        """
        pass

    @abstractmethod
    def save(self, conf: ConfigType, *, mode: str | None = None) -> None:
        """
        保存一个 Config 对象到磁盘.

        :param conf: 配置实例.
        :param mode: 显式指定 mode. None 时用 store 默认 mode.
        """
        pass

    @abstractmethod
    def invalidate(self, conf_type_or_name: Optional[Type[ConfigType] | str] = None) -> None:
        """
        手动清理缓存. 传类型或名称清理单项, 不传清空全部.

        缓存 key 不含 mode (一 store 一 mode 缓存假设), 因此本方法无 mode 参数.
        显式 mode 读写路径本身不动缓存, 不需要 invalidate.
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

    显式 mode kwarg 语义 (§Config-2 kwargs 扩展):
      - mode=None → 用 store 默认 mode (self._mode_name).
      - mode ≠ 默认 mode → 绕过缓存, 直接走文件读写. 保 "一 store 一 mode 缓存"
        不变量, 避免多 mode 共享同一 conf_name 缓存槽产生脏读.
      - fallback=False → 严格只读指定 mode 文件, 缺就抛 FileNotFoundError.
        用于 mode 层必须自持配置的场景.
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

    def _effective_mode(self, mode: str | None) -> str:
        # None → 用 store 默认 mode. 显式传空串 '' 表示强制 base 视图.
        return self._mode_name if mode is None else mode

    def _uses_cache(self, mode: str | None) -> bool:
        # 缓存只对默认 mode 生效. 显式 mode 参数总是绕过缓存.
        return mode is None or mode == self._mode_name

    def _resolve_write_filename(self, config_name: str, effective_mode: str = '') -> str:
        """写入目标文件名: mode 存在时写到 mode-specific 文件."""
        return self._make_config_filename(config_name, effective_mode)

    def _resolve_read_path(
            self,
            config_name: str,
            effective_mode: str = '',
            fallback: bool = True,
    ) -> pathlib.Path:
        """读取时 mode-first 查找: {name}.{mode}.yml → {name}.yml (若 fallback)."""
        root = self._storage.abspath()
        if effective_mode:
            mode_file = root / self._make_config_filename(config_name, effective_mode)
            if mode_file.exists() or not fallback:
                # fallback=False 时不管文件存不存在, 只返回 mode path; 上层 get
                # 用 path.exists() 判断决定是否抛 FileNotFoundError.
                return mode_file
        return root / self._make_config_filename(config_name)

    def get_config_path(self, config_name: str, *, mode: str | None = None) -> str:
        """公开方法: 指定 mode 下的预期写入文件路径."""
        effective_mode = self._effective_mode(mode)
        filename = self._resolve_write_filename(config_name, effective_mode)
        return str(self._storage.abspath().joinpath(filename).absolute())

    # -- core operations -----------------------------------------------

    def get(
            self,
            conf_type: Type[CONF_TYPE],
            *,
            mode: str | None = None,
            fallback: bool = True,
    ) -> CONF_TYPE:
        conf_name = conf_type.conf_name()
        use_cache = self._uses_cache(mode)
        if use_cache and conf_name in self._cache:
            return self._cache[conf_name]

        effective_mode = self._effective_mode(mode)
        path = self._resolve_read_path(conf_name, effective_mode, fallback)
        if not path.exists():
            raise FileNotFoundError(
                f"Config file not found: {conf_type} "
                f"(expected {path})"
            )

        content = path.read_bytes()
        data = self._unmarshal(content)
        instance = conf_type(**data)
        resolved = instance.resolve(environ=self._environ)
        if use_cache:
            self._cache[conf_name] = resolved
        return resolved

    def set_config(
            self,
            conf: ConfigType,
            override: bool = False,
            *,
            mode: str | None = None,
    ) -> None:
        conf_name = conf.conf_name()
        if override:
            self._save(conf, mode=mode)
            return
        # cache-only 分支: 显式非默认 mode 时不动缓存 (语义一致性: 显式 mode
        # 走"绕过缓存"路径, cache-only 变成 no-op — 消费者传显式 mode 意在避免
        # cache 副作用, override=False 又想只写 cache 是矛盾姿态).
        if not self._uses_cache(mode):
            return
        self._cache[conf_name] = conf.resolve(environ=self._environ)
        if self._on_save is not None:
            self._on_save(conf_name)

    def get_or_create(
            self,
            conf: CONF_TYPE,
            *,
            mode: str | None = None,
            fallback: bool = True,
    ) -> CONF_TYPE:
        conf_type = type(conf)
        conf_name = conf_type.conf_name()

        use_cache = self._uses_cache(mode)
        if use_cache and conf_name in self._cache:
            return self._cache[conf_name]

        # mode-aware: 先检查 mode-specific 文件，再 fallback base (由 fallback 控制).
        effective_mode = self._effective_mode(mode)
        read_path = self._resolve_read_path(conf_name, effective_mode, fallback)
        if read_path.exists():
            return self.get(conf_type, mode=mode, fallback=fallback)

        return self._save(conf, mode=mode)

    def _save(self, conf: ConfigType, *, mode: str | None = None) -> ConfigType:
        """保存配置到磁盘并同步缓存 (若适用)."""
        conf_type = type(conf)
        conf_name = conf_type.conf_name()
        data = conf.model_dump(exclude_none=True)
        marshaled = self._marshal(data, conf_type)

        effective_mode = self._effective_mode(mode)
        filename = self._resolve_write_filename(conf_name, effective_mode)
        self._storage.put(filename, marshaled)

        resolved = conf.resolve(environ=self._environ)
        if self._uses_cache(mode):
            self._cache[conf_name] = resolved
            if self._on_save is not None:
                self._on_save(conf_name)
        return resolved

    def save(self, conf: ConfigType, *, mode: str | None = None) -> None:
        self._save(conf, mode=mode)

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
            mode: str | None = None,
    ):
        self._configs = list(configs)
        self._on_save = on_save
        self._mode = mode

    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> ConfigStore:
        # 构造逻辑已收口到 Project._configs — 惰性 import 规避 core↔contracts 循环.
        from ghoshell_moss.core.blueprint.project import Project
        project = con.force_fetch(Project)
        return project._configs(
            on_save=self._on_save,
            mode_name=self._mode or '',
            configs=self._configs,
        )


class ConfigInstanceRegisterBootstrapper(Bootstrapper):
    """
    register config type instance when container bootstrapping.
    """

    def __init__(self, *configs: ConfigType, mode: str | None = None, fallback: bool = False) -> None:
        self._configs = list(configs)
        self._mode = mode
        self._fallback = fallback

    def bootstrap(self, container: IoCContainer) -> None:
        store = container.force_fetch(ConfigStore)
        for config in self._configs:
            store.get_or_create(config, mode=self._mode, fallback=self._fallback)
