from typing import Generic, TypeVar
from pathlib import Path
from ghoshell_moss.core.blueprint.project import Manifest

T = TypeVar('T')

__all__ = ['ScannedManifest']


class ScannedManifest(Manifest[T], Generic[T]):
    """通用的 `Manifest[T]` 实现 — 把扫描到的任意 Python 对象包装为自描述 Manifest.

    ``update_container`` 默认为 no-op. 需要注册到 IoC 容器的类型 (如 Provider)
    可以继承并覆盖.

    正常 manifest 传 ``value``; 扫描出错的 manifest 传 ``error``.
    ``value()`` 在 ``is_error()`` 为 True 时会 raise.
    """

    def __init__(
            self,
            *,
            name: str,
            found_at: Path,
            value: T | None = None,
            error: Exception | None = None,
            import_path: str | None = None,
            description: str = '',
            source: str = '',
            detail: str = '',
    ):
        self._name = name
        self._value = value
        self._error = error
        self._found_at = found_at
        self._import_path = import_path
        self._description = description
        self._source = source
        self._detail = detail

    def found_at(self) -> Path:
        return self._found_at

    def import_path(self) -> str | None:
        return self._import_path

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return self._description

    def source(self) -> str:
        return self._source

    def value(self) -> T:
        if self._error is not None:
            raise self._error
        return self._value  # type: ignore[return-value]

    def detail(self) -> str:
        return self._detail

    def is_error(self) -> bool:
        return self._error is not None

    def error(self) -> Exception | None:
        return self._error
