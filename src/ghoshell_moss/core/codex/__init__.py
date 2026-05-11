from typing import Any
from types import ModuleType
from importlib import import_module
from .reflector import reflect_module, reflect_module_by_import_path, reflect_any_by_import_path, Reflector
from .compiler import Compiler
from .executor import Executor
from ._features import (
    parse_frontmatter,
    list_features,
    list_archived_features,
    get_feature,
    create_feature,
    update_feature_status,
    archive_feature,
    init_features,
    VALID_STATUSES,
)

__all__ = [
    'Reflector',
    'reflect_module', 'reflect_module_by_import_path', 'reflect_any_by_import_path',
    'Compiler',
    'compile',
    'Executor',
    # Features
    'parse_frontmatter',
    'list_features',
    'list_archived_features',
    'get_feature',
    'create_feature',
    'update_feature_status',
    'archive_feature',
    'init_features',
    'VALID_STATUSES',
]


def compile(
        module: str | ModuleType | None,
        append_source: str,
        *,
        module_name: str | None = None,
        local_injections: dict[str, Any] | None = None,
) -> Compiler:
    """
    基于当前运行时进行编译.
    """
    if module is None:
        pass
    elif isinstance(module, str):
        module_name = module_name or module
        module = import_module(module)
    elif isinstance(module, ModuleType):
        module_name = module_name or module.__name__
    else:
        raise AttributeError(f"module {module!r} is not a str or module")

    complier = Compiler(
        origin=module,
        source=append_source,
        modulename=module_name,
        local_injections=local_injections,
        compile_soon=True,
    )
    return complier
