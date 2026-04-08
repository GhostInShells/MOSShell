from typing import Any
from typing_extensions import is_protocol
from types import ModuleType
from ._utils import is_typing
import inspect

__all__ = ['Compiler']


def _escape_python_indent(source: str) -> str:
    if not source:
        return source

    lines = source.splitlines()

    # 1. 找到最小缩进（忽略空行或仅含空格的行）
    min_indent = None

    for line in lines:
        content = line.replace('\t', '    ').lstrip()
        if not content:  # 忽略空行
            continue

        # 计算当前行的领先空格数
        # 建议先将 tab 统一替换为空格，避免切片偏移错误
        indent = len(line) - len(content)

        if min_indent is None or indent < min_indent:
            min_indent = indent

    # 2. 如果没找到有效行，或者最小缩进为 0，直接返回
    if min_indent is None or min_indent == 0:
        return source

    # 3. 移除缩进
    return '\n'.join([line[min_indent:] if line.strip() else "" for line in lines])


class Compiler:
    """
    在运行时, 为一个存在的 Module 编译一段新代码, 不直接污染原来的 module.
    提供 Module 级别的运行时容器, 复制原始 module 的类型, 但不复制属性和实例.
    注意编译后的名字是可控的.
    """

    def __init__(
            self,
            *,
            source: str,
            origin: ModuleType | None = None,
            modulename: str | None = None,
            filename: str = '<moss_codex_temp_module>',
            local_injections: dict[str, Any] | None = None,
            compile_soon: bool = True,
    ):
        self._source = _escape_python_indent(source)
        self._source = "from __future__ import annotations\n" + self._source
        self._origin = origin
        self._local_injections = local_injections or {}
        self._filename = filename
        if modulename is None:
            if origin is not None:
                modulename = origin.__name__
            else:
                modulename = 'moss_codex_temp_module'
        self._modulename = modulename
        self._compiled: ModuleType | None = None
        if compile_soon:
            self._compiled = self._compile()

    @property
    def compiled(self) -> ModuleType:
        if self._compiled is None:
            self._compiled = self._compile()
        return self._compiled

    def get(self, attr_name: str) -> Any:
        """
        获取一个已有的属性.
        """
        if attr_name not in self.compiled.__dict__:
            raise AttributeError(f"'{self._modulename}' has no attribute '{attr_name}'")
        value = self.compiled.__dict__[attr_name]
        return value

    def _compile(self) -> ModuleType:
        module = ModuleType(self._modulename)
        if self._origin:
            _locals = self._filter_origin_attrs(self._origin)
            module.__dict__.update(_locals)
        if self._local_injections:
            module.__dict__.update(self._local_injections)
        module.__file__ = self._filename
        try:
            compiled = compile(self._source, self._modulename, "exec")
            exec(compiled, module.__dict__)
        except SyntaxError as e:
            raise e
        except Exception as e:
            raise SyntaxError(f"Compile {self._modulename} failed: {e}")
        if self._origin:
            inherit_attrs = self._filter_origin_must_inherit_attrs(self._origin)
            module.__dict__.update(inherit_attrs)
        return module

    @staticmethod
    def _filter_origin_attrs(origin: ModuleType) -> dict[str, Any]:
        """
        todo: 逐步完善.
        """
        from copy import deepcopy
        result = {}
        for attr_name, attr_value in origin.__dict__.items():
            if attr_name.startswith("__"):
                continue
            elif is_protocol(attr_value):
                result[attr_name] = attr_value
            elif inspect.ismodule(attr_value):
                result[attr_name] = attr_value
            elif inspect.isclass(attr_value) or inspect.isfunction(attr_value) or inspect.isbuiltin(attr_value):
                result[attr_name] = attr_value
            elif is_typing(attr_value):
                result[attr_name] = attr_value
            elif isinstance(attr_value, object):
                result[attr_name] = deepcopy(attr_value)
            else:
                result[attr_name] = attr_value
        return result

    @staticmethod
    def _filter_origin_must_inherit_attrs(origin: ModuleType) -> dict[str, Any]:
        """
        为编译后的 Module 复制必须继承的对象或类型, 避免类型判断出错.
        :param origin: 原始的 module
        """
        result = {}
        for attr_name, attr_value in origin.__dict__.items():
            if attr_name.startswith("__"):
                continue
            if inspect.isclass(attr_value) or inspect.isfunction(attr_value):
                if attr_value.__module__ != origin.__name__:
                    continue
                result[attr_name] = attr_value
        return result
