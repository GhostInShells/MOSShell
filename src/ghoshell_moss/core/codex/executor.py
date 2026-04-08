from typing import Any, Optional, NamedTuple, Iterator
from types import ModuleType
from .compiler import Compiler
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
import io

_LocalAttrName = str
_KwArgName = str

__all__ = ['ExecutionResult', 'Executor']

@dataclass
class ExecutionResult:
    """
    result of the execution
    """
    returns: Any
    std_output: str


class Executor:
    """
    运行时里为一个 Module 创建一个运行时容器,
    可以为它增加代码, 基于类似的上下文运行.
    可以运行很多次, 其中 `__all__` 定义的变量还会在每一次调用时继承.
    但不会污染原始的 Module.
    """

    EXECUTE_MODULE_NAME = "__execute__"
    """
    执行时编译的临时模块, 默认使用的 modulename. 可以基于这种规则定义执行: 
    
    >>> if __name__ == "__execute__":
    >>>     __result__ = 123
    """
    RESULT_VARIABLE = "__result__"

    def __init__(
            self,
            origin: ModuleType,
            local_injections: dict[str, Any] | None = None,
    ):
        self._origin = origin
        self._local_injections = local_injections or {}

    def execute(
            self,
            code: str = "",
            func_name: str = '',
            *,
            with_local_args: Optional[list[_LocalAttrName]] = None,
            with_local_kwargs: Optional[dict[_KwArgName, _LocalAttrName]] = None,
            args: Optional[list[Any]] = None,
            kwargs: Optional[dict[_KwArgName, Any]] = None,
    ) -> ExecutionResult:
        """
        在原始的 module 下面编译一段代码, 并且立刻执行或者挑选一个函数执行.
        :param code: 追加的代码
        :param func_name: 需要执行的函数. 为空则以编译为准.
        :param with_local_args: 函数依赖的本地参数作为 args
        :param with_local_kwargs: 函数依赖的
        :param args:
        :param kwargs:
        :return:
        """
        result = ExecutionResult(returns=None, std_output='')
        with self._redirect_stdout(result):
            if code:
                compiler = Compiler(
                    source=code,
                    origin=self._origin,
                    modulename=self.EXECUTE_MODULE_NAME,
                    local_injections=self._local_injections,
                )
                module = compiler.compiled
            else:
                module = self._origin

            if not func_name:
                result.returns = module.__dict__.get(self.RESULT_VARIABLE, None)
                return result

            fn = module.__dict__.get(func_name, None)
            if fn is None:
                raise AttributeError(f'"{func_name}" is not found')
            if not callable(fn):
                raise TypeError(f'"{func_name}" is not callable')

            _args = []
            _kwargs = {}
            if with_local_args:
                for attr_name in with_local_args:
                    if not hasattr(module, attr_name):
                        raise AttributeError(f'"{attr_name}" is not defined')
                    _args.append(module.__dict__.get(attr_name))
            if with_local_kwargs:
                for key, attr_name in with_local_kwargs.items():
                    if not hasattr(module, attr_name):
                        raise AttributeError(f'"{attr_name}" is not defined')
                    _kwargs[key] = module.__dict__.get(attr_name)

            if args:
                _args.extend(args)
            if kwargs:
                _kwargs.update(kwargs)

            result.returns = fn(*_args, **_kwargs)

            _all = module.__dict__.get('__all__')
            if _all:
                for attr_name in _all:
                    self._local_injections[attr_name] = module.__dict__.get(attr_name)
            return result

    @contextmanager
    def _redirect_stdout(self, result: ExecutionResult) -> Iterator[None]:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            yield
            result.std_output += str(buffer.getvalue())
