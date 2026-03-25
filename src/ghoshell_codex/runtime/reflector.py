from typing import Iterable
from typing_extensions import Self
from types import ModuleType
from functools import lru_cache
import inspect
from ghoshell_common.helpers import import_from_path

__all__ = [
    'RuntimeModuleReflector',
    'reflect_module',
    'reflect_module_by_import_path',
    'reflect_any_by_import_path',
]

_AttrName = str
_Prompt = str


def reflect_module(module: ModuleType) -> str:
    """
    generate llm-oriented prompt from runtime module
    """
    return RuntimeModuleReflector.from_module(module).reflect()


def reflect_any_by_import_path(import_path: str) -> str:
    """
    :param import_path: [module.path][:attribute]
    :return: value
    """
    from ghoshell_codex.runtime._reflect import reflect_prompt_from_value
    value = import_from_path(import_path)
    if isinstance(value, ModuleType):
        return reflect_module(value)
    data = reflect_prompt_from_value(value)
    if data is None:
        data = repr(value)
    return data


def reflect_module_by_import_path(import_path: str) -> str:
    """
    根据 module path 反射一个 module.
    :param import_path:
    """
    import importlib
    module = importlib.import_module(import_path)
    return reflect_module(module)


class RuntimeModuleReflector:
    """
    reflect module source code in runtime.
    """

    def __init__(
            self,
            module: ModuleType,
            *,
            modulename: str | None = None,
            source: str | None = None,
    ):
        self._module = module
        self._modulename = modulename or module.__name__
        self._source = source or inspect.getsource(module)
        self._prompt: str | None = None

    @classmethod
    @lru_cache(maxsize=100)
    def from_module(cls, module: ModuleType) -> Self:
        return RuntimeModuleReflector(module)

    @property
    def source(self) -> str:
        """
        :return: source code of the module
        """
        return self._source

    @property
    def modulename(self) -> str:
        """
        :return: name of the module
        """
        return self._modulename

    def reflect(self) -> str:
        """
        :return: generated prompt of the module
        """
        if self._prompt is None:
            self._prompt = self._make_prompt()
        return self._prompt

    def _make_prompt(self) -> str:
        from ._reflect import reflect_imported_locals_by_modulename
        from ._utils import escape_string_quotes
        attr_prompts = reflect_imported_locals_by_modulename(
            self._modulename,
            self._module.__dict__
        )
        attr_prompts_str = self.join_attr_prompts(attr_prompts)
        escaped_attr_prompts_str = escape_string_quotes(attr_prompts_str, '"""')
        attr_prompt_part = ("# more attr information are list below (quoted by <attr></attr>):\n"
                            '"""\n'
                            f"{escaped_attr_prompts_str}\n"
                            '"""\n\n'
                            )

        return "\n\n".join([
            self.source,
            attr_prompt_part,
        ])

    @staticmethod
    def join_attr_prompts(attr_prompts: Iterable[tuple[_AttrName, _Prompt]]) -> str:
        """
        joint attr prompts.
        """
        prompts = []
        for name, prompt in attr_prompts:
            if not prompt:
                continue
            prompt = prompt.strip()
            if not prompt:
                continue
            attr_prompt = (f"# <attr:{name}>\n"
                           f"{prompt}\n"
                           f"</attr:{name}>\n")
            prompts.append(attr_prompt)
        return "\n".join(prompts)
