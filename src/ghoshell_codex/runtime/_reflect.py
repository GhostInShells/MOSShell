import abc
from typing import Any, Optional, Dict, Tuple, Iterable, Protocol
from typing_extensions import is_typeddict
from ghoshell_codex.runtime._utils import (
    get_modulename_of_value,
    get_callable_definition,
    is_pydantic_type,
    is_typing,
    is_class_method,
)

from dataclasses import is_dataclass
import inspect

# 将上下文引用的 变量/方法/类型 反射成大模型可以理解的 Prompt.
# 在运行时中完成反射.
#
# 主要解决一个问题, 如何让大模型在 python 运行时中理解一个 python module 怎么使用.
# 包含的讯息不仅有当前的源代码, 还要包含当前源代码的引用对象.
#
# 本质上有三种机制:
# + 类: 展示折叠的, 或者全部的源码.
# + 方法: 展示折叠的, 或者全部的源码.
# + 属性: 展示属性的 typehint. 又有几种做法:
#     - 赋值: 类似 `x:int=123` 的形式展示.
#     - 类型: 没有赋值, 只有 `x: foo` 的方式展示.
#     - 字符串类型: 用字符串的方式来描述类型. 比如 `x: "<foo.Bar>"`. 其类型说明是打印结果.
#     - doc: 在 python 的规范里, 属性可以在其下追加字符串作为它的说明.
#
# 预计有以下几种机制:
#
# 1. 在代码里手写注释或者字符串说明.
# 2. 如果变量拥有 __prompt__ 属性, 通过它 (可以是方法或字符串) 生成 prompt.

__all__ = [
    'reflect_prompt_from_value',
    'reflect_imported_locals_by_modulename', 'reflect_class_with_public_methods',
    'join_prompt_lines', 'join_attr_prompts',
    'AttrPrompts',
]

AttrPrompts = Iterable[Tuple[str, str]]
"""
描述多个属性的代码, 作为 prompt 提供给 LLM. 
每个元素代表一个属性. 
元素的值为 Tuple[name, prompt] 
name 是为了去重, prompt 应该就是原有的 prompt. 

如果 attr_name 存在, 使用 f"{name}{prompt}" 格式化, 预计结构是:`name[: typehint] = value[\n\"""doc\"""]`
如果 attr_name 不存在, 则直接使用 prompt. 

多条 prompt 用 "\n\n".join(prompts) 的方式拼接. 
"""

ignore_modules = {
    "pydantic",
}


class ReflectError(Exception):
    pass


class SelfPrompter(Protocol):
    """
    some class that can prompt itself
    """

    @abc.abstractmethod
    def __prompt__(self) -> str:
        pass


def get_value_self_prompt(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "__prompt__"):
        prompter = value.__prompt__
        if inspect.isclass(value) and not is_class_method(prompter):
            return None
        if callable(prompter):
            return prompter()
        elif isinstance(prompter, str):
            return prompter
    return None


def reflect_imported_locals_by_modulename(
        modulename: str,
        local_values: Dict[str, Any],
) -> AttrPrompts:
    """
    MOSS 系统自带的反射方法, 对一个module 的本地变量做最小化的反射展示.
    基本原理:
    1. 当前模块变量:
       - 当前模块的变量默认不展示, 因为本地变量的 prompt 可以直接写在代码里.
       - 如果定义了 __prompt__ 方法, 则会展示出来.
    2. 不反射任何 `_` 开头的本地变量.
    3. 不反射 builtin 类型.
    4. 如果目标是 module
        - 包含 __prompt__ 方法时嵌套展示
        - 否则不展示. 避免递归问题.
    5. 如果目标是 function
        - 包含 __prompt__ 方法时使用它生成,
        - 否则返回 function 的 definition + doc
    6. 如果目标是 class
        - 包含 __class_prompt__ 方法时, 用它生成.
        - __is_abstract__ 的 class, 直接返回源码.
    7. 如果目标是其它 attr
        _ 只有包含 prompt 方法时才展示.

    :param modulename: 当前模块名. 所有当前模块的变量默认不展示.
    :param local_values: 传入的上下文变量.
    """
    for name, value in local_values.items():
        try:
            prompt = reflect_imported_attr(name, value, modulename)
            if prompt is not None:
                yield name, prompt
        except ReflectError:
            yield '', ''


def reflect_class_with_public_methods(cls: type) -> str:
    """
    reflect class with all its method signatures.
    """
    from inspect import getsource
    from ._utils import make_class_prompt, get_callable_definition
    source = getsource(cls)
    attrs = []
    for name in dir(cls):
        if name.startswith("_"):
            continue
        method = getattr(cls, name)
        if inspect.ismethod(method) or inspect.isfunction(method):
            block = get_callable_definition(method)
            attrs.append(block)
    return make_class_prompt(source=source, attrs=attrs)


def reflect_imported_attr(
        name: str,
        value: Any,
        current_module: str,
) -> Optional[str]:
    """
    反射其中的一个值.
    """
    if name.startswith('_'):
        # 私有变量不展示.
        return None
    elif inspect.isbuiltin(value):
        # 系统内置的, 都不展示.
        return None

    prompt = get_value_self_prompt(value)
    if prompt is not None:
        return prompt

    # module 相关的过滤逻辑.
    value_modulename = get_modulename_of_value(value)
    if not value_modulename:
        return None
    elif '.' not in value_modulename:
        # builtin
        return None
    elif value_modulename == current_module:
        return None
    for ignore_module_name in ignore_modules:
        if value_modulename.startswith(ignore_module_name):
            return None

    return reflect_prompt_from_value(value)


def reflect_prompt_from_value(value: Any, throw: bool = False) -> Optional[str]:
    """
    get prompt from value.
    only:
    1. predefined PromptAble
    2. abstract class
    3. function or method
    will generate prompt
    """
    try:
        if inspect.isbuiltin(value):
            return None
        elif is_typing(value):
            return str(value)

        if inspect.isclass(value):
            # only reflect abstract class
            if inspect.isabstract(value) or is_pydantic_type(value) or is_dataclass(value) or is_typeddict(value):
                source = inspect.getsource(value)
                if source:
                    return source
        elif inspect.isfunction(value) or inspect.ismethod(value):
            # 默认都给方法展示 definition.
            return get_callable_definition(value)

        return None
    except Exception as e:
        if throw:
            raise ReflectError() from e
        return None


def join_prompt_lines(*prompts: Optional[str]) -> str:
    """
    将多个可能为空的 prompt 合并成一个 python 代码风格的 prompt.
    """
    result = []
    for prompt in prompts:
        line = prompt.rstrip()
        if line:
            result.append(prompt)
    return '\n\n'.join(result)


def join_attr_prompts(attr_prompts: AttrPrompts) -> str:
    """
    joint attr prompts.
    """
    prompts = []
    for name, prompt in attr_prompts:
        prompt = prompt.strip()
        if not prompt:
            continue
        attr_prompt = f'''# <attr name=`{name}`>
{prompt}
# </attr>'''
        prompts.append(attr_prompt)
    return join_prompt_lines(*prompts)
