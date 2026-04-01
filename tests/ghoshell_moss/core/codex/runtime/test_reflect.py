from typing import TypedDict
import inspect
from ghoshell_moss.core.codex.runtime import _reflect
from ghoshell_moss.core.codex.runtime._reflect import reflect_imported_locals_by_modulename, reflect_prompt_from_value


class Foo(TypedDict):
    foo: int


def test_reflect_locals_imported_baseline():
    assert inspect.ismodule(_reflect)
    # inspect 也被 prompts 库引用了.
    assert not inspect.isbuiltin(inspect)
    attr_prompts = reflect_imported_locals_by_modulename("ghoshell_codex.runtime.reflect", _reflect.__dict__)
    data = {}
    array = []
    for name, prompt in attr_prompts:
        array.append((name, prompt))
        data[name] = prompt
    # 从 utils 模块里定义的.
    assert "get_callable_definition" in data
    # typing 库本身的不会出现.
    assert "Optional" not in data
    # 引用的抽象类应该存在.


def test_typed_dict_reflect_code():
    pr = reflect_prompt_from_value(Foo)
    source = inspect.getsource(Foo)
    assert len(source) > 0
    assert len(pr) > 0
