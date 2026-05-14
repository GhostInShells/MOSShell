from types import ModuleType

import pytest

from ghoshell_moss.channels.module_channel import new_module_channel


def _make_module(name, **funcs):
    mod = ModuleType(name)
    for fname, func in funcs.items():
        setattr(mod, fname, func)
    return mod


# ---- 基线: 无 __all__ 从 dir() 取非私有 callable ---- #

@pytest.mark.asyncio
async def test_default_fallback_to_dir():
    mod = _make_module("mod", add=lambda a, b: a + b, greet=lambda name: f"hello {name}")
    chan = new_module_channel(mod)

    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        meta = runtime.self_meta()
        cmd_names = {c.name for c in meta.commands}
        assert cmd_names == {"add", "greet"}

        assert await runtime.execute_command("add", args=(1, 2)) == 3
        assert await runtime.execute_command("greet", args=("world",)) == "hello world"


# ---- respect_all: 有 __all__ 则尊重, 没有则 fallback ---- #

@pytest.mark.asyncio
async def test_respect_all():
    mod = _make_module("mod", public=lambda: "ok", internal=lambda: "no")
    mod.__all__ = ["public"]

    chan = new_module_channel(mod)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        cmd_names = {c.name for c in runtime.self_meta().commands}
        assert cmd_names == {"public"}
        assert runtime.get_command("internal") is None


# ---- respect_all=False 强制忽略 __all__ ---- #

@pytest.mark.asyncio
async def test_ignore_all():
    mod = _make_module("mod", visible=lambda: "v", hidden=lambda: "h")
    mod.__all__ = ["visible"]

    chan = new_module_channel(mod, respect_all=False)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        cmd_names = {c.name for c in runtime.self_meta().commands}
        assert cmd_names == {"visible", "hidden"}


# ---- include / exclude ---- #

@pytest.mark.asyncio
async def test_include():
    mod = _make_module("mod", a=lambda: 1, b=lambda: 2, c=lambda: 3)
    chan = new_module_channel(mod, include=["a", "c"])

    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        cmd_names = {c.name for c in runtime.self_meta().commands}
        assert cmd_names == {"a", "c"}


@pytest.mark.asyncio
async def test_exclude():
    mod = _make_module("mod", a=lambda: 1, b=lambda: 2)
    chan = new_module_channel(mod, exclude=["b"])

    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        cmd_names = {c.name for c in runtime.self_meta().commands}
        assert cmd_names == {"a"}


# ---- 私有函数始终排除 ---- #

@pytest.mark.asyncio
async def test_private_always_excluded():
    mod = _make_module("mod", ok=lambda: "ok", _secret=lambda: "no", __dunder=lambda: "no")
    chan = new_module_channel(mod)

    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        cmd_names = {c.name for c in runtime.self_meta().commands}
        assert cmd_names == {"ok"}


# ---- __all__ 中的非 callable 被跳过 ---- #

@pytest.mark.asyncio
async def test_non_callable_in_all_skipped():
    mod = _make_module("mod", real=lambda: "real")
    mod.__all__ = ["real", "CONSTANT"]
    mod.CONSTANT = 42

    chan = new_module_channel(mod)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        cmd_names = {c.name for c in runtime.self_meta().commands}
        assert cmd_names == {"real"}


# ---- 字符串 import ---- #

@pytest.mark.asyncio
async def test_string_import():
    chan = new_module_channel("json", include=["dumps", "loads"])

    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        cmd_names = {c.name for c in runtime.self_meta().commands}
        assert cmd_names == {"dumps", "loads"}
        result = await runtime.execute_command("dumps", args=({"a": 1},))
        assert '"a"' in result


# ---- custom name / description ---- #

@pytest.mark.asyncio
async def test_custom_meta():
    mod = _make_module("original", greet=lambda: "hi")
    chan = new_module_channel(mod, name="renamed", description="desc")

    async with chan.bootstrap() as runtime:
        meta = runtime.self_meta()
        assert meta.name == "renamed"
        assert meta.description == "desc"
        assert runtime.get_command("greet") is not None


# ---- 边界: math 模块 (真实模块, 无 __all__, dir() fallback) ---- #

@pytest.mark.asyncio
async def test_math_module():
    """真实模块反射: math 没有 __all__, 走 dir() fallback."""
    chan = new_module_channel("math", include=["sqrt", "ceil", "floor"])

    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        cmd_names = {c.name for c in runtime.self_meta().commands}
        assert cmd_names == {"sqrt", "ceil", "floor"}

        # 实际执行
        assert await runtime.execute_command("sqrt", args=(4,)) == 2.0
        assert await runtime.execute_command("ceil", args=(3.14,)) == 4
        assert await runtime.execute_command("floor", args=(3.14,)) == 3
