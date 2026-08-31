"""MemoryParameters — 单进程参考实现的协议行为测试."""

import asyncio

import pytest

from ghoshell_moss.core.blueprint.parameter import ParameterModel
from ghoshell_moss.core.parameter import MemoryParameters


class GhostPersona(ParameterModel):
    name: str = "Echo"
    temperature: float = 0.7

    @classmethod
    def parameter_key(cls) -> str:
        return "ghost_persona"


@pytest.mark.asyncio
async def test_declare_returns_default_value():
    params = MemoryParameters()
    async with params:
        decl = await params.declare(GhostPersona())
        assert decl.key == "ghost_persona"
        assert decl.value.name == "Echo"


@pytest.mark.asyncio
async def test_subscribe_gets_initial_value_and_updates():
    params = MemoryParameters()
    async with params:
        decl = await params.declare(GhostPersona())
        sub = await params.subscribe(GhostPersona)
        assert sub.value.name == "Echo"  # 初值 = 声明者的当前值

        seen = []
        sub.on_change(lambda v: seen.append(v.name))

        decl.set(GhostPersona(name="Nova", temperature=0.9))
        await asyncio.sleep(0.01)
        assert sub.value.name == "Nova"
        assert seen == ["Nova"]


@pytest.mark.asyncio
async def test_close_stops_updates():
    params = MemoryParameters()
    async with params:
        decl = await params.declare(GhostPersona())
        sub = await params.subscribe(GhostPersona)

        seen = []
        sub.on_change(lambda v: seen.append(v.name))

        sub.close()
        decl.set(GhostPersona(name="X"))
        await asyncio.sleep(0.01)
        assert seen == []


@pytest.mark.asyncio
async def test_custom_key():
    params = MemoryParameters()
    async with params:
        decl = await params.declare(GhostPersona(), key="alt_persona")
        assert decl.key == "alt_persona"

        sub = await params.subscribe(GhostPersona, key="alt_persona")
        assert sub.value.name == "Echo"

        seen = []
        sub.on_change(lambda v: seen.append(v.name))
        decl.set(GhostPersona(name="custom"))
        await asyncio.sleep(0.01)
        assert seen == ["custom"]


@pytest.mark.asyncio
async def test_declared_lists_schemas():
    params = MemoryParameters()
    async with params:
        await params.declare(GhostPersona())
        schemas = params.declared()
        assert any(s.name == "ghost_persona" for s in schemas)
