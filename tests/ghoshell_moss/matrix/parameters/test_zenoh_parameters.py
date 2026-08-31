"""ZenohParameters — 点对点 declare/subscribe 的集成测试 (单 session, 双实例)."""

import asyncio

import pytest
import zenoh

from ghoshell_moss.depends import depend_matrix

depend_matrix()

from ghoshell_moss.core.blueprint.parameter import ExampleParameter
from ghoshell_moss.matrix.zenoh_helper import MatrixNamespace
from ghoshell_moss.matrix.parameters import ZenohParameters


@pytest.mark.asyncio
async def test_declare_subscribe_push_close():
    session = zenoh.open(zenoh.Config())
    ns = MatrixNamespace("test_scope")
    try:
        decl = ZenohParameters(session, ns, address="cell/declarer")
        sub = ZenohParameters(session, ns, address="cell/subscriber")
        async with decl, sub:
            d = await decl.declare(ExampleParameter())
            d.set(ExampleParameter(example="hello"))
            await asyncio.sleep(0.1)  # publish loop 推出去

            s = await sub.subscribe(ExampleParameter, address="cell/declarer")
            # 初值 = query 拉到的声明者当前值
            assert s.value is not None
            assert s.value.example == "hello"

            seen = []
            s.on_change(lambda v: seen.append(v.example))

            d.set(ExampleParameter(example="second"))
            await asyncio.sleep(0.1)
            assert seen == ["second"]

            s.close()
            d.set(ExampleParameter(example="third"))
            await asyncio.sleep(0.1)
            assert seen == ["second"]  # 退订后不再收
    finally:
        session.close()


@pytest.mark.asyncio
async def test_subscribe_without_declarer_gets_none():
    session = zenoh.open(zenoh.Config())
    ns = MatrixNamespace("test_scope_empty")
    try:
        sub = ZenohParameters(session, ns, address="cell/subscriber")
        async with sub:
            s = await sub.subscribe(ExampleParameter, address="cell/ghost")
            assert s.value is None  # 无声明者 → None
    finally:
        session.close()
