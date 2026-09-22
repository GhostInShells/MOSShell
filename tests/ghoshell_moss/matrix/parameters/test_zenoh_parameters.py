"""Parameters 的 zenoh transport 测试 — 单测 (key 映射) + 集成 (真 session)."""

import asyncio

import pytest
import zenoh

from ghoshell_moss.depends import depend_matrix

depend_matrix()

from ghoshell_moss.core.blueprint.parameter import ExampleParameter
from ghoshell_moss.core.parameter import TruthHostParameters, WorkerParameters
from ghoshell_moss.matrix.zenoh_helper import MatrixNamespace
from ghoshell_moss.matrix.parameters import ParameterNamespace, ZenohParametersBroadcaster


def test_parameter_namespace_key_groups():
    # key 组全部由 namespace 派生, 无魔法值.
    ns = ParameterNamespace(MatrixNamespace("test_scope"))
    base = "MOSS/matrix/scopes/test_scope/parameters"
    assert ns.truth_key("ghost") == f"{base}/host/truth/ghost"
    assert ns.declaration_key("ghost") == f"{base}/worker/declaration/ghost"
    assert ns.declaration_wildcard() == f"{base}/worker/declaration/**"
    assert ns.liveness_key("host/name/uid") == f"{base}/host/liveness/host/name/uid"


@pytest.mark.asyncio
async def test_host_declare_worker_subscribe_converges():
    session = zenoh.open(zenoh.Config())
    ns = MatrixNamespace("test_scope_converge")
    try:
        host = TruthHostParameters("host/name/uid1", ZenohParametersBroadcaster(session, ns))
        worker = WorkerParameters("worker/name/uid2", ZenohParametersBroadcaster(session, ns))
        async with host, worker:
            decl = await host.declare(ExampleParameter(example="calm"))
            sub = await worker.subscribe(ExampleParameter)
            await asyncio.sleep(0.3)

            assert sub.value is not None
            assert sub.value.example == "calm"

            seen = []
            sub.on_change(lambda v: seen.append(v.example))
            decl.set(ExampleParameter(example="excited"))
            await asyncio.sleep(0.3)
            assert seen == ["excited"]

            sub.close()
            decl.set(ExampleParameter(example="third"))
            await asyncio.sleep(0.3)
            assert seen == ["excited"]  # 退订后不再收
    finally:
        session.close()
