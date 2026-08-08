"""Counter caller node — discovers counter service, calls inc + echo.

Start:  moss nodes run .moss/system_test_nodes/counter_caller

V1 validation: this script exercises the operator's discovery + get surface
against a running counter_service node on the same zenoh network.
"""

import asyncio

from ghoshell_moss.core.blueprint.matrix import Matrix


async def main(matrix: Matrix):
    op = await matrix.service_operator()

    # -- wait for counter service to appear on the network --
    services: list = []
    while not services:
        services = await op.get_services_by_kind("counter")
        if not services:
            matrix.logger.info("[counter_caller] waiting for counter service...")
            await asyncio.sleep(1.0)

    meta = services[0]
    matrix.logger.info(
        "[counter_caller] found counter at %s", meta["address"],
    )

    # -- inc -------------------------------------------------------------
    replies = await op.get("counter", "inc", None, meta)
    assert replies, "inc: no reply from counter service"
    count = replies[0]["payload"].decode()
    matrix.logger.info("[counter_caller] inc -> %s", count)

    # -- inc again (verify stateful counter) -----------------------------
    replies = await op.get("counter", "inc", None, meta)
    assert replies, "inc#2: no reply"
    count2 = replies[0]["payload"].decode()
    matrix.logger.info("[counter_caller] inc -> %s", count2)
    assert int(count2) == int(count) + 1, (
        f"counter not stateful: {count} -> {count2}"
    )

    # -- echo ------------------------------------------------------------
    replies = await op.get("counter", "echo", b"hello world", meta)
    assert replies, "echo: no reply"
    echoed = replies[0]["payload"].decode()
    matrix.logger.info("[counter_caller] echo -> %s", echoed)
    assert echoed == "hello world", f"echo mismatch: {echoed!r}"

    matrix.logger.info("[counter_caller] V1 validation PASSED")


if __name__ == "__main__":
    Matrix.discover().run(main)
