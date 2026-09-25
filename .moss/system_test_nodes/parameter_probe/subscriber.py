"""
parameter_probe subscriber — 订阅一个 parameter, 打印真值变化.

用法 (host 由 moss mcp 提供):

    python subscriber.py
"""

import asyncio

from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.blueprint.parameter import ParameterModel


class ProbeValue(ParameterModel):
    counter: int = 0
    message: str = ""

    @classmethod
    def parameter_key(cls) -> str:
        return "parameter_probe"


async def main(matrix: Matrix):
    params = matrix.session.parameters
    sub = await params.subscribe(ProbeValue)

    def _on_truth(data):
        p = data.payload
        print(
            f"[subscriber] truth v={data.version} epoch={data.epoch[:12]!r} "
            f"counter={p.get('counter')} message={p.get('message')!r}",
            flush=True,
        )

    sub.on_truth(_on_truth)
    print(
        f"[subscriber] subscribed counter={sub.value.counter} is_host={params.is_host()}",
        flush=True,
    )

    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    Matrix.new("parameter_subscriber", description="parameter probe subscriber").run(main)
