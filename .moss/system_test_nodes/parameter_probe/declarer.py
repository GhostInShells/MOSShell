"""
parameter_probe declarer — 声明一个 parameter, 每隔 N 秒改一次值.

用法 (host 由 moss mcp 提供):

    python declarer.py
"""

import asyncio

from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.blueprint.parameter import ParameterModel

_INTERVAL = 3.0


class ProbeValue(ParameterModel):
    counter: int = 0
    message: str = ""

    @classmethod
    def parameter_key(cls) -> str:
        return "parameter_probe"


async def main(matrix: Matrix):
    params = matrix.session.parameters
    decl = await params.declare(ProbeValue(counter=0, message="init"))

    def _on_truth(data):
        p = data.payload
        print(
            f"[declarer] truth v={data.version} epoch={data.epoch[:12]!r} "
            f"counter={p.get('counter')} message={p.get('message')!r}",
            flush=True,
        )

    decl.on_truth(_on_truth)
    print(f"[declarer] declared key={decl.key} is_host={params.is_host()}", flush=True)

    i = 0
    while True:
        await asyncio.sleep(_INTERVAL)
        i += 1
        decl.set(ProbeValue(counter=i, message=f"tick-{i}"))
        print(f"[declarer] set counter={i}", flush=True)


if __name__ == "__main__":
    Matrix.new("parameter_declarer", description="parameter probe declarer").run(main)
