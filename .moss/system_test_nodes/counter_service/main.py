"""Counter service node — provides inc + echo queryables via service operator.

Start:  moss nodes run .moss/system_test_nodes/counter_service
"""

from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.services.counter import CounterServer


async def main(matrix: Matrix):
    counter = CounterServer.new(matrix)
    matrix.logger.info("[counter_service] providing counter service...")
    async with counter:
        matrix.logger.info("[counter_service] running — press Ctrl+C to stop")
        await matrix.wait_closed()


if __name__ == "__main__":
    Matrix.discover().run(main)
