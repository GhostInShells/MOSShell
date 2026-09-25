"""signal_receiver — 订阅 session signal 总线, drain 打印每条 signal 的完整 JSON.

不带 channel: 直接订阅 ``matrix.session.on_signal`` (跨进程经 zenoh), 回调在订阅
线程上触发, 经 janus 转进 asyncio 后 drain 打印. 每条 signal 打印一次完整 JSON.

Start:  moss nodes run .moss/system_test_nodes/signal_receiver/
"""

import asyncio

from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.blueprint.mindflow import Signal


async def main(matrix: Matrix):
    import janus

    _queue: janus.Queue[Signal] = janus.Queue(maxsize=200)

    def _on_signal(signal: Signal) -> None:
        _queue.sync_q.put(signal)

    async def _drain() -> None:
        while True:
            signal = await _queue.async_q.get()
            print(signal.to_json(), flush=True)

    matrix.session.on_signal(_on_signal)
    asyncio.create_task(_drain())
    matrix.logger.info("[signal_receiver] draining session signal bus, Ctrl-C to stop")

    try:
        while True:
            await asyncio.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    Matrix.discover().run(main)
