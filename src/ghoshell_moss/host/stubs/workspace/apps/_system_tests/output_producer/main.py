import asyncio
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.message import Message


async def producer_task(matrix: Matrix):
    session = matrix.session
    i = 0
    while True:
        # 模拟生成一条消息
        msg = Message.new().with_content(f"MOSS system signal impulse #{i}")
        # 通过 session 发送，这会通过 Zenoh 广播出去
        session.output('log', msg)
        print("output: %s" % msg.to_content_string())

        # 偶尔丢一个 error，测试 UI 高亮
        if i % 5 == 0:
            err = Message.new().with_content(f"Minor system glitch detected at tick {i}")
            session.output('error', err)
            print("output: %s" % err.to_content_string())

        i += 1
        await asyncio.sleep(1)


if __name__ == "__main__":
    Matrix.discover().run(producer_task)
