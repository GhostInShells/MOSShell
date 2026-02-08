import anyio
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream


def test_basic_producer_and_consumer():
    async def producer(send: MemoryObjectSendStream[str]):
        async with send:
            for c in "hello world":
                await send.send(c)

    async def consumer(recv: MemoryObjectReceiveStream[str]) -> str:
        text = ""
        async with recv:
            async for c in recv:
                text += c
        return text

    async def main() -> None:
        sender, receiver = anyio.create_memory_object_stream()
        async with anyio.create_task_group() as tg:
            tg.start_soon(producer, sender)
            text = await consumer(receiver)
            assert text == "hello world"

    anyio.run(main)


# def test_stream_in_defer_thread():
#     async def producer(send: MemoryObjectSendStream[str]):
#         async with send:
#             for c in "hello world":
#                 await send.send(c)
#
#     async def consumer(recv: MemoryObjectReceiveStream[str]) -> str:
#         text = ""
#         async with recv:
#             async for c in recv:
#                 text += c
#         return text
#
#     def producer_thread(send: MemoryObjectSendStream[str]):
#         text = anyio.run(producer, send)
#         assert text == "hello world"
#
#     def consumer_thread(recv: MemoryObjectReceiveStream[str]):
#         anyio.run(consumer, recv)
#
#     sender, receiver = anyio.create_memory_object_stream()
#
#     threads = [Thread(target=producer_thread, args=(sender,)), Thread(target=consumer_thread, args=(receiver,))]
#     for t in threads:
#         t.start()
#     for t in threads:
#         t.join()
