import inspect
from typing import Optional, Callable, Coroutine, Union, Iterable
from ghoshell_moss.concepts.command import Command, BaseCommandTask, CommandMeta, RESULT, CommandCall
from ghoshell_moss.helpers.func_parser import parse_function_interface
from ghoshell_moss.helpers.result import ThreadSafeResult
from ghoshell_common.helpers import uuid
from collections import deque
from threading import Event
from queue import Queue
import asyncio


class PyCommand(Command):

    def __init__(self, meta: CommandMeta, chan: Channel):
        self._meta = meta

    def meta(self) -> CommandMeta:
        return self._meta

    def __prompt__(self) -> str:
        return self._meta.interface

    async def __call__(self, *args, **kwargs) -> RESULT:
        pass
# class DeltaStream:
#
#     def __init__(self, loop: asyncio.AbstractEventLoop | None = None):
#         self._queue: asyncio.Queue[str | None] = asyncio.Queue()
#         self.loop = loop or asyncio.get_running_loop()
#         self.done = asyncio.Event()
#         self.buffer = ""
#
#     def append(self, delta: str | None):
#         if delta is None:
#             self.done = True
#             return
#         if not self.done:
#             self.buffer += delta
#             self.loop.call_soon_threadsafe(self._queue.put_nowait, delta)
#
#     async def __aiter__(self) -> Iterable[str]:
#         while True:
#             value = self._queue.get()
#             if value is None:
#                 raise StopAsyncIteration
#             yield value
#
#
# class PyCommandTask(CommandTask):
#
#     def __init__(
#             self,
#             meta: CommandMeta,
#             func: Callable[..., Coroutine],
#             call: CommandCall,
#             kwargs: dict,
#     ):
#         self._func = func
#         self._meta = meta
#         self.call = call
#         self.kwargs = kwargs
#         self._result = ThreadSafeResult(call.cid)
#         self._task: Optional[asyncio.Task] = None
#         self._loop: Optional[asyncio.AbstractEventLoop] = None
#         self._stream = None
#         if self._meta.delta_arg is not None:
#             self._stream = DeltaStream(self._loop)
#
#     def is_done(self) -> bool:
#         return self._result.is_done()
#
#     def cancel(self):
#         self._result.cancel("cancelled")
#         if self._task is not None and self._loop is not None:
#             self._loop.call_soon_threadsafe(self._task.cancel)
#
#     async def write(self, delta: str) -> None:
#         if self._stream is not None:
#             self._stream.append(delta)
#
#     async def result(self, timeout: float | None = None) -> RESULT:
#         return await self._result.wait_async(timeout)
#
#     def __await__(self) -> RESULT:
#         if self._result.is_done():
#             # 不重入.
#             return self._result.wait_async().__await__()
#         if self._loop is not None:
#             raise RuntimeError(f"the command call {self.call} has already been awaited")
#         # 注册 loop, 方便线程安全地 cancel 它.
#         self._loop = asyncio.get_running_loop()
#
#         async def wait_until_done() -> RESULT:
#             kwargs = self.kwargs
#
#             # with delta
#
#             if self._stream is not None:
#                 if self._meta.delta_arg == "__stream__":
#                     kwargs[self._meta.delta_arg] = self._stream
#                 else:
#                     await self._stream.done.wait()
#                     kwargs[self._meta.delta_arg] = self._stream.buffer
#
#             # 将它 task 化, 方便 cancel.
#             task = asyncio.create_task(self._func(**self.kwargs))
#             self._task = task
#             # 通过 _result 进行通知.
#             try:
#                 result = await task
#                 # broadcast the result thread-safely
#                 self._result.resolve(result)
#                 return result
#             # todo: 记录异常.
#             except Exception as e:
#                 # always notice the task is excepted.
#                 self._result.cancel(str(e))
#                 raise e
#             finally:
#                 self._task = None
#                 self._loop = None
#
#         return wait_until_done().__await__()
#
#
# class PyCommand(Command):
#     """
#     将 python 的 Coroutine 函数封装成 Command
#     通过反射获取 interface.
#     """
#
#     def __init__(
#             self,
#             chan: str,
#             func: Callable[..., Coroutine],
#             alias: Optional[str] = None,
#             interface: Union[Callable[[], str], None, str] = None,
#             doc: Union[Callable[[], str], None, str] = None,
#             uuid_fn: Callable[[], str] = None,
#     ):
#         self._chan = chan
#         self._name = alias or func.__name__
#         self._func = func
#         self._interface_fn = interface
#         self._doc_fn = doc
#         self._func_itf = parse_function_interface(func)
#         self._uuid_fn = uuid_fn or uuid
#
#     def meta(self) -> CommandMeta:
#
#         return CommandMeta(
#             chan=self._chan,
#             name=self._name,
#             doc=self._get_doc() or self._func_itf.docstring,
#             interface=self.__prompt__(),
#         )
#
#     def _get_doc(self):
#         doc = ""
#         if self._doc_fn is not None:
#             if isinstance(self._doc_fn, str):
#                 doc = self._doc_fn
#             else:
#                 doc = self._doc_fn()
#         return doc
#
#     def __prompt__(self) -> str:
#         if self._interface_fn is not None:
#             if isinstance(self._interface_fn, str):
#                 return self._interface_fn
#             else:
#                 return self._interface_fn()
#         doc = self._get_doc()
#         name = self._name
#         func_itf = self._func_itf
#         func_itf.name = name
#         func_itf.doc = doc
#         return func_itf.to_interface()
#
#     def __call__(self, *args, **kwargs) -> CommandTask:
#         real_kwargs = self._func_itf.prepare_kwargs(*args, **kwargs)
#         call = CommandCall(
#             name=self._name,
#             chan=self._chan,
#             cid=self._uuid_fn(),
#             args=args,
#             kwargs=kwargs,
#         )
#         return PyCommandTask(self._func, call, real_kwargs)
