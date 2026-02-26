import json
import asyncio
import threading
import queue
import sys
import os
import time
from typing import Optional
import logging

from ghoshell_moss import ChannelEvent
from ghoshell_moss.core.duplex import Connection, ConnectionClosedError
from pydantic import ValidationError
from ghoshell_common.contracts import LoggerItf
from ghoshell_common.helpers import Timeleft
from ghoshell_moss.core.helpers import ThreadSafeEvent
from .constants import CHANNEL_EVENT_PREFIX

__all__ = ['ChildProcessConnection']


class ChildProcessConnection(Connection):
    """
    子进程连接，从标准输入读取父进程消息，向标准输出写入响应
    使用线程处理阻塞IO，通过队列与异步循环交互
    """

    def __init__(
            self,
            logger: LoggerItf | None = None,
    ):
        """
        初始化子进程连接

        Args:
            logger: 日志记录器
        """
        self._logger = logger or logging.getLogger('moss')
        self._started = False
        self._closed = False

        # 异步循环和队列
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._read_queue: Optional[asyncio.Queue[str | None]] = None
        self._write_queue: Optional[asyncio.Queue[str | None]] = None

        # 线程相关
        self._stdin_reader_thread: Optional[threading.Thread] = None
        self._stdout_writer_thread: Optional[threading.Thread] = None

        # 事件
        self._closed_event = ThreadSafeEvent()

        # 父进程 ID（用于检测父进程是否变化）
        self._parent_process_id = os.getppid()

        # 线程安全队列（用于线程和异步循环之间通信）
        self._stdin_thread_queue: queue.Queue[str | None] = queue.Queue()
        self._stdout_thread_queue: queue.Queue[str | None] = queue.Queue()

        # 异步任务
        self._stdin_processor_task: Optional[asyncio.Task] = None
        self._stdout_processor_task: Optional[asyncio.Task] = None

    async def recv(self, timeout: float | None = None) -> ChannelEvent:
        """
        从父进程接收 ChannelEvent
        """
        if self.is_closed():
            raise ConnectionClosedError("Connection is closed")

        timeleft = Timeleft(timeout or 0.0)
        while True:
            try:
                # 从异步队列获取消息
                item = await asyncio.wait_for(
                    self._read_queue.get(),
                    timeout=timeleft.left() or None
                )
                if item is None:  # 关闭信号
                    raise ConnectionClosedError("Connection closed by parent process")

                try:
                    data = json.loads(item)
                    return ChannelEvent(**data)
                except (ValidationError, json.decoder.JSONDecodeError) as e:
                    self._logger.error(f"Failed to parse ChannelEvent from parent: {e}, data: {item}")
                    # 继续读取下一条
                    continue
            except asyncio.TimeoutError:
                raise ConnectionClosedError("Receive timeout")

    async def send(self, event: ChannelEvent) -> None:
        """
        向父进程发送 ChannelEvent
        """
        if self.is_closed():
            raise ConnectionClosedError("Connection is closed")

        try:
            data = json.dumps(event.dict())
            await self._write_queue.put(data)
        except (TypeError, AttributeError) as e:
            self._logger.error(f"Failed to serialize ChannelEvent: {e}, event: {event}")
        except Exception as e:
            self._logger.error(f"Unexpected error in send: {e}")
            raise

    def _read_from_stdin_thread(self):
        """
        线程函数：从 stdin 读取消息
        阻塞读取，直到遇到 EOF 或连接关闭
        """
        self._logger.debug("Stdin reader thread started")

        try:
            while not self.is_closed():
                # 检查父进程是否变化
                try:
                    current_ppid = os.getppid()
                    if current_ppid != self._parent_process_id:
                        self._logger.warning(f"Parent process changed from {self._parent_process_id} to {current_ppid}")
                        self._parent_process_id = current_ppid
                        # 父进程变化，可以视为连接中断
                        break
                except (OSError, AttributeError):
                    # 在某些情况下可能无法获取父进程ID
                    pass

                try:
                    # 读取一行（阻塞操作）
                    line = sys.stdin.readline()
                    if not line:  # EOF
                        self._logger.info("Stdin closed (EOF)")
                        break

                    # 检查前缀
                    if not line.startswith(CHANNEL_EVENT_PREFIX):
                        # 非前缀行，记录警告并丢弃
                        self._logger.warning(f"Received non-prefix line from parent: {line.rstrip()}")
                        continue

                    # 去除前缀和换行符
                    content = line[len(CHANNEL_EVENT_PREFIX):].rstrip('\n\r')

                    # 放入线程队列，供异步循环消费
                    self._stdin_thread_queue.put(content)

                except (EOFError, IOError, OSError) as e:
                    self._logger.error(f"Error reading from stdin: {e}")
                    break
                except Exception as e:
                    self._logger.error(f"Unexpected error in stdin reader: {e}")
                    # 短暂休眠后继续，避免CPU忙循环
                    time.sleep(0.01)

        except Exception as e:
            self._logger.error(f"Stdin reader thread error: {e}")
        finally:
            # 放入 None 表示结束
            self._stdin_thread_queue.put(None)
            self._logger.debug("Stdin reader thread ended")

    def _write_to_stdout_thread(self):
        """
        线程函数：向 stdout 写入消息
        从队列获取消息并写入标准输出
        """
        self._logger.debug("Stdout writer thread started")

        try:
            while not self.is_closed():
                try:
                    # 从线程队列获取消息（阻塞操作，带超时）
                    data = self._stdout_thread_queue.get(timeout=0.1)
                    if data is None:  # 关闭信号
                        break

                    # 构建完整消息：前缀 + JSON + 换行
                    full_message = f"{CHANNEL_EVENT_PREFIX}{data}\n"
                    sys.stdout.write(full_message)
                    sys.stdout.flush()  # 确保立即发送

                except queue.Empty:
                    # 队列为空，继续检查关闭状态
                    continue
                except (IOError, BrokenPipeError, OSError) as e:
                    self._logger.error(f"Error writing to stdout: {e}")
                    break
                except Exception as e:
                    self._logger.error(f"Unexpected error in stdout writer: {e}")
                    # 短暂休眠后继续，避免CPU忙循环
                    time.sleep(0.01)

        except Exception as e:
            self._logger.error(f"Stdout writer thread error: {e}")
        finally:
            # 确保刷新缓冲区
            try:
                sys.stdout.flush()
            except:
                pass
            self._logger.debug("Stdout writer thread ended")

    async def _process_stdin_queue(self):
        """
        异步任务：处理从线程队列来的 stdin 消息
        将线程队列中的消息转移到异步队列中
        """
        try:
            while not self.is_closed():
                try:
                    # 从线程队列获取消息（非阻塞）
                    try:
                        item = self._stdin_thread_queue.get_nowait()
                    except queue.Empty:
                        # 队列为空，短暂休眠后继续
                        await asyncio.sleep(0.01)
                        continue

                    if item is None:  # 关闭信号
                        break

                    # 放入异步队列供 recv 方法消费
                    await self._read_queue.put(item)

                except Exception as e:
                    self._logger.error(f"Error processing stdin queue: {e}")
                    await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            self._logger.debug("Stdin queue processor cancelled")
        finally:
            # 放入 None 表示结束
            if self._read_queue:
                await self._read_queue.put(None)
            self._logger.debug("Stdin queue processor ended")

    async def _process_stdout_queue(self):
        """
        异步任务：处理要发送到线程队列的 stdout 消息
        将异步队列中的消息转移到线程队列中
        """
        try:
            while not self.is_closed():
                try:
                    # 从异步队列获取消息
                    data = await asyncio.wait_for(
                        self._write_queue.get(),
                        timeout=0.1
                    )
                    if data is None:  # 关闭信号
                        break

                    # 放入线程队列供写入线程消费
                    self._stdout_thread_queue.put(data)
                    self._write_queue.task_done()

                except asyncio.TimeoutError:
                    # 超时，继续检查关闭状态
                    continue
                except Exception as e:
                    self._logger.error(f"Error processing stdout queue: {e}")
                    await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            self._logger.debug("Stdout queue processor cancelled")
        finally:
            # 放入 None 表示结束
            self._stdout_thread_queue.put(None)
            self._logger.debug("Stdout queue processor ended")

    def is_closed(self) -> bool:
        """
        检查连接是否已关闭
        """
        return self._closed or self._closed_event.is_set()

    def is_connected(self) -> bool:
        """
        检查连接是否已建立且活跃
        通过检查父进程 ID 是否变化来判断
        """
        if self.is_closed():
            return False

        try:
            current_ppid = os.getppid()
            return current_ppid == self._parent_process_id
        except (OSError, AttributeError):
            # 在某些情况下可能无法获取父进程ID
            return not self.is_closed()

    async def close(self) -> None:
        """
        关闭连接，清理资源
        """
        if self.is_closed():
            return

        self._logger.debug("Closing child process connection")
        self._closed = True
        self._closed_event.set()

        # 取消异步任务
        if self._stdin_processor_task:
            self._stdin_processor_task.cancel()
        if self._stdout_processor_task:
            self._stdout_processor_task.cancel()

        # 向队列发送关闭信号
        if self._stdin_thread_queue:
            self._stdin_thread_queue.put(None)
        if self._stdout_thread_queue:
            self._stdout_thread_queue.put(None)

        # 等待线程结束
        if self._stdin_reader_thread and self._stdin_reader_thread.is_alive():
            self._stdin_reader_thread.join(timeout=2.0)
            if self._stdin_reader_thread.is_alive():
                self._logger.warning("Stdin reader thread did not terminate gracefully")

        if self._stdout_writer_thread and self._stdout_writer_thread.is_alive():
            self._stdout_writer_thread.join(timeout=2.0)
            if self._stdout_writer_thread.is_alive():
                self._logger.warning("Stdout writer thread did not terminate gracefully")

        self._logger.debug("Child process connection closed")

    async def start(self) -> None:
        """
        启动连接，开始读取 stdin 和写入 stdout
        """
        if self._started:
            return

        self._logger.debug("Starting child process connection")

        try:
            # 获取当前事件循环
            self._loop = asyncio.get_running_loop()

            # 初始化队列
            self._read_queue = asyncio.Queue()
            self._write_queue = asyncio.Queue()

            # 启动 stdin 读取线程
            self._stdin_reader_thread = threading.Thread(
                target=self._read_from_stdin_thread,
                name="moss-stdin-reader",
                daemon=True
            )
            self._stdin_reader_thread.start()

            # 启动 stdout 写入线程
            self._stdout_writer_thread = threading.Thread(
                target=self._write_to_stdout_thread,
                name="moss-stdout-writer",
                daemon=True
            )
            self._stdout_writer_thread.start()

            # 启动异步队列处理器
            self._stdin_processor_task = asyncio.create_task(self._process_stdin_queue())
            self._stdout_processor_task = asyncio.create_task(self._process_stdout_queue())

            self._started = True
            self._logger.debug("Child process connection started successfully")

        except Exception as e:
            self._logger.error(f"Failed to start child process connection: {e}")
            await self.close()
            raise
