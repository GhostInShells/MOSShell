import json
import asyncio
from typing import List, Optional, Dict, Union
import logging

from ghoshell_moss import ChannelEvent
from ghoshell_moss.core.duplex import Connection, ConnectionClosedError
from pydantic import ValidationError
from ghoshell_common.contracts import LoggerItf
from ghoshell_common.helpers import Timeleft
from ghoshell_moss.core.helpers import ThreadSafeEvent
from .constants import CHANNEL_EVENT_PREFIX, CHANNEL_EVENT_PREFIX_BYTES, NEWLINE_BYTES

__all__ = ['ParentProcessConnection']


class ParentProcessConnection(Connection):
    """
    父进程连接，负责启动子进程并建立基于 stdio 的双工通信
    """

    def __init__(
            self,
            command: Union[str, List[str]],
            cwd: Optional[str] = None,
            env: Optional[Dict[str, str]] = None,
            logger: LoggerItf | None = None,
    ):
        """
        初始化父进程连接

        Args:
            command: 要执行的命令，可以是字符串或列表
                    字符串：如 "python -m my_module"
                    列表：如 ["python", "-m", "my_module"]
            cwd: 子进程的工作目录
            env: 子进程的环境变量
            logger: 日志记录器
        """
        self._logger = logger or logging.getLogger('moss')
        self._command = command
        self._cwd = cwd
        self._env = env
        self._started = False
        self._closed = False

        # 子进程相关
        self._process: Optional[asyncio.subprocess.Process] = None
        self._process_pid: Optional[int] = None

        # 读写队列
        self._read_queue: Optional[asyncio.Queue[str | None]] = None
        self._write_queue: Optional[asyncio.Queue[str | None]] = None

        # 任务
        self._read_task: Optional[asyncio.Task] = None
        self._write_task: Optional[asyncio.Task] = None

        # 事件
        self._closed_event = ThreadSafeEvent()

    async def recv(self, timeout: float | None = None) -> ChannelEvent:
        """
        从子进程接收 ChannelEvent
        """
        if self.is_closed():
            raise ConnectionClosedError("Connection is closed")

        timeleft = Timeleft(timeout or 0.0)
        while True:
            try:
                # 从队列获取消息
                item = await asyncio.wait_for(
                    self._read_queue.get(),
                    timeout=timeleft.left() or None
                )
                if item is None:  # 关闭信号
                    raise ConnectionClosedError("Connection closed by child process")

                try:
                    data = json.loads(item)
                    return ChannelEvent(**data)
                except (ValidationError, json.decoder.JSONDecodeError) as e:
                    self._logger.error(f"Failed to parse ChannelEvent from child: {e}, data: {item}")
                    # 继续读取下一条
                    continue
            except asyncio.TimeoutError:
                raise ConnectionClosedError("Receive timeout")

    async def send(self, event: ChannelEvent) -> None:
        """
        向子进程发送 ChannelEvent
        """
        if self.is_closed():
            raise ConnectionClosedError("Connection is closed")

        try:
            data = json.dumps(event.dict())
            await self._write_queue.put(data)
        except (TypeError, AttributeError) as e:
            self._logger.error(f"Failed to serialize ChannelEvent: {e}, event: {event}")

    async def _recv_content(self) -> str:
        """
        从子进程的 stdout 读取内容
        使用约定的前缀开始，以 \n 结束
        """
        if not self._process or not self._process.stdout:
            raise ConnectionClosedError("Process or stdout is not available")

        while True:
            try:
                # 读取一行
                line_bytes = await self._process.stdout.readline()
                if not line_bytes:  # EOF
                    self._logger.info("Child process stdout closed (EOF)")
                    return None

                # 检查前缀
                if not line_bytes.startswith(CHANNEL_EVENT_PREFIX_BYTES):
                    # 非前缀行，记录警告并丢弃
                    self._logger.warning(
                        f"Received non-prefix line from child: {line_bytes.decode('utf-8', errors='ignore')}")
                    continue

                # 去除前缀和换行符
                content_bytes = line_bytes[len(CHANNEL_EVENT_PREFIX_BYTES):].rstrip(b'\n\r')
                return content_bytes.decode('utf-8')

            except asyncio.CancelledError:
                raise
            except Exception as e:
                self._logger.error(f"Error reading from child stdout: {e}")
                raise ConnectionClosedError(f"Error reading from child: {e}")

    async def _send_bytes(self, data: str) -> None:
        """
        向子进程的 stdin 发送数据
        使用约定的前缀开始，以 \n 结束
        """
        if not self._process or not self._process.stdin:
            raise ConnectionClosedError("Process or stdin is not available")

        try:
            # 构建完整消息：前缀 + JSON + 换行
            full_message = f"{CHANNEL_EVENT_PREFIX}{data}\n"
            self._process.stdin.write(full_message.encode('utf-8'))
            await self._process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as e:
            self._logger.error(f"Failed to send to child stdin (pipe broken): {e}")
            raise ConnectionClosedError("Pipe to child process broken")
        except Exception as e:
            self._logger.error(f"Error writing to child stdin: {e}")
            raise ConnectionClosedError(f"Error writing to child: {e}")

    async def _read_loop(self):
        """
        读取循环：从子进程 stdout 读取消息并放入队列
        """
        try:
            while not self.is_closed() and self._process and self._process.stdout:
                try:
                    content = await self._recv_content()
                    if content is None:  # EOF
                        break
                    await self._read_queue.put(content)
                except ConnectionClosedError:
                    break
                except Exception as e:
                    self._logger.error(f"Error in read loop: {e}")
                    # 短暂休眠后继续
                    await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            self._logger.debug("Read loop cancelled")
        finally:
            # 发送 None 表示读取结束
            if self._read_queue:
                await self._read_queue.put(None)
            self._logger.debug("Read loop ended")

    async def _write_loop(self):
        """
        写入循环：从队列获取消息并发送到子进程 stdin
        """
        try:
            while not self.is_closed() and self._process and self._process.stdin:
                try:
                    data = await self._write_queue.get()
                    if data is None:  # 关闭信号
                        break
                    await self._send_bytes(data)
                    self._write_queue.task_done()
                except ConnectionClosedError:
                    break
                except Exception as e:
                    self._logger.error(f"Error in write loop: {e}")
                    # 短暂休眠后继续
                    await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            self._logger.debug("Write loop cancelled")
        finally:
            # 发送 None 表示写入结束
            if self._write_queue:
                await self._write_queue.put(None)
            self._logger.debug("Write loop ended")

    async def _monitor_process(self):
        """
        监控子进程状态
        """
        try:
            if self._process:
                return_code = await self._process.wait()
                self._logger.info(f"Child process exited with code: {return_code}")

                # 进程退出，关闭连接
                if not self.is_closed():
                    await self.close()
        except asyncio.CancelledError:
            self._logger.debug("Process monitor cancelled")
        except Exception as e:
            self._logger.error(f"Error monitoring process: {e}")

    def is_closed(self) -> bool:
        """检查连接是否已关闭"""
        return self._closed or self._closed_event.is_set()

    def is_connected(self) -> bool:
        """检查连接是否已建立且活跃"""
        return (
                self._started
                and not self.is_closed()
                and self._process
                and self._process.returncode is None
        )

    async def close(self) -> None:
        """关闭连接，终止子进程"""
        if self.is_closed():
            return

        self._logger.debug("Closing parent process connection")
        self._closed = True
        self._closed_event.set()

        # 取消任务
        if self._read_task:
            self._read_task.cancel()
        if self._write_task:
            self._write_task.cancel()

        # 关闭进程
        if self._process:
            try:
                # 尝试优雅终止
                if self._process.returncode is None:
                    self._process.terminate()
                    try:
                        await asyncio.wait_for(self._process.wait(), timeout=2.0)
                    except asyncio.TimeoutError:
                        self._logger.warning("Child process did not terminate gracefully, killing")
                        self._process.kill()
                        await self._process.wait()
            except ProcessLookupError:
                pass  # 进程已经退出
            except Exception as e:
                self._logger.error(f"Error closing child process: {e}")
            finally:
                # 确保进程资源被清理
                if self._process.stdin:
                    self._process.stdin.close()
                self._process = None

        self._logger.debug("Parent process connection closed")

    async def start(self) -> None:
        """启动子进程并建立连接"""
        if self._started:
            return

        self._logger.debug(f"Starting child process with command: {self._command}")

        try:
            # 创建子进程
            if isinstance(self._command, str):
                # 使用 shell 执行命令
                self._process = await asyncio.create_subprocess_shell(
                    self._command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=None,  # 继承父进程的 stderr
                    cwd=self._cwd,
                    env=self._env,
                )
            else:
                # 直接执行命令列表
                self._process = await asyncio.create_subprocess_exec(
                    *self._command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=None,  # 继承父进程的 stderr
                    cwd=self._cwd,
                    env=self._env,
                )

            self._process_pid = self._process.pid
            self._logger.info(f"Child process started with PID: {self._process_pid}")

            # 初始化队列
            self._read_queue = asyncio.Queue()
            self._write_queue = asyncio.Queue()

            # 启动读写循环
            self._read_task = asyncio.create_task(self._read_loop())
            self._write_task = asyncio.create_task(self._write_loop())

            # 启动进程监控
            _ = asyncio.create_task(self._monitor_process())

            self._started = True
            self._logger.debug("Parent process connection started successfully")

        except Exception as e:
            self._logger.error(f"Failed to start child process: {e}")
            await self.close()
            raise
