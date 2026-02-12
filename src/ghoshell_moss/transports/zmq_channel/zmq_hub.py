import asyncio
import logging
import os
import signal
import sys
import time
from contextlib import AsyncExitStack
from typing import Optional

import psutil
from ghoshell_common.contracts import LoggerItf
from pydantic import BaseModel, Field

from ghoshell_moss import CommandErrorCode
from ghoshell_moss.core import PyChannel
from ghoshell_moss.transports.zmq_channel.zmq_channel import ZMQChannelProxy

__all__ = [
    "ZMQChannelHub",
    "ZMQChannelProxy",
    "ZMQHubConfig",
    "ZMQProxyConfig",
]


class ManagedProcess:
    """
    异步进程资源管理器。
    实现上下文管理器协议，确保退出时进程一定被关闭。
    """

    def __init__(self, name: str, script_path: str, env: dict, logger: logging.Logger):
        self.name = name
        self.script_path = script_path
        self.env = env
        self.logger = logger

        self.process: Optional[asyncio.subprocess.Process] = None
        self.start_time: float = 0.0
        self._monitor_task: Optional[asyncio.Task] = None

    async def __aenter__(self):
        self.logger.info("--- 启动子进程: %s", self.name)
        self.start_time = time.time()

        # 启动子进程
        # Unix下使用 start_new_session=True 创建进程组，方便 killpg 一把全杀
        creationflags = 0
        start_new_session = False

        if sys.platform == "win32":
            # Windows 特定设置
            creationflags = asyncio.subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            start_new_session = True

        self.process = await asyncio.create_subprocess_exec(
            sys.executable,
            self.script_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self.env,
            start_new_session=start_new_session,
            creationflags=creationflags,
        )

        # 启动后台日志监控任务
        self._monitor_task = asyncio.create_task(self._monitor_logs())
        return self

    async def _monitor_logs(self):
        """后台任务：读取并打印子进程日志"""

        async def read_stream(stream, level):
            while True:
                line = await stream.readline()
                if not line:
                    break
                decoded = line.decode("utf-8", errors="ignore").rstrip()
                self.logger.log(level, "[%s] %s", self.name, decoded)

        try:
            await asyncio.gather(
                read_stream(self.process.stdout, logging.INFO), read_stream(self.process.stderr, logging.ERROR)
            )
        except asyncio.CancelledError:
            pass
        except Exception:
            self.logger.exception("监控子进程 %s 日志时出错", self.name)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """退出上下文时，确保清理进程"""
        if self._monitor_task:
            self._monitor_task.cancel()

        if not self.process or self.process.returncode is not None:
            return

        self.logger.info("--- 正在关闭子进程: %s (PID: %s)", self.name, self.process.pid)

        try:
            # 1. 尝试优雅关闭 (SIGTERM / CTRL_BREAK)
            if sys.platform == "win32":
                self.process.terminate()
            else:
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass

            # 等待退出
            try:
                await asyncio.wait_for(self.process.wait(), timeout=3.0)
                self.logger.info("子进程 %s 已优雅退出", self.name)
            except asyncio.TimeoutError:
                # 2. 强制关闭 (SIGKILL)
                self.logger.warning("子进程 %s 响应超时，正在强制关闭...", self.name)
                if sys.platform == "win32":
                    self.process.kill()
                else:
                    try:
                        os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                await self.process.wait()
                self.logger.info("子进程 %s 已强制关闭", self.name)

        except Exception:
            self.logger.exception("关闭子进程 %s 时发生错误", self.name)

    @property
    def pid(self) -> Optional[int]:
        return self.process.pid if self.process else None

    @property
    def is_running(self) -> bool:
        if not self.process:
            return False
        if self.process.returncode is not None:
            return False
        # 二次检查
        try:
            return psutil.pid_exists(self.process.pid)
        except:
            return False


class ZMQProxyConfig(BaseModel):
    script: str = Field(description="the script filename of the zmq channel provider")
    description: str = Field(description="the description of the zmq channel provider")
    address: str = Field(default="", description="the address of the zmq channel provider")


class ZMQHubConfig(BaseModel):
    name: str = Field(description="name of the hub")
    description: str = Field(description="description of the hub")
    root_dir: str = Field(description="所有子进程脚本所在的目录地址, 用来和 proxy config.script 获取运行路径.")
    proxies: dict[str, ZMQProxyConfig] = Field(
        default_factory=dict, description="the zmq channel provider configurations, from name to config"
    )


class ZMQChannelHub:
    """
    基于 AsyncExitStack 重构的 ZMQ Channel Hub。
    确保子进程生命周期安全，无僵尸进程。
    """

    def __init__(self, config: ZMQHubConfig, logger: LoggerItf | None = None):
        self._config = config
        self._logger = logger or logging.getLogger(__name__)

        # 核心：主资源栈，管理 Hub 的生命周期
        self._main_exit_stack = AsyncExitStack()

        # 状态管理：映射 channel_name -> (ChildStack, ManagedProcessInstance)
        # 这样我们可以单独关闭某一个 channel
        self._active_channels: dict[str, tuple[AsyncExitStack, ManagedProcess]] = {}

    def channel_description(self) -> str:
        """生成通道描述，包括所有已配置的子通道及其状态"""
        description = self._config.description
        config_lines = ["已配置的子通道："]

        for name, config in self._config.proxies.items():
            status = "❌ 未运行"

            # 检查运行状态
            if name in self._active_channels:
                _, managed_proc = self._active_channels[name]
                if managed_proc.is_running:
                    runtime = time.time() - managed_proc.start_time
                    runtime_str = self._format_runtime(runtime)
                    status = f"✅ 运行中 (PID: {managed_proc.pid}, 运行时间: {runtime_str})"
                else:
                    # 进程对象存在但已退出（异常情况）
                    status = "⚠️  已退出"

            channel_line = f"- {name}: {status}"
            desc_line = f"  描述: {config.description}"
            config_lines.extend([channel_line, desc_line, ""])

        config_section = "\n".join(config_lines)
        return "\n\n".join([description, config_section])

    def _format_runtime(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}秒"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}分钟"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}小时"

    async def connect_or_reconnect_sub_channel_process(self, name: str, config: ZMQProxyConfig) -> None:
        """启动或重启子进程"""

        # 1. 如果已存在，先关闭旧的
        if name in self._active_channels:
            await self.terminate_sub_channel_process(name)

        # 2. 准备路径和环境
        script_path = os.path.join(self._config.root_dir, config.script)
        if not os.path.exists(script_path):
            raise CommandErrorCode.NOT_FOUND.error(f"子 Channel {name} 脚本不存在: {script_path}")

        env = os.environ.copy()
        env["MOSHELL_PARENT_PID"] = str(os.getpid())

        # 3. 创建一个新的上下文栈，用于单独管理这个子进程
        # 将这个子栈压入主栈，确保 Hub 关闭时也能关闭它
        child_stack = await self._main_exit_stack.enter_async_context(AsyncExitStack())

        try:
            # 4. 创建并启动进程资源
            managed_proc = ManagedProcess(name, script_path, env, self._logger)
            await child_stack.enter_async_context(managed_proc)

            # 5. 记录状态
            self._active_channels[name] = (child_stack, managed_proc)

        except Exception:
            self._logger.exception("启动子通道 %s 失败", name)
            # 如果启动失败，立即清理子栈
            await child_stack.aclose()
            raise

    async def terminate_sub_channel_process(self, name: str) -> None:
        """关闭单个子 Channel"""
        if name not in self._active_channels:
            return

        self._logger.info("正在终止子通道: %s", name)
        child_stack, _ = self._active_channels.pop(name)

        # 关闭子栈会触发 ManagedProcess.__aexit__
        await child_stack.aclose()

    async def close(self):
        """关闭整个 Hub，清理所有子进程"""
        self._logger.info("正在关闭 Hub 并清理所有子进程...")
        # 关闭主栈会自动以 LIFO 顺序关闭所有注册的子栈
        await self._main_exit_stack.aclose()
        self._active_channels.clear()
        self._logger.info("所有子进程已清理完成")

    def is_sub_channel_running(self, name: str) -> bool:
        if name not in self._active_channels:
            return False
        _, proc = self._active_channels[name]
        return proc.is_running

    # --- 以下为 PyChannel 交互逻辑 ---

    async def start_sub_channel(self, name: str, timeout: float = 15.0) -> str:
        """PyChannel Command: 开启子节点"""
        if not name:
            raise CommandErrorCode.VALUE_ERROR.error("channel name cannot be empty")
        proxy_conf = self._config.proxies.get(name)
        if proxy_conf is None:
            raise CommandErrorCode.VALUE_ERROR.error(f"sub channel {name} not registered")

        await self.connect_or_reconnect_sub_channel_process(name, proxy_conf)

        # 等待 ZMQ 连接就绪
        current_chan = PyChannel.get_from_context()
        sub_channel = current_chan.get_channel(name)
        try:
            await asyncio.wait_for(sub_channel.broker.wait_connected(), timeout=timeout)
        except asyncio.TimeoutError:
            # 如果连接超时，应该把刚启动的进程杀掉，避免残留
            await self.terminate_sub_channel_process(name)
            raise CommandErrorCode.TIMEOUT.error(f"start channel {name} timeout")

        return ""

    async def close_channel(self, name: str, timeout: float = 5.0) -> str:
        """PyChannel Command: 关闭子节点"""
        if not name:
            raise CommandErrorCode.VALUE_ERROR.error("channel name cannot be empty")
        try:
            await asyncio.wait_for(self.terminate_sub_channel_process(name), timeout=timeout)
        except asyncio.TimeoutError:
            raise CommandErrorCode.TIMEOUT.error(f"close channel {name} timeout")
        except Exception as e:
            raise CommandErrorCode.UNKNOWN_CODE.error(f"close channel {name} error: {e}")

        return f"Channel {name} closed."

    def as_channel(self) -> PyChannel:
        _channel = PyChannel(
            name=self._config.name,
            description=self._config.description,
            block=True,
        )

        for name, config in self._config.proxies.items():
            # 如果config没有指定address，在路径下创建socket文件作为通信地址
            if not config.address:
                sock_path = os.path.join(self._config.root_dir, config.script + ".sock")
                if os.path.exists(sock_path):
                    os.remove(sock_path)
                config.address = f"ipc://{sock_path}"

            sub_channel = ZMQChannelProxy(
                name=name,
                address=config.address,
                logger=self._logger,
            )
            _channel.import_channels(sub_channel)

        _channel.build.with_description()(self.channel_description)
        _channel.build.command()(self.start_sub_channel)
        _channel.build.command()(self.close_channel)

        # 注册异步关闭钩子
        _channel.build.on_stop(self.close)

        return _channel
