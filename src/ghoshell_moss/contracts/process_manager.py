"""子进程管理器 contract — shell / exec / task / background 四模式进程治理.

ProcessManager 是 MOSS 的进程基础设施. 三层设计:

1. ManagedProcess  — 治理包装.  承诺: 生命周期随 manager, 调用记录入栈, 元信息, 可监控.
                     其余透传 asyncio 子进程 API.
2. ProcessTask     — 一次性任务.  在 ManagedProcess 之上承诺: 输出 buffer + 落 tmp 文件,
                     可二次查询. 执行完即止.
3. BackgroundTask  — 持续性任务. 在 ProcessTask 之上承诺: loop/once/on_prompt 调度,
                     sleep 间隔, refresh 生命周期.

三层对应三种使用姿态:
- "启动一个进程, 不用管输出"          → execute / shell
- "启动一个进程, 把输出存起来, 之后读" → execute_task / shell_task
- "持续运行的后台任务, 随 refresh 更新" → execute_task(background_run=('loop', 0))
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, Field
from typing_extensions import Self

__all__ = [
    "ProcessManager",
    "ProcessMeta",
    "ProcessTask", "BackgroundTask", "ManagedProcess",
    'BackgroundRunType', 'LoopTimes', 'ErrorInfo',
]

# -- 类型字面量 --

BackgroundRunType = Literal["once", "loop", "on_prompt"]
"""后台任务的执行策略.

- ``once``:      执行一次, 冻结. 用于静态快照.
- ``loop``:      持续循环. loop=0 无限, loop=N 执行 N 次.
- ``on_prompt``: 每轮 refresh 时执行一次. 用于需要每帧更新的状态.
                 对应 ProjectManager 的 RefreshMode.
"""

LoopTimes = int
"""循环次数. 0 = 无限循环. >0 = 执行指定次数后停止."""

ErrorInfo = str
"""kill / killpg 等操作返回的错误信息. None 表示成功."""


# -- 元信息 --

class ProcessMeta(BaseModel):
    """子进程的元信息. 所有通过 ProcessManager 启动的进程都有此记录."""

    index: int = Field(
        description="在 ProcessManager 中的唯一序号",
    )
    pid: int = Field(
        description="操作系统进程 ID",
    )
    exit_code: int | None = Field(
        default=None,
        description="退出码. None 表示尚未退出",
    )
    command: str = Field(
        description="命令行字符串 (shell 模式) 或 argv 拼接 (exec 模式)",
    )
    name: str = Field(
        description="进程名称, 用于日志和监控",
    )
    description: str = Field(
        default="",
        description="进程用途的简短描述",
    )
    cwd: str = Field(
        description="工作目录",
    )
    with_os_env: bool = Field(
        default=True,
        description="是否继承 OS 环境变量",
    )
    extra_env: dict[str, str] = Field(
        default_factory=dict,
        description="追加的环境变量",
    )
    start_new_session: bool = Field(
        default=True,
        description="是否创建新会话 (setsid). 平台相关, 不支持时降级",
    )
    created: float = Field(
        default_factory=time.time,
        description="启动时间戳",
    )
    updated: float = Field(
        default_factory=time.time,
        description="最后状态更新时间戳",
    )
    task_id: str | None = Field(
        default=None,
        description="关联的 ProcessTask id",
    )
    background_task_id: str | None = Field(
        default=None,
        description="关联的 BackgroundTask id",
    )


# -- 三层抽象 --

# Layer 1: ManagedProcess

@dataclass
class ManagedProcess:
    """子进程的治理包装.

    比裸 asyncio.subprocess.Process 多出:
    - 生命周期随 ProcessManager (统一启停, 整体清空)
    - 调用记录入栈 (executed 历史)
    - 元信息可查询 (ProcessMeta)
    - 进程标识 (name / description)

    其余能力 (wait / send_signal / terminate / kill / pid / returncode)
    直接通过 ``process`` 字段使用, 和普通 asyncio 子进程一致.
    """

    meta: ProcessMeta
    """元信息."""

    process: asyncio.subprocess.Process
    """底层 asyncio 子进程. 可直接操作."""


# Layer 2: ProcessTask

class ProcessTask(ABC):
    """一次性子进程任务 — 输出自动落 tmp 文件.

    在 ManagedProcess 之上承诺:
    - stdout/stderr 一路 buffer 到内存 (有限窗口), 一路写入文件 (完整)
    - 执行完毕后文件保留, 可二次查询
    - 通过 send_signal / terminate / kill / wait 管理生命周期
    """

    @property
    @abstractmethod
    def id(self) -> str:
        """任务唯一标识."""
        ...

    @property
    @abstractmethod
    def meta(self) -> ProcessMeta:
        """进程元信息."""
        ...

    @abstractmethod
    def stdout_buffer(self, *, offset: int = 0, limit: int = 0) -> str:
        """内存中的 stdout 缓存窗口. limit=0 表示全部."""
        ...

    @abstractmethod
    def stderr_buffer(self, *, offset: int = 0, limit: int = 0) -> str:
        """内存中的 stderr 缓存窗口."""
        ...

    @property
    @abstractmethod
    def stdout_file(self) -> Path:
        """stdout 完整输出文件路径."""
        ...

    @property
    @abstractmethod
    def stderr_file(self) -> Path:
        """stderr 完整输出文件路径."""
        ...

    @property
    @abstractmethod
    def is_alive(self) -> bool:
        """进程是否仍在运行."""
        ...

    @property
    @abstractmethod
    def return_code(self) -> int | None:
        """退出码. None 表示尚未退出."""
        ...

    @abstractmethod
    def send_signal(self, signal: int) -> None:
        """发送信号."""
        ...

    @abstractmethod
    def terminate(self) -> None:
        """发送 SIGTERM."""
        ...

    @abstractmethod
    def kill(self) -> None:
        """发送 SIGKILL."""
        ...

    @abstractmethod
    async def wait(self) -> None:
        """阻塞到进程退出."""
        ...


# Layer 3: BackgroundTask

class BackgroundTask(ABC):
    """持续性后台任务 — 按策略反复执行, 持有最后一次 ProcessTask.

    在 ProcessTask 之上承诺:
    - 按 BackgroundRunType 自动调度
    - sleep 间隔控制执行频率
    - 随 ProcessManager.refresh_background() 进入下一轮
    - 生命周期可独立启停
    """

    @property
    @abstractmethod
    def id(self) -> str:
        """任务唯一标识."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """任务名称."""
        ...

    @property
    @abstractmethod
    def type(self) -> BackgroundRunType:
        """执行策略."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """任务用途描述."""
        ...

    @property
    @abstractmethod
    def loop(self) -> int:
        """总循环次数. 0 = 无限."""
        ...

    @property
    @abstractmethod
    def executed(self) -> int:
        """已完成次数."""
        ...

    @property
    @abstractmethod
    def sleep(self) -> float:
        """两次执行之间的间隔秒数. loop 模式下生效."""
        ...

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """当前是否正在执行 (有活跃的 ProcessTask)."""
        ...

    @property
    @abstractmethod
    def last(self) -> ProcessTask | None:
        """最近一次执行的任务. None 表示尚未执行过."""
        ...

    @abstractmethod
    def last_stdout(self, *, offset: int = 0, limit: int = 0) -> str:
        """最近一次执行的 stdout 窗口."""
        ...

    @abstractmethod
    def last_stderr(self, *, offset: int = 0, limit: int = 0) -> str:
        """最近一次执行的 stderr 窗口."""
        ...


# -- ProcessManager --

class ProcessManager(ABC):
    """子进程管理器.

    统一管理 exec / shell 两种启动模式, Process / ProcessTask / BackgroundTask
    三层抽象. 持有所有进程运行状态, 提供进程组级别的生命周期治理.

    使用方式::

        # async with ProcessManagerImpl(root=...) as pm:
        #     proc = await pm.execute("echo", "hello")
        #     await proc.process.wait()
        #     task = await pm.execute_task("find", ".", output_file=Path("/tmp/out.txt"))
        #     print(task.stdout_buffer())

    ``root`` / ``pwd`` / ``cd`` 提供目录作用域 — 所有 spawn 的进程默认
    继承 pwd, 约束在 root 子树内.
    """

    # -- 目录作用域 --

    @property
    @abstractmethod
    def root(self) -> Path:
        """根目录. 所有路径操作相对于此."""
        ...

    @property
    @abstractmethod
    def pwd(self) -> Path:
        """当前工作目录."""
        ...

    @abstractmethod
    def cd(self, path: str, *, from_pwd: bool = True) -> Path:
        """切换工作目录.

        from_pwd=True: path 相对于 pwd 解析.
        返回切换后的绝对路径.
        """
        ...

    # -- 进程查询 --

    @abstractmethod
    def executing(self) -> dict[int, ProcessMeta]:
        """正在运行的进程. key = index."""
        ...

    @abstractmethod
    def get_executing(self, index: int) -> ManagedProcess | None:
        """按 index 获取运行中的进程."""
        ...

    @abstractmethod
    def executed(self) -> list[ProcessMeta]:
        """已结束的进程历史. 保留 maxsize 条, FIFO."""
        ...

    # -- Layer 1: 执行 (exec / shell) --

    @abstractmethod
    async def execute(
            self,
            *args: str,
            name: str | None = None,
            description: str | None = None,
            cwd: str | Path | None = None,
            extra_env: dict | None = None,
            stdin: int | None = None,
            stdout: int | None = None,
            stderr: int | None = None,
            start_new_session: bool = True,
            with_os_env: bool = True,
            **kwargs,
    ) -> ManagedProcess:
        """exec 模式启动子进程.

        *args 直接传递给 asyncio.create_subprocess_exec.
        不经过 shell 解析. 推荐用于大多数场景.
        """
        ...

    @abstractmethod
    async def shell(
            self,
            cmd: str,
            *,
            name: str | None = None,
            description: str | None = None,
            cwd: str | Path | None = None,
            extra_env: dict | None = None,
            stdin: int | None = None,
            stdout: int | None = None,
            stderr: int | None = None,
            start_new_session: bool = True,
            with_os_env: bool = True,
            **kwargs,
    ) -> ManagedProcess:
        """shell 模式启动子进程.

        cmd 通过 shell 解析. 仅在需要管道 / 重定向 / glob 等 shell 特性时使用.
        """
        ...

    # -- Layer 2: 一次性任务 (exec / shell + 输出管理) --

    @abstractmethod
    async def execute_task(
            self,
            *args: str,
            name: str | None = None,
            description: str | None = None,
            cwd: str | Path | None = None,
            extra_env: dict | None = None,
            output_file: Path | None = None,
            with_os_env: bool = True,
            start_new_session: bool = True,
            background_run: tuple[BackgroundRunType, LoopTimes] | None = None,
            sleep: float = 0.0,
            callback: Callable[[ProcessTask], None] | None = None,
            buffer_lines: int = 100,
            **kwargs,
    ) -> ProcessTask:
        """exec 模式 + 输出管理.

        stdout/stderr 同时 buffer (内存有限窗口) + 写入文件 (完整).
        output_file 指定输出路径, 为空则使用 tmp 目录下的自动命名文件.

        background_run 不为空时, 任务注册为 BackgroundTask:
        - 1-tuple ``(type,)``: 使用默认 loop 值 (once=1, loop=0=无限, on_prompt=1)
        - 2-tuple ``(type, loop)``: loop=0 无限, loop=N 执行 N 次
        """
        ...

    @abstractmethod
    async def shell_task(
            self,
            cmd: str,
            *,
            name: str | None = None,
            description: str | None = None,
            cwd: str | Path | None = None,
            extra_env: dict | None = None,
            output_file: Path | None = None,
            with_os_env: bool = True,
            start_new_session: bool = True,
            background_run: tuple[BackgroundRunType, LoopTimes] | None = None,
            sleep: float = 0.0,
            callback: Callable[[ProcessTask], None] | None = None,
            buffer_lines: int = 100,
            **kwargs,
    ) -> ProcessTask:
        """shell 模式 + 输出管理. 同 execute_task, 但 cmd 通过 shell 解析."""
        ...

    # -- Layer 3: 后台任务管理 --

    @abstractmethod
    def background_tasks(self) -> list[BackgroundTask]:
        """当前活跃的后台任务."""
        ...

    @abstractmethod
    async def refresh_background(self) -> list[BackgroundTask]:
        """触发一轮后台任务刷新.

        on_prompt 类任务在此轮执行.
        返回本轮更新的任务列表.
        """
        ...

    @abstractmethod
    async def stop_background_task(self, task_id: str) -> bool:
        """停止指定后台任务. 杀掉当前运行中的进程 (如果有). 返回是否成功."""
        ...

    @abstractmethod
    def stopped_background_tasks(self) -> list[BackgroundTask]:
        """已停止的后台任务历史."""
        ...

    # -- 进程组治理 --

    @abstractmethod
    def kill(self, pid: int) -> ErrorInfo | None:
        """kill 单个进程 (SIGKILL). 返回 None 表示成功."""
        ...

    @abstractmethod
    def killpg(self, process_group: int, signal: int) -> ErrorInfo | None:
        """kill 进程组.

        覆盖 start_new_session 后的所有未分离子孙进程.
        主动 setsid/setpgid 脱离的守护进程不受影响 — 那是子进程自己的责任.

        平台:
        - Linux/macOS: os.killpg
        - Windows: 平台不支持进程组, 降级为 kill 单个 pid.
          不抛异常, 返回 Error 字符串描述降级行为.
        """
        ...

    # -- 生命周期 --

    @abstractmethod
    async def __aenter__(self) -> Self:
        """启动 ProcessManager. 初始化 tmp 目录, 恢复历史状态等."""
        ...

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """关闭 ProcessManager. 终止所有活跃进程 (先 SIGINT, 超时 SIGKILL)."""
        ...
