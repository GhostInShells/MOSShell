"""Subprocesses contract — 子进程的纯机制层治理.

Subprocesses 的一句话承诺: "spawn 并治理一组不比 owner 活得久的子进程".

三层结构:

1. SubprocessFacade — 纯 spawn (execute/shell). 消费者 spawn 面 — 不含查询/信号/生命周期,
                     结构性杜绝误关共享实例.
2. Subprocesses     — SubprocessFacade + 查询 (executing/get/executed) + 信号
                     (kill/killpg) + 生命周期 (async with). owner 治理面 — 台账归此层.
3. ManagedProcess   — 子进程富句柄. meta / process / output / stop / add_done_callback.

输出捕获不是独立的任务抽象, 是 spawn 时的可选参数: ``capture=CaptureSpec(...)``.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field
from typing_extensions import Self

from ghoshell_moss.message.message import Additional

__all__ = [
    "CaptureSpec",
    "ErrorInfo",
    "ManagedProcess",
    "ProcessMeta",
    "ProcessOutput",
    "SubprocessFacade",
    "Subprocesses",
]

ErrorInfo = str
"""kill / killpg 等操作返回的错误信息. None 表示成功."""


# -- 元信息 --

class ProcessMeta(BaseModel):
    """子进程的元信息. 所有通过 Subprocesses 启动的进程都有此记录."""

    index: int = Field(
        description="在 Subprocesses 中的唯一序号",
    )
    pid: int = Field(
        description="操作系统进程 ID",
    )
    pgid: int | None = Field(
        default=None,
        description="进程组 ID. start_new_session=True 时等于 pid; 平台不支持时为 None",
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
        description="工作目录 (绝对路径)",
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
    additional: Additional = Field(
        default=None,
        description="Addition 挂载点 (HasAdditional). 调用方可绑强类型附加数据"
                    " (message.AdditionType), 随 meta 流经 on_exit 回调等通路.",
    )


# -- 输出捕获 --

class CaptureSpec(BaseModel):
    """spawn 时的输出捕获声明. 传入 execute/shell 的 capture 参数.

    声明后 stdout/stderr 被接管 (PIPE + 后台 drain), 通过
    ``ManagedProcess.output`` 读取. 与手动传 stdout/stderr 互斥.
    """

    buffer_lines: int = Field(
        default=100,
        description="内存 tail 窗口行数. 0 = 不维护内存窗口",
    )
    stdout_file: Path | None = Field(
        default=None,
        description="stdout 完整输出落盘路径. None = 由实现分配 tmp 文件",
    )
    stderr_file: Path | None = Field(
        default=None,
        description="stderr 完整输出落盘路径. None = 由实现分配 tmp 文件",
    )


class ProcessOutput(ABC):
    """一个子进程的输出视图 — 内存 tail 窗口 + 完整落盘文件.

    进程退出后对象保留, 可继续读 (二次查询).
    """

    @abstractmethod
    def stdout(self, *, offset: int = 0, limit: int = 0) -> str:
        """内存中的 stdout tail 窗口. limit=0 表示窗口内全部."""
        ...

    @abstractmethod
    def stderr(self, *, offset: int = 0, limit: int = 0) -> str:
        """内存中的 stderr tail 窗口."""
        ...

    @property
    @abstractmethod
    def stdout_file(self) -> Path | None:
        """stdout 完整输出文件路径. 未落盘时为 None."""
        ...

    @property
    @abstractmethod
    def stderr_file(self) -> Path | None:
        """stderr 完整输出文件路径. 未落盘时为 None."""
        ...

    @abstractmethod
    async def wait_drained(self) -> None:
        """阻塞到输出流全部排空 (进程退出且 drain 完成). 之后读取是完整的."""
        ...


# -- 子进程富句柄 --

class ManagedProcess(ABC):
    """子进程的富句柄.

    比裸 asyncio.subprocess.Process 多出:
    - 生命周期随 Subprocesses owner (统一启停, owner 退出即清场)
    - 元信息可查询 (meta)
    - 输出捕获 (output, spawn 时声明 capture 才有)
    - 优雅停止 (stop)
    - on-exit 回调 (add_done_callback)

    其余能力 (wait / send_signal / terminate / kill / pid / returncode)
    直接通过 ``process`` 字段使用, 和普通 asyncio 子进程一致.
    """

    @property
    @abstractmethod
    def meta(self) -> ProcessMeta:
        """元信息."""
        ...

    @property
    @abstractmethod
    def process(self) -> asyncio.subprocess.Process:
        """底层 asyncio 子进程. 可直接操作."""
        ...

    @property
    @abstractmethod
    def output(self) -> ProcessOutput | None:
        """输出捕获视图. spawn 时未声明 capture 则为 None."""
        ...

    @abstractmethod
    async def stop(self, timeout: float = 5.0) -> None:
        """优雅停止: 先温和信号 (SIGINT), 超时后升级 SIGKILL (覆盖进程组).

        进程已退出则立即返回. 幂等.
        """
        ...

    @abstractmethod
    def add_done_callback(self, callback: Callable[[ProcessMeta], None]) -> None:
        """注册进程退出的一次性回调.

        进程已退出则立刻同步 fire. 否则注册, 由 owner 在回收阶段顺序 fire.
        语义同 asyncio.Future.add_done_callback —
        callback 在 asyncio loop 线程触发, 调用方不必 thread-safe.
        callback 异常被吞 (写 logger), 不影响其它 callback / 回收流程.
        """
        ...


# -- Subprocesses --

class SubprocessFacade(ABC):

    # -- spawn --
    @abstractmethod
    async def execute(
            self,
            *args: str,
            name: str | None = None,
            description: str | None = None,
            cwd: str | Path | None = None,
            extra_env: dict[str, str] | None = None,
            capture: CaptureSpec | None = None,
            stdin: int | None = None,
            stdout: int | None = None,
            stderr: int | None = None,
            start_new_session: bool = True,
            with_os_env: bool = True,
            on_exit: Callable[[ProcessMeta], None] | None = None,
    ) -> ManagedProcess:
        """exec 模式启动子进程.

        *args 直接传递给 asyncio.create_subprocess_exec, 不经过 shell 解析.
        推荐用于大多数场景.

        cwd: 工作目录. None 时使用实现的默认目录 (构造时指定).
        capture: 声明输出捕获, 与手动传 stdout/stderr 互斥.
        on_exit: 进程退出回调, 等价于启动后立刻 add_done_callback(cb).
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
            extra_env: dict[str, str] | None = None,
            capture: CaptureSpec | None = None,
            stdin: int | None = None,
            stdout: int | None = None,
            stderr: int | None = None,
            start_new_session: bool = True,
            with_os_env: bool = True,
            on_exit: Callable[[ProcessMeta], None] | None = None,
    ) -> ManagedProcess:
        """shell 模式启动子进程.

        cmd 通过 shell 解析. 仅在需要管道 / 重定向 / glob 等 shell 特性时使用.
        其余参数同 execute.
        """
        ...


class Subprocesses(SubprocessFacade, ABC):
    """子进程机制灶台 — spawn / 查询 / 信号 / 生命周期.

    铁律: **子进程不比 owner 活得久**. 通过三件套保证:
    1. start_new_session (setsid) + 关停时 killpg — 覆盖未主动脱离的子孙进程
    2. pipe fencing — capture 模式下管道随 owner 关闭, 子进程写管道即收 SIGPIPE
    3. 回收 polling — 每个 spawn 的进程都有 reclaim 协程 await 其退出

    主动 setsid/setpgid 脱离的守护进程不在承诺内 — 那是子进程自己的责任.

    使用方式::

        async with SubprocessesImpl(...) as sp:
            proc = await sp.execute("echo", "hello")
            await proc.process.wait()

            proc = await sp.execute("find", ".", capture=CaptureSpec())
            await proc.output.wait_drained()
            print(proc.output.stdout())

    所有 spawn 收显式 cwd 参数; 本抽象不持有目录状态 (无 cd/pwd).
    """

    # 一 owner 一实例: Subprocesses 是 IoC 非单例工厂产物, owner 生命周期
    # 为其所有子进程划界 (治理=所有权). 无全局进程板.

    # -- 查询 --

    @abstractmethod
    def executing(self) -> dict[int, ProcessMeta]:
        """正在运行的进程. key = index."""
        ...

    @abstractmethod
    def get(self, index: int) -> ManagedProcess | None:
        """按 index 获取运行中的进程句柄."""
        ...

    @abstractmethod
    def executed(self) -> list[ProcessMeta]:
        """已结束的进程历史. 保留有限条数, FIFO."""
        ...

    # -- 信号 --

    @abstractmethod
    def kill(self, pid: int) -> ErrorInfo | None:
        """kill 单个进程 (SIGKILL). 返回 None 表示成功."""
        ...

    @abstractmethod
    def killpg(self, process_group: int, signal: int) -> ErrorInfo | None:
        """kill 进程组.

        覆盖 start_new_session 后的所有未分离子孙进程.

        平台:
        - Linux/macOS: os.killpg
        - Windows: 平台不支持进程组, 降级为 kill 单个 pid.
          不抛异常, 返回 Error 字符串描述降级行为.
        """
        ...

    # -- 生命周期 --

    @abstractmethod
    def is_running(self) -> bool:
        """本实例是否处于运行中 (已 __aenter__ 且未关闭).

        消费方 (如 channel) 据此决定生命周期归属: 已 running 的实例
        由其 owner 治理, 只用不管; 未启动的实例由消费方托管 async with.
        """
        ...

    @abstractmethod
    async def __aenter__(self) -> Self:
        """启动 Subprocesses."""
        ...

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """关闭 Subprocesses. 终止所有活跃进程 (先 SIGINT, 超时 SIGKILL killpg)."""
        ...
