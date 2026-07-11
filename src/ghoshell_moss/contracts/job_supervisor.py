"""JobSupervisor contract — 短命进程的重复执行与可观测 fold.

JobSupervisor 的一句话承诺: "按声明的调度重复执行短命进程, 并把 N 次执行
fold 成一个可观测的 Job".

- ``submit(JobSpec) → Job``: 调度即数据 (interval / times / at), 不做模式多态.
- Job 是 N 次执行的 fold: 状态 / 执行计数 / 最近一次输出窗口, 通过
  ``Job.snapshot() → JobSnapshot`` 读取 — 呈现层 (channel / _pin /
  get_context_messages) 消费统一快照结构.
- owner 死, jobs 死. 无全局任务板.
"""

# 技术目标 (reviewer 上下文, 契约演进见 FEATURE.md matrix-cell-governance §TT-4):
#
# 本抽象是 BackgroundTask 三分后 loop 语义的归宿 (once→Subprocesses.execute,
# on_prompt→channel.get_context_messages). 设计约束:
#
# 1. 调度即数据 — JobSpec 用字段 (interval/times/at) 表达调度, 不用
#    BackgroundRunType 式的模式多态; 与 TT-15 "状态即数据" 同纪律.
#    interval-only 用 asyncio loop 即可 (~100 行胶水), cron 需求真实出现
#    才引 APScheduler, 现在不预留其参数面.
# 2. Job = fold — Job 的存在资格是 fold-identity: 单次执行的句柄是
#    ManagedProcess, 已由 Subprocesses 提供; Job 只为"跨 N 次执行的累积
#    可观测性"存在.
# 3. 认知胶囊三层 (TT-4): Subprocesses(机制, 不知道胶囊) → JobSupervisor
#    (fold, 胶囊原料) → channel/_pin/get_context_messages (呈现, 胶囊投影).
#    JobSnapshot 是层间的纯数据流, 显式化为 pydantic model 防融合回潮 —
#    呈现层不得直接摸 Job 内部状态或 Subprocesses.
# 4. per-owner: IoC 非单例工厂, 每 owner 一实例, 底层组合该 owner 的
#    Subprocesses; owner 关闭 → 全部 job 停止 → 子进程被 Subprocesses 清场.
#
# MCP 无状态 request/response 装不下"活着的快照" — 这是此层不能外包给
# MCP 生态的根本原因 (TT-4).

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field
from typing_extensions import Self

__all__ = [
    "JobSupervisor",
    "Job",
    "JobSpec",
    "JobSnapshot",
    "JobStatus",
]


# -- 调度声明 --

class JobSpec(BaseModel):
    """一个 job 的完整声明 — 执行什么 + 怎么调度. 调度即数据.

    执行体是一条命令 (exec 或 shell 模式二选一), 每轮作为短命进程 spawn,
    输出自动捕获进快照窗口.
    """

    name: str = Field(
        description="job 名称, 用于快照与日志",
    )
    description: str = Field(
        default="",
        description="job 用途的简短描述",
    )

    # -- 执行体 --
    args: tuple[str, ...] = Field(
        default=(),
        description="exec 模式 argv. 与 shell_cmd 二选一",
    )
    shell_cmd: str | None = Field(
        default=None,
        description="shell 模式命令. 与 args 二选一",
    )
    cwd: str | None = Field(
        default=None,
        description="工作目录 (绝对路径). None 用 owner 默认目录",
    )
    extra_env: dict[str, str] = Field(
        default_factory=dict,
        description="追加的环境变量",
    )

    # -- 调度 --
    interval: float = Field(
        default=0.0,
        description="两轮执行之间的间隔秒数 (上轮结束到下轮开始). 0 = 结束即开始下一轮",
    )
    times: int = Field(
        default=0,
        description="总执行轮数. 0 = 无限",
    )
    at: float | None = Field(
        default=None,
        description="首轮执行的最早时间戳 (time.time() 语义). None = 立即",
    )

    # -- 观测 --
    buffer_lines: int = Field(
        default=100,
        description="快照中输出窗口的行数 (最近一轮). 0 = 不保留输出",
    )
    on_failure: Literal["pause", "continue"] = Field(
        default="pause",
        description="某轮退出码非 0 时: pause = 暂停等 resume, continue = 照常调度下一轮",
    )


# -- 状态 --

class JobStatus(str, Enum):
    """job 的调度状态. 封闭 enum — 状态即数据."""

    PENDING = "pending"
    """已提交, 首轮尚未开始 (等 at 或等调度)."""

    RUNNING = "running"
    """当前有一轮正在执行."""

    SLEEPING = "sleeping"
    """轮与轮之间的间隔期."""

    PAUSED = "paused"
    """因失败暂停 (on_failure=pause), 等 resume."""

    FINISHED = "finished"
    """times 轮全部执行完毕."""

    STOPPED = "stopped"
    """被 stop() 终止, 不再调度."""


class JobSnapshot(BaseModel):
    """job 的观测快照 — 跨 N 轮执行的 fold 结果.

    呈现层 (channel instructions / context messages / CLI 视图) 消费的
    唯一数据结构. 读到的是拍快照瞬间的一致状态.
    """

    id: str = Field(
        description="job 唯一标识",
    )
    name: str = Field(
        description="job 名称",
    )
    description: str = Field(
        default="",
        description="job 用途描述",
    )
    status: JobStatus = Field(
        description="当前调度状态",
    )
    executed: int = Field(
        default=0,
        description="已完成轮数",
    )
    times: int = Field(
        default=0,
        description="总轮数声明. 0 = 无限",
    )
    last_exit_code: int | None = Field(
        default=None,
        description="最近一轮的退出码. None = 尚未完成过任何一轮",
    )
    last_started: float | None = Field(
        default=None,
        description="最近一轮的启动时间戳",
    )
    last_finished: float | None = Field(
        default=None,
        description="最近一轮的结束时间戳",
    )
    stdout_tail: str = Field(
        default="",
        description="最近一轮的 stdout 窗口 (buffer_lines 行)",
    )
    stderr_tail: str = Field(
        default="",
        description="最近一轮的 stderr 窗口",
    )
    created: float = Field(
        default_factory=time.time,
        description="快照生成时间戳",
    )


# -- Job --

class Job(ABC):
    """一个已提交 job 的句柄 — N 次短命进程执行的 fold.

    观测走 snapshot(), 控制走 stop()/resume(). 单轮执行的进程细节
    不在此暴露 — 那是 Subprocesses 层的东西.
    """

    @property
    @abstractmethod
    def id(self) -> str:
        """job 唯一标识."""
        ...

    @property
    @abstractmethod
    def spec(self) -> JobSpec:
        """提交时的声明. 只读."""
        ...

    @abstractmethod
    def snapshot(self) -> JobSnapshot:
        """拍一份当前状态快照."""
        ...

    @abstractmethod
    async def stop(self, timeout: float = 5.0) -> None:
        """终止 job: 不再调度, 杀掉当前运行中的一轮 (如有). 幂等."""
        ...

    @abstractmethod
    def resume(self) -> None:
        """从 PAUSED 恢复调度. 非 PAUSED 状态下调用是 no-op."""
        ...

    @abstractmethod
    async def wait(self) -> JobSnapshot:
        """阻塞到 job 进入终态 (FINISHED / STOPPED). 返回终态快照.

        无限 job (times=0) 只会因 stop() 到达终态.
        """
        ...


# -- JobSupervisor --

class JobSupervisor(ABC):
    """job 灶台 — 提交 / 查询 / 生命周期.

    每 owner 一实例. owner 关闭 (__aexit__) 时全部 job 停止,
    运行中的子进程随底层 Subprocesses 清场.

    使用方式::

        async with JobSupervisorImpl(subprocesses=sp) as jobs:
            job = jobs.submit(JobSpec(
                name="health-check",
                args=("curl", "-sf", "http://localhost:8080/health"),
                interval=30.0,
            ))
            snap = job.snapshot()
    """

    @abstractmethod
    def submit(self, spec: JobSpec) -> Job:
        """提交一个 job, 立即进入调度. 返回句柄."""
        ...

    @abstractmethod
    def jobs(self) -> list[Job]:
        """本 supervisor 持有的全部 job (含终态的, 保留有限条数)."""
        ...

    @abstractmethod
    def get(self, job_id: str) -> Job | None:
        """按 id 获取 job 句柄."""
        ...

    # -- 生命周期 --

    @abstractmethod
    async def __aenter__(self) -> Self:
        """启动 JobSupervisor."""
        ...

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """关闭 JobSupervisor. 停止全部 job."""
        ...
