"""
跨域共享的 host 级名词: WorkspaceView / JobView.

WorkspaceView 源自 workspace.ts, JobView 源自 jobs.ts. 两者都被 events.py 的
HostFrame/MuxFrame 引用, 故先于此声明, 避免循环 import.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "WorkspaceId",
    "WorkspaceView",
    "JobId",
    "JobView",
]

WorkspaceId = str
JobId = str


class WorkspaceView(BaseModel):
    """一个 workspace 行: 每个 workspace.* value 携带的记录投影."""

    model_config = ConfigDict(extra="allow")

    workspaceId: WorkspaceId = Field(default="")
    path: str = Field(default="", description="host-side realpath 规范化目录路径.")
    title: str = Field(default="", description="显示标题 (create 时默认取 basename).")
    sessionIds: list[str] = Field(default_factory=list, description="归属会话, 手工排序.")
    createdAt: str = Field(default="", description="ISO-8601 创建时刻.")
    updatedAt: str = Field(default="", description="ISO-8601 最后变更时刻.")


class JobView(BaseModel):
    """后台 job 的客户端可见视图. 注册表 live record 不上线, 每次 push 现铸视图."""

    model_config = ConfigDict(extra="allow")

    id: JobId = Field(default="", description="<kind>-N 形式, 任务全程稳定.")
    kind: str = Field(default="", description="producer 类型 (bash/pwsh/pty-send/subagent…), 开放字符串.")
    label: str = Field(default="", description="单行标签: 命令或委托描述.")
    status: str | Literal["running", "stopping", "completed", "killed", "failed"] = Field(default="running")
    detail: str | None = Field(default=None, description="kind 特定状态详情 (如 exit code).")
    startedAt: int = Field(default=0, description="注册时刻 epoch ms.")
    finishedAt: int | None = Field(default=None, description="settled 时刻 epoch ms, 存活期间缺省.")
