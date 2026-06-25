"""ProjectManager contract — Ghost 在文件系统项目中的操作中枢.

从 Project.root 实例化, 通过 channel_builder 构建为 Channel,
提供给 Ghost 作为第一个能力模块.

核心职责:
1. 目录作用域 (pwd / cd)
2. 命令执行 (execute_command / background_command)
3. Pin 屏幕阵列 (pin / unpin / pinned / read_pin)
4. 文件操作 (read_file)
5. 认知上下文 (instruction / context_messages)
6. 认知框架持久化 (dump_status / load_status)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ghoshell_moss.message import Message

RefreshMode = Literal["once", "loop", "on_prompt"]
"""pin 的刷新策略:

- ``once``: 执行一次, 冻结. 用于静态身份信息 (uname, hostname, 版本号).
- ``loop``: 后台持续运行, stdout 流式写入 tmp. 用于 tail -f / 监控.
- ``on_prompt``: 每轮思考帧前执行一次. 用于 git status / 目录快照.
"""


@dataclass
class PinInfo:
    """一个 pin 的运行时状态."""

    id: str
    """pin 唯一标识, 由 command 字符串哈希生成."""

    command: str
    """生成屏幕内容的 shell 命令."""

    refresh: RefreshMode = "once"
    """刷新策略."""

    tmp_file: str = ""
    """stdout 写入的临时文件路径."""

    line_count: int = 0
    """当前缓存的总行数."""

    last_update: float = 0.0
    """最后更新时间戳."""

    alive: bool = True
    """进程是否还活着 (loop 模式下有意义)."""


@dataclass
class BackgroundTaskInfo:
    """一个后台任务的运行时状态."""

    id: int
    """任务唯一标识."""

    command: str
    """执行的 shell 命令."""

    loop: int = 1
    """剩余循环次数. 0 表示无限循环."""

    executed: int = 0
    """已完成次数."""

    alive: bool = True
    """进程是否还活着."""


class ProjectManager(ABC):
    """Ghost 在文件系统中的操作中枢.

    模型通过它感知项目、执行命令、管理认知屏幕.
    ProjectManager 从 Project.root 实例化自身, 然后通过
    ``new_channel()`` + ``chan.build.*`` 构建为 Channel.

    术语:
    - pin: 有屏幕的后台命令 — 输出自动注入 context_messages
    - background_command: 无屏幕的后台命令 — 纯副作用
    - screen: 模型在每一帧思考时能看到的信息窗口
    """

    # -- 目录作用域 --

    @property
    @abstractmethod
    def root(self) -> Path:
        """项目根目录. 所有路径操作相对于此."""
        ...

    @property
    @abstractmethod
    def pwd(self) -> Path:
        """当前工作目录."""
        ...

    @abstractmethod
    def cd(self, path: str) -> str:
        """切换工作目录.

        返回切换后的绝对路径. path 可以是相对路径 (相对于 pwd)
        或绝对路径 (必须在 root 子树内).
        """
        ...

    # -- 命令执行 --

    @abstractmethod
    async def execute_command(self, command: str, *, timeout: float = 60.0) -> str:
        """同步执行 shell 命令, 阻塞等待结果.

        底层使用 subprocess. 适合需要结果才能继续的操作.
        返回 stdout + stderr + exit code 的组合文本.
        """
        ...

    @abstractmethod
    async def background_command(
        self, command: str, *, loop: int = 1, notify: bool = False
    ) -> int:
        """启动后台命令. 立即返回 task_id.

        loop=0 表示无限循环, 每次执行完立刻重跑.
        notify=True 时, 命令异常退出会通过 Mindflow impulse 通知模型.
        后台命令的输出不进入 context_messages (没有屏幕).
        需要屏幕 → 用 pin().
        """
        ...

    @abstractmethod
    def background_tasks(self) -> dict[int, BackgroundTaskInfo]:
        """返回所有活跃的后台任务."""
        ...

    @abstractmethod
    async def cancel_background(self, task_id: int) -> str:
        """取消后台任务."""
        ...

    # -- Pin: 认知屏幕阵列 --

    @abstractmethod
    def pin(self, command: str, *, refresh: RefreshMode = "once") -> str:
        """添加一块认知屏幕.

        屏幕标题 = command 自身. 去重由 command 哈希保证 —
        相同 command 重复 pin 返回已有 pin_id, 不创建新屏幕.

        refresh 策略:
        - ``once``: 执行一次, stdout 冻结在 tmp 文件
        - ``loop``: 后台持续运行 (tail -f 模式), stdout 持续更新
        - ``on_prompt``: 每轮思考帧前执行一次, 更新 tmp 文件

        所有输出写入 tmp 文件. context_messages 展示 tail -n,
        完整内容通过 read_pin() 按需拉取.

        返回 pin_id (command 哈希).
        """
        ...

    @abstractmethod
    def unpin(self, pin_id: str) -> str:
        """移除一块屏幕, 清理 tmp 文件, 停止关联进程."""
        ...

    @abstractmethod
    def pinned(self) -> dict[str, PinInfo]:
        """返回所有活跃屏幕的状态."""
        ...

    @abstractmethod
    def read_pin(self, pin_id: str, *, offset: int = 0, limit: int = 0) -> str:
        """读取屏幕完整输出.

        limit=0 表示全量. offset 从 0 开始.
        """
        ...

    # -- 文件操作 --

    @abstractmethod
    async def read_file(self, path: str, *, offset: int = 0, limit: int = 200) -> str:
        """读取项目内文件.

        path 相对于 root. 返回带行号的文本片段.
        """
        ...

    # -- 认知上下文 --

    @abstractmethod
    def instruction(self) -> str:
        """生成静态认知 — session 启动时调用一次.

        默认实现:
        1. 读取向上查找到的 CLAUDE.md
        2. 补充宿主信息 (uname, hostname, pwd, 目录结构)
        """
        ...

    @abstractmethod
    async def context_messages(self) -> list[Message]:
        """生成动态认知 — 每轮思考帧调用.

        渲染所有屏幕:
        - 每块屏幕: [pin_id] command → tail -n 行 + tmp 文件路径
        - 当前 pwd
        - 活跃后台任务摘要
        """
        ...

    # -- 认知框架持久化 --

    @abstractmethod
    def dump_status(self, filename: str) -> str:
        """把当前认知框架序列化到文件.

        保存 pins 配置 + cwd + 后台任务配置.
        不保存 tmp 文件内容 (可重建).
        """
        ...

    @abstractmethod
    def load_status(self, filename: str) -> str:
        """从文件恢复认知框架.

        恢复 pins + cwd, 重建后台任务, 重跑 once 类 pin.
        """
        ...
