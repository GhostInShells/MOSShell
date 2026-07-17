"""Constrained CTML control surface for Data's current Memento branch."""

from collections.abc import Callable

from ghoshell_moss.core.blueprint.channel_builder import MutableChannel, new_channel

from ._memory import DataMemory

__all__ = ["new_memento_channel"]


def new_memento_channel(
    memory: DataMemory,
    *,
    on_reflect: Callable[[], None] | None = None,
) -> MutableChannel:
    """Expose only the Data owner's current branch; cross-owner writes stay impossible."""
    channel = new_channel(
        name="ghost",
        description="Data Ghost 的 Memento 控制面：查看、锚定、改写与分叉当前记忆。",
    )

    @channel.build.command(always_observe=True)
    def memory_inspect() -> dict:
        """查看当前 Memento 分支、暂存区、commit 与反思追赶状态。"""
        return memory.inspect()

    @channel.build.command(always_observe=True)
    def memory_log() -> str:
        """列出当前时间线的 commit 锚点和释义。"""
        views = memory.branch.all_commits()
        lines = [f"commits={len(views)} staging={len(memory.branch.staging())}"]
        lines.extend(
            f"seq={view.seq} id={view.id} kind={view.note.kind()} summary={view.summary() or '[empty]'}"
            for view in views
        )
        return "\n".join(lines)

    @channel.build.command(always_observe=True)
    def memory_staging() -> str:
        """查看尚未冻结的完成 Moment；原文仍在 owner 的 staging 中。"""
        records = memory.branch.staging()
        return "\n".join(f"moment={record.id} type={record.type}" for record in records) or "staging is empty"

    @channel.build.command(always_observe=True)
    def memory_show(commit: str) -> str:
        """按 commit 序号或唯一 id 前缀展开冻结的原始 Moment。"""
        return memory.describe_commit(commit)

    @channel.build.command(always_observe=True)
    def memory_commit(summary: str) -> str:
        """把当前 staging 显式冻结为 semantic commit；summary 不能为空。"""
        view = memory.semantic_commit(summary)
        return f"semantic commit seq={view.seq} id={view.id}"

    @channel.build.command(always_observe=True)
    def memory_reinterpret(commit: str, summary: str) -> str:
        """改写一个本 owner commit 的当前释义；原始 Moment 和旧 note 保留。"""
        view = memory.reinterpret(commit, summary)
        return f"reinterpreted seq={view.seq} id={view.id}"

    @channel.build.command(always_observe=True)
    def memory_branches() -> list[dict[str, str]]:
        """列出本 owner 的时间线。"""
        return memory.branches()

    @channel.build.command(always_observe=True)
    def memory_fork(commit: str, name: str = "") -> str:
        """从冻结 commit 创建并切换到新时间线；不能从 staging 出生。"""
        branch = memory.fork(commit, name)
        return f"switched to fork id={branch.meta.branch_id} name={branch.meta.name}"

    @channel.build.command(always_observe=True)
    def memory_switch(branch: str) -> str:
        """按唯一 branch id 前缀切换本 owner 的当前时间线。"""
        selected = memory.switch(branch)
        return f"switched to branch id={selected.meta.branch_id} name={selected.meta.name}"

    @channel.build.command(always_observe=True)
    def memory_reflect() -> str:
        """请求后台追赶尚未反思的 mechanical commit，不阻塞当前 CTML 回合。"""
        if on_reflect is None:
            return "reflection is disabled"
        on_reflect()
        return "reflection catch-up scheduled"

    return channel
