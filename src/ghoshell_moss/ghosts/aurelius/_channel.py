"""Constrained CTML control surface for Aurelius's current Memento branch."""

from collections.abc import Callable

from ghoshell_moss.core.blueprint.channel_builder import MutableChannel, new_channel

from ._desktop import AureliusDesktop
from ._memory import AureliusMemory

__all__ = ["new_memento_channel"]


def new_memento_channel(
    memory: AureliusMemory,
    *,
    desktop: AureliusDesktop | None = None,
    on_reflect: Callable[[], None] | None = None,
    on_curate: Callable[[], None] | None = None,
) -> MutableChannel:
    """Expose only the Aurelius owner's current branch; cross-owner writes stay impossible."""
    channel = new_channel(
        name="ghost",
        description=(
            "Aurelius 的认知控制面。过去发生的一切都在 Memento 轨迹里，可用 memory_search / "
            "memory_show 逐字查回原文。回答具体事实前若上下文不明确可见，先查证再作答。"
        ),
    )

    # These three read the trajectory to answer the current question, so their result
    # MUST feed the next Re-Act cycle. Without always_observe the model — following the
    # discipline instruction to "search first, then answer" — would emit memory_search,
    # get the hits, and never observe them back: the turn settles silently. That is the
    # exact failure on a canonical-key question like "ORBIT-004".
    @channel.build.command(always_observe=True)
    def memory_search(keyword: str, limit: int = 20) -> list[dict]:
        """在本 owner 的记忆轨迹里逐字检索关键词，返回命中的稳定地址（commit/moment）与片段。"""
        return [hit.model_dump(mode="json") for hit in memory.search(keyword, limit=limit)]

    @channel.build.command(always_observe=True)
    def memory_log() -> str:
        """列出当前时间线的 commit 锚点和释义，像 git log。"""
        views = memory.branch.all_commits()
        lines = [f"commits={len(views)} staging={len(memory.branch.staging())}"]
        lines.extend(
            f"seq={view.seq} id={view.id} kind={view.note.kind()} summary={view.summary() or '[empty]'}"
            for view in views
        )
        return "\n".join(lines)

    @channel.build.command(always_observe=True)
    def memory_show(commit: str) -> str:
        """按 commit 序号或唯一 id 前缀展开冻结的原始 Moment，逐字取回证据。"""
        return memory.describe_commit(commit)

    @channel.build.command(visible=False)
    def memory_inspect() -> dict:
        """查看当前 Memento 分支、暂存区、commit 与反思追赶状态。"""
        return memory.inspect()

    @channel.build.command(visible=False)
    def memory_staging() -> str:
        """查看尚未冻结的完成 Moment；原文仍在 owner 的 staging 中。"""
        records = memory.branch.staging()
        return "\n".join(f"moment={record.id} type={record.type}" for record in records) or "staging is empty"

    @channel.build.command(visible=False)
    def memory_commit(summary: str) -> str:
        """显式冻结已有 staging；不含当前尚未完成的对话帧，普通事实无需调用。"""
        view = memory.semantic_commit(summary)
        return f"semantic commit seq={view.seq} id={view.id}"

    @channel.build.command(visible=False)
    def memory_reinterpret(commit: str, summary: str) -> str:
        """改写一个本 owner commit 的当前释义；原始 Moment 和旧 note 保留。"""
        view = memory.reinterpret(commit, summary)
        return f"reinterpreted seq={view.seq} id={view.id}"

    @channel.build.command(visible=False)
    def memory_branches() -> list[dict[str, str]]:
        """列出本 owner 的时间线。"""
        return memory.branches()

    @channel.build.command(visible=False)
    def memory_fork(commit: str, name: str = "") -> str:
        """从冻结 commit 创建并切换到新时间线；不能从 staging 出生。"""
        branch = memory.fork(commit, name)
        return f"switched to fork id={branch.meta.branch_id} name={branch.meta.name}"

    @channel.build.command(visible=False)
    def memory_switch(branch: str) -> str:
        """按唯一 branch id 前缀切换本 owner 的当前时间线。"""
        selected = memory.switch(branch)
        return f"switched to branch id={selected.meta.branch_id} name={selected.meta.name}"

    @channel.build.command(visible=False)
    def memory_reflect() -> str:
        """请求后台追赶尚未反思的 mechanical commit，不阻塞当前 CTML 回合。"""
        if on_reflect is None:
            return "reflection is disabled"
        on_reflect()
        return "reflection catch-up scheduled"

    @channel.build.command(visible=False)
    def memory_curate() -> str:
        """请求后台旁路策展：从冻结轨迹重写记忆笔记文件，不阻塞当前回合。"""
        if on_curate is None:
            return "curation is disabled"
        on_curate()
        return "curation scheduled"

    if desktop is not None:

        @channel.build.command(always_observe=True)
        async def desktop_open(directory: str = ".", label: str = "") -> str:
            """在 Aurelius workspace 边界内打开一个 Ground。"""
            ground = await desktop.open(directory, label=label or None)
            return f"opened ground label={ground.label} root={ground.root}"

        @channel.build.command(always_observe=True)
        async def desktop_close(label: str) -> str:
            """关闭一个 Ground，并仅 sediment 其 Pin 清单。"""
            await desktop.close(label)
            return f"closed ground label={label}"

        @channel.build.command(always_observe=True)
        def desktop_pin(label: str, addr: str, note: str = "") -> dict:
            """把 Ground 内的地址 pin 到当前工作表面；不保存文件快照。"""
            return desktop.pin(label, addr, note).model_dump(mode="json")

        @channel.build.command(always_observe=True)
        def desktop_unpin(label: str, addr: str) -> str:
            """从当前 Ground 工作表面移除一枚 Pin。"""
            desktop.unpin(label, addr)
            return f"unpinned {addr} from {label}"

        @channel.build.command(always_observe=True)
        async def desktop_update(label: str, addr: str) -> dict:
            """显式承认 Pin 指向对象的当前变化。"""
            return (await desktop.update(label, addr)).model_dump(mode="json")

        @channel.build.command(always_observe=True)
        async def desktop_frame(label: str) -> str:
            """重绘指定 Ground 的当前帧。"""
            return await desktop.frame(label)

    return channel
