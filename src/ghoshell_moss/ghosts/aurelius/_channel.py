"""Constrained CTML control surface for Aurelius's current Memento branch."""

from collections.abc import Callable

from ghoshell_moss.core.blueprint.channel_builder import MutableChannel, new_channel

from ._desktop import AureliusDesktop
from ._knowledge import MemoryProjection
from ._memory import AureliusMemory

__all__ = ["new_memento_channel"]


def _memory_claims_view(knowledge: MemoryProjection, *, detail: bool, limit: int) -> dict:
    if limit < 1 or limit > 100:
        raise ValueError("limit must be between 1 and 100")
    snapshot = knowledge.snapshot()
    result = knowledge.inspect()
    selected_claims = snapshot.claims[:limit]
    if detail:
        result.update({
            "claims": [claim.model_dump(mode="json") for claim in selected_claims],
            "candidates": [item.model_dump(mode="json") for item in snapshot.candidates[:limit]],
            "claims_truncated": len(snapshot.claims) > limit,
            "candidates_truncated": len(snapshot.candidates) > limit,
        })
        return result
    result.update({
        "claims": [
            {
                "key": claim.key,
                "value": claim.value,
                "kind": claim.kind,
                "status": claim.status,
                "subject_scope": claim.subject_scope,
                "confidence": claim.confidence,
            }
            for claim in selected_claims
        ],
        "claims_truncated": len(snapshot.claims) > limit,
        "detail": "set detail=true to include bounded evidence and candidates",
    })
    return result


def new_memento_channel(
    memory: AureliusMemory,
    *,
    knowledge: MemoryProjection | None = None,
    desktop: AureliusDesktop | None = None,
    on_reflect: Callable[[], None] | None = None,
) -> MutableChannel:
    """Expose only the Aurelius owner's current branch; cross-owner writes stay impossible."""
    channel = new_channel(
        name="ghost",
        description=(
            "Aurelius 的受限认知控制面。完成的 Moment 和事实 Recall 已自动处理；"
            "memory_* 是仅供用户显式运维/审计的隐藏命令，普通对话不得主动调用。"
        ),
    )

    @channel.build.command(visible=False)
    def memory_inspect() -> dict:
        """查看当前 Memento 分支、暂存区、commit 与反思追赶状态。"""
        state = memory.inspect()
        if knowledge is not None:
            state["knowledge"] = knowledge.inspect()
        return state

    @channel.build.command(visible=False)
    def memory_log() -> str:
        """列出当前时间线的 commit 锚点和释义。"""
        views = memory.branch.all_commits()
        lines = [f"commits={len(views)} staging={len(memory.branch.staging())}"]
        lines.extend(
            f"seq={view.seq} id={view.id} kind={view.note.kind()} summary={view.summary() or '[empty]'}"
            for view in views
        )
        return "\n".join(lines)

    @channel.build.command(visible=False)
    def memory_staging() -> str:
        """查看尚未冻结的完成 Moment；原文仍在 owner 的 staging 中。"""
        records = memory.branch.staging()
        return "\n".join(f"moment={record.id} type={record.type}" for record in records) or "staging is empty"

    @channel.build.command(visible=False)
    def memory_show(commit: str) -> str:
        """按 commit 序号或唯一 id 前缀展开冻结的原始 Moment。"""
        return memory.describe_commit(commit)

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

    if knowledge is not None:

        @channel.build.command(visible=False)
        def memory_recall(query: str) -> dict:
            """按 canonical field 召回本 owner/current branch 的可验证 Claim。"""
            return knowledge.recall(query).model_dump(mode="json")

        @channel.build.command(visible=False)
        def memory_claims(detail: bool = False, limit: int = 20) -> dict:
            """有界审计；默认仅摘要，detail=true 才包含证据与候选。"""
            return _memory_claims_view(knowledge, detail=detail, limit=limit)

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
