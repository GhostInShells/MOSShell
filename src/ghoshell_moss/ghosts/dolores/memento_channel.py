"""Dolores 的记忆器官 —— memento 反身 channel | 集成 | alpha

把 ``EgoMementoManager`` 的读面反射成 ghost 自己的器官: 看见自己有哪些轨迹线、读某一段的
原文、和某一段的上下文对话. 由 ghost 组装进 ``ghost`` 根 channel, CTML 上是它的子节点.

三层暴露, 边界见 ``channel_builder`` 的 tier:

- cold ``instruction``: 自解释 —— 坐标模型、摘要与原文的区别、轨迹的磁盘位置.
- warm ``notice``: branch 一览, 一条线一行. 每 refresh 重算, 但 shell trajectory 会 diff
  增量重供 —— 没变即零成本; commits_total / latest 本来每次落锚就变, 是温数据.
- hot: 不占. 刻意自省, 不做每帧轮询.

命令全是读接口 (``always_observe=True``): 结果是"信息", 模型要基于内容做下一步推理.
预期失败 (坐标不存在 / 坏 commit / 时间格式错) 直接返回文本提示, 不抛异常 —— 抛错只留给
真正的意外 (兜底).

面向模型的文本 (instruction / notice / 命令 docstring 与返回值) 一律英文 —— 与 dolores
的提示词体系一致; 中文只留给开发者注释.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ghoshell_moss.core.blueprint.channel_builder import MutableChannel, new_channel
from ghoshell_moss.ghosts.dolores._ego_memento import BrokenCommitError, EgoMementoManager

__all__ = ["build_memento_channel"]


def _instruction_text(manager: EgoMementoManager, storage_root: Path | None) -> str:
    """自解释 —— 只讲控制面 (分支 / 坐标 / 两条读路径). memento 概念由系统指令解释, 这里不重复."""
    lines = [
        "## Memento (control)",
        "",
        "Reflexive control over your memento trajectory.",
        "",
        f"- You are on branch `{manager.config.branch_name}`; the notice lists every branch.",
        "- You point at a place on the line by its coordinate: `{branch_index}-{seq}`, e.g. `1-27`.",
    ]
    if storage_root is not None:
        lines.append(
            f"- Durable at `{storage_root}`; compacting your context does not touch it."
        )
    lines.extend([
        "",
        "read returns a span's raw conversation. chat talks with the context of a span — the other",
        "side is not the commit but the conversation that had happened by the time it was made. chat",
        "runs a real round of inference, so it is slow; read first when you only need to recall.",
    ])
    return "\n".join(lines)


def _branches_text(manager: EgoMementoManager) -> str:
    """notice —— 全部 branch 一行一条; 标出当前活着的那条."""
    infos = manager.list_branches()
    if not infos:
        return "[memento] no branch yet."
    current = manager.config.branch_name
    lines = ["[memento] branches:"]
    for info in infos:
        latest = info.latest_coord if info.latest_coord else "(empty)"
        mark = " <- the one you are living on" if info.name == current else ""
        lines.append(
            f"  {info.name} #{info.index} commits={info.commits_total} latest={latest}{mark}"
        )
    return "\n".join(lines)


def _parse_date(value: str | None, field: str) -> datetime | None:
    """ISO 时间串 → datetime; 空 = None. 解析不了抛 ValueError (命令侧转成文本提示)."""
    if value is None or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.strip())
    except ValueError:
        raise ValueError(
            f"{field} is not an ISO timestamp: {value!r} "
            "(e.g. 2026-09-01 or 2026-09-01T10:00:00)"
        ) from None


def build_memento_channel(
    manager: EgoMementoManager,
    *,
    storage_root: str | Path | None = None,
    name: str = "memento",
    description: str | None = None,
) -> MutableChannel:
    """组装 memento 反身 channel —— ghost 的记忆器官.

    :param manager: ghost 持有的 EgoMementoManager (锚点 / 阈值 / 旁路都归它治理).
    :param storage_root: 轨迹的磁盘位置, 进 instruction 自解释. None = 略去那一行.
    :param name: CTML 标签名.
    :param description: 覆盖默认描述.
    """
    if description is None:
        description = (
            "Memento — your own memory: trajectory branches, coordinates (branch_index-seq), "
            "raw reads (read), and conversation with a span's context (chat)."
        )
    root = Path(storage_root).resolve() if storage_root is not None else None

    chan = new_channel(name=name, description=description)

    @chan.build.instruction
    async def _instruction() -> str:
        return _instruction_text(manager, root)

    @chan.build.notice
    async def _branches() -> str:
        return _branches_text(manager)

    @chan.build.command(name="view", always_observe=True)
    async def view(branch: str | None = None, limit: int | None = None) -> str:
        """Look at one branch's view — the most recent commit summaries.

        :param branch: branch name. None = the one you are living on.
        :param limit: at most how many, counted back from the most recent. None = the default window.
        """
        message = manager.view_message(branch, n=limit)
        if message is None:
            return f"[memento] no branch `{branch}`"
        return message.to_content_string()

    @chan.build.command(name="read", always_observe=True)
    async def read(coord: str) -> str:
        """Read a span's raw conversation.

        :param coord: commit coordinate, e.g. `1-27`.
        """
        try:
            text = await manager.read(coord)
        except KeyError:
            return f"[memento] no commit at `{coord}`"
        return text or f"[memento] nothing was said in that span ({coord})"

    @chan.build.command(name="history", always_observe=True)
    async def history(
        branch: str | None = None,
        from_date: str | None = None,
        until_date: str | None = None,
    ) -> str:
        """List one branch's commits over a time range — like `git log --oneline`, with detail.

        :param branch: branch name. None = the one you are living on.
        :param from_date: start of the range, inclusive; ISO such as `2026-09-01` or `2026-09-01T10:00:00`.
        :param until_date: end of the range, inclusive; same format.
        """
        try:
            start = _parse_date(from_date, "from_date")
            end = _parse_date(until_date, "until_date")
        except ValueError as error:
            return f"[memento] {error}"
        digests = manager.list_commits(branch, from_date=start, until_date=end)
        if not digests:
            return f"[memento] no commits in {branch or manager.config.branch_name} for that range"
        lines: list[str] = []
        for digest in digests:
            broken = " (broken placeholder, chat unavailable)" if digest.broken else ""
            lines.append(f"{digest.coord} {digest.title}{broken}")
            if digest.body:
                lines.extend(f"    {line}" for line in digest.body.splitlines())
        return "\n".join(lines)

    @chan.build.command(name="chat", always_observe=True)
    async def chat(coord: str, prompt: str) -> str:
        """Talk one round with the context of a span.

        The other side is not the commit but the conversation that had happened by the time it was
        made, so anything after it is outside its view. This runs a real round of inference: slow
        and token-hungry.

        :param coord: commit coordinate, e.g. `1-27`.
        :param prompt: what you want to ask.
        """
        try:
            return await manager.chat(coord, prompt)
        except KeyError:
            return f"[memento] no commit at `{coord}`"
        except BrokenCommitError:
            return f"[memento] `{coord}` has no context to talk to; read it instead"

    return chan
