"""Ground channel — 认知场的运行时落点 | 集成 | alpha

Example:
    from ghoshell_moss import new_shell_main_channel
    from ghoshell_moss.channels.ground_channel import build_grounds_channel

    main = new_shell_main_channel()
    main.import_channels(build_grounds_channel())
"""

from __future__ import annotations

from pathlib import Path

from ghoshell_container import IoCContainer

from ghoshell_moss.core.blueprint.channel_builder import (
    MutableChannel,
    ChannelFactory,
    new_channel,
)
from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.ground import DefaultGroundSet

__all__ = ["new_grounds_channel", "build_grounds_channel"]


def new_grounds_channel(
    grounds: list[str | Path],
    *,
    workspace_root: str | Path | None = None,
    name: str = "grounds",
    description: str | None = None,
) -> MutableChannel:
    """组装 grounds channel — 持一个 DefaultGroundSet, 启动时打开场.

    :param grounds: 启动时默认打开的场目录列表 (相对 workspace_root 或绝对).
    :param workspace_root: 相对路径解析基点. None = 进程 cwd.
    :param name: CTML 标签名.
    :param description: 覆盖默认描述.
    """
    root = Path(workspace_root).resolve() if workspace_root else Path.cwd().resolve()
    groundset = DefaultGroundSet(workspace_root=root)

    if description is None:
        description = (
            "Ground — navigate directories marked by GROUND.md (frontmatter "
            "identity + body law + pins)."
        )

    chan = new_channel(name=name, description=description)

    @chan.build.startup
    async def _startup() -> None:
        for d in grounds:
            await groundset.open(d)

    @chan.build.instruction
    def _instruction() -> str:
        open_grounds = ", ".join(sorted(groundset.active()))
        return (
            "## grounds\n"
            "A ground is a directory marked by GROUND.md: frontmatter (identity + "
            "pins), body (law), pins (first-person gaze).\n"
            f"Open grounds: {open_grounds}"
        )

    @chan.build.command(name="render", always_observe=True)
    async def render(label: str = "") -> str:
        """Render a ground's full frame (body + pin results).

        :param label: ground label (see instruction for open grounds). Empty = render all open grounds.
        """
        active = groundset.active()
        if label:
            g = groundset.get(label)
            if g is None:
                return f"[grounds] unknown ground {label!r}. open: {sorted(active)}"
            return str(await g.render())
        if not active:
            return "[grounds] no open grounds"
        out: list[str] = []
        for l, g in active.items():
            out.append(f"<!-- ground:{l} -->\n{await g.render()}")
        return "\n\n".join(out)

    @chan.build.command(name="walk", always_observe=True)
    async def walk(dir: str) -> str:
        """Open a ground by directory and render it.

        :param dir: directory path (relative to workspace root, or absolute).
        """
        g = await groundset.open(dir)
        return str(await g.render())

    return chan


def build_grounds_channel(
    *,
    grounds: list[str | Path] | None = None,
    workspace_root: str | Path | None = None,
    name: str = "grounds",
    description: str | None = None,
) -> ChannelFactory:
    """IoC 集成工厂 — 解析项目根, grounds=None 时默认 [项目根]."""
    def factory(container: IoCContainer) -> Channel:
        resolved_root = workspace_root
        if not resolved_root:
            from ghoshell_moss.core.blueprint.project import Project
            project = Project.discover()
            resolved_root = str(project.root)
        resolved_grounds = grounds or [resolved_root]
        return new_grounds_channel(
            resolved_grounds,
            workspace_root=resolved_root,
            name=name,
            description=description,
        )
    return factory
