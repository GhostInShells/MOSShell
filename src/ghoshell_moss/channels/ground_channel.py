"""Reflects a GROUND.md cognitive field as a render surface | integration | beta

One home, one channel. The channel holds a root and a boundary, and answers one
question: *render this position*. That is the only thing it can honestly add to
a shell — a render pushed into the ghost's live context. Everything else the old
channel did (open/close, pin editing, stats, template previews, a tree of gated
child channels) is either a file operation or a render, and both already have
owners.

The face itself is decided by the position (SPEC §6.1): a directory with its own
GROUND.md renders as itself; otherwise the nearest GROUND.md up to the boundary
renders it as a walk face. Children named by `inline` render recursively — an
unlisted child is never drawn.

Example:
    from ghoshell_moss import new_shell_main_channel
    from ghoshell_moss.channels.ground_channel import new_ground_channel
    main = new_shell_main_channel()
    main.import_channels(new_ground_channel("/path/to/home", name="project_ground"))
"""

from __future__ import annotations

from pathlib import Path

from ghoshell_moss.core.blueprint.channel_builder import MutableChannel, new_channel
from ghoshell_moss.ground import RenderLimits, render_position

__all__ = ["new_ground_channel"]


def new_ground_channel(
    root: str | Path,
    *,
    boundary: str | Path | None = None,
    name: str = "ground",
    description: str | None = None,
    instruction: str | None = None,
    limits: RenderLimits | None = None,
) -> MutableChannel:
    """装配一个 ground channel — 一个 home, 一条命令.

    :param root: 场根 (也是默认边界). 相对路径按 cwd 解析.
    :param boundary: 向上探索的边界. None = root 本身, 永不越出这个 home.
    :param instruction: 覆盖默认说明 (None = 机制 prose, "" = 整段不出现).
    :param limits: 默认渲染预算 (字节 / 墙钟). None = SPEC 的默认值.
    """
    base = Path(root).expanduser().resolve()
    edge = Path(boundary).expanduser().resolve() if boundary is not None else base

    if description is None:
        description = (
            "Ground — a cognitive field: a directory marked by GROUND.md whose content "
            "survives compaction. `render` draws the face of a position."
        )

    chan = new_channel(name=name, description=description)

    @chan.build.instruction
    async def _instruction() -> str:
        if instruction is not None:
            return instruction
        return (
            "## Ground (cognitive field)\n"
            f"Rooted at `{base}`. `render` draws the face of a position: a directory with its "
            "own GROUND.md renders as itself, one without it renders through the nearest "
            "GROUND.md up to this root — the walk face carries `<cwd>` and no body. Children "
            "named by `inline` are drawn inside the parent; unlisted ones are not drawn at "
            "all. There is no open/close/pin command: edit GROUND.md with your file tools, "
            "and a render is always a cold read."
        )

    @chan.build.command(name="render", always_observe=True)
    async def render(path: str = "", template: str = "") -> str:
        """Draw the face of a position under this home.

        The position decides the face, and a render is a cold read — it re-reads
        GROUND.md and re-runs exec pins every time.

        :param path: A directory relative to this home. Empty = the home root.
        :param template: Draw through a render table instead of the ground's own:
            a template name from `<ground>/.grounds/`, or a `.md` file path.
        """
        target = Path(path).expanduser()
        position = target if target.is_absolute() else (base / target)
        return await render_position(
            position,
            boundary=edge,
            template=template or None,
            limits=limits,
        )

    return chan
