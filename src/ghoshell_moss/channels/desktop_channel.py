"""Shell 操作面的 OS 工具组织层 | 系统管理 | alpha

desktop 是 MOSS Shell 操作面的工具集成点。挂载 bash / file_editor（未来 + ground），
与 matrix (cell 治理) 平级。无 own commands，极简自我介绍。

Example:
    from ghoshell_moss import new_shell_main_channel
    from ghoshell_moss.channels.desktop_channel import build_desktop_channel
    main = new_shell_main_channel()
    main.import_channels(build_desktop_channel())
"""

from __future__ import annotations

from ghoshell_container import IoCContainer

from ghoshell_moss.core.blueprint.channel_builder import (
    ChannelFactory,
    new_channel,
)
from ghoshell_moss.core.concepts.channel import Channel

__all__ = [
    "build_desktop_channel",
    "new_desktop_channel",
]


def new_desktop_channel(
    *,
    name: str = "desktop",
    description: str | None = None,
) -> Channel:
    """极简集成 channel, import_channels 平级挂 bash + file_editor.

    未来加 ground 时, 在 caller 侧 import 传 extra_children.
    """

    default_desc = (
        "Shell operation surface: OS tools for subprocess execution and "
        "filesystem access. Children: bash (terminal), file_editor."
    )
    chan = new_channel(name=name, description=description or default_desc)

    @chan.build.instruction
    def desktop_instruction() -> str:
        return (
            "Desktop tools: bash (subprocess execution) and file_editor "
            "(filesystem access). Use desktop.bash:exec for shell commands, "
            "desktop.file_editor:view for reading files."
        )

    return chan


def build_desktop_channel(
    *,
    name: str = "desktop",
    description: str | None = None,
    with_bash: bool = True,
    with_file_editor: bool = True,
    extra_children: tuple[Channel | ChannelFactory, ...] = (),
) -> ChannelFactory:
    """High-order factory: config -> ChannelFactory.

    从 container 解析 IoC 依赖, 组合 bash + file_editor 为 desktop 的静态子 channel.

    :param name: desktop channel tag (default 'desktop')
    :param description: override default description
    :param with_bash: attach ``terminal_channel`` as ``desktop.bash``
    :param with_file_editor: attach ``file_editor_channel`` as ``desktop.file_editor``
    :param extra_children: additional Channel or ChannelFactory to import
    """

    def factory(container: IoCContainer) -> Channel:
        desktop = new_desktop_channel(name=name, description=description)

        children: list[Channel | ChannelFactory] = list(extra_children)
        if with_bash:
            from ghoshell_moss.channels.terminal_channel import build_terminal_channel
            children.append(build_terminal_channel())
        if with_file_editor:
            from ghoshell_moss.channels.file_editor_channel import (
                build_file_editor_channel,
            )
            children.append(build_file_editor_channel())

        desktop.import_channels(*children)
        return desktop

    return factory
