from typing import Any
from ghoshell_moss.message import Message
from ghoshell_moss.core.concepts.channel import ChannelMeta, ChannelFullPath
from ghoshell_moss.core.helpers.xml import xml_start_tag, xml_end_tag

__all__ = [
    'make_interfaces',
    'make_context_messages',
    'make_instruction_messages',
    'MAIN_CHANNEL_NAME',
    'CTML_INTERFACE',
    'CTML_CONTEXT',
    'CTML_INSTRUCTIONS',
]

MAIN_CHANNEL_NAME = '__main__'
CTML_INTERFACE = 'ctml_interface'
CTML_CONTEXT = 'ctml_context'
CTML_INSTRUCTIONS = 'ctml_instructions'


def make_interfaces(metas: dict[ChannelFullPath, ChannelMeta], *, name: str | None = None) -> list[Message]:
    """
    实现 CTML v1.0.0 的 interface 描述.
    :param metas:
    :param name: moss shell name
    """
    message = Message.new(tag=CTML_INTERFACE, name=name)
    message.with_content("```python\n")
    blocks = []

    for channel_path, channel_meta in metas.items():
        # 跳过 command meta 为空的.
        if len(channel_meta.commands) == 0:
            continue
        # 如果不是 available, 就快速描述不可用.
        attributes: dict[str, Any] = {'name': channel_path or MAIN_CHANNEL_NAME}
        if not channel_meta.available:
            attributes['available'] = channel_meta.available
            blocks.append('# ' + xml_start_tag('channel', attributes, self_close=True))
            continue
        # 添加 channel 的开始和结束.
        blocks.append(xml_start_tag('channel', attributes, self_close=False))
        commands = channel_meta.commands
        not_available_commands = []
        for cmd_meta in commands:
            if not cmd_meta.available:
                not_available_commands.append(cmd_meta.name)
                continue
            if not cmd_meta.blocking:
                blocks.append("# not blocking")
            if cmd_meta.priority != 0:
                blocks.append(f"# priority {cmd_meta.priority}")
            blocks.append(cmd_meta.interface)
            blocks.append("\n")

        # with not available commands
        if len(not_available_commands) > 0:
            blocks.append("# not available: " + ','.join(not_available_commands))
        blocks.append(xml_end_tag('channel'))

    message.with_content('\n'.join(blocks))
    message.with_content("\n```")
    return message


def make_context_messages(metas: dict[ChannelFullPath, ChannelMeta], *, name: str | None = None)-> list[Message]:
    """
    按照 ctml 1.0.0 规则, 生成 context messages.
    """
    message = Message.new(tag=CTML_CONTEXT, name=name)
    for channel_path, channel_meta in metas.items():
        path_name = channel_path or MAIN_CHANNEL_NAME
        if len(channel_meta.context) == 0:
            continue
        message.with_content(xml_start_tag('channel', {'name': path_name}, self_close=False))
        for content_message in channel_meta.context:
            # 追加到上下文里.
            message.with_content(*content_message.as_contents())
        message.with_content(xml_end_tag('channel'))
    return [message]


def make_instruction_messages(metas: dict[ChannelFullPath, ChannelMeta], *, name: str | None = None) -> list[Message]:
    """
    按照 ctml 1.0.0 规则, 生成 instruction messages.
    """
    message = Message.new(tag=CTML_INSTRUCTIONS, name=name)
    for channel_path, channel_meta in metas.items():
        path_name = channel_path or MAIN_CHANNEL_NAME
        if len(channel_meta.instructions) == 0:
            continue
        message.with_content(xml_start_tag('channel', {'name': path_name}, self_close=False))
        for content_message in channel_meta.instructions:
            # 追加到上下文里.
            message.with_content(*content_message.as_contents())
        message.with_content(xml_end_tag('channel'))
    return [message]