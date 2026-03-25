from typing import Any, Dict
from ghoshell_moss.message import Message, Content
from ghoshell_moss.core.concepts.channel import ChannelMeta, ChannelFullPath, Channel
from ghoshell_moss.core.helpers.xml import xml_start_tag, xml_end_tag

__all__ = [
    'make_interfaces',
    'make_context_messages',
    'make_instruction_messages',
    'MAIN_CHANNEL_NAME',
    'MOSS_CONTEXT',
    'MOSS_INSTRUCTIONS',
    'generate_channel_tree',
]

MAIN_CHANNEL_NAME = '__main__'
MOSS_CONTEXT = 'moss_context'
MOSS_INSTRUCTIONS = 'moss_instructions'


def generate_channel_tree(channels: Dict[ChannelFullPath, ChannelMeta], with_desc: bool = False) -> str:
    """
    根据 channel 路径字典生成树形字符串。
    """
    # 1. 标准化路径：空字符串 -> '__main__'
    nodes = {}
    for path, meta in channels.items():
        key = '__main__' if path == '' else path
        nodes[key] = _Node(key, meta.description)

    # 2. 构建父子关系
    root_paths = set()  # 记录父节点不存在的节点（根级节点）
    for full in nodes:
        if full == '__main__':
            root_paths.add(full)
        else:
            parts = full.split('.')
            parent = '.'.join(parts[:-1])
            if parent in nodes:
                # 父节点存在，建立父子关系
                nodes[parent].children.append(nodes[full])
            else:
                root_paths.add(full)

    # 3. 确保 __main__ 节点存在
    if '__main__' not in nodes:
        nodes['__main__'] = _Node('__main__', '')
        root_paths.add('__main__')

    main_node = nodes['__main__']

    # 将除 __main__ 本身以外的根级节点作为 __main__ 的子节点
    for path in root_paths:
        if path != '__main__':
            main_node.children.append(nodes[path])

    # 4. 递归生成树形字符串
    lines = []

    # 输出 __main__ 节点（根）
    desc_part = f" `{main_node.desc}`" if main_node.desc and with_desc else ""
    lines.append(main_node.full + desc_part)

    # 输出子节点
    def _print_children(children: list['_Node'], prefix: str, bloodline: str):
        for i, child in enumerate(children):
            is_last = (i == len(children) - 1)
            connector = "└── " if is_last else "├── "
            _desc_part = ''
            if child.desc and with_desc:
                desc = child.desc.replace('\n', ';')
                _desc_part = f": `{desc}`"
            name = child.full[len(bloodline):]
            name = name.lstrip('.')
            new_bloodline = Channel.join_channel_path(bloodline, name)
            lines.append(prefix + connector + name + _desc_part)
            # 递归子节点的子节点
            child_prefix = prefix + ("    " if is_last else "│   ")
            _print_children(child.children, child_prefix, bloodline=new_bloodline)

    _print_children(main_node.children, "", bloodline='')

    return "\n".join(lines)


class _Node:
    __slots__ = ('full', 'desc', 'children')

    def __init__(self, full: str, desc: str = ""):
        self.full = full
        self.desc = desc
        self.children: list[_Node] = []


def make_interfaces(channel_meta: ChannelMeta) -> str:
    """
    实现 CTML v1.0.0 的 interface 描述.
    """
    # 如果不是 available, 就快速描述不可用.
    commands = channel_meta.commands
    if len(commands) == 0:
        return ''
    blocks = ['<interface>']
    available_commands = 0
    blocks.append("```python")
    for cmd_meta in commands:
        if not cmd_meta.available:
            continue
        available_commands += 1
        if not cmd_meta.blocking:
            blocks.append("# not blocking")
        if cmd_meta.priority != 0:
            blocks.append(f"# priority {cmd_meta.priority}")
        blocks.append(cmd_meta.interface)

    # with not available commands
    if available_commands == 0:
        return ''

    blocks.append('```')
    blocks.append('</interface>')
    return '\n'.join(blocks)


def make_context_messages(metas: dict[ChannelFullPath, ChannelMeta], *, name: str | None = None) -> list[Message]:
    """
    按照 ctml 1.0.0 规则, 生成 context messages.
    """
    if len(metas) == 0:
        return []
    message = Message.new(tag=MOSS_CONTEXT, name=name, timestamp=False)
    for channel_path, channel_meta in metas.items():
        path_name = channel_path or MAIN_CHANNEL_NAME
        message.with_content(xml_start_tag('channel', {'name': path_name}, self_close=False))
        # add with instruction or failure
        if channel_meta.failure:
            message.with_content(xml_start_tag('failure'))
            message.with_content(channel_meta.failure)
            message.with_content(xml_end_tag('failure'))
        if len(channel_meta.context) > 0:
            message.with_content(xml_start_tag('context'))
            for content_message in channel_meta.context:
                # 追加到上下文里.
                message.with_content(*content_message.as_contents())
            message.with_content(xml_end_tag('context'))
        # make channel interface
        interface = make_interfaces(channel_meta)
        if interface:
            message.with_content(interface)
        message.with_content('\n' + xml_end_tag('channel'))
    return [message]


def make_instruction_messages(metas: dict[ChannelFullPath, ChannelMeta], *, name: str | None = None) -> str:
    """
    按照 ctml 1.0.0 规则, 生成 instruction messages.
    """
    if len(metas) == 0:
        return ''
    message = Message.new(tag=MOSS_INSTRUCTIONS, name=name, timestamp=False)
    for channel_path, channel_meta in metas.items():
        path_name = channel_path or MAIN_CHANNEL_NAME
        if len(channel_meta.instruction) == 0 and not channel_meta.description:
            # 忽略没有 instructions 的.
            continue
        message.with_content(xml_start_tag('channel', {'name': path_name}, self_close=False))
        if channel_meta.description:
            # description.
            message.with_content(xml_start_tag('description'))
            message.with_content(channel_meta.description)
        message.with_content(xml_end_tag('description'))
        # add with instruction
        if channel_meta.instruction:
            message.with_content(xml_start_tag('instruction'))
            message.with_content(channel_meta.instruction)
            message.with_content(xml_end_tag('instruction'))
        message.with_content(xml_end_tag('channel'))
    return message.to_xml()
