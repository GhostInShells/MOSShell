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


def make_interfaces(channel_meta: ChannelMeta, *, dynamic: bool = True, sustain: bool = True) -> str:
    """
    实现 CTML v1.0.0 的 interface 描述.
    """
    # 如果不是 available, 就快速描述不可用.
    commands = channel_meta.commands
    if len(commands) == 0:
        return ''
    available_commands = 0
    blocks = []
    blocks.append("```python")
    for cmd_meta in commands:
        if not cmd_meta.available:
            continue
        if cmd_meta.dynamic and not dynamic:
            # 排除掉非动态的 command meta.
            continue
        if not cmd_meta.dynamic and not sustain:
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
    return '\n'.join(blocks)


class ChannelMetaPrompter:

    def __init__(self, path: ChannelFullPath, meta: ChannelMeta):
        self.path = path or MAIN_CHANNEL_NAME
        self.meta = meta
        # 是否是虚拟节点.
        self.virtual = meta.virtual

    def make_full_block(self) -> Message | None:
        channel_container = Message.new(tag="channel", name=self.path, timestamp=False)
        if description := self.description_message():
            channel_container.with_messages(description, timestamp=False)
        if instruction := self.instruction_message():
            channel_container.with_messages(instruction, timestamp=False)
        if failure := self.failure_message():
            channel_container.with_messages(failure, timestamp=False)
            return channel_container
        if states := self.states_message():
            channel_container.with_messages(states, timestamp=False)
        if context := self.context_messages():
            channel_container.with_messages(*context, timestamp=True, with_meta=True)
        if interface := self.interface_message(dynamic=True, sustain=True):
            channel_container.with_messages(interface, timestamp=False)
        if channel_container.is_empty():
            return None
        return channel_container

    def make_instruction_block(self) -> Message | None:
        """
        virtual 类型的节点没有资格生成 instruction.
        """
        if self.virtual:
            return None
        channel_instruction_container = Message.new(tag="channel", name=self.path, timestamp=False)
        # 先添加 description.
        if description := self.description_message():
            channel_instruction_container.with_messages(description, timestamp=False)
        if instruction := self.instruction_message():
            channel_instruction_container.with_messages(instruction, timestamp=False)
        dynamic = False
        # 只展示可持续消息.
        sustain = True
        if interface_msg := self.interface_message(dynamic=dynamic, sustain=sustain):
            channel_instruction_container.with_messages(interface_msg, timestamp=False)
        if channel_instruction_container.is_empty():
            return None
        return channel_instruction_container

    def make_context_block(self) -> Message | None:
        """
        生成 Channel Context 的标准逻辑.
        """
        channel_context_message_container = Message.new(
            tag="channel",
            name=self.path,
            timestamp=False,
            # 只添加 refreshed 的最后时间戳.
            attributes={'refreshed': self.meta.created.isoformat()},
        )
        if failure := self.failure_message():
            channel_context_message_container.with_messages(failure)
            return channel_context_message_container
        # virtual 时添加的信息.
        if self.virtual:
            if description := self.description_message():
                channel_context_message_container.with_messages(description, timestamp=False)
            if instruction := self.instruction_message():
                channel_context_message_container.with_messages(instruction, timestamp=False)

        # 正常添加 interface.
        sustain = self.virtual
        dynamic = True
        # 正常添加 context.
        if states := self.states_message():
            channel_context_message_container.with_messages(states, timestamp=False)
        context_messages = self.context_messages()
        if len(context_messages) > 0:
            channel_context_message_container.with_messages(*context_messages)
        if channel_context_message_container.is_empty():
            # 如果容器为空, 什么消息体都没有.
            return None
        interface_msg = self.interface_message(dynamic=dynamic, sustain=sustain)
        if interface_msg is not None:
            channel_context_message_container.with_messages(interface_msg, timestamp=False)
        if channel_context_message_container.is_empty():
            return None
        return channel_context_message_container

    def failure_message(self) -> Message | None:
        if not self.meta.failure:
            return None
        failure_message = Message.new(tag="failure", timestamp=False)
        failure_message.with_content(self.meta.failure)
        return failure_message

    def context_messages(self) -> list[Message]:
        return self.meta.context

    def instruction_message(self) -> Message | None:
        """
        生成的系统指令.
        """
        if not self.meta.instruction:
            return None
        return Message.new(tag="instruction", timestamp=False).with_content(self.meta.instruction)

    def states_message(self) -> Message | None:
        """
        状态相关的消息.
        """
        if not self.meta.states:
            return None
        message_container = Message.new(tag="states", timestamp=False)
        message_container.with_content("States of the channel:\n")
        # 生成 states 的描述.
        for name, desc in self.meta.states.items():
            desc = desc.replace('\n', ';')
            message_container.with_content(f"- {name}: {desc}\n")

        if self.meta.current_state:
            message_container.with_content(f"Current state: {self.meta.current_state}")
        return message_container

    def description_message(self) -> Message | None:
        if not self.meta.description:
            return None
        return Message.new(tag="description", timestamp=False).with_content(self.meta.description)

    def interface_message(self, dynamic: bool, sustain: bool) -> Message | None:
        interface = make_interfaces(self.meta, dynamic=dynamic, sustain=sustain)
        if not interface:
            return None
        return Message.new(tag="interface", timestamp=False).with_content(interface)


def make_context_messages(metas: dict[ChannelFullPath, ChannelMeta], *, name: str | None = None) -> list[Message]:
    """
    按照 ctml 1.0.0 规则, 生成 context messages.
    """
    if len(metas) == 0:
        return []
    # 用单一容器包裹所有的消息. 并且标记自身时间戳.
    context_message_container = Message.new(tag=MOSS_CONTEXT, name=name, timestamp=True)
    for channel_path, channel_meta in metas.items():
        # 如果是 virtual, 则需要展示所有讯息.
        prompter = ChannelMetaPrompter(channel_path, channel_meta)
        if block := prompter.make_context_block():
            context_message_container.with_messages(block, with_meta=True, timestamp=True)
    return [context_message_container]


def make_instruction_messages(metas: dict[ChannelFullPath, ChannelMeta], *, name: str | None = None) -> str:
    """
    按照 ctml 1.0.0 规则, 生成 instruction messages.
    """
    if len(metas) == 0:
        return ''
    message = Message.new(tag=MOSS_INSTRUCTIONS, name=name, timestamp=False)
    for channel_path, channel_meta in metas.items():
        # 如果是 virtual, 则需要展示所有讯息.
        prompter = ChannelMetaPrompter(channel_path, channel_meta)
        if block := prompter.make_instruction_block():
            message.with_content(block)
    return message.to_xml()
