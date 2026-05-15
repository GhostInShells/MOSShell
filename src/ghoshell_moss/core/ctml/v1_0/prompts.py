from typing import Dict
from ghoshell_moss.message import Message
from ghoshell_moss.core.concepts.channel import ChannelMeta, ChannelFullPath, Channel
from .constants import MOSS_DYNAMIC, MOSS_STATIC, MAIN_CHANNEL_NAME
import datetime
import dateutil

__all__ = [
    'make_interfaces',
    'make_dynamic_messages',
    'make_static_messages',
]


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

    def __init__(
            self,
            path: ChannelFullPath,
            meta: ChannelMeta,
            *,
            virtual: bool | None = None,
    ):
        self.path = path or MAIN_CHANNEL_NAME
        self.meta = meta
        # 是否是虚拟节点.
        self.virtual = virtual if virtual is not None else meta.virtual

    def _wrap_block(self, messages: list[Message]) -> list[Message]:
        if len(messages) == 0:
            return []
        result = [
            Message.new(tag="", timestamp=False).with_content(
                f'<channel name="{self.path}">'
            )
        ]
        result.extend(messages)
        result.append(Message.new(tag="", timestamp=False).with_content(f'</channel>'))
        return result

    def make_full_block(self) -> list[Message]:
        """
        生成完整的消息 block.
        """
        result = []
        if description := self.description_message():
            result.append(description)
        if instruction := self.instruction_message():
            result.append(instruction)
        if failure := self.failure_message():
            result.append(failure)
            return self._wrap_block(result)
        if states := self.states_message():
            result.append(states)
        if context := self.context_messages():
            result.extend(context)
        if interface := self.interface_message(dynamic=True, sustain=True):
            result.append(interface)
        return self._wrap_block(result)

    def make_static_block(self) -> list[Message]:
        """
        virtual 类型的节点没有资格生成 instruction.
        """
        if self.virtual:
            # 虚拟节点不配返回静态信息.
            return []
        result = []
        # 先添加 description.
        if description := self.description_message():
            result.append(description)
        if instruction := self.instruction_message():
            result.append(instruction)
        dynamic = False
        # 只展示可持续消息.
        sustain = True
        if interface := self.interface_message(dynamic=dynamic, sustain=sustain):
            result.append(interface)
        return self._wrap_block(result)

    def make_dynamic_block(self) -> list[Message]:
        """
        生成 Channel Context 的标准逻辑.
        """
        result = []
        if failure := self.failure_message():
            result.append(failure)
            return self._wrap_block(result)
        # virtual 时添加的信息.
        if self.virtual:
            if description := self.description_message():
                result.append(description)
            if instruction := self.instruction_message():
                result.append(instruction)

        # 正常添加 interface.
        sustain = self.virtual
        dynamic = True
        # 正常添加 context.
        if states := self.states_message():
            result.append(states)
        if context_messages := self.context_messages():
            result.extend(context_messages)
        interface_msg = self.interface_message(dynamic=dynamic, sustain=sustain)
        if interface_msg is not None:
            result.append(interface_msg)
        return self._wrap_block(result)

    def failure_message(self) -> Message | None:
        if not self.meta.failure:
            return None
        failure_message = Message.new(tag="failure", timestamp=False)
        failure_message.with_content(self.meta.failure)
        return failure_message

    def context_messages(self) -> list[Message]:
        result = []
        if len(self.meta.context) > 0:
            result.append(Message.new(tag="").with_content("<context>"))
            result.extend(self.meta.context)
            result.append(Message.new(tag="").with_content("</context>"))
        return result

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


def make_dynamic_messages(metas: dict[ChannelFullPath, ChannelMeta]) -> list[Message]:
    """
    按照 ctml 1.0.0 规则, 生成 context messages.
    """
    if len(metas) == 0:
        return []
    # 用单一容器包裹所有的消息. 并且标记自身时间戳.
    result = []
    for channel_path, channel_meta in metas.items():
        # 如果是 virtual, 则需要展示所有讯息.
        prompter = ChannelMetaPrompter(channel_path, channel_meta)
        if block := prompter.make_dynamic_block():
            result.extend(block)
    if len(result) == 0:
        return result
    refresh_at = datetime.datetime.now(dateutil.tz.gettz()).isoformat(timespec="seconds")
    result.insert(
        0,
        Message.new(tag="", timestamp=False).with_content(f'<{MOSS_DYNAMIC} refreshed="{refresh_at}">')
    )
    result.append(Message.new(tag='').with_content(f"</{MOSS_DYNAMIC}>"))
    return result


def make_static_messages(metas: dict[ChannelFullPath, ChannelMeta]) -> str:
    """
    按照 ctml 1.0.0 规则, 生成 instruction messages.
    """
    if len(metas) == 0:
        return ''
    lines = [f'<{MOSS_STATIC}>']
    for channel_path, channel_meta in metas.items():
        # 如果是 virtual, 则需要展示所有讯息.
        prompter = ChannelMetaPrompter(channel_path, channel_meta)
        if block := prompter.make_static_block():
            for msg in block:
                lines.append(msg.to_content_string())
    lines.append(f'</{MOSS_STATIC}>')
    return '\n'.join(lines)
