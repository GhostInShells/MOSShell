import os

from ghoshell_moss.message import Message
from ghoshell_moss.core.concepts.channel import ChannelMeta, ChannelFullPath
from ghoshell_moss.core.concepts.command import Command
from .constants import MOSS_DYNAMIC, MOSS_STATIC, MAIN_CHANNEL_NAME, CONTENT_COMMAND_NAME
import datetime
import dateutil

__all__ = [
    'make_interfaces',
    'make_dynamic_messages',
    'make_static_messages',
    'ChannelMetaPrompter',
]


def make_interfaces(
        channel_meta: ChannelMeta,
        *,
        dynamic: bool = True,
        sustain: bool = True,
        ordered: bool = False,
) -> str:
    """
    实现 CTML v1.0.0 的 interface 描述.
    """
    # 如果不是 available, 就快速描述不可用.
    commands = channel_meta.commands
    if ordered:
        commands = sorted(commands, key=lambda meta: meta.name)
    if len(commands) == 0:
        return ''
    available_commands = 0
    blocks = ["```python"]
    for cmd_meta in commands:
        if not cmd_meta.visible:
            # ignore invisible
            continue
        elif not cmd_meta.available:
            continue
        elif cmd_meta.dynamic and not dynamic:
            # 排除掉非动态的 command meta.
            continue
        elif not cmd_meta.dynamic and not sustain:
            continue
        # 除了 CONTENT Command 外, 所有的魔术方法都隐藏描述. 但是提供实现.
        elif Command.is_magic_command(cmd_meta.name) and cmd_meta.name != CONTENT_COMMAND_NAME:
            continue

        available_commands += 1
        if not cmd_meta.blocking:
            blocks.append("@nonblocking")
        if cmd_meta.always_observe:
            blocks.append("@observe")
        if cmd_meta.macro:
            blocks.append("@macro")
        blocks.append(cmd_meta.interface)

    # with not available commands
    if available_commands == 0:
        return ''

    blocks.append('```')
    return '\n'.join(blocks)


def _visible_named_notices(named_notices: dict[str, str]) -> dict[str, str]:
    """渲染前的片段过滤: 空值是生产者的静默信号 — 不渲染, 也不宣告变更."""
    return {name: text for name, text in named_notices.items() if text}


def _notice_fragment(name: str, text: str) -> str:
    return f"<{name}>{text}</{name}>"


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
        if interface := self.interface_message(dynamic=True, sustain=True):
            result.append(interface)
        if context := self.context_messages():
            result.extend(context)
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
        parts = []
        if notice := self.notice_text():
            parts.append(notice)
        interface = make_interfaces(self.meta, dynamic=dynamic, sustain=sustain)
        if interface:
            parts.append(interface)
        if not parts:
            return None
        return Message.new(tag="interface", timestamp=False).with_content('\n'.join(parts))

    # --- shell trajectory 版本上下文构建, 针对上下文缓存做优化 --- #

    def notice_text(self) -> str:
        body = self._notice_body()
        if not body:
            return ""
        return "<notice>\n" + body + "\n</notice>"

    def _notice_body(self) -> str:
        """notice 全量正文: 无名 notice 在前, 有名片段按 name 稳定排序在后."""
        parts = []
        if self.meta.notice:
            parts.append(self.meta.notice)
        visible = _visible_named_notices(self.meta.named_notices)
        parts.extend(_notice_fragment(name, visible[name]) for name in sorted(visible))
        return "\n".join(parts)

    def _notice_delta(self, target: ChannelMeta) -> str:
        """逐 name 比较 notice, 只发射变化的片段.

        notice 是温数据, 逐片段比较后一个片段变动不重发其它片段. 空值不渲染也不宣告
        变更 — 模型保留上次读到的内容; 生产者要让模型知道片段没了, 返回 "removed" 这类
        标记文本即可, 那只是内容, 这里不做任何识别.
        """
        lines = []
        if target.notice and target.notice != self.meta.notice:
            lines.append(target.notice)
        prev_visible = _visible_named_notices(self.meta.named_notices)
        next_visible = _visible_named_notices(target.named_notices)
        for name in sorted(set(prev_visible) | set(next_visible)):
            if name in next_visible and next_visible[name] != prev_visible.get(name):
                lines.append(_notice_fragment(name, next_visible[name]))
        if not lines:
            return ""
        return "<notice>\n" + "\n".join(lines) + "\n</notice>"

    def failure_text(self) -> str:
        if self.meta.failure:
            return "<failure>\n" + self.meta.failure + "\n</failure>"
        return ""

    def state_text(self) -> str:
        status_message = self.states_message()
        if status_message:
            return status_message.to_content_string()
        return ""

    def commands_interface_text(self) -> str:
        """commands interface."""
        if len(self.meta.commands) == 0:
            return ""
        interface_blocks = ["<interface>"]
        interface = make_interfaces(self.meta, dynamic=True, sustain=True, ordered=True)
        interface_blocks.append(interface)
        interface_blocks.append("</interface>")
        return '\n'.join(interface_blocks)

    def _make_channel_facade(self, body: str) -> str:
        if not body:
            return ""
        return f'<channel path="{self.path}">\n{body}\n</channel>'

    def _make_facade_body(self, failure: str, states: str, notice: str, interface: str) -> str:
        """四个文本块组装成 facade body. failure 非空时短路, 只返回 failure."""
        if failure:
            return failure
        sections = [section for section in (states, notice, interface) if section]
        return '\n'.join(sections)

    def facade_body(self) -> str:
        """channel 的可变表面"""
        return self._make_facade_body(
            self.failure_text(),
            self.state_text(),
            self.notice_text(),
            self.commands_interface_text(),
        )

    def full_facade(self) -> str:
        """计算 facade"""
        body_parts = []
        if self.meta.instruction:
            body_parts.append(f"<instruction>\n" + self.meta.instruction + "\n</instruction>")
        if facade_text := self.facade_body():
            body_parts.append(facade_text)
        if len(body_parts) == 0:
            return ""
        return self._make_channel_facade("\n".join(body_parts))

    def diff_facade(self, channel_meta: ChannelMeta) -> str:
        """逐 section 对比, 只发射变化的文本块.

        failure 是短路面: 两边不等 → 发 target 全板 (新板即 failure, 或 failure 被清除后的健康板);
        相同且非空 → 该板只有 failure, 无差可发. 其余 states / interface 整体比较, notice 再往下
        逐 name 比较 (见 _notice_delta), 只把变化的文本拼进 delta.
        """
        if channel_meta.created == self.meta.created:
            return ""
        target = ChannelMetaPrompter(self.path, channel_meta)

        self_failure = self.failure_text()
        target_failure = target.failure_text()
        if self_failure != target_failure:
            return target._make_channel_facade(target.facade_body())
        if self_failure:
            # 两边 failure 相同且非空 → facade 只含 failure, 已相等.
            return ""

        sections = []
        if self.state_text() != target.state_text():
            sections.append(target.state_text())
        if notice_delta := self._notice_delta(channel_meta):
            sections.append(notice_delta)
        if self.commands_interface_text() != target.commands_interface_text():
            sections.append(target.commands_interface_text())
        if not sections:
            return ""
        return target._make_channel_facade("\n".join(sections))

    def dynamic_context_messages(self) -> list[Message]:
        if not self.meta.context:
            return []
        result = [
            Message.new(tag="", timestamp=False).with_content(
                f'<channel path="{self.path}">'
            )
        ]
        result.extend(self.meta.context)
        result.append(Message.new(tag="", timestamp=False).with_content(f'</channel>'))
        return result


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
