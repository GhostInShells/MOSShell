from pathlib import Path
from ghoshell_moss.message import Message
from ghoshell_moss.core.concepts.channel import ChannelMeta

VERSION = "v0_2_0.zh"

__all__ = [
    'get_moss_ctml_meta_instruction',
]


def get_moss_ctml_meta_instruction(version: str = VERSION) -> str:
    path = Path(__file__).parent.joinpath(f"prompts/ctml_{version}.md")
    with path.open() as f:
        return f.read()


def make_channel_context_messages(channel_path: str, channel_meta: ChannelMeta) -> list[Message]:
    path_name = channel_path or "__main__"
    message = Message.new(role="system")
    pass


def make_channel_instruction_messages(channel_path: str, channel_meta: ChannelMeta) -> list[Message]:
    messages = []
    interface_message = Message.new(role="system")
    # 生成代码 interface.
    for channel_path, channel_meta in self._channel_metas.items():
        path_name = channel_path or "__main__"
        not_available = "" if channel_meta.available else "(not available)"
        interface_message.with_content(
            f"=== interface:{path_name} {not_available}===\n",
            channel_meta.description,
            "\n\n```python\n" + make_command_interface(channel_meta.commands) + "\n```\n",
            f"\n=== end interface:{path_name} ===\n",
        )
    messages.append(interface_message)
    for channel_path, channel_meta in self._channel_metas.items():
        path_name = channel_path or "__main__"
        if not channel_meta.available:
            continue
        if len(channel_meta.instructions) > 0:
            first = None
            last = None
            for channel_instruction_message in channel_meta.instructions:
                if not channel_instruction_message.is_done():
                    continue
                elif first is None:
                    first = channel_instruction_message.get_copy()
                    first.contents.insert(0, Text.new(f"\n=== instructions:{path_name} ===\n").to_content())
                    messages.append(first)
                    last = first
                    continue
                else:
                    last = channel_instruction_message.get_copy()
                    messages.append(last)
            if last:
                last.contents.append(
                    Text.new(f"\n=== end instructions:{path_name} ===\n").to_content(),
                )
    return messages
