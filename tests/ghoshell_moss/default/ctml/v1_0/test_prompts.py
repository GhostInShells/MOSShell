"""
CTML v1.0.0 prompts unit tests.

Direct assertions on `make_interfaces` — the pure function that renders a
ChannelMeta into the `@decorator + signature` block the model sees.
Kept separate from `test_ctml_v1.py` (which exercises full CTML flows).
"""
import datetime

from ghoshell_moss.core.concepts.channel import ChannelMeta
from ghoshell_moss.core.concepts.command import CommandMeta
from ghoshell_moss.core.ctml.v1_0.prompts import make_interfaces, ChannelMetaPrompter


def _aware(second: int) -> datetime.datetime:
    return datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc) + datetime.timedelta(seconds=second)


def _cmd(name: str, *, always_observe: bool = False, blocking: bool = True) -> CommandMeta:
    return CommandMeta(
        name=name,
        interface=f"async def {name}() -> str:\n    ...",
        always_observe=always_observe,
        blocking=blocking,
    )


def test_make_interfaces_emits_observe_for_always_observe_command():
    """KD11: `always_observe=True` command MUST emit `@observe` line."""
    meta = ChannelMeta(name='vision', commands=[_cmd('look', always_observe=True)])
    rendered = make_interfaces(meta)
    assert "@observe" in rendered
    assert "async def look" in rendered


def test_make_interfaces_no_observe_for_plain_command():
    """Plain command MUST NOT get `@observe` — only opt-in commands do."""
    meta = ChannelMeta(name='calc', commands=[_cmd('add')])
    rendered = make_interfaces(meta)
    assert "@observe" not in rendered
    assert "async def add" in rendered


def test_make_interfaces_observe_and_nonblocking_stack():
    """Both decorators land on the same command; @nonblocking precedes @observe."""
    meta = ChannelMeta(name='mixed', commands=[_cmd('probe', always_observe=True, blocking=False)])
    rendered = make_interfaces(meta)
    nonblock_idx = rendered.index("@nonblocking")
    observe_idx = rendered.index("@observe")
    def_idx = rendered.index("async def probe")
    assert nonblock_idx < observe_idx < def_idx


# --- help rendering ---


def test_interface_message_wraps_help_in_tags():
    """Help text is wrapped in <help> tags inside the interface message."""
    meta = ChannelMeta(
        name="mcp",
        help="3 servers connected: github (12 tools), filesystem (5 tools)",
    )
    prompter = ChannelMetaPrompter("mcp", meta)
    msg = prompter.interface_message(dynamic=True, sustain=True)
    assert msg is not None
    content = msg.to_content_string()
    assert "<help>" in content
    assert "3 servers connected" in content
    assert "</help>" in content


def test_interface_message_no_help_no_tags():
    """When help is empty, no <help> tag is emitted."""
    meta = ChannelMeta(name="shell", commands=[_cmd("exec")])
    prompter = ChannelMetaPrompter("shell", meta)
    msg = prompter.interface_message(dynamic=True, sustain=True)
    assert msg is not None
    content = msg.to_content_string()
    assert "<help>" not in content


def test_interface_message_help_only_no_commands():
    """Help renders even when there are no visible commands."""
    meta = ChannelMeta(name="cli", help="available: git, docker, moss")
    prompter = ChannelMetaPrompter("cli", meta)
    msg = prompter.interface_message(dynamic=True, sustain=True)
    assert msg is not None
    content = msg.to_content_string()
    assert "<help>" in content
    assert "git" in content
    assert "```python" not in content


def test_interface_message_help_and_commands_ordering():
    """Help block precedes command signatures in interface message."""
    meta = ChannelMeta(
        name="mcp",
        help="github: 12 tools",
        commands=[_cmd("list", always_observe=True)],
    )
    prompter = ChannelMetaPrompter("mcp", meta)
    msg = prompter.interface_message(dynamic=True, sustain=True)
    assert msg is not None
    content = msg.to_content_string()
    help_idx = content.index("<help>")
    cmd_idx = content.index("```python")
    assert help_idx < cmd_idx, "help must appear before command signatures"


# --- facade / diff_facade ---


def test_diff_facade_skips_when_created_equal():
    """created 相等 → 快速跳过, 即使 body 已变化也返回空串(存疑的 created 闸门)."""
    created = _aware(0)
    prev = ChannelMeta(name='a', help='v1', created=created)
    cur = ChannelMeta(name='a', help='v2', created=created)
    prompter = ChannelMetaPrompter('a', prev)
    assert prompter.diff_facade(cur) == ""


def test_diff_facade_no_emit_when_body_unchanged():
    """created 不同但 facade body 文本一致 → 不发射."""
    prev = ChannelMeta(name='a', help='same', created=_aware(0))
    cur = ChannelMeta(name='a', help='same', created=_aware(1))
    prompter = ChannelMetaPrompter('a', prev)
    assert prompter.diff_facade(cur) == ""


def test_diff_facade_emits_new_facade_on_change():
    """created 不同且 body 变化 → 发射新 facade, 不含旧内容."""
    prev = ChannelMeta(name='a', help='old help', created=_aware(0))
    cur = ChannelMeta(name='a', help='new help', created=_aware(1))
    prompter = ChannelMetaPrompter('a', prev)
    delta = prompter.diff_facade(cur)
    assert '<channel path="a"' in delta
    assert 'new help' in delta
    assert 'old help' not in delta


def test_facade_body_renders_states_as_string():
    """facade_body 是纯文本投影: states 必须被序列化为字符串, 不混入 Message 对象."""
    meta = ChannelMeta(
        name='a',
        states={'idle': 'not doing anything'},
        current_state='idle',
    )
    prompter = ChannelMetaPrompter('a', meta)
    body = prompter.facade_body()
    assert '<states>' in body
    assert 'Current state: idle' in body


def test_facade_body_failure_short_circuits():
    """failure 存在时 facade_body 只返回 failure, 抑制 help/states/interface."""
    meta = ChannelMeta(name='a', failure='boom', help='help text')
    prompter = ChannelMetaPrompter('a', meta)
    body = prompter.facade_body()
    assert '<failure>' in body
    assert 'boom' in body
    assert 'help text' not in body


def test_full_facade_includes_instruction_and_help():
    """full_facade 包 instruction + facade body, 外层 channel 标记."""
    meta = ChannelMeta(name='a', instruction='do the thing', help='help text')
    prompter = ChannelMetaPrompter('a', meta)
    facade = prompter.full_facade()
    assert '<channel path="a"' in facade
    assert '<instruction>' in facade
    assert 'do the thing' in facade
    assert '<help>' in facade
