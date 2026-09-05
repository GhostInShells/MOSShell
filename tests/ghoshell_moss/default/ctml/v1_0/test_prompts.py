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


# --- notice rendering ---


def test_interface_message_wraps_notice_in_tags():
    """Notice text is wrapped in <notice> tags inside the interface message."""
    meta = ChannelMeta(
        name="mcp",
        notice="3 servers connected: github (12 tools), filesystem (5 tools)",
    )
    prompter = ChannelMetaPrompter("mcp", meta)
    msg = prompter.interface_message(dynamic=True, sustain=True)
    assert msg is not None
    content = msg.to_content_string()
    assert "<notice>" in content
    assert "3 servers connected" in content
    assert "</notice>" in content


def test_interface_message_no_notice_no_tags():
    """When notice is empty, no <notice> tag is emitted."""
    meta = ChannelMeta(name="shell", commands=[_cmd("exec")])
    prompter = ChannelMetaPrompter("shell", meta)
    msg = prompter.interface_message(dynamic=True, sustain=True)
    assert msg is not None
    content = msg.to_content_string()
    assert "<notice>" not in content


def test_interface_message_notice_only_no_commands():
    """Notice renders even when there are no visible commands."""
    meta = ChannelMeta(name="cli", notice="available: git, docker, moss")
    prompter = ChannelMetaPrompter("cli", meta)
    msg = prompter.interface_message(dynamic=True, sustain=True)
    assert msg is not None
    content = msg.to_content_string()
    assert "<notice>" in content
    assert "git" in content
    assert "```python" not in content


def test_interface_message_notice_and_commands_ordering():
    """Notice block precedes command signatures in interface message."""
    meta = ChannelMeta(
        name="mcp",
        notice="github: 12 tools",
        commands=[_cmd("list", always_observe=True)],
    )
    prompter = ChannelMetaPrompter("mcp", meta)
    msg = prompter.interface_message(dynamic=True, sustain=True)
    assert msg is not None
    content = msg.to_content_string()
    notice_idx = content.index("<notice>")
    cmd_idx = content.index("```python")
    assert notice_idx < cmd_idx, "notice must appear before command signatures"


# --- facade / diff_facade ---


def test_diff_facade_skips_when_created_equal():
    """created 相等 → 快速跳过, 即使 body 已变化也返回空串(存疑的 created 闸门)."""
    created = _aware(0)
    prev = ChannelMeta(name='a', notice='v1', created=created)
    cur = ChannelMeta(name='a', notice='v2', created=created)
    prompter = ChannelMetaPrompter('a', prev)
    assert prompter.diff_facade(cur) == ""


def test_diff_facade_no_emit_when_body_unchanged():
    """created 不同但 facade body 文本一致 → 不发射."""
    prev = ChannelMeta(name='a', notice='same', created=_aware(0))
    cur = ChannelMeta(name='a', notice='same', created=_aware(1))
    prompter = ChannelMetaPrompter('a', prev)
    assert prompter.diff_facade(cur) == ""


def test_diff_facade_emits_new_facade_on_change():
    """created 不同且 body 变化 → 发射新 facade, 不含旧内容."""
    prev = ChannelMeta(name='a', notice='old help', created=_aware(0))
    cur = ChannelMeta(name='a', notice='new help', created=_aware(1))
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
    """failure 存在时 facade_body 只返回 failure, 抑制 notice/states/interface."""
    meta = ChannelMeta(name='a', failure='boom', notice='notice text')
    prompter = ChannelMetaPrompter('a', meta)
    body = prompter.facade_body()
    assert '<failure>' in body
    assert 'boom' in body
    assert 'notice text' not in body


def test_full_facade_includes_instruction_and_notice():
    """full_facade 包 instruction + facade body, 外层 channel 标记."""
    meta = ChannelMeta(name='a', instruction='do the thing', notice='notice text')
    prompter = ChannelMetaPrompter('a', meta)
    facade = prompter.full_facade()
    assert '<channel path="a"' in facade
    assert '<instruction>' in facade
    assert 'do the thing' in facade
    assert '<notice>' in facade


# --- diff_facade 逐 section 粒度 (decoupled delta) ---


def test_diff_facade_emits_only_changed_notice():
    """只 notice 变 → delta 只含 <notice>, 不含 <interface>/<states>."""
    prev = ChannelMeta(name='a', notice='old', created=_aware(0))
    cur = ChannelMeta(name='a', notice='new', created=_aware(1))
    delta = ChannelMetaPrompter('a', prev).diff_facade(cur)
    assert '<channel path="a"' in delta
    assert '<notice>' in delta
    assert 'new' in delta
    assert '<interface>' not in delta


def test_diff_facade_emits_only_changed_interface():
    """只 interface 变 → delta 只含 <interface>, 不含 <notice>."""
    prev = ChannelMeta(name='a', notice='same', commands=[_cmd('list')], created=_aware(0))
    cur = ChannelMeta(name='a', notice='same', commands=[_cmd('list'), _cmd('extra')], created=_aware(1))
    delta = ChannelMetaPrompter('a', prev).diff_facade(cur)
    assert '<interface>' in delta
    assert 'async def extra' in delta
    assert '<notice>' not in delta


def test_diff_facade_emits_only_changed_states():
    """只 states 变 → delta 只含 states 块, 不含 <notice>/<interface>."""
    prev = ChannelMeta(name='a', states={'idle': 'nothing'}, current_state='idle', created=_aware(0))
    cur = ChannelMeta(name='a', states={'idle': 'nothing', 'busy': 'working'}, current_state='busy', created=_aware(1))
    delta = ChannelMetaPrompter('a', prev).diff_facade(cur)
    assert '<states>' in delta
    assert 'Current state: busy' in delta
    assert '<notice>' not in delta
    assert '<interface>' not in delta


def test_diff_facade_emits_both_when_notice_and_interface_change():
    """notice 与 interface 同时变 → 两者都进 delta."""
    prev = ChannelMeta(name='a', notice='old', commands=[_cmd('list')], created=_aware(0))
    cur = ChannelMeta(name='a', notice='new', commands=[_cmd('list'), _cmd('extra')], created=_aware(1))
    delta = ChannelMetaPrompter('a', prev).diff_facade(cur)
    assert '<notice>' in delta
    assert 'new' in delta
    assert '<interface>' in delta
    assert 'async def extra' in delta


def test_diff_facade_failure_change_short_circuits():
    """failure 变 → delta 只发 failure (含 target 健康板), 不发 notice/interface."""
    prev = ChannelMeta(name='a', failure='boom-old', notice='n', created=_aware(0))
    cur = ChannelMeta(name='a', failure='boom-new', notice='n', created=_aware(1))
    delta = ChannelMetaPrompter('a', prev).diff_facade(cur)
    assert '<failure>' in delta
    assert 'boom-new' in delta
    assert '<notice>' not in delta
