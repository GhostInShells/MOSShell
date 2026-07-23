"""
CTML v1.0.0 prompts unit tests.

Direct assertions on `make_interfaces` — the pure function that renders a
ChannelMeta into the `@decorator + signature` block the model sees.
Kept separate from `test_ctml_v1.py` (which exercises full CTML flows).
"""
from ghoshell_moss.core.concepts.channel import ChannelMeta
from ghoshell_moss.core.concepts.command import CommandMeta
from ghoshell_moss.core.ctml.v1_0.prompts import make_interfaces


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
