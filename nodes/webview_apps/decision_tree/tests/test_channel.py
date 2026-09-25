import pytest

from ghoshell_decision_tree.channel import build_decision_tree_channel
from ghoshell_decision_tree.store import DecisionTreeStore

from fakes import FakeSurface, SignalRecorder


@pytest.fixture
def store(tmp_path):
    return DecisionTreeStore(root=tmp_path)


def _channel(store):
    surf = FakeSurface()
    sig = SignalRecorder()
    chan = build_decision_tree_channel(store, surface=surf, signaler=sig)
    return chan, surf, sig


@pytest.mark.asyncio
async def test_channel_exposes_commands(store):
    chan, _, _ = _channel(store)
    async with chan.bootstrap() as runtime:
        for name in (
            "create_tree", "open_tree", "trees",
            "create_node", "link_node", "update_node",
            "read", "focus", "history",
        ):
            assert runtime.get_command(name) is not None, name


@pytest.mark.asyncio
async def test_create_tree_and_node_via_commands(store):
    chan, surf, _ = _channel(store)
    async with chan.bootstrap() as runtime:
        out = await runtime.execute_command(
            "create_tree", args=('{"root":"t","name":"t","title":"Tree"}',)
        )
        assert "created" in out
        out = await runtime.execute_command(
            "create_node", args=('{"tree":"t","name":"a","title":"A"}',)
        )
        assert "created" in out
        out = await runtime.execute_command("read", args=("t", "a"))
        assert "A" in out
        assert "open" in out
        assert any(f["type"] == "state" for f in surf.frames)


@pytest.mark.asyncio
async def test_invalid_status_surfaces_as_observe(store):
    chan, _, _ = _channel(store)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command(
            "create_tree", args=('{"root":"t","name":"t","title":"Tree"}',)
        )
        await runtime.execute_command(
            "create_node", args=('{"tree":"t","name":"a","title":"A"}',)
        )
        with pytest.raises(Exception):
            await runtime.execute_command(
                "update_node", args=('{"tree":"t","name":"a","status":"nope"}',)
            )


@pytest.mark.asyncio
async def test_notice_reports_trees_and_url(store):
    chan, surf, _ = _channel(store)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command(
            "create_tree", args=('{"root":"t","name":"t","title":"Tree"}',)
        )
        await runtime.refresh_metas()
        named = runtime.self_meta().named_notices
        assert named["url"] == surf.url
