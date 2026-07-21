from ghoshell_moss.core.duplex.protocol import CommandCallEvent, ChannelEventSerializedError
import pytest


def test_command_call_event_with_unserializable_args():
    async def gen():
        for i in range(100):
            yield i

    event_model = CommandCallEvent(
        name="foo",
        chan="main",
        args=[],
        kwargs={"invalid": gen()}
    )
    assert event_model.name == "foo"
    with pytest.raises(ChannelEventSerializedError):
        event = event_model.to_channel_event()
