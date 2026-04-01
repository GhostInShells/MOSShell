from ghoshell_moss.core.concepts.channel import ChannelMeta
import json


def test_channel_meta_serialize() -> None:
    meta = ChannelMeta()
    js = meta.model_dump_json()
    data = json.loads(js)
    new_meta = ChannelMeta(**data)
    assert new_meta == meta
