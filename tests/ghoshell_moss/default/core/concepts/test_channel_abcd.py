from ghoshell_moss.core.concepts.channel import ChannelMeta, Channel
import json


def test_channel_meta_serialize() -> None:
    meta = ChannelMeta()
    js = meta.model_dump_json()
    data = json.loads(js)
    new_meta = ChannelMeta(**data)
    assert new_meta == meta

def test_channel_path_split() -> None:
    main = ''
    paths = Channel.split_channel_path_to_names(main)
    assert len(paths) == 0
    assert Channel.join_channel_path('', *paths) == main