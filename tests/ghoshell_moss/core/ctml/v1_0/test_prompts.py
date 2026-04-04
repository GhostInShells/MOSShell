from ghoshell_moss.core.ctml.v1_0.prompts import generate_channel_tree
from ghoshell_moss.core.concepts.channel import ChannelMeta


def test_generate_channel_tree() -> None:
    channels = {
        '': 'main',
        'a.b.c': 'a.b.c\na.b.c',
        'a.b': 'a.b',
        'e.f': 'e.f',
        'g': 'g',
    }
    metas = {}
    for key, value in channels.items():
        metas[key] = ChannelMeta(
            name=key,
            description=value,
        )

    value = generate_channel_tree(metas, with_desc=True)
    assert len(value.split('\n')) == len(channels)
