from ghoshell_moss.core.concepts.channel import Channel


class AdapterChannel(Channel):
    """
    用来给 Channel 做别名和修改.
    """

    def __init__(
        self,
        name: str,
        description: str,
        origin: Channel,
    ) -> None:
        pass
