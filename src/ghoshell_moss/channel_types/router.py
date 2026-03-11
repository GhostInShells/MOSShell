from ghoshell_moss.core.concepts.channel import MutableChannel


class RouterChannel(MutableChannel):
    """
    todo: 可以路由到多个子 Channel. 通过打开和关闭, 切换展示出来的子 Channel.
        可以认为是 PyChannel 的一种升级版.
    """

    async def open(self, *channels: str) -> None:
        pass

    async def hide(self, *channels: str) -> None:
        pass
