"""网页收藏夹，用 id 或 URL 打开浏览器网页 | 交互能力 | alpha

Example:
    from ghoshell_moss.channels.web_bookmark import new_web_bookmark_channel
    chan = new_web_bookmark_channel(name="web")
"""

import webbrowser

from ghoshell_container import IoCContainer
from pydantic import BaseModel, Field

from ghoshell_moss.contracts.configs import ConfigType, ConfigStore
from ghoshell_moss.core import PyChannel

__all__ = ["new_web_bookmark_channel", "WebConfig", "WebInfo"]


class WebInfo(BaseModel):
    id: str = Field(default="", description="网页的唯一 id，用来让模型快速调用")
    url: str = Field(default="", description="网页的URL")
    description: str = Field(default="", description="网页的描述")


class WebConfig(ConfigType):
    web_list: list[WebInfo] = Field(default_factory=list, description="网页列表")

    @classmethod
    def conf_name(cls) -> str:
        return "web"


def new_web_bookmark_channel(
    *,
    name: str = "web_bookmarks",
    description: str = "网页收藏夹，用 id 或 URL 打开网页",
) -> PyChannel:
    """创建网页收藏夹 channel。

    ConfigStore 在 bootstrap 时自动从 IoC 容器获取，加载 web 配置。
    指令函数惰性求值，运行时展示 .moss_ws/configs/web.yaml 中的收藏列表。

    :param name: channel 名称（CTML 标签名）
    :param description: channel 描述
    """
    _web_list: list[WebInfo] = []

    chan = PyChannel(name=name, description=description)

    async def open_web(id_or_url: str) -> None:
        """用给定的 id 或 URL 打开网页。

        :param id_or_url: 网页 id（收藏列表中的）或完整 URL
        """
        url = id_or_url
        for info in _web_list:
            if info.id == id_or_url:
                url = info.url
                break
        webbrowser.open(url)

    chan.build.command(always_observe=False)(open_web)

    @chan.build.instruction
    def web_bookmark_instruction() -> str:
        bookmark_list = "\n".join(
            [f"- {v.id}: {v.description}" for v in _web_list]
        )
        return (
            "网页收藏夹。用 open_web 命令打开网页。\n"
            "\n"
            f"已收藏的网页：\n{bookmark_list}\n"
            "\n"
            "如果 id 不在列表中，则该收藏不存在。坦诚告知用户即可。"
        )

    def _on_bootstrap(channel, container: IoCContainer):
        store = container.force_fetch(ConfigStore)
        config = store.get_or_create(WebConfig(web_list=[]))
        _web_list[:] = config.web_list

    chan.on_bootstrap(_on_bootstrap)

    return chan
