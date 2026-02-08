import webbrowser

from ghoshell_common.contracts import Workspace, WorkspaceConfigs, YamlConfig
from ghoshell_container import IoCContainer
from pydantic import BaseModel, Field

from ghoshell_moss import PyChannel

"""
预计实现一个网页收藏夹, 让 AI 来收藏网页, 并且可以随时打开.
在 MOSS 介绍自身项目的时候, 可以随时打开网页. 

现阶段收藏夹的配置需要人工指定, 耦合目录, 不太好. 

Beta 版本希望实现的是: 
1. 将网页收藏夹改造为 ChannelApp 范式. 
2. 用 pyqt6 实现一个简单的界面, 也允许人添加/删除/修改
3. 支持收藏夹目录, AI 可以看到目录树. 
4. 支持 AI 去 pin 不超过 n 个地址. 
5. 指定目录, 用来存放 AI 生成的收藏数据. 
"""

web_chan = PyChannel(name="web_bookmarks", description="这是一个网页收藏夹. 可以用来打开指定的网页. ")


class WebInfo(BaseModel):
    id: str = Field(default="", description="网页的唯一 id，用来让模型快速调用")
    url: str = Field(default="", description="网页的URL")
    description: str = Field(default="", description="网页的描述")


class WebConfig(YamlConfig):
    relative_path = "web.yaml"

    web_list: list[WebInfo] = Field(default_factory=list, description="网页列表")

    @classmethod
    def load(cls, container: IoCContainer):
        ws = container.force_fetch(Workspace)
        configs = WorkspaceConfigs(ws.configs())
        return configs.get_or_create(WebConfig())

    def to_str(self):
        return "\n".join([f"- {v.id}: {v.description}" for i, v in enumerate(self.web_list)])

    def to_web_info_map(self) -> dict[str, WebInfo]:
        web_info_map: dict[str, WebInfo] = {}
        for web_info in self.web_list:
            web_info_map[web_info.id] = web_info
        return web_info_map


def build_web_bookmark_chan(container: IoCContainer) -> PyChannel:
    web_config = WebConfig.load(container)
    web_info_map = web_config.to_web_info_map()

    async def open_web(id_or_url: str):
        url = id_or_url
        if id_or_url in web_info_map:
            url = web_info_map[id_or_url].url
        webbrowser.open(url)

    open_web_docstring = f"""
用给定的 id 去打开一个网页。存在的网页 id：
{web_config.to_str()}

:param id_or_url: 要打开的网页的URL, 或者一个指定的 web id。

注意！！id 不存在于以上列表的，就并不存在。
你需要坦诚的理解到这点。
"""

    web_chan.build.command(doc=open_web_docstring)(open_web)

    return web_chan
