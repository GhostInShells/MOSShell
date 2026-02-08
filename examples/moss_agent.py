import os.path
import pathlib
import asyncio

from ghoshell_common.contracts import Workspace, LoggerItf
from ghoshell_container import Container

from ghoshell_moss_contrib.example_ws import workspace_container, get_example_speech
from ghoshell_moss.channels.mac_channel import new_mac_control_channel
from ghoshell_moss_contrib.channels.mermaid_draw import new_mermaid_chan
from ghoshell_moss_contrib.channels.web_bookmark import build_web_bookmark_chan

from ghoshell_moss_contrib.agent import SimpleAgent, ModelConf, ConsoleChat
from ghoshell_moss.core.shell import new_shell
from ghoshell_moss.transports.zmq_channel.zmq_hub import ZMQChannelHub, ZMQHubConfig, ZMQProxyConfig

# 不着急删除, 方便自测时开启.
from ghoshell_moss_contrib.channels.screen_capture import ScreenCapture
from ghoshell_moss.transports.zmq_channel.zmq_hub import ZMQChannelProxy

"""
说明: 

目标是提供一个 MOSS 项目自解释的 Agent, 并作为通用的自解释 Agent 的模板.
预计未来的项目自解释 Agent (py版), 可以独立配置在任意项目中, 可能用 __project_agent__.py 之类的方式约定.

但由于 MOSS 的 Alpha 版本诸多核心功能尚未完成, 所以先人工定义这个模式, 方便测试验证. 
"""

CURRENT_DIR = pathlib.Path(__file__).parent
WORKSPACE_DIR = CURRENT_DIR.joinpath('.workspace').absolute()


def load_instructions(con: Container, files: list[str]) -> str:
    """
    读取 agent 的 instructions
    TODO: 暂时先这么做. Beta 版本会做一个正式的 Agent. Alpha 版本先临时用测试的 simple agent 攒一个.
    """
    ws = con.force_fetch(Workspace)
    instru_storage = ws.configs().sub_storage('moss_instructions')
    instructions = []
    for filename in files:
        content = instru_storage.get(filename)
        instructions.append(content.decode('utf-8'))

    return "\n\n".join(instructions)


def run_moss_agent(container: Container):
    """
    运行 moss agent.
    todo: 是一个临时的实现. Beta 版本要做成文件级别自动发现的 agent.
    """
    logger = container.get(LoggerItf)

    # hub channel
    zmq_hub = ZMQChannelHub(
        config=ZMQHubConfig(
            name="hub",
            description="可以启动指定的子通道并运行.",

            # todo: 当前版本全部基于约定来做. 快速验证.
            root_dir=str(CURRENT_DIR.joinpath('moss_zmq_channels').absolute()),

            # todo:
            #    zmq hub 不是 MOSS 架构的目标范式, Alpha 版本未完成 LocalChannelApplications 模块
            #    所以先用 zmq hub 来验证跨进程打开的效果.
            #    已知的问题: 异常退出时, 相关子进程会无法关闭. 需要手动 kill. 可以 ps aux | grep channel 查看.
            #    目标仍然是实现一个本地化的, 可以自动发现 channel 的父进程管理器.
            #    目前需要人为匹配好 子channel, 包括监听的端口. 未来会全部改成自动的.
            proxies={
                "miku": ZMQProxyConfig(
                    script="miku_app.py",
                    description="可以打开你的 live2d 数字人 miku (初音) 的躯体.",
                    address="tcp://localhost:5555",
                ),
                "vision": ZMQProxyConfig(
                    script="vision_app.py",
                    description="你的视觉通道, 通过这个通道你可以使用摄像头看见",
                    address="tcp://localhost:5557",
                ),
                # todo: 暂时未实装, 预计实装它.
                # "no_ppt": ZMQProxyConfig(
                #     script="no_ppt_channel.py",
                #     description="",
                #     address="tcp://localhost:5558",
                # )
            },
        ),
        logger=logger,
    )

    speech = get_example_speech(container)
    shell = new_shell(container=container, speech=speech)
    shell.main_channel.import_channels(
        zmq_hub.as_channel(),
        # 浏览器
        build_web_bookmark_chan(container),
        new_mermaid_chan(),

        # todo: 开启这个模块, 可以让 Agent 通过 JXA 操作 mac 电脑. 不过配套的 prompt 并不完善.
        # new_mac_control_channel(description="使用 jxa 语法来操作当前所在 mac, 有明确 mac 操作命令要求时才允许使用."),
        # todo: 开启这个模块, 可以让 Agent 选择屏幕截图.
        # ScreenCapture(logger=logger).as_channel(),

        # todo: 如果有 Jetarm demo 的话... 可以开启, 让 moss 可以同时控制数字人.
        # ZMQChannelProxy(
        #     name="jetarm",
        #     address="tcp://192.168.1.15:9527",
        # ),
    )

    instructions = load_instructions(
        container,
        [
            'persona.md',
            'behaviors.md',
        ]
    )

    agent = SimpleAgent(
        talker="",  # 这里可以放入自己的名字.
        container=container,
        instruction=instructions,
        chat=ConsoleChat(logger=logger),
        model=ModelConf(
            kwargs=dict(
                thinking=dict(
                    type="disabled",
                )
            ),
        ),
        shell=shell,
    )

    async def run_agent():
        agent_task = asyncio.create_task(agent.run())
        try:
            await agent_task
        except KeyboardInterrupt:
            pass
        finally:
            if not agent_task.done():
                agent_task.cancel()
                try:
                    await agent_task
                except asyncio.CancelledError:
                    pass

    asyncio.run(run_agent())


if __name__ == '__main__':
    with workspace_container(WORKSPACE_DIR) as _container:
        run_moss_agent(_container)
