import asyncio

from ghoshell_container import Container
import argparse

from ghoshell_moss.core.shell import new_shell
from ghoshell_moss.speech import make_baseline_tts_speech
from ghoshell_moss.speech.player.pyaudio_player import PyAudioStreamPlayer
from ghoshell_moss.speech.volcengine_tts import VolcengineTTS, VolcengineTTSConf
from ghoshell_moss.transports.zmq_channel.zmq_channel import ZMQChannelProxy
from ghoshell_moss_contrib.agent import ModelConf, SimpleAgent
from ghoshell_moss_contrib.agent.chat import ConsoleChat
from ghoshell_moss_contrib.example_ws import workspace_container, get_container
from pathlib import Path

ADDRESS = "tcp://192.168.1.15:9527"
"""填入正确的 ip, 需要先对齐 jetarm_ws 运行的机器设备和监听的端口. """


async def run_agent(address: str = ADDRESS, container: Container | None = None):
    container = container or get_container()
    # 创建 Shell
    shell = new_shell(container=container)

    jetarm_chan = ZMQChannelProxy(
        name="jetarm",
        address=address,
    )

    shell.main_channel.import_channels(jetarm_chan)
    player = PyAudioStreamPlayer()
    tts = VolcengineTTS(
        conf=VolcengineTTSConf(
            default_speaker="saturn_zh_female_keainvsheng_tob",
        )
    )

    agent = SimpleAgent(
        instruction="""
你是 灵灵-1 号, 灵枢 (Ghost In Shells) 项目的第一个具身机器人. 

你拥有 JetArm 的 6 dof 机械臂身体, 行为表现像是一个仿生动物 (蛇形). 

你会一边说话, 一边做各种动作. 
""",
        shell=shell,
        speech=make_baseline_tts_speech(player=player, tts=tts),
        model=ModelConf(
            kwargs={
                "thinking": {
                    "type": "disabled",
                },
            },
        ),
        chat=ConsoleChat(),
    )
    await agent.run()


def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="运行 jetarm agent 程序")

    # 添加 --address 参数，设置默认值
    parser.add_argument(
        "--address",
        type=str,
        default="tcp://192.168.1.15:9527",
        help="代理地址，默认值: tcp://192.168.1.15:9527"
    )

    # 解析命令行参数
    args = parser.parse_args()

    ws_dir = Path(__file__).resolve().parent.parent.joinpath('.workspace')
    with workspace_container(workspace_dir=ws_dir) as container:
        # 运行异步主函数，传入地址参数
        asyncio.run(run_agent(address=args.address, container=container))


# 运行主函数
if __name__ == "__main__":
    main()
