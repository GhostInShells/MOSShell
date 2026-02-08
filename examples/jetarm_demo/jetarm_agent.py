import asyncio

from ghoshell_container import Container

from ghoshell_moss.core.shell import new_shell
from ghoshell_moss.speech import make_baseline_tts_speech
from ghoshell_moss.speech.player.pyaudio_player import PyAudioStreamPlayer
from ghoshell_moss.speech.volcengine_tts import VolcengineTTS, VolcengineTTSConf
from ghoshell_moss.transports.zmq_channel.zmq_channel import ZMQChannelProxy
from ghoshell_moss_contrib.agent import ModelConf, SimpleAgent
from ghoshell_moss_contrib.agent.chat import ConsoleChat

container = Container(name="jetarm_agent_container")

ADDRESS = "tcp://192.168.1.15:9527"
"""填入正确的 ip, 需要先对齐 jetarm_ws 运行的机器设备和监听的端口. """


async def run_agent():
    # 创建 Shell
    shell = new_shell(container=container)

    jetarm_chan = ZMQChannelProxy(
        name="jetarm",
        address=ADDRESS,
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
    # 运行异步主函数
    asyncio.run(run_agent())


# 运行主函数
if __name__ == "__main__":
    main()
