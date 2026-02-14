import asyncio
import importlib.util
import os
import sys
from os.path import dirname, join

import live2d.v3 as live2d
import pygame
from ghoshell_container import Container

from ghoshell_moss.speech import Speech, make_baseline_tts_speech
from ghoshell_moss.speech.player.pyaudio_player import PyAudioStreamPlayer
from ghoshell_moss.speech.volcengine_tts import VolcengineTTS, VolcengineTTSConf
from ghoshell_moss_contrib.agent import ModelConf, SimpleAgent

current_dir = os.path.dirname(os.path.abspath(__file__))
if importlib.util.find_spec("miku_channels") is None:
    # 加载当前路径.
    sys.path.append(current_dir)

import pathlib

from miku_channels.arm import left_arm_chan, right_arm_chan
from miku_channels.body import body_chan
from miku_channels.elbow import left_elbow_chan, right_elbow_chan
from miku_channels.expression import expression_chan
from miku_channels.eye import eye_chan
from miku_channels.eyebrow import eyebrow_left_chan, eyebrow_right_chan
from miku_channels.head import head_chan
from miku_channels.leg import left_leg_chan, right_leg_chan
from miku_channels.necktie import necktie_chan
from miku_provider import init_live2d, init_pygame

from ghoshell_moss.core.shell import new_shell
from ghoshell_moss_contrib.example_ws import get_example_speech, workspace_container

# 全局状态
WIDTH = 600
HEIGHT = 800

model: live2d.LAppModel | None = None

# 初始化Pygame和Live2D


async def speak(duration: float = 5.0, speed: float = 1.0, max_open: float = 0.9, min_open: float = 0.0):
    """
    说话的嘴部动作

    @param duration: 动画持续时间，单位秒
    @param speed: 开合速度参数，值越大速度越快，默认1.0
    @param max_open: 最大张开程度，0 到 1 之间的浮点数，默认为0.7
    @param min_open: 最小张开程度，0 到 1 之间的浮点数，默认为0.0
    """
    from miku_channels.motions import open_close

    PARAM = "ParamMouthOpenY"
    # 特殊处理嘴部动作，说话通常从张开开始

    # 调用通用动画函数，注意初始方向设置为打开
    final_value = await open_close(
        model=model,
        param_name=PARAM,
        duration=duration,
        speed=speed,
        max_value=max_open,
        min_value=min_open,
        initial_direction="open",  # 说话从打开开始
    )
    # 确保最终状态是完全闭合
    model.SetParameterValue(PARAM, 0.0)


speaking_event = asyncio.Event()


def start_speak(*args):
    speaking_event.set()


def stop_speak(*args):
    speaking_event.clear()


async def run_agent(container: Container, speech: Speech | None = None):
    loop = asyncio.get_running_loop()

    # 创建 Shell
    shell = new_shell(container=container)

    async def speaking():
        try:
            while not shell.is_close():
                if speaking_event.is_set():
                    await speak(duration=0.3)
                else:
                    await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass

    speaking_task = loop.create_task(speaking())

    head_chan.import_channels(
        expression_chan,
        # mouth_chan, 不把嘴巴给大模型调用.
        eye_chan,
        eyebrow_left_chan,
        eyebrow_right_chan,
    )
    body_chan.import_channels(
        head_chan,
        left_arm_chan,
        right_arm_chan,
        left_elbow_chan,
        right_elbow_chan,
        necktie_chan,
        left_leg_chan,
        right_leg_chan,
    )
    shell.main_channel.import_channels(body_chan)
    player = PyAudioStreamPlayer()
    player.on_play(start_speak)
    player.on_play_done(stop_speak)
    speech = speech or container.get(Speech)
    if speech is None:
        tts = VolcengineTTS(conf=VolcengineTTSConf(default_speaker="saturn_zh_female_keainvsheng_tob"))
        speech = make_baseline_tts_speech(player=player, tts=tts)

    agent = SimpleAgent(
        instruction="你是miku, 拥有 live2d 数字人躯体. 你是可爱和热情的数字人. ",
        shell=shell,
        speech=speech,
        model=ModelConf(
            kwargs={
                "thinking": {
                    "type": "disabled",
                },
            },
        ),
        container=container,
    )
    await agent.run()
    speaking_task.cancel()
    await speaking_task


async def run_agent_and_render(container: Container, speech: Speech | None = None):
    # 初始化 Pygame 和 Live2D
    global model
    screen, display = init_pygame()
    model_path = join(dirname(__file__), "model/miku.model3.json")
    model = init_live2d(model_path, container)

    # 保持窗口打开，直到用户关闭
    running = True
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    # 创建一个任务来运行 agent
    agent_task = asyncio.create_task(run_agent(container, speech))

    try:
        while running and not agent_task.done():
            # 处理pygame事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # 更新和绘制模型
            model.Update()
            live2d.clearBuffer(255, 255, 255, 150.0)
            model.Draw()

            # 显示完成信息
            info_text = font.render("演示完成，按关闭按钮退出", True, (255, 255, 255))
            screen.blit(info_text, (10, 10))

            pygame.display.flip()
            clock.tick(60)

            # 让出控制权给其他协程，确保agent也能运行
            await asyncio.sleep(0.01)
    except asyncio.CancelledError:
        pass
    finally:
        # 取消agent任务
        if not agent_task.done():
            agent_task.cancel()
            try:
                await agent_task
            except asyncio.CancelledError:
                pass

        # 清理资源
        live2d.dispose()
        pygame.quit()


WORKSPACE_DIR = pathlib.Path(__file__).parent.parent.joinpath(".workspace")


def main():
    # 运行异步主函数
    with workspace_container(WORKSPACE_DIR) as container:
        speech = get_example_speech(container, default_speaker="saturn_zh_female_keainvsheng_tob")
        asyncio.run(run_agent_and_render(container, speech))


# 运行主函数
if __name__ == "__main__":
    main()
