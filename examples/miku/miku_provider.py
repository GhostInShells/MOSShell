import asyncio
import importlib.util
import os
import sys
from os.path import dirname, join

import live2d.v3 as live2d
import pygame
from ghoshell_container import Container, get_container

current_dir = os.path.dirname(os.path.abspath(__file__))
if importlib.util.find_spec(miku_channels) is None:
    sys.path.append(current_dir)

from miku_channels.arm import left_arm_chan, right_arm_chan
from miku_channels.body import body_chan
from miku_channels.elbow import left_elbow_chan, right_elbow_chan
from miku_channels.expression import expression_chan
from miku_channels.eye import eye_chan
from miku_channels.eyebrow import eyebrow_left_chan, eyebrow_right_chan
from miku_channels.head import head_chan
from miku_channels.leg import left_leg_chan, right_leg_chan
from miku_channels.necktie import necktie_chan

from ghoshell_moss import Channel
from ghoshell_moss.transports.zmq_channel import ZMQChannelProvider

# 全局状态
model: live2d.LAppModel | None = None
WIDTH = 600
HEIGHT = 800


# 初始化Pygame和Live2D
def init_pygame():
    pygame.init()
    display = (WIDTH, HEIGHT)
    screen = pygame.display.set_mode(display, pygame.DOUBLEBUF | pygame.OPENGL)
    pygame.display.set_caption("Digital Human Demo with PyChannel")
    return screen, display


# 初始化Live2D模型
def init_live2d(model_path: str, con: Container | None = None) -> live2d.LAppModel:
    global model
    live2d.init()
    live2d.glInit()
    model = live2d.LAppModel()
    model.LoadModelJson(model_path)
    model.Resize(WIDTH, HEIGHT)
    # model.SetAutoBlinkEnable(False)
    # model.SetAutoBreathEnable(True)
    con = con or get_container()
    con.bind(live2d.LAppModel, model)
    return model


def miku_body() -> Channel:
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
    return body_chan


async def run_game_with_zmq_provider(address: str = "tcp://localhost:5555", con: Container | None = None):
    # 初始化 Pygame 和 Live2D
    screen, display = init_pygame()
    model_path = join(dirname(__file__), "model/miku.model3.json")
    init_live2d(model_path, con=con)

    # 保持窗口打开，直到用户关闭
    running = True
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    # 创建一个任务来运行 agent
    provider = ZMQChannelProvider(
        address=address,
        container=con,
    )
    _miku = miku_body()
    task = asyncio.create_task(provider.run_until_closed(_miku))

    try:
        while running:
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
    except KeyboardInterrupt:
        pass
    finally:
        # 关闭 provider.
        try:
            provider.close()
            await task
        except asyncio.CancelledError:
            pass
        # 清理资源
        live2d.dispose()
        pygame.quit()


async def run_provider(address: str = "tcp://localhost:5555"):
    _body_chan = miku_body()
    provider = ZMQChannelProvider(
        address=address,
        container=get_container(),
    )

    try:
        await provider.run_until_closed(_body_chan)
    except KeyboardInterrupt:
        pass
