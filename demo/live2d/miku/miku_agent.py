from ghoshell_moss.agent.simple_agent import SimpleAgent

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import pygame
import live2d.v3 as live2d
import asyncio
from os.path import join, dirname
from ghoshell_moss.shell import new_shell
from ghoshell_container import  Container
from channels.body import body_chan
from channels.expression import expression_chan
from channels.arm import left_arm_chan, right_arm_chan
from channels.elbow import left_elbow_chan, right_elbow_chan
from channels.necktie import necktie_chan
from channels.head import head_chan
from channels.mouth import mouth_chan
from channels.leg import left_leg_chan, right_leg_chan


# 全局状态
model: live2d.LAppModel | None = None
container = Container()
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
def init_live2d(model_path: str):
    global model
    live2d.init()
    live2d.glInit()
    model = live2d.LAppModel()
    model.LoadModelJson(model_path)
    model.Resize(WIDTH, HEIGHT)
    # model.SetAutoBlinkEnable(False)
    # model.SetAutoBreathEnable(True)
    container.bind(live2d.LAppModel, model)


async def run_agent():
    # 创建 Shell
    shell = new_shell(container=container)
    shell.main_channel.include_channels(body_chan)
    body_chan.include_channels(
        head_chan,
        left_arm_chan,
        right_arm_chan,
        left_elbow_chan,
        right_elbow_chan,
        necktie_chan,
        left_leg_chan,
        right_leg_chan,
    )
    head_chan.include_channels(
        expression_chan,
        mouth_chan,
    )
    agent = SimpleAgent(instruction="你是miku", shell=shell)
    await agent.run()


async def run_agent_and_render():
    # 初始化 Pygame 和 Live2D
    screen, display = init_pygame()
    model_path = join(dirname(__file__), "model/miku.model3.json")
    init_live2d(model_path)

    # 保持窗口打开，直到用户关闭
    running = True
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    # 创建一个任务来运行 agent
    agent_task = asyncio.create_task(run_agent())

    try:
        while running:
            # 处理pygame事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # 更新和绘制模型
            model.Update()
            live2d.clearBuffer()
            model.Draw()

            # 显示完成信息
            info_text = font.render("演示完成，按关闭按钮退出", True, (255, 255, 255))
            screen.blit(info_text, (10, 10))

            pygame.display.flip()
            clock.tick(60)
            
            # 让出控制权给其他协程，确保agent也能运行
            await asyncio.sleep(0.01)
    finally:
        # 取消agent任务
        agent_task.cancel()
        try:
            await agent_task
        except asyncio.CancelledError:
            pass
        
        # 清理资源
        live2d.dispose()
        pygame.quit()


def main():
    # 运行异步主函数
    asyncio.run(run_agent_and_render())


# 运行主函数
if __name__ == "__main__":
    main()
