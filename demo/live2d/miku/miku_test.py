import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import pygame
import live2d.v3 as live2d
import asyncio
import time
from os.path import join, dirname
import ghoshell_moss
from ghoshell_moss.shell import new_shell
import threading
from ghoshell_container import Container
from channels.body import body_chan
from channels.expression import expression_chan
from channels.arm import left_arm_chan, right_arm_chan
from channels.elbow import left_elbow_chan, right_elbow_chan
from channels.necktie import necktie_chan
from channels.head import head_chan
from channels.mouth import mouth_chan
from channels.leg import left_leg_chan, right_leg_chan
from channels.eye import eye_chan
from channels.eyebrow import eyebrow_left_chan, eyebrow_right_chan

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


container = Container()
# 创建 Shell
shell = new_shell(container=container)
shell.main_channel.import_channels(body_chan)
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
head_chan.import_channels(
    expression_chan,
    mouth_chan,
    eye_chan,
    eyebrow_left_chan,
    eyebrow_right_chan,
)


async def run_demo_sequence(_shell: ghoshell_moss.MOSSShell):
    """使用 CTML 用例数组运行演示序列"""
    # CTML 演示用例数组
    demo_cases = [
        # 用例 1: Motion执行
        {
            "name": "测试 motion 能力",
            "ctml": """
            <body:gentle_torso_twist duration="5.0" />
            """,
            "description": "测试 motion 能力的执行",
        },
        # 用例 2: Expression执行
        {
            "name": "测试 expression 能力",
            "ctml": """
            <expression:blush duration="5" />
            """,
            "description": "测试 expression 能力的执行",
        },
        # 用例 3: Arm执行
        {
            "name": "测试 arm 能力",
            "ctml": """
            <left_arm:move duration="0.5" angle="10.0" />
            <left_arm:move duration="0.5" angle="5.0" />
            <right_arm:move duration="0.5" angle="10.0" />
            <right_arm:move duration="0.5" angle="5.0" />
            """,
            "description": "测试 arm 能力的执行",
        },
        # 用例 4: Hand执行
        {
            "name": "测试 hand 能力",
            "ctml": """
            <left_elbow:move duration="0.5" angle="10.0" />
            <left_elbow:move duration="0.5" angle="-30.0" />
            <right_elbow:move duration="0.5" angle="10.0" />
            <right_elbow:move duration="0.5" angle="-30.0" />
            """,
            "description": "测试 elbow 能力的执行",
        },
        # 用例 5: Tie执行
        {
            "name": "测试 tie 能力",
            "ctml": """
            <necktie:flutter duration="5.0"     />
            """,
            "description": "测试 tie 能力的执行",
        },
        # 用例 6: Mouth执行
        {
            "name": "测试 mouth 能力",
            "ctml": """
            <mouth:speek duration="5.0" />
            """,
            "description": "测试 mouth 能力的执行",
        },
        {
            "name": "测试 body 能力",
            "ctml": """
            <body:activate_body duration="5.0" />
            """,
            "description": "测试 body 能力的执行",
        },

        # 用例 7: Leg执行
        {
            "name": "测试 leg 能力",
            "ctml": """
            <left_leg:move duration="0.5" angle="10.0" />
            <left_leg:move duration="0.5" angle="0.0" />
            <right_leg:move duration="0.5" angle="-10.0" />
            <right_leg:move duration="0.5" angle="0.0" />
            """,
            "description": "测试 leg 能力的执行",
        },

        # 用例 8: Eye执行
        {
            "name": "测试 eye 能力",
            "ctml": """
            <eye:gaze x="-1.0" duration="1.0" />
            <eye_left:blink duration="3.0" />
            <eye:gaze y="-1.0" duration="1.0" />
            <eye:gaze x="1.0" duration="1.0" />
            <eye:gaze y="1.0" duration="1.0" />
            """,
            "description": "测试 eye 能力的执行",
        },
        # 用例 9: Eyebrow执行
        {
            "name": "测试 eyebrow 能力",
            "ctml": """
            <eye_left:blink duration="5.0" speed="2.0" />
            <eye_right:blink duration="5.0" speed="2.0" />
            <mouth:speek duration="5.0" speed="2.0"/>

            <eyebrow_left:move x="-1.0" y="0.0" angle="0.0" speed="2.0" />
            <eyebrow_left:move x="1.0" y="0.0" angle="0.0" speed="2.0" />
            <eyebrow_left:move x="0.0" y="0.0" angle="0.0" speed="2.0" />
            <eyebrow_left:move x="0.0" y="1.0" angle="0.0" speed="2.0" />
            <eyebrow_left:move x="0.0" y="-1.0" angle="0.0" speed="2.0" />
            <eyebrow_left:move x="0.0" y="0.0" angle="0.0" speed="2.0" />
            <eyebrow_left:move x="0.0" y="0.0" angle="1.0" speed="2.0" />
            <eyebrow_left:move x="0.0" y="0.0" angle="-1.0" speed="2.0" />
            <eyebrow_left:move x="0.0" y="0.0" angle="0.0" speed="2.0" />
            <eyebrow_left:move x="1.0" y="1.0" angle="1.0" speed="2.0" />
            <eyebrow_left:move x="-1.0" y="-1.0" angle="-1.0" speed="2.0" />
            <eyebrow_left:move x="0.0" y="0.0" angle="0.0" speed="2.0" />

            <eyebrow_right:move x="-1.0" y="0.0" angle="0.0" speed="2.0" />
            <eyebrow_right:move x="1.0" y="0.0" angle="0.0" speed="2.0" />
            <eyebrow_right:move x="0.0" y="0.0" angle="0.0" speed="2.0" />
            <eyebrow_right:move x="0.0" y="1.0" angle="0.0" speed="2.0" />
            <eyebrow_right:move x="0.0" y="-1.0" angle="0.0" speed="2.0" />
            <eyebrow_right:move x="0.0" y="0.0" angle="0.0" speed="2.0" />
            <eyebrow_right:move x="0.0" y="0.0" angle="1.0" speed="2.0" />
            <eyebrow_right:move x="0.0" y="0.0" angle="-1.0" speed="2.0" />
            <eyebrow_right:move x="0.0" y="0.0" angle="0.0" speed="2.0" />
            <eyebrow_right:move x="1.0" y="1.0" angle="1.0" speed="2.0" />
            <eyebrow_right:move x="-1.0" y="-1.0" angle="-1.0" speed="2.0" />
            <eyebrow_right:move x="0.0" y="0.0" angle="0.0" speed="2.0" />
            """,
            "description": "测试 eyebrow 能力的执行",
        },
        # 用例 10: Arm执行
        {
            "name": "测试 arm 能力",
            "ctml": """
            <body.left_arm:akimbo duration="1.5"/>
            """,
            "description": "测试 arm 能力的执行",
        },
    ]

    demo_cases = [
        {
            "name": "测试 motion 能力",
            "ctml": """
 <body:set_default_policy policy="Sad"/>
<body:sad duration="5.0"/>
<body.left_arm:move duration="1.5" angle="0.0"/>
<body.right_arm:move duration="1.5" angle="0.0"/>
<body.left_elbow:move duration="1.5" angle="-30.0"/>
<body.right_elbow:move duration="1.5" angle="-30.0"/>
<body.left_leg:move duration="1.5" angle="0.0"/>
<body.right_leg:move duration="1.5" angle="0.0"/>
<body.necktie:flutter duration="5.0"/>
            """,
            "description": "测试 motion 能力的执行",
        },
    ]

    all_results = {}

    # 执行每个用例
    for i, case in enumerate(demo_cases, 1):
        print(f"\n=== 用例 {i}: {case['name']} ===")
        print(f"描述: {case['description']}")

        # 使用 shell 的 interpreter 解析和执行 CTML
        async with await _shell.interpreter() as interpreter:
            # 输入 CTML 内容
            interpreter.feed(case["ctml"])

            # 等待所有命令执行完成
            tasks = await interpreter.wait_execution_done(timeout=30.0)

            # 记录执行结果
            case_results = {}
            for task_name, task in tasks.items():
                case_results[task_name] = {
                    "success": task.success(),
                    "channel": task.exec_chan,
                    "state": task.state,
                    "result": task.result()
                }

            all_results[case["name"]] = case_results

            # 输出执行结果摘要
            success_count = sum(1 for t in tasks.values() if t.success())
            print(f"执行结果: {success_count}/{len(tasks)} 个任务成功")

            # 显示每个任务的详细信息
            for task_name, task in tasks.items():
                status = "成功" if task.success() else "失败"
                print(f"  - {task_name}: {status} (通道: {task.exec_chan})")

    # 输出总体统计
    print("\n=== 演示序列完成 ===")
    total_cases = len(demo_cases)
    successful_cases = sum(1 for results in all_results.values()
                           if all(task["success"] for task in results.values()))

    total_tasks = sum(len(results) for results in all_results.values())
    successful_tasks = sum(1 for results in all_results.values()
                           for task in results.values() if task["success"])

    print(f"用例通过率: {successful_cases}/{total_cases}")
    print(f"任务成功率: {successful_tasks}/{total_tasks}")

    # 按通道统计执行情况
    channel_stats = {}
    for results in all_results.values():
        for task_info in results.values():
            chan = task_info["channel"]
            if chan not in channel_stats:
                channel_stats[chan] = {"total": 0, "success": 0}
            channel_stats[chan]["total"] += 1
            if task_info["success"]:
                channel_stats[chan]["success"] += 1

    print("\n各通道执行统计:")
    for chan, stats in channel_stats.items():
        success_rate = stats["success"] / stats["total"] * 100
        print(f"  {chan}: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")

    return all_results


async def run_shell_with_example():
    async with shell:
        await run_demo_sequence(shell)


def run_shell_example():
    asyncio.run(run_shell_with_example())


# 演示序列
def main():
    # 初始化 Pygame 和 Live2D
    screen, display = init_pygame()
    model_path = join(dirname(__file__), "model/miku.model3.json")
    init_live2d(model_path)

    # 保持窗口打开，直到用户关闭
    running = True
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    # 启动 Shell 并运行演示序列
    t = threading.Thread(target=run_shell_example)
    t.start()

    while running:
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
        time.sleep(0.01)

    # 清理资源
    live2d.dispose()
    pygame.quit()
    t.join()


# 运行主函数
if __name__ == "__main__":
    main()
