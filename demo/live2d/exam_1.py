import pygame

import asyncio
import time
from os.path import join, dirname
import ghoshell_moss
from ghoshell_moss.shell import new_shell
from ghoshell_moss.depends import check_demo

if check_demo():
    import live2d.v3 as live2d_v3
    from live2d.v2 import StandardParams
    import math
    import threading

# 全局状态
model: live2d_v3.LAppModel | None = None


# 初始化Pygame和Live2D
def init_pygame():
    pygame.init()
    display = (300, 400)
    screen = pygame.display.set_mode(display, pygame.DOUBLEBUF | pygame.OPENGL)
    pygame.display.set_caption("Digital Human Demo with PyChannel")
    return screen, display


# 初始化Live2D模型
def init_live2d(model_path: str):
    global model
    live2d_v3.init()
    live2d_v3.glInit()
    model = live2d_v3.LAppModel()
    model.LoadModelJson(model_path)
    model.Resize(300, 400)
    model.SetAutoBlinkEnable(True)
    model.SetAutoBreathEnable(True)


# 创建 Shell
shell = new_shell()

# 注册各个轨道的命令函数
gaze_chan = shell.main_channel.new_child('gaze')
face_chan = shell.main_channel.new_child('face')
pose_chan = shell.main_channel.new_child('pose')
spine_chan = shell.main_channel.new_child('spine')
arms_chan = shell.main_channel.new_child('arms')


# 注册命令函数 (与之前相同)
@gaze_chan.build.command()
async def look_left_right(duration: float = 3.0):
    """左右看"""
    start_time = time.time()
    while time.time() - start_time < duration:
        progress = (time.time() - start_time) / duration
        angle = -30 + 60 * math.sin(progress * 2 * math.pi)
        model.SetParameterValue(StandardParams.ParamAngleX, angle)
        await asyncio.sleep(0.016)


@gaze_chan.build.command()
async def blink(duration: float = 0.5):
    """眨眼"""
    start_time = time.time()
    while time.time() - start_time < duration:
        progress = (time.time() - start_time) / duration
        if progress < 0.5:
            eye_open = 1.0 - 2 * progress
        else:
            eye_open = 2 * (progress - 0.5)

        model.SetParameterValue(StandardParams.ParamEyeLOpen, eye_open)
        model.SetParameterValue(StandardParams.ParamEyeROpen, eye_open)
        await asyncio.sleep(0.016)


@face_chan.build.command()
async def smile(duration: float = 2.0):
    """微笑"""
    start_time = time.time()
    while time.time() - start_time < duration:
        progress = (time.time() - start_time) / duration
        model.SetParameterValue(StandardParams.ParamMouthOpenY, progress)
        model.SetParameterValue(StandardParams.ParamMouthForm, 0.5 + 0.5 * progress)
        await asyncio.sleep(0.016)


@face_chan.build.command()
async def surprise(duration: float = 1.5):
    """惊讶表情"""
    start_time = time.time()
    while time.time() - start_time < duration:
        progress = (time.time() - start_time) / duration
        model.SetParameterValue(StandardParams.ParamEyeLOpen, 1.0 + 0.5 * progress)
        model.SetParameterValue(StandardParams.ParamEyeROpen, 1.0 + 0.5 * progress)
        model.SetParameterValue(StandardParams.ParamMouthOpenY, progress)
        await asyncio.sleep(0.016)


@pose_chan.build.command()
async def lean_forward(duration: float = 2.0):
    """身体前倾"""
    start_time = time.time()
    while time.time() - start_time < duration:
        progress = (time.time() - start_time) / duration
        model.SetParameterValue(StandardParams.ParamAngleX, 20 * progress)
        model.SetParameterValue(StandardParams.ParamBodyAngleX, 10 * progress)
        await asyncio.sleep(0.016)


@pose_chan.build.command()
async def lean_back(duration: float = 2.0):
    """身体后仰"""
    start_time = time.time()
    while time.time() - start_time < duration:
        progress = (time.time() - start_time) / duration
        model.SetParameterValue(StandardParams.ParamAngleX, -20 * progress)
        model.SetParameterValue(StandardParams.ParamBodyAngleX, -10 * progress)
        await asyncio.sleep(0.016)


@spine_chan.build.command()
async def nod(duration: float = 1.5):
    """点头"""
    start_time = time.time()
    while time.time() - start_time < duration:
        progress = (time.time() - start_time) / duration
        if progress < 0.5:
            head_y = -20 * (progress * 2)
        else:
            head_y = -20 + 20 * ((progress - 0.5) * 2)

        model.SetParameterValue(StandardParams.ParamAngleY, head_y)
        await asyncio.sleep(0.016)


@spine_chan.build.command()
async def tilt_head(duration: float = 2.0):
    """头部倾斜"""
    start_time = time.time()
    while time.time() - start_time < duration:
        progress = (time.time() - start_time) / duration
        tilt = -15 + 30 * progress
        model.SetParameterValue(StandardParams.ParamAngleZ, tilt)
        await asyncio.sleep(0.016)


@arms_chan.build.command()
async def wave_hand(duration: float = 2.0):
    """挥手"""
    start_time = time.time()
    while time.time() - start_time < duration:
        progress = (time.time() - start_time) / duration
        angle = 90 * math.sin(progress * 2 * math.pi * 2)
        model.SetParameterValue('ParamArmRA', angle)
        await asyncio.sleep(0.016)


@arms_chan.build.command()
async def raise_arm(duration: float = 1.5):
    """抬手"""
    start_time = time.time()
    while time.time() - start_time < duration:
        progress = (time.time() - start_time) / duration
        angle = 90 * progress
        model.SetParameterValue('ParamArmRA', angle)
        await asyncio.sleep(0.016)


async def run_demo_sequence(_shell: ghoshell_moss.MOSSShell):
    """使用 CTML 用例数组运行演示序列"""
    # CTML 演示用例数组
    demo_cases = [
        # 用例 1: 单个函数执行 - 微笑
        {
            "name": "单个微笑表情",
            "ctml": '<face:smile duration="2.0" />',
            "description": "测试单个表情函数的执行"
        },

        # 用例 2: 单个函数执行 - 点头
        {
            "name": "单个点头动作",
            "ctml": '<spine:nod duration="1.5" />',
            "description": "测试单个脊柱动作的执行"
        },

        # 用例 3: 同轨顺序执行 - 连续表情变化
        {
            "name": "同轨顺序表情变化",
            "ctml": """
            <face:smile duration="1.5" />
            <face:surprise duration="1.0" />
            <face:smile duration="1.5" />
            """,
            "description": "测试同一轨道内多个函数的顺序执行"
        },

        # 用例 4: 异轨并行执行 - 表情+头部动作
        {
            "name": "异轨并行执行",
            "ctml": """
            <face:smile duration="2.0" />
            <spine:nod duration="1.5" />
            """,
            "description": "测试不同轨道函数的并行执行"
        },

        # 用例 5: 复杂多轨组合 - 完整的小场景
        {
            "name": "多轨组合场景",
            "ctml": """
            <gaze:look_left_right duration="3.0" />
            <gaze:blink duration="0.5" />
            <face:smile duration="2.0" />
            <spine:nod duration="1.5" />
            <arms:wave_hand duration="2.0" />
            """,
            "description": "测试多轨道组合执行的复杂场景"
        },

        # 用例 6: 带参数的不同时长组合
        {
            "name": "不同时长组合",
            "ctml": """
            <face:smile duration="1.0" />
            <spine:nod duration="2.0" />
            <arms:wave_hand duration="3.0" />
            """,
            "description": "测试不同执行时长的函数组合"
        },

        # 用例 7: 极简组合 - 验证基本功能
        {
            "name": "极简组合验证",
            "ctml": """
            <face:smile duration="0.5" />
            <gaze:blink duration="0.5" />
            """,
            "description": "极简组合验证基本功能"
        }
    ]

    all_results = {}

    # 执行每个用例
    for i, case in enumerate(demo_cases, 1):
        print(f"\n=== 用例 {i}: {case['name']} ===")
        print(f"描述: {case['description']}")

        # 使用 shell 的 interpreter 解析和执行 CTML
        async with _shell.interpreter() as interpreter:
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
    model_path = join(dirname(__file__), "models/hiyori/runtime/hiyori_pro_t11.model3.json")
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
        live2d_v3.clearBuffer()
        model.Draw()

        # 显示完成信息
        info_text = font.render("演示完成，按关闭按钮退出", True, (255, 255, 255))
        screen.blit(info_text, (10, 10))

        pygame.display.flip()
        clock.tick(60)
        time.sleep(0.01)

    # 清理资源
    live2d_v3.dispose()
    pygame.quit()
    t.join()


# 运行主函数
if __name__ == "__main__":
    main()
