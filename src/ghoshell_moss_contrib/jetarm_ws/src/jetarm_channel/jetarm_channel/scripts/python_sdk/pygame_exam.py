"""
极简的 Pygame 测试脚本 for NVIDIA Jetson
功能：验证图形显示和电容屏触摸输入
"""

import sys

import pygame

# 初始化 Pygame
pygame.init()

# 设置窗口尺寸 - 通常设置为屏幕分辨率，这里假设一个常见尺寸，您可以根据您的屏幕修改
screen_width, screen_height = 1280, 800
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Jetson Touch & Display Test")

# 设置字体
font = pygame.font.Font(None, 36)  # 使用默认字体，大小36

# 颜色定义
BACKGROUND = (30, 30, 40)  # 深蓝灰色背景
TEXT_COLOR = (255, 255, 255)  # 白色文字
TOUCH_COLOR = (0, 255, 0)  # 绿色触摸点


# 主循环
def main():
    touch_positions = []  # 存储触摸点位置和ID

    print("PyGame Touch Test 开始运行!")
    print(f"窗口尺寸: {screen_width} x {screen_height}")
    print("请在屏幕上触摸来测试...")
    print("按 ESC 键或关闭窗口退出")

    running = True
    while running:
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
            elif event.type == pygame.FINGERDOWN:
                # 手指按下事件
                x = event.x * screen_width
                y = event.y * screen_height
                touch_id = event.finger_id
                touch_positions.append((x, y, touch_id))
                print(f"手指按下: ID {touch_id} 在 ({x:.1f}, {y:.1f})")
            elif event.type == pygame.FINGERUP:
                # 手指抬起事件
                touch_id = event.finger_id
                # 移除该手指的触摸点
                touch_positions = [pos for pos in touch_positions if pos[2] != touch_id]
                print(f"手指抬起: ID {touch_id}")
            elif event.type == pygame.FINGERMOTION:
                # 手指移动事件
                touch_id = event.finger_id
                x = event.x * screen_width
                y = event.y * screen_height
                # 更新该手指的位置
                for i, (_, _, id_val) in enumerate(touch_positions):
                    if id_val == touch_id:
                        touch_positions[i] = (x, y, touch_id)
                print(f"手指移动: ID {touch_id} 到 ({x:.1f}, {y:.1f})")

        # 清空屏幕
        screen.fill(BACKGROUND)

        # 绘制所有触摸点
        for x, y, touch_id in touch_positions:
            pygame.draw.circle(screen, TOUCH_COLOR, (int(x), int(y)), 25)
            # 显示触摸点ID
            id_text = font.render(str(touch_id), True, TEXT_COLOR)
            screen.blit(id_text, (int(x) - 10, int(y) - 10))

        # 显示说明文字
        instructions = ["Jetson 图形与触摸测试", f"触摸点: {len(touch_positions)}", "按 ESC 退出"]

        for i, line in enumerate(instructions):
            text_surface = font.render(line, True, TEXT_COLOR)
            screen.blit(text_surface, (20, 20 + i * 40))

        # 更新显示
        pygame.display.flip()

    # 退出Pygame
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
