"""
安全测试脚本：让指定舵机在当前位置附近的小范围内缓慢来回摆动。
"""

import time
import traceback

from jetarm_channel.ros_robot_controller_sdk import Board

# 配置参数
DEVICE_PATH = "/dev/ttyUSB0"  # 您的串口设备
BAUDRATE = 1000000
TARGET_SERVO_ID = 1  # 选择要测试的舵机ID (1-5)
SWING_RANGE = 20  # 摆动范围 (±20个原始单位，非常安全)
MOVE_DURATION = 2  # 单次运动时间（秒），缓慢移动
CYCLE_COUNT = 10  # 摆动循环次数


def main():
    print(f"开始安全测试：舵机 #{TARGET_SERVO_ID} 将进行 {CYCLE_COUNT} 次小幅摆动.")
    print("按 Ctrl+C 可随时终止测试.")

    try:
        # 1. 初始化板卡
        board = Board(device=DEVICE_PATH, baudrate=BAUDRATE)
        board.enable_reception(True)
        time.sleep(1)  # 等待稳定

        # 2. 获取舵机当前位置，作为摆动的中心点
        print("正在读取舵机初始位置...")
        pos = board.bus_servo_read_position(TARGET_SERVO_ID)
        current_pos = pos[0]
        if current_pos is None:
            print(f"错误：无法读取舵机 #{TARGET_SERVO_ID} 的位置。请检查ID是否正确。")
            return

        center_position = current_pos
        print(f"舵机 #{TARGET_SERVO_ID} 的初始位置: {center_position}")

        # 3. 计算摆动的目标位置
        high_target = center_position + SWING_RANGE
        low_target = center_position - SWING_RANGE
        print(f"摆动范围: {low_target} <--> {high_target}")

        # 4. 开始缓慢摆动循环
        print("开始摆动...")
        for i in range(CYCLE_COUNT):
            print(f"循环 {i + 1}/{CYCLE_COUNT}: 移动到 {high_target}")
            # 移动到高点
            board.bus_servo_set_position(MOVE_DURATION, [[TARGET_SERVO_ID, high_target]])
            time.sleep(MOVE_DURATION + 0.1)  # 等待移动完成

            print(f"循环 {i + 1}/{CYCLE_COUNT}: 移动到 {low_target}")
            # 移动到低点
            board.bus_servo_set_position(MOVE_DURATION, [[TARGET_SERVO_ID, low_target]])
            time.sleep(MOVE_DURATION + 0.1)  # 等待移动完成

        # 5. 测试结束，返回中心位置
        print(f"测试完成，返回中心位置 {center_position}")
        board.bus_servo_set_position(MOVE_DURATION, [[TARGET_SERVO_ID, center_position]])
        time.sleep(MOVE_DURATION)

    except KeyboardInterrupt:
        print("\n用户中断测试。尝试返回中心位置...")
        # 如果可能，尝试让舵机回到安全位置
        try:
            board.bus_servo_set_position(1.0, [[TARGET_SERVO_ID, center_position]])
            time.sleep(1.0)
        except Exception:
            traceback.print_exc()
        print("测试已终止")
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        traceback.print_exc()
    finally:
        print("安全测试结束")


if __name__ == "__main__":
    main()
