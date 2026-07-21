"""
JetArm 总线舵机并行控制验证脚本
目的：验证 bus_servo_set_position 方法的并行性与阻塞特性。
"""

import time

from jetarm_channel.ros_robot_controller_sdk import Board

# 配置：请根据实际硬件调整这些参数
SERVO_IDS = [1, 2, 3, 4, 5, 10]  # 要测试的舵机ID，例如 [1, 2] 代表两个肩部舵机


def main():
    board = Board(device="/dev/ttyUSB0", baudrate=1000000, timeout=5)
    board.enable_reception(True)

    while line := input(">"):
        if line == "/q":
            exit(0)
        if not line.startswith("/"):
            print("unknown command")
            continue

        parts = line.split(" ", 3)
        if len(parts) != 3:
            print("unknown command: server position duration")
            continue
        servo_id = int(parts[0].lstrip("/"))
        if servo_id not in SERVO_IDS:
            print("unknown servo id")
            continue
        position = int(parts[1])
        duration = float(parts[2])
        board.bus_servo_set_position(duration, [[servo_id, position]])

        time.sleep(1)
        r = {}
        for servo_id in SERVO_IDS:
            r[servo_id] = board.bus_servo_read_position(servo_id)
        print("servo positions: {}".format(r))


if __name__ == "__main__":
    main()
