"""
JetArm 总线舵机并行控制验证脚本
目的：验证 bus_servo_set_position 方法的并行性与阻塞特性。
"""

import time

from jetarm_channel.ros_robot_controller_sdk import Board

# 配置：请根据实际硬件调整这些参数
SERVO_IDS = [1, 2]  # 要测试的舵机ID，例如 [1, 2] 代表两个肩部舵机
SAFE_POSITION_A = 500  # 测试位置A (脉宽，如500)
SAFE_POSITION_B = 300  # 测试位置B (脉宽，如300)
MOVEMENT_DURATION = 2.0  # 运动时间（秒）
TEST_CYCLES = 5  # 往复运动测试次数


def main():
    print("初始化 JetArm Board 控制器...")
    board = Board(device="/dev/ttyUSB0", baudrate=1000000, timeout=5)
    board.enable_reception(True)

    # 参考您提供的初始化代码：设置偏移和电机速度（确保安全）
    # 如果已知偏移量，可以取消注释。否则，默认可能已是0。
    # for servo_id in SERVO_IDS:
    #     board.pwm_servo_set_offset(servo_id, 0)
    # 停止所有电机（虽然舵机控制可能不需要，但是个好习惯）
    board.set_motor_speed([[1, 0], [2, 0], [3, 0], [4, 0]])

    print(f"开始并行控制验证，测试舵机: {SERVO_IDS}")
    time.sleep(1.0)  # 短暂延迟，让下位机稳定

    # 准备测试数据：让所有测试舵机同时从当前位置运动到位置A
    target_positions_a = [[sid, SAFE_POSITION_A] for sid in SERVO_IDS]
    target_positions_b = [[sid, SAFE_POSITION_B] for sid in SERVO_IDS]

    print("\n=== Test A: 单次并行运动 (观察阻塞特性) ===")
    start_time = time.time()
    print(f"指令下发前时间: {start_time:.6f}")

    # 关键调用：发送并行运动指令
    board.bus_servo_set_position(MOVEMENT_DURATION, target_positions_a)

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"指令返回后时间: {end_time:.6f}")
    print(f"函数调用耗时: {elapsed:.6f} 秒")

    if elapsed < 0.001:
        print("-> 结果: 函数为异步非阻塞 (立即返回)")
    else:
        print(f"-> 结果: 函数为阻塞调用 (阻塞了 {elapsed:.2f} 秒)")
        print("   这意味着它会等待舵机开始运动或完成？需要进一步观察运动本身。")

    # 等待一段时间，确保能看到运动完成
    time.sleep(MOVEMENT_DURATION + 0.5)

    print("\n=== Test B: 往复运动压力测试 (观察并行性与稳定性) ===")
    for i in range(TEST_CYCLES):
        print(f"\n--- 循环 {i + 1}/{TEST_CYCLES} ---")
        cycle_start = time.time()

        # 运动到B
        board.bus_servo_set_position(MOVEMENT_DURATION / 2, target_positions_b)
        t1 = time.time()
        # 立即记录时间，判断是否阻塞
        time.sleep(MOVEMENT_DURATION / 2)  # 等待大致完成

        # 运动回A
        board.bus_servo_set_position(MOVEMENT_DURATION / 2, target_positions_a)
        t2 = time.time()
        time.sleep(MOVEMENT_DURATION / 2)

        cycle_end = time.time()
        total_cycle_time = cycle_end - cycle_start
        expected_cycle_time = MOVEMENT_DURATION  # 两次运动时间之和

        print(f"  单循环总耗时: {total_cycle_time:.3f}s [预期: ~{expected_cycle_time:.3f}s]")
        print(f"  运动B调用耗时: {t1 - cycle_start:.6f}s")
        print(f"  运动A调用耗时: {t2 - (cycle_start + MOVEMENT_DURATION / 2):.6f}s")

        # 简单判断并行性：如果总时间远大于单次运动时间x2，可能是串行
        if total_cycle_time > expected_cycle_time * 1.2:
            print("  **注意: 循环总耗时显著高于预期，舵机控制可能非完全并行！**")
        else:
            print("  循环时间符合预期，并行性良好。")

    print("\n=== 验证完成 ===")


if __name__ == "__main__":
    main()
