"""
目的：直接通过串口与JetArm下位机通信，读取所有舵机的当前位置。
说明：此脚本不依赖ROS，仅使用pyserial和提供的SDK协议。
"""

import time

import serial

from jetarm_channel.ros_robot_controller_sdk import Board  # 假设这个SDK文件在同一目录下


def main():
    # 1. 初始化板卡连接
    # 注意：串口设备路径和波特率可能需要根据您的实际设备调整
    # 常见路径：'/dev/ttyUSB0', '/dev/ttyACM0'
    device_path = "/dev/ttyUSB0"
    baudrate = 1000000

    print(f"尝试连接设备: {device_path}, 波特率: {baudrate}")

    try:
        # 初始化Board类，它会自动连接串口
        board = Board(device=device_path, baudrate=baudrate)
        # 启用数据接收
        board.enable_reception(True)
        print("连接成功！等待1秒让设备稳定...")
        time.sleep(1)

        # 2. 定义要查询的舵机ID范围
        # 常见配置：基础舵机ID 1-4，夹爪舵机ID 5
        servo_ids = [1, 2, 3, 4, 5]

        print("\n开始读取舵机位置... (Ctrl+C 停止)")
        print("舵机位置单位：度 (需根据实际转换)")
        print("-" * 40)

        # 3. 循环读取并显示舵机位置
        while True:
            for servo_id in servo_ids:
                try:
                    # 读取指定舵机的位置
                    # 注意：bus_servo_read_position返回的是原始计数值，需要转换为角度
                    # 常见转换公式：角度 = (计数值 - 500) * 0.24 或类似
                    position_info = board.bus_servo_read_position(servo_id)

                    if position_info is not None:
                        # position_info 是返回的原始数据，通常是脉冲宽度或编码器计数值
                        # 您需要根据下位机协议文档将其转换为角度
                        raw_value = position_info[0] if isinstance(position_info, (list, tuple)) else position_info
                        print(f"舵机 {servo_id}: 原始值 = {raw_value}", end=" | ")
                    else:
                        print(f"舵机 {servo_id}: 读取失败", end=" | ")

                except Exception as e:
                    print(f"舵机 {servo_id}: 错误 {e}", end=" | ")

            print()  # 换行
            time.sleep(1)  # 每秒读取一次

    except serial.serialutil.SerialException as e:
        print(f"串口连接失败: {e}")
        print("请检查:")
        print("1. 串口设备路径是否正确")
        print("2. 当前用户是否有权限访问该设备 (尝试: sudo chmod a+rw /dev/ttyUSB0)")
        print("3. 是否有其他程序占用了该串口")
        return
    except KeyboardInterrupt:
        print("\n用户中断测试")
    except Exception as e:
        print(f"发生未知错误: {e}")
    finally:
        print("测试结束")


if __name__ == "__main__":
    main()
