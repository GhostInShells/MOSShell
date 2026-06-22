#include "stm32_protocol.hpp"
#include <iostream>
#include <chrono>
#include <thread>

int main() {
    try {
        jetarm_control::STM32Protocol board("/dev/ttyUSB0", 1000000, 1000);
        board.enable_reception(true);

        std::cout << "Testing servo position reading..." << std::endl;

        // 先等待一下让接收线程稳定运行
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        // 测试读取位置
        std::cout << "\nTesting position reading..." << std::endl;
        for (int i = 1; i <= 5; i++) {
            std::cout << "Reading servo " << i << "..." << std::endl;
            auto position = board.bus_servo_read_position(i, 2000);
            if (position) {
                std::cout << "Servo " << i << " position: " << *position << std::endl;
            } else {
                std::cout << "Failed to read servo " << i << " position" << std::endl;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}