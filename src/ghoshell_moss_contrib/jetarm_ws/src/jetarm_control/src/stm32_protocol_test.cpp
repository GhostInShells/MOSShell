#include "stm32_protocol.hpp"
#include <csignal>
#include <chrono>
#include <thread>
#include <iostream>
#include <iomanip>

using namespace std::chrono_literals;

static volatile bool g_running = true;
void sigint_handler(int) { g_running = false; }

void read_and_print_position(jetarm_control::STM32Protocol& board, uint8_t id, const std::string& context) {
    auto pos_opt = board.bus_servo_read_position(id);
    if (pos_opt) {
        std::cout << "  舵机 " << +id << " " << context << ": " << *pos_opt << std::endl;
    } else {
        std::cout << "  舵机 " << +id << " " << context << ": 读取失败" << std::endl;
    }
}

int main() {
    const std::string DEVICE_PATH = "/dev/ttyUSB0";
    const uint32_t    BAUDRATE    = 1000000;
    const std::vector<uint8_t> SERVO_IDS = {1, 2, 3, 4, 5, 10};
    const int16_t     SWING_RANGE = 20;
    const uint16_t    MOVE_MS     = 2000;
    const size_t      LOOPS      = 2;

    std::signal(SIGINT, sigint_handler);

    try {
        jetarm_control::STM32Protocol board(DEVICE_PATH, BAUDRATE);

        // 启用接收（关键！）
        board.enable_reception(true);

        // 等待数据
        std::this_thread::sleep_for(1000ms);


        // 显式启动协议栈
        std::cout << "STM32 协议栈已启动，开始查询舵机状态..." << std::endl;

        // 等待一会儿让查询循环积累一些数据
        std::this_thread::sleep_for(1000ms);

        std::cout << "=== 开始舵机测试 ===" << std::endl;

        // 先读取所有舵机初始状态
        std::cout << "\n=== 初始状态查询 ===" << std::endl;
        for (uint8_t id : SERVO_IDS) {
            read_and_print_position(board, id, "初始位置");
        }

        // 测试单个舵机运动
        for (uint8_t id : SERVO_IDS) {
            if (!g_running) break;

            std::cout << "\n=== 测试舵机 " << +id << " ===" << std::endl;

            auto pos_opt = board.bus_servo_read_position(id);
            if (!pos_opt) {
                std::cerr << "无法读取舵机 " << +id << " 的位置，跳过测试" << std::endl;
                continue;
            }

            int16_t center = *pos_opt;
            int16_t high = center + SWING_RANGE;
            int16_t low = center - SWING_RANGE;

            // 限制在有效范围内
            high = std::max(0, std::min(1000, static_cast<int>(high)));
            low = std::max(0, std::min(1000, static_cast<int>(low)));

            std::cout << "  中心: " << center << ", 范围: " << low << "~" << high << std::endl;

            // 运动测试
            for (int i = 0; i < 4 && g_running; i++) {
                int16_t target = (i % 2 == 0) ? high : low;
                if (i == 3) target = center;

                std::cout << "\n  运动 " << (i+1) << "/4: 目标 " << target << std::endl;

                // 运动前读取
                read_and_print_position(board, id, "运动前");

                // 发送运动指令
                try {
                    board.bus_servo_set_position(MOVE_MS, {{id, static_cast<uint16_t>(target)}});
                    std::cout << "  指令已发送" << std::endl;
                } catch (const std::exception& e) {
                    std::cerr << "  发送失败: " << e.what() << std::endl;
                    break;
                }

                // 等待运动完成
                std::this_thread::sleep_for(std::chrono::milliseconds(MOVE_MS + 500));

                if (!g_running) break;

                // 运动后读取
                read_and_print_position(board, id, "运动后");

                // 检查误差
                auto final_pos = board.bus_servo_read_position(id);
                if (final_pos) {
                    int16_t diff = std::abs(*final_pos - target);
                    std::cout << "  误差: " << diff;
                    if (diff > 5) {
                        std::cout << " ⚠️ 偏差较大" << std::endl;
                    } else {
                        std::cout << " ✅ 良好" << std::endl;
                    }
                }
            }

            // 最终确认
            std::this_thread::sleep_for(500ms);
            read_and_print_position(board, id, "最终位置");
        }

        // 停止协议栈
        std::cout << "\nSTM32 协议栈已停止" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\n=== 测试完成 ===" << std::endl;
    return 0;
}