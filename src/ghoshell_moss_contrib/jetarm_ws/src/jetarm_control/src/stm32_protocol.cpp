#include "stm32_protocol.hpp"
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <cstring>
#include <stdexcept>
#include <chrono>
#include <thread>
#include <iostream>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <queue>
#include <unordered_map>

namespace jetarm_control {

// 包控制器状态枚举
enum class PacketControllerState {
    STARTBYTE1 = 0,
    STARTBYTE2 = 1,
    LENGTH = 2,
    FUNCTION = 3,
    ID = 4,
    DATA = 5,
    CHECKSUM = 6
};

// 包功能枚举
enum class PacketFunction : uint8_t {
    SYS = 0,
    LED = 1,
    BUZZER = 2,
    MOTOR = 3,
    PWM_SERVO = 4,
    BUS_SERVO = 5,
    KEY = 6,
    IMU = 7,
    GAMEPAD = 8,
    SBUS = 9,
    OLED = 10,
    NONE = 11
};

// 总线舵机子命令
enum class BusServoSubCmd : uint8_t {
    SET_POSITION = 0x01,
    STOP = 0x03,
    READ_POSITION = 0x05,
    ENABLE_TORQUE = 0x0B,
    DISABLE_TORQUE = 0x0C
};

// CRC8校验表
static const uint8_t CRC8_TABLE[256] = {
    0, 94, 188, 226, 97, 63, 221, 131, 194, 156, 126, 32, 163, 253, 31, 65,
    157, 195, 33, 127, 252, 162, 64, 30, 95, 1, 227, 189, 62, 96, 130, 220,
    35, 125, 159, 193, 66, 28, 254, 160, 225, 191, 93, 3, 128, 222, 60, 98,
    190, 224, 2, 92, 223, 129, 99, 61, 124, 34, 192, 158, 29, 67, 161, 255,
    70, 24, 250, 164, 39, 121, 155, 197, 132, 218, 56, 102, 229, 187, 89, 7,
    219, 133, 103, 57, 186, 228, 6, 88, 25, 71, 165, 251, 120, 38, 196, 154,
    101, 59, 217, 135, 4, 90, 184, 230, 167, 249, 27, 69, 198, 152, 122, 36,
    248, 166, 68, 26, 153, 199, 37, 123, 58, 100, 134, 216, 91, 5, 231, 185,
    140, 210, 48, 110, 237, 179, 81, 15, 78, 16, 242, 172, 47, 113, 147, 205,
    17, 79, 173, 243, 112, 46, 204, 146, 211, 141, 111, 49, 178, 236, 14, 80,
    175, 241, 19, 77, 206, 144, 114, 44, 109, 51, 209, 143, 12, 82, 176, 238,
    50, 108, 142, 208, 83, 13, 239, 177, 240, 174, 76, 18, 145, 207, 45, 115,
    202, 148, 118, 40, 171, 245, 23, 73, 8, 86, 180, 234, 105, 55, 213, 139,
    87, 9, 235, 181, 54, 104, 138, 212, 149, 203, 41, 119, 244, 170, 72, 22,
    233, 183, 85, 11, 136, 214, 52, 106, 43, 117, 151, 201, 74, 20, 246, 168,
    116, 42, 200, 150, 21, 75, 169, 247, 182, 232, 10, 84, 215, 137, 107, 53
};

uint8_t checksum_crc8(const std::vector<uint8_t>& data) {
    uint8_t check = 0;
    for (uint8_t byte : data) {
        check = CRC8_TABLE[check ^ byte];
    }
    return check;
}

class STM32Protocol::Impl {
public:
    Impl(const std::string& device, uint32_t baudrate, uint32_t timeout_ms)
        : fd_(-1), timeout_ms_(timeout_ms), running_(false), enable_recv_(false) {

        open_serial_port(device, baudrate);
        running_ = true;
        receiver_thread_ = std::thread(&Impl::recv_task, this);
    }

    ~Impl() {
        running_ = false;
        if (receiver_thread_.joinable()) {
            receiver_thread_.join();
        }
        if (fd_ >= 0) {
            close(fd_);
        }
    }

    void enable_reception(bool enable) {
        enable_recv_ = enable;
    }

    void bus_servo_set_position(uint16_t duration_ms, const std::vector<ServoPosition>& positions) {
        std::vector<uint8_t> data;
        data.push_back(static_cast<uint8_t>(BusServoSubCmd::SET_POSITION));
        data.push_back(duration_ms & 0xFF);
        data.push_back((duration_ms >> 8) & 0xFF);
        data.push_back(positions.size());

        for (const auto& pos : positions) {
            data.push_back(pos.id);
            data.push_back(pos.position & 0xFF);
            data.push_back((pos.position >> 8) & 0xFF);
        }

        send_packet(static_cast<uint8_t>(PacketFunction::BUS_SERVO), data);
    }

    void bus_servo_stop(const std::vector<uint8_t>& servo_ids) {
        std::vector<uint8_t> data;
        data.push_back(static_cast<uint8_t>(BusServoSubCmd::STOP));
        data.push_back(servo_ids.size());
        data.insert(data.end(), servo_ids.begin(), servo_ids.end());

        send_packet(static_cast<uint8_t>(PacketFunction::BUS_SERVO), data);
    }

    void bus_servo_enable_torque(uint8_t servo_id, bool enable) {
        std::vector<uint8_t> data;
        data.push_back(enable ? static_cast<uint8_t>(BusServoSubCmd::ENABLE_TORQUE)
                             : static_cast<uint8_t>(BusServoSubCmd::DISABLE_TORQUE));
        data.push_back(servo_id);

        send_packet(static_cast<uint8_t>(PacketFunction::BUS_SERVO), data);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    std::optional<int16_t> bus_servo_read_position(uint8_t servo_id, int timeout_ms) {
        // 清空当前队列中的旧数据
        {
            std::lock_guard<std::mutex> lock(bus_servo_mutex_);
            while (!bus_servo_queue_.empty()) {
                bus_servo_queue_.pop();
            }
        }

        // 发送读取请求
        std::vector<uint8_t> request_data;
        request_data.push_back(static_cast<uint8_t>(BusServoSubCmd::READ_POSITION));
        request_data.push_back(servo_id);

        send_packet(static_cast<uint8_t>(PacketFunction::BUS_SERVO), request_data);

        // 等待响应
        std::unique_lock<std::mutex> lock(bus_servo_mutex_);
        if (bus_servo_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                                 [this] { return !bus_servo_queue_.empty(); })) {

            auto response = bus_servo_queue_.front();
            bus_servo_queue_.pop();

            // 解析响应数据
            if (response.size() >= 4) {
                uint8_t resp_servo_id = response[0];
                uint8_t resp_cmd = response[1];
                int8_t success = static_cast<int8_t>(response[2]);


                if (resp_servo_id == servo_id && resp_cmd == static_cast<uint8_t>(BusServoSubCmd::READ_POSITION)) {
                    if (success == 0 && response.size() >= 5) {
                        int16_t position = static_cast<int16_t>(response[3] | (response[4] << 8));
                        return position;
                    } else {
                    }
                }
            }
        } else {
        }

        return std::nullopt;
    }

private:
    void open_serial_port(const std::string& device, uint32_t baudrate) {
        fd_ = open(device.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
        if (fd_ < 0) {
            throw std::runtime_error("Failed to open serial port: " + device);
        }

        struct termios tty;
        memset(&tty, 0, sizeof(tty));

        if (tcgetattr(fd_, &tty) != 0) {
            close(fd_);
            throw std::runtime_error("Failed to get serial attributes");
        }

        cfsetospeed(&tty, B1000000);
        cfsetispeed(&tty, B1000000);

        tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
        tty.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL | IXON);
        tty.c_oflag &= ~OPOST;
        tty.c_lflag &= ~(ECHO | ECHONL | ICANON | ISIG | IEXTEN);
        tty.c_cc[VMIN] = 0;
        tty.c_cc[VTIME] = 1;

        tty.c_cflag |= (CLOCAL | CREAD);
        tty.c_cflag &= ~(PARENB | PARODD);
        tty.c_cflag &= ~CSTOPB;
        tty.c_cflag &= ~CRTSCTS;

        if (tcsetattr(fd_, TCSANOW, &tty) != 0) {
            close(fd_);
            throw std::runtime_error("Failed to set serial attributes");
        }

        tcflush(fd_, TCIOFLUSH);
    }

    void send_packet(uint8_t func, const std::vector<uint8_t>& data) {
        std::lock_guard<std::mutex> lock(write_mutex_);

        std::vector<uint8_t> packet;
        packet.push_back(0xAA);
        packet.push_back(0x55);
        packet.push_back(func);
        packet.push_back(data.size());
        packet.insert(packet.end(), data.begin(), data.end());

        uint8_t crc = checksum_crc8(std::vector<uint8_t>(packet.begin() + 2, packet.end()));
        packet.push_back(crc);

        ssize_t written = write(fd_, packet.data(), packet.size());
        if (written != static_cast<ssize_t>(packet.size())) {
            std::cerr << "Warning: Failed to write complete packet" << std::endl;
        }

        tcdrain(fd_);
    }

    void recv_task() {
        PacketControllerState state = PacketControllerState::STARTBYTE1;
        uint8_t current_func = 0;
        uint8_t current_length = 0;
        uint8_t bytes_received = 0;
        std::vector<uint8_t> current_frame;

        while (running_) {
            if (!enable_recv_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            uint8_t byte;
            ssize_t n = read(fd_, &byte, 1);

            if (n == 1) {
                switch (state) {
                    case PacketControllerState::STARTBYTE1:
                        if (byte == 0xAA) {
                            state = PacketControllerState::STARTBYTE2;
                            current_frame.clear();
                            current_frame.push_back(byte);
                        }
                        break;

                    case PacketControllerState::STARTBYTE2:
                        current_frame.push_back(byte);
                        if (byte == 0x55) {
                            state = PacketControllerState::FUNCTION;
                        } else {
                            state = PacketControllerState::STARTBYTE1;
                        }
                        break;

                    case PacketControllerState::FUNCTION:
                        current_frame.push_back(byte);
                        current_func = byte;
                        state = PacketControllerState::LENGTH;
                        break;

                    case PacketControllerState::LENGTH:
                        current_frame.push_back(byte);
                        current_length = byte;
                        bytes_received = 0;

                        if (current_length == 0) {
                            state = PacketControllerState::CHECKSUM;
                        } else {
                            state = PacketControllerState::DATA;
                        }
                        break;

                    case PacketControllerState::DATA:
                        current_frame.push_back(byte);
                        bytes_received++;
                        if (bytes_received >= current_length) {
                            state = PacketControllerState::CHECKSUM;
                        }
                        break;

                    case PacketControllerState::CHECKSUM:
                        {
                            // 计算校验和（从功能号开始到数据结束）
                            std::vector<uint8_t> check_data(current_frame.begin() + 2, current_frame.end());
                            uint8_t crc_calc = checksum_crc8(check_data);

                            if (crc_calc == byte) {
                                current_frame.push_back(byte);
                                process_packet(current_func, std::vector<uint8_t>(current_frame.begin() + 4, current_frame.end() - 1));
                            } else {
                                std::cerr << "CRC mismatch" << std::endl;
                            }
                            state = PacketControllerState::STARTBYTE1;
                        }
                        break;
                }
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }

    void process_packet(uint8_t func, const std::vector<uint8_t>& data) {

        if (func == static_cast<uint8_t>(PacketFunction::BUS_SERVO)) {
            // 将数据放入总线舵机队列并通知等待的线程
            std::lock_guard<std::mutex> lock(bus_servo_mutex_);
            bus_servo_queue_.push(data);
            bus_servo_cv_.notify_one();
        }
        // 可以在这里添加其他功能类型的处理
    }

    int fd_;
    uint32_t timeout_ms_;
    std::atomic<bool> running_;
    std::atomic<bool> enable_recv_;

    std::thread receiver_thread_;
    std::mutex write_mutex_;

    // 总线舵机专用队列和同步变量
    std::mutex bus_servo_mutex_;
    std::condition_variable bus_servo_cv_;
    std::queue<std::vector<uint8_t>> bus_servo_queue_;
};

// STM32Protocol 公共接口实现
STM32Protocol::STM32Protocol(const std::string& device, uint32_t baudrate, uint32_t timeout_ms)
    : pImpl_(new Impl(device, baudrate, timeout_ms)) {}

STM32Protocol::~STM32Protocol() {
    delete pImpl_;
}

void STM32Protocol::enable_reception(bool enable) {
    pImpl_->enable_reception(enable);
}

void STM32Protocol::bus_servo_set_position(uint16_t duration_ms, const std::vector<ServoPosition>& positions) {
    pImpl_->bus_servo_set_position(duration_ms, positions);
}

void STM32Protocol::bus_servo_stop(const std::vector<uint8_t>& servo_ids) {
    pImpl_->bus_servo_stop(servo_ids);
}

void STM32Protocol::bus_servo_enable_torque(uint8_t servo_id, bool enable) {
    pImpl_->bus_servo_enable_torque(servo_id, enable);
}

std::optional<int16_t> STM32Protocol::bus_servo_read_position(uint8_t servo_id, int timeout_ms) {
    return pImpl_->bus_servo_read_position(servo_id, timeout_ms);
}

} // namespace jetarm_control