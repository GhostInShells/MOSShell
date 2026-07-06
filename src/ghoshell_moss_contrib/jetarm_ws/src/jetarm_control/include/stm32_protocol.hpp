// stm32_protocol.hpp
#ifndef STM32_PROTOCOL_HPP
#define STM32_PROTOCOL_HPP

#include <string>
#include <vector>
#include <optional>
#include <cstdint>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <unordered_map>

namespace jetarm_control {

struct ServoPosition {
    uint8_t id;
    uint16_t position;
};

class STM32Protocol {
public:
    STM32Protocol(const std::string& device, uint32_t baudrate = 1000000, uint32_t timeout_ms = 1000);
    ~STM32Protocol();

    void enable_reception(bool enable);

    void bus_servo_set_position(uint16_t duration_ms, const std::vector<ServoPosition>& positions);
    void bus_servo_stop(const std::vector<uint8_t>& servo_ids);
    void bus_servo_enable_torque(uint8_t servo_id, bool enable);

    std::optional<int16_t> bus_servo_read_position(uint8_t servo_id, int timeout_ms = 1000);

private:
    class Impl;
    Impl* pImpl_;
};

} // namespace jetarm_control

#endif // STM32_PROTOCOL_HPP