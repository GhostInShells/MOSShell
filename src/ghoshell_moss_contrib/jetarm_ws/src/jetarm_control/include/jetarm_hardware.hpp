#ifndef JETARM_CONTROL__JETARM_HARDWARE_HPP_
#define JETARM_CONTROL__JETARM_HARDWARE_HPP_

#include <memory>
#include <string>
#include <vector>

#include "hardware_interface/system_interface.hpp"
#include "rclcpp/macros.hpp"
#include "rclcpp_lifecycle/state.hpp"
#include "stm32_protocol.hpp"

namespace jetarm_control
{
class JetArmHardware : public hardware_interface::SystemInterface
{
public:
  RCLCPP_SHARED_PTR_DEFINITIONS(JetArmHardware)

  hardware_interface::CallbackReturn on_init(
    const hardware_interface::HardwareInfo & info) override;

  hardware_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;

  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

  hardware_interface::return_type read(
    const rclcpp::Time & time,
    const rclcpp::Duration & period) override;

  hardware_interface::return_type write(
    const rclcpp::Time & time,
    const rclcpp::Duration & period) override;

private:
  std::unique_ptr<jetarm_control::STM32Protocol> board_;

  std::vector<double> hw_positions_;
  std::vector<double> hw_velocities_;
  std::vector<double> hw_efforts_;
  std::vector<double> hw_commands_;
  std::vector<double> last_positions_;  // ← 新增：用于速度估算

  std::vector<uint8_t>  servo_ids_;
  std::vector<double>   direction_;
  std::vector<uint16_t> center_pulse_;   // 每个关节的"指天"脉冲

  std::string device_;
  uint32_t baudrate_;                   // ← 新增：波特率
  uint32_t timeout_ms_;                 // ← 新增：超时时间
};
}  // namespace jetarm_control

#endif  // JETARM_CONTROL__JETARM_HARDWARE_HPP_
