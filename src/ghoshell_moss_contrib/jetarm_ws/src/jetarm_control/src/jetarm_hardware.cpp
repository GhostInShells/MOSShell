#include "jetarm_hardware.hpp"
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <sstream>

#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "rclcpp/rclcpp.hpp"

namespace jetarm_control
{

hardware_interface::CallbackReturn
JetArmHardware::on_init(const hardware_interface::HardwareInfo & info)
{
  if (hardware_interface::SystemInterface::on_init(info) !=
      hardware_interface::CallbackReturn::SUCCESS)
  {
    return hardware_interface::CallbackReturn::ERROR;
  }

  // 从硬件参数读取设备配置
  device_ = info.hardware_parameters.at("device");

  // 读取波特率，默认为 1000000
  baudrate_ = 1000000;
  if (info.hardware_parameters.count("baudrate")) {
    baudrate_ = std::stoul(info.hardware_parameters.at("baudrate"));
  }

  // 读取超时时间，默认为 1000ms
  timeout_ms_ = 1000;
  if (info.hardware_parameters.count("timeout_ms")) {
    timeout_ms_ = std::stoul(info.hardware_parameters.at("timeout_ms"));
  }

  const size_t n_joints = info.joints.size();
  hw_positions_.assign(n_joints, 0.0);
  hw_velocities_.assign(n_joints, 0.0);
  hw_efforts_.assign(n_joints, 0.0);
  hw_commands_.assign(n_joints, 0.0);

  servo_ids_.resize(n_joints);
  direction_.assign(n_joints, 1.0);
  center_pulse_.resize(n_joints, 500);   // 默认 500 = 0 rad

  // 解析 center_pulse
  if (info.hardware_parameters.count("center_pulse"))
  {
    std::stringstream ss(info.hardware_parameters.at("center_pulse"));
    for (size_t i = 0; i < n_joints; ++i)
    {
      ss >> center_pulse_[i];
      if (ss.peek() == ',') ss.ignore();
    }
  }

  // 解析其余关节级参数
  for (size_t i = 0; i < n_joints; ++i)
  {
    const auto & j = info.joints[i];
    servo_ids_[i]   = std::stoi(j.parameters.at("servo_id"));
    direction_[i]   = std::stod(j.parameters.at("direction"));
  }

  // 初始化上一次位置用于速度估算
  last_positions_ = hw_positions_;

  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn
JetArmHardware::on_configure(const rclcpp_lifecycle::State &)
{
  try
  {
    board_ = std::make_unique<STM32Protocol>(device_, baudrate_, timeout_ms_);

    // 启动后台接收线程
    board_->enable_reception(true);

    RCLCPP_INFO(rclcpp::get_logger("JetArmHardware"),
                "Successfully opened device: %s at %d baud",
                device_.c_str(), baudrate_);
  }
  catch (const std::exception & e)
  {
    RCLCPP_ERROR(rclcpp::get_logger("JetArmHardware"),
                 "Serial open failed: %s", e.what());
    return hardware_interface::CallbackReturn::ERROR;
  }
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn
JetArmHardware::on_activate(const rclcpp_lifecycle::State &)
{
  // 首先读取当前位置
  this->read(rclcpp::Time(0), rclcpp::Duration(0, 0));

  // 启用所有舵机扭矩
  for (const auto id : servo_ids_) {
    try {
      board_->bus_servo_enable_torque(id, true);
      RCLCPP_DEBUG(rclcpp::get_logger("JetArmHardware"),
                   "Enabled torque for servo %d", id);
    } catch (const std::exception & e) {
      RCLCPP_ERROR(rclcpp::get_logger("JetArmHardware"),
                   "Failed to enable torque for servo %d: %s", id, e.what());
    }
  }

  // 将当前位置设为命令目标，避免跳变
  for (size_t i = 0; i < hw_commands_.size(); ++i) {
    hw_commands_[i] = hw_positions_[i];
    last_positions_[i] = hw_positions_[i];
  }

  RCLCPP_INFO(rclcpp::get_logger("JetArmHardware"),
              "Hardware activated. Initial positions set.");

  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn
JetArmHardware::on_deactivate(const rclcpp_lifecycle::State &)
{
  // 禁用所有舵机扭矩
  for (const auto id : servo_ids_) {
    try {
      board_->bus_servo_enable_torque(id, false);
      RCLCPP_DEBUG(rclcpp::get_logger("JetArmHardware"),
                   "Disabled torque for servo %d", id);
    } catch (const std::exception & e) {
      RCLCPP_ERROR(rclcpp::get_logger("JetArmHardware"),
                   "Failed to disable torque for servo %d: %s", id, e.what());
    }
  }

  // 释放资源
  board_.reset();
  RCLCPP_INFO(rclcpp::get_logger("JetArmHardware"), "Hardware deactivated");

  return hardware_interface::CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface>
JetArmHardware::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> state_interfaces;
  for (size_t i = 0; i < info_.joints.size(); ++i)
  {
    state_interfaces.emplace_back(
      info_.joints[i].name, hardware_interface::HW_IF_POSITION, &hw_positions_[i]);
    state_interfaces.emplace_back(
      info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &hw_velocities_[i]);
    state_interfaces.emplace_back(
      info_.joints[i].name, hardware_interface::HW_IF_EFFORT, &hw_efforts_[i]);
  }
  return state_interfaces;
}

std::vector<hardware_interface::CommandInterface>
JetArmHardware::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> command_interfaces;
  for (size_t i = 0; i < info_.joints.size(); ++i)
  {
    command_interfaces.emplace_back(
      info_.joints[i].name, hardware_interface::HW_IF_POSITION, &hw_commands_[i]);
  }
  return command_interfaces;
}

hardware_interface::return_type
JetArmHardware::read(const rclcpp::Time & /*time*/, const rclcpp::Duration & period)
{
  // 一次性读取所有关节位置，提高效率
  for (size_t i = 0; i < servo_ids_.size(); ++i)
  {
    uint8_t servo_id = servo_ids_[i];

    // 请求并读取位置
    auto position = board_->bus_servo_read_position(servo_id);

    if (position) {
      int32_t t = std::clamp<int32_t>(*position, 0, 1000);

      // 正确的弧度换算公式
      double new_position = direction_[i] *
                           (t - center_pulse_[i]) *
                           (M_PI / 500.0);

      // 估算速度（如果周期有效）
      if (period.seconds() > 0.001) {
        hw_velocities_[i] = (new_position - last_positions_[i]) / period.seconds();
      }

      hw_positions_[i] = new_position;
      last_positions_[i] = new_position;

      // 调试输出
      RCLCPP_DEBUG(rclcpp::get_logger("JetArmHardware"),
                   "Joint %zu: pulse=%d -> rad=%.3f, vel=%.3f rad/s",
                   i, t, hw_positions_[i], hw_velocities_[i]);
    } else {
      RCLCPP_DEBUG(rclcpp::get_logger("JetArmHardware"),
                   "No new position data for servo %d, using cached value", servo_id);
    }

    hw_efforts_[i] = 0.0;  // 总线舵机通常不提供力矩反馈
  }

  return hardware_interface::return_type::OK;
}

hardware_interface::return_type
JetArmHardware::write(const rclcpp::Time & /*time*/, const rclcpp::Duration & period)
{
  // 计算运动时间（毫秒），确保最小1ms
  const uint16_t duration_ms = static_cast<uint16_t>(
    std::max(1.0, period.seconds() * 1000.0));

  std::vector<jetarm_control::ServoPosition> targets;
  targets.reserve(servo_ids_.size());

  for (size_t i = 0; i < servo_ids_.size(); ++i)
  {
    try {
      // 正确的弧度到脉冲换算公式
      double pulse_d = center_pulse_[i] +
                      (hw_commands_[i] * direction_[i]) *
                      (500.0 / M_PI);

      uint16_t pulse = static_cast<uint16_t>(
        std::clamp(std::round(pulse_d), 0.0, 1000.0));

      targets.push_back({servo_ids_[i], pulse});

      // 调试输出
      RCLCPP_DEBUG(rclcpp::get_logger("JetArmHardware"),
                   "Joint %zu: rad=%.3f -> pulse=%d, duration=%dms",
                   i, hw_commands_[i], pulse, duration_ms);
    } catch (const std::exception & e) {
      RCLCPP_ERROR(rclcpp::get_logger("JetArmHardware"),
                   "Exception preparing command for servo %d: %s",
                   servo_ids_[i], e.what());
      return hardware_interface::return_type::ERROR;
    }
  }

  try {
    // 批量发送命令到所有舵机
    board_->bus_servo_set_position(duration_ms, targets);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(rclcpp::get_logger("JetArmHardware"),
                 "Exception sending commands: %s", e.what());
    return hardware_interface::return_type::ERROR;
  }

  return hardware_interface::return_type::OK;
}

}  // namespace jetarm_control

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(
  jetarm_control::JetArmHardware,
  hardware_interface::SystemInterface)