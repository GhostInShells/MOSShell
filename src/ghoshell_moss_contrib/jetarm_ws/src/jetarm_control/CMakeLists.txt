cmake_minimum_required(VERSION 3.8)
project(jetarm_control)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 为所有目标启用位置无关代码（解决 ARM64 架构的链接问题）
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 依赖
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)

# 1) 头文件目录
include_directories(include)

# 2) 协议库（供硬件和测试复用）
add_library(stm32_protocol STATIC
  src/stm32_protocol.cpp
)

# 为 stm32_protocol 显式设置位置无关代码
set_target_properties(stm32_protocol PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

target_include_directories(stm32_protocol
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

# 3) JetArmHardware 插件库
add_library(jetarm_hardware SHARED
  src/jetarm_hardware.cpp
)
ament_target_dependencies(jetarm_hardware
  rclcpp
  hardware_interface
  pluginlib
)
target_link_libraries(jetarm_hardware stm32_protocol)

# 插件描述文件
pluginlib_export_plugin_description_file(hardware_interface jetarm_hardware.xml)

# 4) 测试可执行文件
add_executable(stm32_protocol_test src/stm32_protocol_test.cpp)
# 为测试可执行文件添加必要的依赖
target_link_libraries(stm32_protocol_test stm32_protocol)
# 添加头文件目录
target_include_directories(stm32_protocol_test
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

add_executable(stm32_read_test src/stm32_read_test.cpp)
# 为测试可执行文件添加必要的依赖
target_link_libraries(stm32_read_test stm32_protocol)
# 添加头文件目录
target_include_directories(stm32_read_test
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

# 安装
install(
  TARGETS jetarm_hardware stm32_protocol
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# 单独安装测试可执行文件到 lib/${PROJECT_NAME} 目录
install(
  TARGETS stm32_protocol_test stm32_read_test
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config launch
        DESTINATION share/${PROJECT_NAME})

ament_package()