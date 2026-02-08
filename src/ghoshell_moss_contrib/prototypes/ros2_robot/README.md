# Ros2 Robot

这是验证 ROS2 与 MOSS 打通的用例.

基本原理是对接一个 ROS2 机器人, 用:

1. 用 Trajectory 轨迹动画控制
1. 定义关节参数, 主要目的是让 AI 感知到自己的形体
1. 验证 Command 和 ROS2 Action 打通

具体实现则是 JetArm (幻尔机械臂)
