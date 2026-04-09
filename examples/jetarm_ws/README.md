# 安装

## 连接至 jetarm

官方文档上有相关描述, 连接手段有:

1. 用线直接连接设备

- 保证 macos 上安装了 minicom (串口连接)
- macos 上用扩展坞, 用 usb-typec 连线连接到 jetson 上的串口 (最右边的那个)
- 用 `ls /dev/tty* | grep USB` 查看是否有 USB 连接的设备.
- 用 minicom 按标准串口码率连接 jetson, 连接成功后输入用户名密码 (ubuntu)

2. 用局域网 ssh 连接

- 先按官方文档, 连接 HW 开头的内部网络, 检查 ip `ping 192.168.149.1`
- ping 通后 (有时会 ping 不通, 没有 debug), 运行 `ssh ubuntu@192.168.149.1` 连接目标设备.
- 登录后, 先按官方文档, 修改本地 wifi 的配置, 改为 wifi 地址.
- 运行 `sudo systemctl restart wifi.service` 重启 wifi, 或者直接用管理设备的命令行 (需要询问模型)

选择方案二会方便一些, 以后开机连接顺畅的话, 都会自动连接上. 需要修改路由器, 绑定地址, 不要让目标 ip 地址老变.

## 在 jetson 上做开发

使用 ssh 协议可以做文件的双向同步, 很多 ide 都支持. 不过我个人的习惯是:

1. 在 jetson 上创建一个专门的开发目录，`cd ~ && mkdir MOSShell`
2. `cd MOSShell && git init` 初始化目标仓库.
3. 在本地项目中运行 `git remote add jetson ubuntu@ip地址:~/MOSShell`
4. 在目标目录中通过 `git config receive.denyNonFastForwards false` 设置它允许 push 分支覆盖本地分支
5. 以后通过 git 来同步.

## 确认 ros2 环境

确保在 ubuntu 22.04 中已经安装了 ros2 humble (humble 基于 ubuntu 22.04 提供各种库), 同时没有启动默认的 ros2 路径.

* 检查 ros2 的方法是连接进目标机器, 运行 `ros2` 查看命令.
* 检查没有默认启动的具体方法是修改 jetarm 的 `~/.zshrc` 文件关联的配置文件, 里面有默认启动的 ros2 路径.

## 编译 jetarm_ws

连接到目标机器, 创建好本项目的仓库, 确保代码已经传入设备.

```bash
cd ~/MOSShell
pip install . 
```

然后在 `jetarm_ws` 目录下, 开始编译:

```bash
cd ~/MOSShell/examples/jetarm_ws
source /opt/ros/humble/setup.zsh
colcon build --symbal-link 
```

也有一些常用的命令

```bash
colcon build --package-select 指定包名
```

系统安装各种依赖: jetson 上需要安装很多 ros2 相关的依赖. 由于没有整理, 基本思路是编译时缺什么, 就改什么.
当前 `src` 目录下, 所有文件夹都是独立的包, colcon 编译时会递归寻找 `package.xml`.

完成后记得引入环境:

```bash
# 先引入 ros2 的 zsh. 
source install/setup.zsh # 根据实际使用的 shell 切换. 
```

# 启动 ros2 工程

## 保证 ros2 环境准备

由于在 `~/.zshrc` 修改了 ros2 启动和引用, 所以需要每次 ssh 连接后, 启动 ros 相关能力之前都要运行一下命令.

```bash
source /opt/ros/humble/setup.zsh
source install/setup.zsh
```

这个也可以按需列入 `~/.zshrc` 里, 看偏好. 我个人是不用的.

## 核心目录说明

- `ros2`: ghoshell 验证阶段所有的 ros2 类库
    - `src`: 核心库目录
        - `jetarm_6dof_description`:
          用来存放 jetarm 的机器人描述相关讯息,
          也可以启动 rviz `ros2 launch jetarm_6dof_description view_model.launch.py`
        - `jetarm_driver`:
          是验证 python 驱动的节点, 想用 python 实现 ros2 control 的 interface. 不过现在不用了.
        - `jetarm_control`:
          核心的 ros2 control 实现. 启动这个节点, 机器人就可以驱动了. 详见后面的测试用例. deepseek 等也能给出 ros2
          control 支持的各种命令.
          运行这个节点的脚本是 `ros2 launch jetarm_control jetarm_control.launch.py`
        - `jetarm_moveit2`:
          这个是基于 ros2 control (jetarm_control) 实现的 moveit 节点, 所有的代码应该都由 moveit2 的 assitant 生成.
          具体方法可以问模型, 需要提前安装 moveit2 到全局环境里.

# 常用测试命令

## 从 ros control 控制运动轨迹

```bash
ros2 action send_goal /joint_trajectory_controller/follow_joint_trajectory control_msgs/action/FollowJointTrajectory "
trajectory:
  joint_names: ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'r_joint'] 
  points:
    - positions: [0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
      time_from_start: {sec: 1, nanosec: 0}
    - positions: [0.0, 1.0, -1.57, -1.57, 0.2, 0.0]
      time_from_start: {sec: 3, nanosec: 0}
"
`

# 机器人打通

## 基本能力启动

启动 jetarm_control: `ros2 launch jetarm_control jetarm_control.launch.py`

jetarm_control 目录是将机械臂底层的下位机控制, 封装成标准的 Ros2 Control Hardware. 相关代码可以查阅. 

启动成功后, 可以用 ros2 命令验证运行 (需要从另一个 shell 进来, 并且要 source 相关环境):

```zsh
ros2 topic list  # 查看存在的 topic

ros2 topic echo /joint_states --once # 查看关节位置. 

ros2 action list  #  查看存在的轨迹 action. 
```

## 运行 channel

机器人所有的能力预计用 `jetarm_channel` 对外提供, 需要在运行 `jetarm_control` 的基础上, 运行:

`ros2 run jetarm_channel jetarm_channel_node` 暂时还没有专门实现 launch, 用脚本测试.

启动后, channel 可以直接用 python 调试. 调试前最好使用 ghoshell 库本身. 步骤如下:

1. 用某个 ssh 进入 jetson 上 ghoshell 的目录, 运行 `uv venv` 和 `source .venv/bin/activate` 等进入环境.
2. `which python` 查看 python 是否正确.
3. 在 jetarm channel 启动的前提下, 运行 `python ghoshell/demo/jetarm_demo/connect_pychannel_with_rcply.py`, 检查 channel 通讯.
4. 运行 `python demo/jetarm_demo/jetarm_agent.py` 可以启动连接到 jetarm 的 agent

``