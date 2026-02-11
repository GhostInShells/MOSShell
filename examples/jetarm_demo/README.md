# jetarm demo

用来验证对 幻尔 6dof 机械臂 的控制.

运行前需要:

1. 真的有幻尔 6dof jetarm 机械臂.
1. 在机械臂开发板上, 已经实装了 jetarm_ws, 完成编译可运行, 并启动了 jetarm_channel 和 jetarm_control 节点.
1. 已经完成了 examples 的依赖安装.

运行:

1. 测试 `python connect_pychannel_with_rcply.py --address=jetarm_control监听的地址端口`, 看看是否能运动.
1. 启动 agent `python jetarm_agent.py --address=jetarm_control监听的地址端口`

这个例子不必特别测试. jetarm 本身二次开发难度比较大. 看看样例知道怎么回事就可以了.
