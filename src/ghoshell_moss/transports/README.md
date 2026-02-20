# 关于 transports

本目录放置各种基于 `ghoshell_moss.core.duplex` 实现的双工通讯组件.
用来构建 Channel 到 Shell 的跨进程通讯.

MOSS 架构中, Shell 和 Channel 可以运行在不同的设备, 不同的进程上.
只需要建立通讯通道, shell 就可以持有 channel 的远程连接 (runtime).

基本原理:

1. provider 端: 通过 ChannelProvider 去运行一个本地 Channel
1. shell 端: 通过 ChannelProxy 去对接 provider 的通讯, 得到一个 Channel 实例.

相当于:

- 上行通道: 本地 channel -> provider -> 本地 connection -> shell 侧 connection -> shell 侧 channel proxy
- 下行通道: shell -> channel proxy -> shell 侧 connection -> 本地 connection -> 本地 provider -> 本地 channel.

通过这种方式, 可以将本地的树形 channel 一次性提供给远端.
