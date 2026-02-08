# 目录介绍

- core: ghoshell moss 的核心功能模块
- message: 兼容性的模型消息协议. 暂时放到 ghoshell-moss 库, 未来可能迁出
- transports: 通过 provider -> proxy 范式, 跨进程的构建 channel 之间的双工通讯.
- compatible: 兼容性模块, 用来兼容行业生态. 比如 claude mcp 和 claude skills.
- channels: ghoshell-moss 库认为需要开箱自带的 channel 实现.
