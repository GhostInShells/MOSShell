# 关于 duplex

本目录存放 MOSS Channel 双工通讯链路的基础协议.

基本原理是: `provider.run(channel) -> provider connection -> proxy connection -> proxy`

这里的基础协议在 Alpha 版本中尚未沉淀完. 等到完全成型后, 会成为跨语言 channel 通讯的标准协议.

具体实现计划要有各种版本: 父子进程 / websocket / zmq / redis / mqtt 等等.
