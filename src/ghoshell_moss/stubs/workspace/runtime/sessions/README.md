# Sessions

MOSS 运行时有三个关键的参数: mode - ghost - network. 
分别决定使用什么环境, 什么 ghost, 以及用什么通讯网络. 

这三个参数决定了一个运行状态, 为它们构建一个可复用的存储隔离级别, 就是 session-scope. 
