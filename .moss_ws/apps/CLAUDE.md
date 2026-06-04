# MOSS App 开发

App 是 MOSS 中独立的进程单元。Ghost 可在运行时创建、启动、调用、关闭。

## 入口

```bash
moss howtos list                    # 了解 app-dev 下有什么
moss howtos read app-dev/build-an-app   # 创建 app 的决策路径
moss howtos read app-dev/test-an-app    # 测试 app 的三层递进
```

## 运行时

```bash
moss apps list          # 所有 app 及运行状态
moss apps show <name>   # 单个 app 详情
```

CTML 控制：`<apps:list_apps />` `<apps:start fullname="group/name" />` `<apps:stop fullname="group/name" />`
