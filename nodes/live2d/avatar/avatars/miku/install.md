# miku — 本地测试套件

miku 是驱动框架的**本地冒烟案例**：第一个用来验证"标准 Cubism 包丢进来就能驱动"的
模型。它没有 `channel.py`，走自动映射（`mapper.py`），所以它同时证明了兜底路径成立。

模型资产是经典 IP，**不可分发**——`model/` 整个目录被 gitignore，这里只留这份说明。
把一份 MIKU 模型包放进 `model/`，要求有一个入口 `*.model3.json`（名字不限）：

```
avatars/miku/model/
├── miku.model3.json      # 入口（本仓库验证用的是这个名字）
├── MIKU.moc3
├── MIKU.cdi3.json
├── MIKU.4096/texture_00.png
├── MIKU.physics3.json
└── 表情和动作/*.motion3.json, *.exp3.json
```

本项目验证用的 MIKU 包来自 LiveHime 导出；任何结构等价的 Cubism 包都可以。放好后：

```bash
moss nodes run nodes/live2d/avatar -- --avatar miku
```

> 没有 `model/miku.model3.json` 时，驱动只会报告这套形象不可用（不在发现列表里），
> 不影响其它套件。
