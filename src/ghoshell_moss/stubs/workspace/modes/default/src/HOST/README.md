# HOST — mode 专属声明

只在 **host 节点**启动时加载的配置（host-only）。

- `channels.py` — 主 channel 唯一入口（单文件模块，非 package）。
- `providers/ configs/ topics/ signals/ parameters/ resources/ nuclei/` — mode 专属声明，覆盖/扩展 project 层与 matrix 层。
- 默认内容随 `moss modes overwrite-stubs <name>` 同步。
