# avatar

Live2D 虚拟形象躯体 node。驱动一个形象，人类在浏览器里看见它，Ghost 通过 CTML 驱动它。

文档入口：

- `NODE.md` — 节点机制（Ghost 挂载后读的也是它）
- `avatars/README.md` — 形象套件契约（怎么加/改一套形象）
- `INSTALL.md` + `skills/` — 自建环境与验证
- `../CLAUDE.md` — 方法论（范式真相）

## 运行

```bash
moss nodes run nodes/live2d/avatar -- --avatar <name>
```

打开启动日志里的页面地址。`--avatar` 优先级：命令行 > env `LIVE2D_AVATAR` > 第一个可用套件。
换形象 = 换 `--avatar` 重启。

## 开发

- 驱动框架在 `src/avatar_node/`：`cubism.py`（解析）、`lexicon.py`（共享词表）、
  `avatar.py`（事件面）、`bridge.py`（同源 server）、`mapper.py`（自动映射示范）、
  `discovery.py`（套件发现）。
- 页面在 `src/avatar_node/web/`；渲染库在 `vendor/`（gitignore）。
- 改自动映射的命名/分组 → `lexicon.py` + `cubism.py`，不写 per-model 配置。
