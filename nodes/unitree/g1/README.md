# G1 Body

Unitree G1 人形机器人身体 — MOSS 的具身能力代表。让一个 Ghost 降临到一台真实的 G1: 感知、移动、动手臂、说话, 在人类授予的授权内控制身体。

## node 结构

```
nodes/unitree/g1/
├── control/              # 整机身体 node — 完整 G1 channel 树 (当前单 node, 组分割延后)
│   ├── NODE.md           # 给运行时的 Ghost — 身体认知
│   ├── main.py           # 装配: sdk.bootstrap + provide_channel
│   ├── pyproject.toml    # 独立 venv (ghoshell-moss[matrix] + unitree-sdk2py)
│   ├── INSTALL.md        # 安装
│   └── skills/           # 技巧 (SKILL.md 范式)
├── docs/                 # 硬件 / SDK topic / PC2 装机技术文档
├── scripts/              # 安全原子化验证脚本 (SDK 调研, 各有 SKILL.md)
└── CLAUDE.md             # AI 认知入口 — 范式真相
```

共享能力库在 `src/ghoshell_moss_contrib/unitree/g1/` (框架内建, 未来可独立发布), node 只是薄壳装配。

## 当前状态

node 迁移第一阶段完成: 单 control node 落地, 待 G1 真机验证。**macOS 不可测试** (cyclonedds 不编译), 等价代码在 G1 真机 (PC2) 验证。

## 文档地图

| 文档 | 内容 |
|------|------|
| README.md | 整体 (本文件) |
| CLAUDE.md | 范式真相 — 安全先于设计 / Channel 最简 / macOS 规划 PC2 实装 |
| docs/ | 硬件网络拓扑 / DDS topic / PC2 装机日志 |
| scripts/ | 安全原子化验证脚本 (SDK 调研) |
| control/NODE.md | 给运行时的 Ghost — 身体认知与铁律 |
| control/INSTALL.md | 安装 |
| control/skills/ | 技巧 — 真机验证等 |

## 实机验证

安装见 `control/INSTALL.md`, 验证步骤见 `control/skills/verify/SKILL.md`。

## 参考

- [Unitree SDK2 Python](https://github.com/unitreerobotics/unitree_sdk2_python)
- [Unitree G1 开发者文档](https://support.unitree.com/home/zh/G1_developer/about_G1)
