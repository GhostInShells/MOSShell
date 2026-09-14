---
name: live2d-avatar-add
description: 新增或定制一套 Live2D 形象 — 放模型资产、写 channel.py 定义命令面、启动验证。
---

# 新增 / 定制一套形象

一套形象 = `avatars/<name>/` 一个自包含目录。契约见 `../avatars/README.md`。

## 三步

### 1. 放模型

```bash
mkdir -p avatars/<name>/model
# 放入一个 Cubism 模型包, 必须含 *.model3.json 入口
```

### 2. （可选）写 channel.py 定制命令面

不写则走自动映射（`mapper.py`，示范面）。要设计好用/语义化的命令面就写：

```python
# avatars/<name>/channel.py
from ghoshell_moss.core.blueprint.channel_builder import new_channel

async def build(avatar):
    chan = new_channel(name="<name>", description="...")

    @chan.build.command()
    async def look(x: float = 0.0, y: float = 0.0) -> None:
        """让眼睛转向 (x, y)，取值 -1 到 1"""
        avatar.params({"ParamEyeBallX": x, "ParamEyeBallY": y})

    return chan
```

可用的事件面见 `avatars/README.md`（`param`/`motion`/`expression`/`reset`/`backdrop`）。
参数 id 从 `avatar.spec` 拿；唇形/眨眼绑定读 `avatar.spec.lip_sync` / `eye_blink`。

### 3. 启动验证

```bash
moss nodes run nodes/live2d/avatar -- --avatar <name>
```

换形象 = 换 `--avatar` 重启。

## 常用动作

- 想直接改自动映射的命名/分组规则 → 改 `src/avatar_node/lexicon.py`（共享词表）与
  `cubism.py`（噪声过滤），不写 per-model 配置。
- 想要一个从 stub 起的空白 channel.py 起步 → 参考 `avatars/README.md` 里的最小样例。
