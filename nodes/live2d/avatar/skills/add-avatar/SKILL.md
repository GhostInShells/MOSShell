---
name: live2d-avatar-add
description: 新增或定制一套 Live2D 形象 — 放模型资产、写 channel.py 定义命令面、启动验证。
---

# 新增 / 定制一套形象

一套形象 = `avatars/<name>/` 一个自包含目录。契约见 `../avatars/README.md`。

## 五步

### 1. 放模型

```bash
mkdir -p avatars/<name>/model
# 放入一个 Cubism 模型包, 必须含 *.model3.json 入口
```

### 2. （可选）写 AVATAR.md 人设与 idle 配置

```yaml
---
name: <显示名>
description: 一句冷人设
voice: <TTS 音色名, 如 可爱女生>
groups:                    # 可选: 覆盖某 group 子 channel 的 instruction
  face:
    instruction: "..."
idle:                      # 可选: 待机配置
  delay: 3.0
  loop: { group: Idle, index: 0 }
  parts: { blink: true, breath: true }
---
正文 = 更完整人设, 不进 instruction
```

模型包自带眨眼的（动作曲线驱动眼开闭），把 `idle.parts.blink` 设 `false`，避免和 SDK
默认眨眼双重闪烁。

### 3. （可选）写 channel.py 定制命令面

不写则走自动映射（`mapper.py`，示范面）。要设计好用/语义化的命令面就写：

```python
# avatars/<name>/channel.py
from ghoshell_moss.core.blueprint.states_channel import new_prime_channel

async def build(avatar):
    chan = new_prime_channel(name="<name>", description="...")

    @chan.build.command()
    async def look(x: float = 0.0, y: float = 0.0) -> None:
        """让眼睛转向 (x, y)，取值 -1 到 1"""
        avatar.params({"ParamEyeBallX": x, "ParamEyeBallY": y})

    return chan
```

`build` 必须返回 `PrimeChannel`（`new_prime_channel` 构建），驱动要 `with_module` 挂动画轨迹。

可用的事件面见 `avatars/README.md`（`param`/`play`/`motion`/`expression`/`reset`/`backdrop`/`set_idle_loop`）。
参数 id 从 `avatar.spec` 拿；唇形/眨眼绑定读 `avatar.spec.lip_sync` / `eye_blink`。

### 4. （可选）写 animations.py 动画轨迹

每个 `async def` 是一条动画命令，编译反射到主 channel。注入 `get_avatar()` 与 `asyncio`：

```python
# avatars/<name>/animations.py
async def wave():
    """打招呼: 抬左手挥一挥."""
    avatar = get_avatar()
    await avatar.play("Tap@Body", 0)
    await asyncio.sleep(0.2)
    avatar.param("ParamArmLA", 0.8, manual=True)
```

编辑后调 `reload_animations` 热更新（编译失败保留上一版）。契约见 `avatars/README.md`。

### 5. 启动验证

```bash
moss nodes run nodes/live2d/avatar -- --avatar <name>
```

换形象 = 换 `--avatar` 重启。

## 常用动作

- 想直接改自动映射的命名/分组规则 → 改 `src/avatar_node/lexicon.py`（共享词表）与
  `cubism.py`（噪声过滤），不写 per-model 配置。
- 想要一个从 stub 起的空白 channel.py 起步 → 参考 `avatars/README.md` 里的最小样例。
