# GhostRuntime 架构设计结论

日期: 2026-05-16

## 四种方案

| # | 方案 | 本质 | 判定 |
|---|------|------|------|
| 1 | Moss 包含 Ghost | MossRuntime 内置 ghost slot，`.ghost` 返回 `Ghost \| None` | 否。隐式启动，ghost 生命周期不可独立验证 |
| 2 | Ghost 包含 Moss | GhostRuntime 是完全独立的 Runtime | 否。API 面复制 MossRuntime，ghost 绑死在本地 |
| 3 | 第三方编排 | Host 同时管理 MossRuntime + GhostRuntime 两个生命周期 | 否。多一层间接，且未解决两套 API 的问题 |
| 4 | GhostRuntime 持有 MossRuntime | 组合优于继承，独立 ABC + 薄 adapter | **选定** |

## 核心决策

**1. GhostRuntime 不是 Runtime。** MossRuntime 管理执行（shell/channel/apps）。GhostRuntime 只做一件事：在 MossRuntime 启动前后完成 Ghost 的注册和生命周期编排。命名里的 "Runtime" 容易误导——它本质是一个 Adapter。

**2. 组合优于伪装。** GhostRuntime 不实现 MossRuntime ABC。调用方通过 `.moss` 访问全部 Moss 能力，通过 `.ghost` 访问 Ghost。隐式委托比显式包装更诚实。

**3. API 面克制。** 只暴露 5 个成员。不加 moss 业务方法的路径压缩（如 `.shell`/`.session`/`.matrix`/`.mode`）。`.moss` 是唯一的 Moss 能力入口。防止滑向方案 2 的 API 膨胀。

**4. ghost + mindflow main loop 托管给 Matrix。** 通过 `matrix.create_task()` 注册内部 async 函数，Matrix 退出时自动 cancel。GhostRuntime 不自行管理关闭信号，`close()` 一行委托给 `self.moss.close()`。

**5. GhostRuntime 保持 ABC。** 抽象为可扩展设计：不同 Ghost 原型（Atom 及后续）可能需不同的编排逻辑。同时为第三方零上下文开发保留接口。

## 最终 ABC

```python
class GhostRuntime(ABC):
    # 五个成员: 4 property + close() + __aenter__/__aexit__

    .moss       → MossRuntime    # 全部 moss 能力
    .ghost      → Ghost          # factory 产出的运行时实例
    .meta       → GhostMeta      # 启动前即可访问的元信息
    .container  → IoCContainer   # 快捷路径: moss.matrix.container
    close()     → self.moss.close()
```

文件位置: `ghoshell_moss/core/blueprint/host.py` — 与 `MossRuntime`/`MossSystemPrompter` 同簇。

## 三循环验证路径

Mindflow 测试套件 (`tests/ghoshell_moss/core/mindflow/test_base_mindflow.py`) 已验证三循环可运行：

```
main loop:    mindflow.loop() → Attention → (Articulator, Action) → janus.Queue
articulate:   queue → ghost.articulate(articulator) → send_nowait()
action:       queue → action.received_logos() → CTML 执行 → outcome()
                                                      ↑
                                            moss.moss_exec() 在这里
```

GhostRuntime 的 `__aenter__` 实现负责 wiring：预注入 providers/nuclei → MossRuntime 启动 → ghost 实例化 → 注册 articulate/action loops 为 matrix tasks。

MindflowSuite 已验证的图景:
- 三线程并行 (main + articulate + action)
- 单线程串行 (task 模式)
- 连续 observe 循环 (10 轮不中断)
- 信号过期/suppress 竞争
- incomplete signal 等待 complete

## 实现阶段

当前 step 10 的 ABC 定义已完成。下一步是实现 `GhostRuntimeImpl`，继承 ABC 完成 wiring。
