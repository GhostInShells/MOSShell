---
created: 2026-05-18
depends:
- first-ghost-prototype
description: 可预设返回值的 Mock Ghost，用于测试 GhostRuntime 体系而不依赖真实模型调用。
milestone: null
priority: P0
status: completed
status_note: MockGhost + MockArticulator 原型完成，24 tests pass
step: implementation
title: Mock Ghost Prototype
updated: '2026-05-18'
---

# Mock Ghost Prototype

## 动机

测试 GhostRuntime（三循环调度、session signal 路由、task output 协议）时，
不需要跑一个真实的 Ghost（Atom 需要 Anthropic API key + 模型调用）。
需要一个可控的 mock，预设 `articulate()` 等核心方法的返回值。

## 核心决策

1. **参数随时可替换**：所有预设值都是 public 属性，不是构造时固化。测试中随时 swap。
2. **articulate 预设为队列模式**：`articulate_responses` 是 `list[str]`，按序 yield。
   支持多轮：测试可以中途替换列表模拟不同轮次的模型输出。
3. **放在 `ghosts/mock/`**：遵循 atom 的 package 结构，作为独立 Ghost 原型。
4. **零外部依赖**：不依赖 pydantic_ai、anthropic SDK。MockGhost 是纯 Python。

## 文件结构

```
src/ghoshell_moss/ghosts/mock/
├── __init__.py       # 导出 MockGhost, MockGhostMeta
├── _meta.py          # MockGhostMeta — GhostMeta 的可预设实现
├── _runtime.py       # MockGhost — Ghost 的可预设实现
└── test_mock.py      # package 内测试
```

## 预设参数一览

### MockGhostMeta
| 属性 | 类型 | 默认值 |
|------|------|--------|
| `name` | `str` | `"mock"` |
| `description` | `str` | `"Mock ghost for testing"` |
| `nuclei_metas` | `list[NucleusMeta]` | `[]` |
| `providers` | `list[Provider]` | `[]` |

### MockGhost
| 属性 | 类型 | 默认值 |
|------|------|--------|
| `articulate_responses` | `list[str]` | `[]` |
| `system_prompt` | `str` | `""` |
| `memories` | `list[Message]` | `[]` |
| `channel` | `Channel \| None` | `None` |
| `mindflow` | `Mindflow \| None` | `None` |