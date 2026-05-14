# 02 — 最小原型技术目标

日期：2026-05-14

## 背景

明确了 Ghost 的三层抽象后，需要定义第一个原型（Atom）的最小实现范围。目标不是功能完整，而是跑通全链路，作为后续 Ghost 的参照基线。

## 讨论要点

### 1. 代码位置

- 官方原型统一放在 `src/ghoshell_moss/ghosts/`
- 第一版覆盖 `atom/` 下的 `_meta.py` 和 `_runtime.py`
- 旧文件保留（human 工程师写的例子），开发完毕后删除

### 2. Ghost 专属工作空间

每个 ghost 在 workspace 中有独立治理目录：

```
[workspace]/ghosts/{name}/
```

- 路径通过 IoC 拿到 workspace 后拼接
- 目录内组织由各原型自行决定（Adapter 哲学）
- 隔离边界清晰，互不干扰

### 3. Soul 文件

把"灵魂"从"配置"中解耦：

```
[workspace]/souls/{name}.md
```

- 不放入 manifests——这是对行业现有做法的兼容，不是 MOSS 原生概念
- 一个 soul 可被多个 ghost 实例复用
- 极简版直接读 markdown 全文，不做结构化解析

Soul 引用方式（多态）：

```python
# str: 约定路径，从 workspace/souls/ 下加载文件
soul: str = "atom_default"

# Path: 绝对路径，调试用
soul: Path = Path("/absolute/path/to/soul.md")
```

`str` 是常规路径，`Path` 是调试逃生门。

### 4. 上下文组装管线

```
soul.md  +  moss_meta_instruction (CTML)  +  moss_static (channels/interfaces)
    ↓                    ↓                              ↓
  存在/目的/对齐      如何使用 MOSS 与环境交互         可用能力列表
    ↓                    ↓                              ↓
              完整的 system prompt
```

三层各司其职，互不耦合。

### 5. Moment 二次转义

`Moment` 包含动态运行时快照（perspectives），不能直接存入历史：

```
Moment (含 perspectives: 动态快照)
    │
    ├──→ to_model_request()     第一次转义：拼请求上下文（包含 perspectives）
    │         ↓ 发送给模型
    │         ↓ 拿到 ModelResponse
    │
    └──→ save_model_request()   第二次转义：存入历史前剥离过期动态信息
              perspectives 不可回放，需裁掉
              只保留 logos + outcomes + stop_reason (= Reaction)
```

`ModelContext` 已建模此结构。`to_messages()` 中 `with_history_perspective_turns` 参数控制回放策略。

### 6. Conversation 内存实现

- v0 不做持久化，纯内存
- `ConversationStore` ABC 已存在，但不接入
- 持久化作为后续迭代目标

### 7. 使用默认框架组件

极简版 Ghost 不需要自己定义：
- `mindflow()` — 返回 None，用 MOSS 默认实现
- `channel()` — 返回 None，无反身性控制
- `nuclei()` — 返回 []，无自定义感知模块

### 8. 模型后端

使用 PydanticAI Agent 作为模型调用层。已有 `core/agents/pydantic_agents/` 适配。

## 决策结论

1. 代码位置：`ghosts/atom/`，覆盖 _meta.py + _runtime.py
2. Ghost 工作空间：`[workspace]/ghosts/{name}/`，由 IoC 提供 workspace 路径
3. Soul：`[workspace]/souls/` 下的 markdown 文件，str/Path 多态引用
4. Context 组装：soul + moss_meta_instruction + moss_static → system_prompt
5. Moment → to_model_request() → save_model_request() 两次转义
6. Conversation 纯内存
7. 用默认 Mindflow，无自定义 nuclei/channel
8. 模型后端：PydanticAI Agent
