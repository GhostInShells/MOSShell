# Hub 按需节点管理

**日期**: 2026-05-11
**模型**: DeepSeek V4 (via Claude Code)

## 问题

原来的 `FractalHubChannelState.get_virtual_children()` 对**所有**发现的子节点自动创建 ZenohProxyChannel。这带来两个问题：

1. 连接上不等于要立刻提供——AI 应该有选择权
2. 过多不需要的 proxy 造成资源浪费和 runtime 重连

## 决策

改为**按需打开**模式：

- 新增 `_opened_nodes: set[str]` 跟踪 AI 显式打开的节点
- 新增 `open_node(name)` / `close_node(name)` 两个 PyCommand
- `get_virtual_children()` 只返回已打开节点的 proxy
- Context 消息区分「可用节点」和「已打开节点」

参考 `AppStoreChannelState` 的 PyCommand 模式实现。

## 影响

- AI 需要先查看 context 了解可用节点，再 `open_node` 连接
- 断线节点自动清理出 opened 集合
- 已打开节点的 proxy 仍然复用（和之前的优化一致）

## 待观察

- AI 是否能自然理解 open_node/close_node 的使用模式
- 是否需要 `open_all` / `close_all` 批量命令
