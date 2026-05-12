# Liveness Key 过滤

**日期**: 2026-05-11
**模型**: DeepSeek V4 (via Claude Code)

## 问题

repl 中 `moss.dynamic()` 显示的 fractal hub 节点名出现乱码：

```
- **node/moss/channel_bridge/provider_liveness** (alive=True)
- **node/node/node/moss/channel_bridge/...**
```

根因：`_query_liveness` 使用 `MOSS/fractal/**` wildcard 查询，匹配到了 liveness key 之外的所有 channel bridge 子路径 key。

## 决策

在 `_query_liveness` 中加入过滤：只保留 `MOSS/fractal/{name}` 格式（无额外 `/`）的 key，跳过所有嵌套路径。

```python
name = key[len(prefix) + 1:]
if '/' in name:  # channel bridge 子路径
    continue
```

## 影响

- `connected()` 返回的 cell 名称变为干净的节点名
- FractalHubChannelState 的 context 消息不再污染
