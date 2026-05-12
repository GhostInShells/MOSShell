# Debug 工具链

**日期**: 2026-05-11
**模型**: DeepSeek V4 (via Claude Code)

## 状态

半成品。最小的调试能力已就位。

## 已完成

- `moss-as-fractal` CLI：`ClickHandler` 将 matrix.logger 输出到控制台
- `on_proxy_event` 回调：channel duplex 事件打印到屏幕（心跳聚合、错误高亮）
- 两个流合一，都在 click.echo

## TODO

- [ ] 统一的 moss 调试工具（不只 fractal）
- [ ] `moss debug fractal` 子命令，替代 CLI 里零散的 handler 挂载
- [ ] 结构化日志级别控制 (`--log-level`)
- [ ] 非阻塞的 logger stream（janus queue + 独立 consumer task）
