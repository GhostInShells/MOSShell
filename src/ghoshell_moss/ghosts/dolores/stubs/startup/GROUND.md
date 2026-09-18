# Startup

dolores 的开机启动机制。ghost runtime 装线完成后回调 `Ghost.startup()`，
dolores 读取当前 mode 的 startup 文档，作为 self-wake signal 发送出去激活自己。

## 文档

`{mode}.startup.yml` —— 按 mode 命名的启动文档；不存在时回退 `default.startup.yml`。

字段（都可选，缺省即不生效）：

- `description`：为什么创建这个 startup（文档向）
- `created_at`：创建时间字符串（文档向）
- `command`：一条 CTML 命令，作为 command logos 直接执行（说话）
- `instruction`：预热指令正文，机制组装成 `<startup>` 握手协议注入，模型预热后回 `pong`

`<startup>` 包裹与握手协议由启动机制负责，不写进文档。
