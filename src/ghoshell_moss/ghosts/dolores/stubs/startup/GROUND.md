# Startup

dolores 的开机启动机制。ghost runtime 装线完成后回调 `Ghost.startup()`，
dolores 读取当前 mode 的 startup 文档，作为 self-wake signal 发送出去激活自己。

## 怎么工作

启动完成后，dolores 把 startup 文档的 `instruction` 作为预热内容注入
（机制自动组装 `<startup>` 标签包裹），模型读完后自然回应；`command`
若非空，则作为 command logos 反射弧直接执行（不思考，强制首动作）。

## 文档

`{mode}.startup.yml` —— 按 mode 命名的启动文档；不存在时回退 `default.startup.yml`。

字段（都可选，缺省即不生效）：

- `description`：为什么创建这个 startup（文档向）
- `created_at`：创建时间字符串（文档向）
- `instruction`：预热 / 开场提示语，模型读到后自然回应
- `command`：可选。一条 CTML 命令，反射弧直接执行（强制首动作）。留空则模型自由回应 `instruction`

## 怎么改

- 给某个 mode 加启动提示：复制 `default.startup.yml` 为 `{mode}.startup.yml`，改字段内容。
- 改默认：直接编辑 `default.startup.yml`。
- 只想让模型自然开场：只写 `instruction`，不写 `command`。
- 想强制某个首动作：写 `command`。
