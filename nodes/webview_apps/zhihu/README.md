# zhihu

知乎内容 node：以官方 `zhihu-cli` 为 subprocess 底盘，ghost 在人类授权下读本人私域
（创作/评论/统计/关注/收藏）+ 检索知乎，结果呈现在共享 web 面。

## What it does

- **读本人私域**（需 Access Secret）：`me contents` / `me content` / `me comments` /
  `me stats` / `me content-stats` / `me followees` / `me favorites` / `knowledge`
- **检索公共内容**（无需凭证）：`search zhihu` / `search global` / `hot` / `answer` /
  `question recommend` / `question answers`
- **web 共享面**：凭证引导卡 + action 卡流 + 裁决（单条通过 / 整个 type 自动 / 拒绝）

模型侧只有一个 `action` 命令（text__ = Action JSON），菜单来自 `capabilities`、参数来自
`help`；node 不写死 16 个命令。

## Setup

见 `INSTALL.md`：装 CLI + 授权 + `moss nodes install`。

## Usage

```bash
moss nodes run nodes/webview_apps/zhihu
```

web 面端口默认 ephemeral，从 channel 的 `url` notice 读实际地址；`--port N` 或
`MOSS_ZHIHU_PORT` 固定。

## Development

```
src/ghoshell_zhihu/
  action.py   — Action / ActionResult pydantic + 数据形状提取
  cli.py      — zhihu-cli subprocess 封装（run / capabilities / help / auth_set）
  store.py    — 单一 store：action 记录 + type 授权格 + 凭证态
  channel.py  — 模型侧 channel（action / capabilities / help / read / quota）
  surface.py  — 人类侧 web 面（WS 下行 action 帧、上行 auth/approve/reject/auto）
index.html    — web 主体
main.py       — 装配入口
skill/        — vendored 官方安装底盘（manifest.json + scripts）
```

设计决策见 `.ai_partners/features/workstreams/2026/09/zhihu-node/FEATURE.md`。
