# Dolores Ego 装线 — 结论与 Bug 清单

> dolores 的子任务。2026-09-02 让 running 的 ghost 读自己的代码做 dogfood 评审。
> 原「三方视角」过程记录已删——信息不完整，曾误导后续接手导致重新对齐；完整讨论轨迹见
> git log（`git log -- src/ghoshell_moss/ghosts/dolores/`）。
> 由 `ghost-prototype-dolores` FEATURE.md 关联索引，不追加进主 feature。

## 收敛的设计结论

| 层 | 语义 |
|---|---|
| plain-text（`<\|CTML\|>` 之外） | 外部信息，markdown-first 或 speech-first（mode 二选一） |
| `<\|CTML\|>`（之内） | 控制语法（含 `__content__` 自由文本→语音） |
| tool 追加 CTML | interleaved（思维超前于行为） |

`<|CTML|>` 是模式分隔符（不是 tokenizer 特殊 token）：默认 plain-text 模式（`<` `>` 是字面量），
遇 `<|CTML|>` 切进 CTML 模式（SAX 只看到 CTML 内容）。它解决的是 `<` `>` 字符冲突，不是语音通道问题。

## Bug 清单

> 第一轮 dogfood 产出的 bug，多数已在后续 commit 修复。状态以代码为准，勿把本表当待办。

| # | 问题 | 状态 |
|---|---|---|
| 1 | 回声全量重渲染（facade-delta 未生效） | 已修 `2e57a8f8` |
| 2 | baseline `<key>value</key>` 渲染污染 | 记录错误——key 作 tag 经讨论判定正确，无需改 |
| 3 | yield 返回 "ok" 哑载荷 | 已修 `59f13736`+`ab6aaac1`（moment index 替代 uuid） |
| 4 | fetch_next_moment / wait_next_moment 词汇（when vs who） | 已修 `ab6aaac1`（typed tool surface） |
| 5 | "thinking" 占位符泄漏 | 已修 |
| 6 | dsh UI 生命周期（消息被吞） | 已修 `ea90993a`（pre-step never rejects） |
| 7 | exit 失败闸门残留 | 已修 `ea90993a`（pre-step gate 重构） |
| 8 | inputs_messages 不一致 | 已修（复用 `inputs_messages(with_command_executing=False)`，executing 归 context） |
| 9 | observe 镜像风险 | 已修 `59f13736`（moment index 帧带序号） |
