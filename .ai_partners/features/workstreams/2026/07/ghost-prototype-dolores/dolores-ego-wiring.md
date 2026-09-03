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

| # | 问题 | 归属 | 状态 |
|---|---|---|---|
| 1 | 回声全量重渲染（facade-delta 未生效） | shell-trajectory | 立 bug + 独立单测 |
| 2 | baseline `<key>value</key>` 渲染污染 | dolores epoch | 修（改渲染 + value 转义） |
| 3 | yield 返回 "ok" 哑载荷 | dolores ego | 改 moment_id |
| 4 | fetch_next_moment / wait_next_moment 词汇（轴应是 when 非 who） | dolores tools | 改 docstring |
| 5 | "thinking" 占位符泄漏 | dolores ego | 删/改 |
| 6 | dsh UI 生命周期（消息被吞） | dsh-fusion | 查（疑似 epoch/enter 引入） |
| 7 | exit 失败闸门残留 | dolores ego | 待定 |
| 8 | inputs_messages 不一致 | dolores ego | 验证 |
| 9 | observe 镜像风险 | dolores tools | 验证 |
