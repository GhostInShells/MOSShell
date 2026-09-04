# Dolores Ego 装线 — 收敛的设计结论

> dolores 的子任务。2026-09-02 让 running 的 ghost 读自己的代码做 dogfood 评审。
> 原「三方视角」过程记录已删——信息不完整，曾误导后续接手导致重新对齐；完整讨论轨迹见
> git log（`git log -- src/ghoshell_moss/ghosts/dolores/`）。
> 由 `ghost-prototype-dolores` FEATURE.md 关联索引，不追加进主 feature。

## 收敛的设计结论

| 层 | 语义 |
|---|---|
| plain-text（`<\|CTML\|>` 之外） | 外部信息，markdown（纯视觉通道），不执行、不发声 |
| `<\|CTML\|>`（之内） | 控制语法（含 `__content__` 自由文本→语音） |
| tool 追加 CTML | interleaved（思维超前于行为） |

`<|CTML|>` 是模式分隔符（不是 tokenizer 特殊 token）：默认 plain-text 模式（`<` `>` 是字面量），
遇 `<|CTML|>` 切进 CTML 模式（SAX 只看到 CTML 内容）。它解决的是 `<` `>` 字符冲突，不是语音通道问题。

## Bug 清单

> 第一轮 dogfood 产出的 bug 已折进 [dolores-todo.md](dolores-todo.md)（D8–D16），状态以 todo 为准。
