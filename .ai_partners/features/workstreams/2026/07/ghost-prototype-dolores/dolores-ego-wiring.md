# Dolores Ego 装线 — 收敛的设计结论

> dolores 的子任务。2026-09-02 让 running 的 ghost 读自己的代码做 dogfood 评审。
> 原「三方视角」过程记录已删——信息不完整，曾误导后续接手导致重新对齐；完整讨论轨迹见
> git log（`git log -- src/ghoshell_moss/ghosts/dolores/`）。
> 由 `ghost-prototype-dolores` FEATURE.md 关联索引，不追加进主 feature。

## 收敛的设计结论

| 层 | 语义 |
|---|---|
| CTML（默认流） | 控制语法（含 `__content__` 自由文本 → 语音），MOSS 流式解析执行 |
| `<\|Markdown\|>…</\|Markdown\|>` | markdown（纯视觉通道），dsh web 渲染，不执行、不发声 |
| tool 追加 CTML | interleaved（思维超前于行为） |

`<|Markdown|>` 是成对 escape（不是 tokenizer 特殊 token）：默认整条流是 CTML，
遇 `<|Markdown|>` 切出到 markdown，遇 `</|Markdown|>` 切回。它解决「默认是控制还是展示」的
通道问题 —— CTML 是默认，markdown 是显式 escape。

## Bug 清单

> 第一轮 dogfood 产出的 bug 已折进 [dolores-todo.md](dolores-todo.md)（D8–D16），状态以 todo 为准。
