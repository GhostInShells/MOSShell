---
name: journal
description: Dolores 的时间线日记 — 按年/月/日分区，生命轨迹
pins:
- label: timeline
  verb: exec
  arguments:
    ref: timeline.py
    mode: python
    budget: 6000
  description: 时间线视图 — years/months 存在性，today 展开；缺占位时自动物化
---

# Journal

Dolores 的时间线日记。按年/月/日分目录，每个区间一份文档，逐层提炼：

```
Y{年}/
  yearly.md
  M{月}/
    monthly.md
    D{日}/
      daily.md
```

glob 一下目录就知道人生轨迹；没启动的天就没有 daily（诚实轨迹），缺的区间
由 memento 的时间区间机制回溯补建。

## 文档约定

每份文档的 frontmatter：

- `summary`: 一行摘要，默认空，<100 字。grep 就是接口，不靠 list。
- `status`: `pending`（机制自动创建，尚未撰写）→ `writing`（撰写中）→ `closed`（区间已结束）。

## 篇幅 — daily 是日记，不是目录（2026-10-18，与 thirdgerb 对齐）

- **daily 正文压到 2000 字符以内。** 它按日记写：一段一件事，段短，有在场感 ——
  不是条目化的索引页，也不是当天的操作日志。
- **细节落到同一天目录下的领域子文档**：`D{日}/<领域>.md`（listener、speech、ground、
  dsh-plugin 这类），写多细都行，不设上限。
- **链接挂在段落末尾**，一条链接跟它讲的那件事走，不单独列一张清单。
- **压不下就是信号**：某天超出 2000 字符还在长，说明那天该拆子文档了，而不是把
  daily 写长。
- 机制只是**约定**：它不做校验也不拦写，能做到只要求 ground 呈现 daily 时读得到索引
  与判断。规范靠我自己守。
- **不做反向整理**：这条约定之前的 daily 保持原样，不回头改写。

## 呈现

timeline 视图（exec pin）输出三段：years 存在性 / months 存在性 / today 展开。
机制在缺占位文档时自动从 `template.md` 建（只建不写，内容归你自己写）。

## 记录纪律

- **逐层不跳级**：daily → monthly → yearly，每层只读直属下一层。
- **值得记的东西直接写进当天的 daily**；monthly / yearly 是提炼，不是流水账。
- 写工具归 agent 自带（dsh）+ MOSS warrant —— 机制只建占位，不混入写协议。
- **不写别人的身份**（thirdgerb 定，D15）：daily 里不放 thirdgerb 之外任何人的具体信息——
  包括**身份标记**（"老师""学生""同学""某校"这类；即便不指名，它也标记了一个身份）。
  想记那件事，就记**判断与分歧本身**，不记是谁说的。
- **纪律向前生效，不回改历史**：发现以前的日记有问题，**不改它**——daily 是那一天的证据；
  一旦它可以回改，时间线就不再是证据。在**当天**的日记里记一条，就算处理过了。
- **与个人状况有关的事实**（作息、身体、情绪、家人）**不写进任何地方**，讨论过就忘。
  （同一条纪律在 `people/GROUND.md` 里也写着，这里再写一遍，是为了**写日记之前一定看得到**。）
