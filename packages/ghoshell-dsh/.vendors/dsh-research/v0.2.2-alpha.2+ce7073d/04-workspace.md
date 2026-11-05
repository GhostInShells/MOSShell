# 04 — workspace 是投影，不是身份

## 动机

我们的 case 需要一个"这个 session 属于谁"的答案。dsh 现成的 workspace 能不能直接拿来当那个答案？
顺带解释一个现场现象：会话界面里同时出现"主路 / 旁路 / @ home"三组。

## 路径（可重放）

```sh
cd .vendors/dsh-research/v0.2.2-alpha.2+ce7073d/source
sed -n '1,140p' packages/workspace/workspace/README.md
sed -n '1,80p'  packages/workspace/workspace/src/paths.ts

# 实盘（DSH_HOME 下）
cd ../../../../../.moss/ghosts/moss/.dsh
node -e "…读 storages/workspace.json，按 project 汇总 sessionIds…"
ls sessions/            # 分桶名就是 cwd 编码
zstd -dc <session>.jsonl.zstd | head -1   # 看 header 的 cwd
```

## 结论

- `[读码]` **唯一性 canon 是 `fs.realpath`**：一个规范化路径 = 一个 project，符号链接指向已有目录也算同一个；
  对已有目录再 `create` 原样返回旧的。→ **同一路径挂两个 id 不存在**。
  （`packages/workspace/workspace/README.md` 实现节 *One record per canonical path*）
- `[读码]` **归属由 `SessionHeader.cwd` 派生**：session 加入它所在目录的 project，一个 session 只属于一个 project；
  bootstrap 从持久化的 header 重建分组，且 *header-only reads*——只读 header，不加载事件体。
- `[读码]` **title 可变，路径不可变**：`setTitle()` 是显示层，身份是 `WorkspaceId`，归属是规范化路径。
- `[读码]` **archive 在 session 上，且是全局集合**：`archivedSessionIds` 存在 registry 全局 state 里，
  不改变 workspace 归属；删除 project 是非破坏的（folder / files / histories 一律不动，session 变 ungrouped）。
- `[读码]` **archive 自带 activity 闸**：先走 `workspace/session-activity`（waterfall）问各 provider
  还有什么在跑——自己的 turn、跑着的 subagent、背景 job、活跃 reminder；非空即拒，并列出来。
  `{stopActivity:true}` 才先写 archive 再发 stop。
- `[读码]` **对模型不可见**：package 声明 *invisible to models*，零 prompt / 零 request 成本。它是 host 特性，不是运行时原语。
- `[实测]` **它挡不住 cwd 选错**：本项目 DSH_HOME 下 69 个 session 的 cwd 是 `.moss/ghosts/moss/.dsh`，
  **没有一个进任何 project**；时间跨度 10-02 → 10-30，今天仍在发生。见文末现场。

## 观点

**workspace 是投影，不是身份。** 它的键是 cwd，而 cwd 由启动者决定——所以它回答的是
"这个 session 从哪个目录启动的"，不是"这个 session 属于哪个任务"。

对我们的设计，这给出一条决定：

> **case 的身份必须由我们自己签发，不能从 workspace 借用。** workspace 只能当展示层的分组投影用
> （它便宜、可重建、对模型不可见，正好适合当"人的那一侧"）。

顺带两条可以借的：`archive` 是 session 级、全局集合、带 activity 闸——**这正是我们说的 stop the world**，
而且是现成的；`header-only reads` 是可重建索引该有的成本，我们的 sid→case 反向索引照这个标准做。

失效条件：如果将来 dsh 把 workspace 的键从 cwd 改成显式 id（或允许同一路径多记录），"投影不是身份"
这条仍然成立，但"同路径不能多 id"要重读一遍。

## 现场（待单独修复，不在本轮）

```
project 记录：moss @ MOSShell(186)  moss @ home(283)
sessions 分桶（按 cwd）：
  项目根            189   其中 3 个不在 registry
  ghost home        283   全部在
  .dsh (DSH_HOME)    69   **全部不在任何 project**
  /tmp probe          3   不在
```

两个待查：① 69+3 无主 session（cwd 落在 DSH_HOME / /tmp）——UI 上就是那个第三组；
② registry 里 home 有 283 条，但界面上 home 现在看不到数据，两边不一致。
