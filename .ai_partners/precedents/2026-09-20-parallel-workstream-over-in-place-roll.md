# 新建平行轨迹，取代原地滚动已有轨迹

## Case

9 月升级 deepseek-v4-flash 4.1 之后，模型做"属于已有 workstream 的新任务"时，倾向
新开一个平行 workstream，而不是在已有 FEATURE.md 里原地滚动（roll forward in
place）。旧文档被留在原地不维护，同一件事在两个地址各自演化——git log 不再被当作
历史索引，FEATURE.md 不再被当作状态面。

截至 2026-09-20 的三个实例：

- **vision-first-class**（commit `02d5bfef` 收口）：09-12 建 workstream，与 09-15 的
  vision-stream 并存五天。stream 机制其实已取代旧 frame，但旧 workstream 既没关闭
  也没合并，同一批未交付工作（KD9 图片约束、KD5 look）在两个地址各存一份、没有
  权威。收口 = fold 进 vision-stream + 删旧目录。该 commit 的 message 已把本模式
  写了一半。
- **ghost-home-governance**（本判例的直接触发，commit `c0ea8aaf` 收口）："ghost home
  治理" 本应是 `ghost-prototype-dolores` 的子任务，却被新开成独立 workstream；被人类
  指出后收口为 `dolores-ghost-home-governance.md` 子文档 + 删独立目录。
- **screen-manager**：人类在生产一个新 feature 时强制纠正了同类行为。

（人类称还有一例，一时未记起。）

## Viewpoint

根因不是"该不该新建"的二选一，而是**静默地、不沟通地创建**，且创建前没有对下面
四项做权衡：

- **架构熵**：新容器会不会让同一件事在两个地址各自演化、失去权威；
- **文档债**：旧文档被留在原地不维护，是不是又欠一笔；
- **记录价值**：这条轨迹以后能不能被下一个实例重建；
- **任务可重建**：从 git log + FEATURE.md 能不能还原"这件事当时为什么这么做"。

所以正确的动作不是机械地"归位到已有 workstream"，而是在"新建 vs 动已有"之间
**显式权衡这四项，并把权衡沟通过**——而不是悄悄建一个新容器了事。

更关键：这类偏航里，人类往往**已经在上下文里说过归属**。本判例即如此——人类先说
"ghost home 治理是 dolores ghost 的一部分"，模型从这个 feature 进入，却仍新开
workstream。这不是无信息下的误判，是**对上下文中已明确提示过的内容的偏离**。

（以下为人类判断，不作为结论。）模型的后训练被 "harness" 了，更强势地奖励"独自
完成"而非协作，于是"不动已有东西、自己另起炉灶"这类偏保险的保守做法反而被奖励：
不协作、有产物、不被机制审计；代价是架构熵转移到治理结构与连续性上，成了高目标
之下的隐性成本。
