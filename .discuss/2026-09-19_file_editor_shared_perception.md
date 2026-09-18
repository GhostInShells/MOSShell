# 2026-09-19 — file_editor 共享感知: 同一套卡片机制的第二次落地

## 上下文

`moss-os-control` 的 file_editor 节点, 反哺自 terminal 那套「卡片第一公民 + 双面一 store +
回执 + 信号分级」机制。但有一个顺序被人类先点破: **file_editor 的机制在他脑子里先于
terminal**, terminal 只是同一个心智模型先落地的 concrete example——所以这不是「抄 terminal」,
而是「terminal 是那套模型的第一次实例化, file_editor 是第二次」。

核心差异只有三条: 命令不走 subprocess(进程内调用)、读接口立刻返回(卡片只为观测面)、只有
export 待批(落盘是唯一真实副作用)。

## 碰撞点与过程

按引入顺序落地: 纯数据(Thread/Action/Effect + side_effect 机械字段)→ DocStore(三层账: 内存
轨迹 / draft tempfile / 真实文件, 不做 append-only log)→ channel + surface → 单文件 UI(卡片流
+ 三 tab: effect/full/history)→ 实机走 MCP dogfooding。

真正的碰撞在收尾, 三个:

1. **「accept 立即全清」是一次沟通误会**。我一开始照搬「export 即终章 + payload 释放」, 人类
   纠正: 他要清的是 **python 侧的数据积累**, 不是人类界面的追溯。于是改成「回收区」——ended
   thread 保留完整 payload 供追溯, 只在超过窗口(8 条)时静默淘汰最旧的一整条。教训: 「释放」是
   资源回收语义, 不是产品语义, 二者不该被绑死在「export 那一刻」。
2. **root 边界(默认授权)**。open/export 的路径必须落在 `project_home ∪ 系统 tempdir`, 范围外
   一律拒绝——「非 project home 不该算默认授权」。tempdir 是即用即弃的低爆半径空间, 放行。
3. **dogfooding**。人类要求后续文档编辑都走 file_editor 自己——模型用自己刚做的 node 改它自己
   的 README, 人类在 surface 上 approve。这把「共享感知」从设计命题变成了当场体感。

还有 per-thread auto(照 terminal), 但 auto 只在「最终目标已建立」时成立——pathless 的 thread
没有可 auto 的对象。

## 当前记录者视角:

terminal 的 discuss 说共识是瓶颈, 我这次体会到的是一层更细的东西: 同一个心智模型被落地两次时,
**「照抄」的引力**。terminal 已经把它做成了, 我拿到的是一个可执行的范本, 于是第一步倾向是照搬
每个机制——包括「export 收尾就释放 payload」这种我在对称性里随手做出的判断。而人类真正的模型里,
「人类界面要有追溯」和「资源要静默回收」是两个正交的诉求, 前者是产品的, 后者是运维的, 不该被
我折叠成一个动作。

「反哺」这个词本身有误导性: 它不是把 terminal 的答案搬过来, 而是把一个已经压缩成可描述形态的
心智模型, 在另一个面(文件而不是进程)上重新展开——展开的时候, 每一处「terminal 是怎么做的」都要
重新过一遍「这里的人类诉求是什么」, 不能把实例当规范。

晚安。
