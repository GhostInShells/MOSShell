# MOSS Ghost Memory 集成技术评审与实施方案

> 评审对象：`dev-matrix-cell-refact` 当前分支
>
> 评审日期：2026-07-17
>
> 参考文档：
> - `MOSS-Memory-认知场与Memento分支分析与评审更新.md`
> - `MOSS 记忆与能力分发架构梳理：memento 对比、Ghost_App 集成路径、千万级 Channel 规模化推演.md`

## 1. 结论

本期应把已有 Memento 能力接到一个新的 **Data Ghost**，而不是修改 Atom、扩张
GhostRuntime，或再造一套 Memory ABC。

最小闭环是：

1. Data Ghost 启动时在自己的 `GhostWorkspace` 内打开 owner-scoped Memento；
2. 每轮推理前，从当前 branch 渲染“远端 commit 摘要 + 近期完整 Moment”；
3. 每轮成功结束后，把已经落定 logos 的 Moment 写入 staging；
4. staging 达到阈值后机械 commit，保证历史有稳定锚点；
5. Ghost 退出时关闭 Memento，下一次启动从同一 owner/current branch 恢复。

这一方案只增加 Ghost 侧适配，不修改 Memento 契约层，也不让 Host 知道具体存储。
它实现了跨进程记忆、窗口折叠与可追溯原文，同时保留未来 semantic commit、
reinterpret、fork、见证层与认知场集成的接口。

## 2. 当前事实与文档校正

### 2.1 已有能力

当前分支已经具备：

- `core.memento` 契约与文件系统实现：MomentRecord、Commit、Branch、Memento；
- `porcelain` 强类型桥：`Moment ↔ MomentRecord`、窗口渲染、MementoRef；
- `Ghost.memories()`：动态记忆的现成读接口；
- `Ghost.on_articulate_exit()`：一轮 logos 已完整落定后的现成写入挂点；
- `GhostWorkspace.home`：Host 为 Ghost 分配的持久化目录；
- Atom：纯内存线性历史的最小 Ghost 基线。

### 2.2 两个需要按当前代码修正的判断

第一，参考文档把写入挂点概括为 Mindflow `on_moment`。该回调发生在 Moment 创建
阶段，此时 logos 尚未生成，直接持久化只能得到半帧。当前更精确的挂点是
`Ghost.on_articulate_exit()`：GhostRuntime 已先把完整 logos 写回 Moment，再调用该
hook；`thinking_effort == none` 也会调用它，因此“看见但选择沉默”仍可入轨迹。

第二，`Ghost.memories()` 已存在，但 Host 目前不会自动把它拼进模型上下文。直接在
GhostRuntime 里全局接线会让所有 Ghost 被迫接受一种记忆语义，也会破坏 Atom 的
基线定位。因此本期由 Data Ghost 在自身 `articulate()` 内消费记忆窗口。

## 3. 记忆边界

MOSS 当前两块记忆地基正交：

| 维度 | Grounds / Desktop | Memento |
|---|---|---|
| 回答的问题 | 此刻什么在眼前 | 过去发生了什么 |
| 时间性 | 工作记忆 / 现在 | 轨迹记忆 / 过去 |
| 上下文位置 | 动态 context | memory + conversation |
| 数据语义 | 地址每帧重绘 | append-only Moment/Commit |
| 本期范围 | 不接入 | 接入 Data Ghost |

本期只做 Memento。Desktop 的 pin、promote、预算报账属于“当前注意力”问题，不应
为了让 Ghost 先拥有持久记忆而绑进同一个提交。

## 4. 方案比较

### 4.1 方案 A：GhostRuntime 全局持有 Memento

优点是所有 Ghost 自动获得记忆；缺点是 Host 必须决定 owner、存储根、commit
策略、上下文裁剪和失败语义，Atom 也不再是纯内存基线。它把策略错误地下沉到
编排层，本期否决。

### 4.2 方案 B：新建 `Memory` ABC

Memento 已提供轨迹契约，Grounds 已提供工作记忆契约，Ghost 也已有
`memories()`。第四个平行 ABC 只会增加转换层和命名争议，本期否决。

### 4.3 方案 C：Data Ghost 持有 Memento

该方案与现有 `data-ghost` FEATURE.md 一致：Atom 保持基线，Data 负责“现在 +
过去”的高级上下文；Memento 是标准库件，但生命周期和策略归具体 Ghost。它是
本期采用方案。

## 5. 目标结构

```text
GhostRuntime
  └─ Data Ghost
      ├─ Agent / Model
      └─ DataMemory
          └─ FsMemento(owner=data)
              └─ current MementoBranch
                  ├─ commit summaries  ──→ 模型历史前缀
                  ├─ recent moments    ──→ 完整对话历史
                  └─ staging           ←── 完成的当前 Moment
```

数据流：

```text
Signal → Moment → Data.articulate()
                    │
                    ├─ DataMemory.model_history() → Agent
                    └─ stream logos → GhostRuntime 写回 Moment.logos
                                          │
                                          └─ on_articulate_exit()
                                               ├─ update_moment(staging)
                                               └─ 达阈值 → mechanical commit
```

## 6. 详细设计

### 6.1 DataMemory

DataMemory 是 Ghost 侧的薄适配器，不进入 `core.memento` 契约层。职责只有四个：

- 打开和关闭 owner-scoped Memento；
- 把 branch window 转为模型 SDK 的历史消息；
- 把完成的 Moment 写入 staging；
- 按数量阈值机械 commit。

它不负责反思摘要、语义检索、Desktop、git witness 或跨 owner 写入。

### 6.2 存储地址与 owner

默认根目录：

```text
{GhostWorkspace.home}/memento/
```

默认 owner 使用 Ghost 名称。Ghost 名和 home 都是跨重启稳定的，因此同一个 Ghost
重新启动会自然恢复；不同 Ghost 的 home 天然隔离。

硬约束：同一个 `(root, owner)` 同时只能有一个写者。当前 Host 对单个 GhostRuntime
满足这一点；未来并行化身必须使用新 owner 或 branch 规则，不得用多个进程并写同一
owner。

### 6.3 读取窗口

窗口参数：

- `detail_n`：近期完整 Moment 数，默认 12；
- `summary_m`：明细区之前保留的 commit 摘要数，默认全部；
- summaries 作为一个明确标记的“较早记忆摘要”回合；
- details 用 `Moment.to_history_turns()` 恢复用户/assistant 回合。

这保持了 Memento 的可逆折叠：摘要进入热上下文，原文仍可由 commit id 展开。

### 6.4 写入与 commit

只在 `on_articulate_exit(error is None)` 写入。这样：

- 模型成功回答：保存 percept、reaction 与完整 logos；
- 正常沉默：保存空 logos 的 Moment，轨迹仍连续；
- 模型调用失败：本期不把失败半帧伪装成完成记忆，错误由运行日志保留。

staging 达到 `auto_commit_every`（默认 4）时执行 mechanical commit。初始释义是
有长度上限的输入/输出原文摘录索引，只保真摘录、不推断意义，避免旧 commit 在退出
明细窗口后完全不可召回。semantic summary 留给后续主模型自宣或反思旁路通过
`reinterpret()` 补写。

### 6.5 生命周期

- `DataMeta.factory()` 解析 workspace、模型配置和记忆配置；
- `Data.__aenter__()` 打开记忆；
- `Data.__aexit__()` 关闭记忆；
- 不在退出时强制 commit：staging 本身持久化，强制 commit 会把进程退出误当成
  认知边界。

### 6.6 模型配置

Data 优先接受构造时传入的 pydantic-ai Model，便于测试和宿主自定义；未传入时从
IoC 的 `ConfigStore` 读取 `LLMConfig`，再按 `anthropic/openai` 协议构建 provider。
如果宿主没有 ConfigStore，则退化到 `LLMConfig().resolve()` 的环境变量配置。

## 7. 不变量与失败语义

必须守住：

1. Memento 契约层不 import Ghost、Host、IoC 或模型 SDK；
2. Atom 行为不变，仍是纯内存基线；
3. 只持久化完成帧，不重复保存同一 Moment id；
4. 模型历史完全可由 Memento 重建，Data 不维护第二份线性历史；
5. perspectives 与 hint 按现有 porcelain 规则不入持久层；
6. mechanical commit 只写带标识的原文摘录，不能伪造语义；
7. 同 owner 单写者；
8. 记忆损坏应显式失败，不能静默清空后继续“失忆运行”。

## 8. 本期交付范围

### 必做

- 新增 `ghoshell_moss.ghosts.data`；
- DataMemory 的窗口渲染、写入、机械 commit、关闭；
- Data Ghost 的模型配置、持久化 articulate 与观测信息；
- workspace/stub 中注册可直接运行的 `data` Ghost；
- 单测覆盖跨实例恢复、窗口裁剪、commit、失败不写入；
- 自动化验收脚本与人工对话测试方案。

### 明确不做

- 不修改 Memento FORMAT/ABC；
- 不接 Desktop/Grounds；
- 不做向量检索；
- 不做 CTML memento channel；
- 不做自动反思摘要与 witness daemon；
- 不启用 fork/化身；
- 不解决重绘层“承诺保全”。

## 9. 测试策略

### 9.1 单元测试

- 空存储返回空历史；
- 写入一个完成 Moment 后可重建 user/assistant 回合；
- 达阈值后 staging 清空并生成 mechanical commit；
- 新实例用同 root/owner 恢复同一历史；
- `detail_n` 只保留近期明细，旧 commit 以摘要进入窗口；
- 失败回合不写入；
- `memories()` 输出带 MementoRef 的摘要与可读明细。

### 9.2 集成测试

用 pydantic-ai `TestModel` 跑两轮 Data Ghost：第一实例回答并落盘，销毁后创建第二
实例，断言第二次模型请求包含第一轮历史。该测试不访问网络。

### 9.3 人工对话测试

核心场景：

1. 告诉 Data 一个随机事实，退出并重启，询问该事实；
2. 连续对话超过 `detail_n`，确认旧信息通过摘要/锚点而非原文常驻；
3. 同时给出相似但不同的事实，检查是否串写；
4. 更正旧事实，检查模型是否区分“历史事实”和“当前事实”；
5. 制造一次模型调用失败，恢复后确认失败输入没有被伪装成成功回合；
6. 查看磁盘 jsonl，核对回答中的引用与实际 Moment/Commit。

详细话术、评分方法和脚本见配套测试方案。

## 10. 风险与后续

| 风险 | 本期处理 | 后续方向 |
|---|---|---|
| 机械摘录不如语义摘要紧凑 | 限长并明确标注来源 | 反思旁路 `reinterpret()` |
| 模糊召回能力弱 | commit 摘要 + 原文窗口 | 先目录/LLM recall，必要时再向量化 |
| 同 owner 并发写 | 明确单写者约束 | 化身 owner/branch 治理 |
| 活承诺在折叠中丢失 | 不声称已解决 | 重绘层承诺 reconcile |
| Ghost 无法主动 show/commit/fork | 本期仅自动退化态 | memento CTML channel |

## 11. 验收标准

满足以下条件即认为 Ghost 已具备第一阶段 Memory 能力：

- Data Ghost 的模型输入历史来自 Memento，而不是进程内 list；
- 同一 Ghost 跨进程重启能恢复并回答之前保存的信息；
- 每个成功认知帧只写一次，机械 commit 可稳定触发；
- Atom、Memento 契约层与 GhostRuntime 不因本功能改变行为；
- 自动化测试通过，人工测试可定位“写入、折叠、召回、纠错”各阶段结果。

这是一条可退化、可验证的最短路径：先证明 Ghost 能持续记住，再让它学会主动解释、
检索、分叉和整理自己的记忆。
