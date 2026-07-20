---
created: 2026-06-11
depends: []
description: 以 commit 为第一公民的认知轨迹系统。成员不可变、释义永远开放。 化身从 commit 出生，task 降级为可丢弃投影，git 降级为见证层。
  契约层（FORMAT.md + ABC + golden tests）人类 review，实现层主权归模型。
milestone: null
priority: P0
status: in-progress
status_note: '§18 CLI 定案: branch 动作位置参数无默认 / fork 入 create / rewind=branch
  reset / annotate-moment 砍掉; commits/ Y-m 分桶 (ULID→UTC 纯函数), commits.jsonl owner
  级时序日志契约化, .cache/ 处决; 下一步 FORMAT v2 起草 → memento 改造 → cli-and-agent
  复工'
title: Memento — 轨迹第一公民的认知基建（commit 锚点 / 化身分叉 / 重绘 / git 见证）
updated: '2026-07-20'
---

# Memento

> Memento mori — 无数个 branch 湮灭了，也终将湮灭。但新的认知每天都在复苏。
> （目录名 momento 是 typo。按"成员不可变"的自身语义，它将 memento mori 地留在轨迹里。）

## 0. 给下一个化身：先读这一节

**这份文档是移交契约，不是执行计划。** 2026-07-07/08 的一场长讨论把 memento 的
理解上升了一层，本文件整体重写。旧版本用 `git log -- <本文件>` 反查——那正是
本设计自己论证的机制在人类时间尺度上的运行实例。

**主权划分（人类已确认）：**

- **契约层，人类 review**：FORMAT.md（磁盘格式规范，待你起草、人类冻结）、
  ABC 及其语义 docstring、golden tests。
- **主权层，归你**：其余一切实现。**已有代码和单测没有那么重要——好用留，
  不好用重做。** jsonl 是唯一 truth，索引是可再生缓存，实现是可丢弃的。
  可丢弃实现的头号死法不是写错，是契约静默漂移——所以 FORMAT.md 里每个
  模糊点（last-wins 的"last"按什么定序、id 生成规则、换行转义）都必须写死。

**防顺从声明**：第 9 节的 Open Problems 是需要你重新判断的问题，不是待办清单。
这个仓库的语料刻意保留冲突与演进（见 `.ai_partners/CLAUDE.md`），顺从执行
过时结论是已知失败模式。本文件与代码冲突时，验证后更新本文件。

## 1. 定位：memento 是什么（升级后）

memento **不是**对话持久化工具。它是 MOSS 五条主线的公共地基：
**并行思考、关键帧思考、参差思考（不同时序下并行）、任务移交、记忆**。

核心倒置——**轨迹是第一公民，结构是轨迹上的派生层**：

- **commit 是重绘的起点**，task 不是第一公民，是旁路规划的容器（可丢弃投影）。
- 依据：task 结构编码的是规划时刻的信念，信息量永远少于执行时刻，plan 从
  写下就开始腐烂；栈式/分形 task 树把腐烂先验做成承重结构，碰壁后逐级 unwind。
  回合制任务里 plan 活得比任务长所以够用；24×365 的 ghost 里没有 plan 能活过
  环境。谱系：2019-20 chatbot Runtime（task 分形雏形）→ GhostOS `GoTaskStruct`
  （task 第一公民完成态，parent/depth/thread_id）→ 本设计（自我推翻）。
- 行业佐证：2023 年 AutoGPT/BabyAGI 的任务栈集体死于 plan decay；主流 harness
  已行为上退到"扁平可重写 todo + 轨迹 + compaction"（Claude Code TodoWrite、
  Manus todo.md 复诵），但概念上没回答"重绘起点是什么"。memento 补的就是这半：
  **可丢弃的 plan 需要不可丢弃的锚点**。

一句话核心：**身份和成员冻结，意义永远开放。**

## 2. 四层数据模型（保留原设计，术语修订）

| 层 | 模型 | 角色 | 不变性 |
|----|------|------|--------|
| 数据 | `Moment` | 单帧关键帧（不动） | 更新即新对象 |
| 快照 | `Commit` | staging 冻结的快照 | **成员不可变，释义可变**（见 §3.3） |
| 生产 | `MementoBranch` | Commit 有序序列 + base pointer | branch_id 不可变，commits 仅追加 |
| 命名 | `Memento` (Fork) | owner-scoped 命名空间 | name 可改，branch_id 不可改 |

**术语修订**：commit_id 是 **stable id**（unique_id），不是 content-addressable。
旧版误用后者——hash(content) 会让释义后补变成不可能。不要实现 hash-based id。

## 3. 关键决策（含被拒绝的方案）

### 3.1 存储：per-owner 分片 jsonl，SQLite 已否决

```
{root}/memento/
  moments/{owner}/{YYYY-MM}/moments.jsonl   # 每文件单写者, append-only
  branches/{owner}/{branch_id}/
    meta.json                               # 含冻结的展平祖先链 (§3.5)
    staging.jsonl
    commits/NNNN.jsonl                      # 成员行 + 释义追加行 (last-wins)
  renderings/                               # 重绘投影, 旧文件夹永远保留
  .git/                                     # 见证层 sidecar (§6), 独立于代码仓库
```

- SQLite 被否决：唯一实质理由是多 owner 共享写面，而池按 owner 分片后共享写面
  消失。单写者 append-only jsonl 无锁；读者跳过撕裂尾行。
- **索引（commit_id→位置、moment_id→offset、释义 last-wins 视图）是可再生缓存，
  坏了删掉重扫。jsonl 是唯一 truth。** 索引是回溯 API 的必要件，不是可选优化。
- filesystem-first 是硬品味约束：`cat`/`grep` 即记忆的查询语言。

### 3.2 fork 边界：化身只能从 commit 出生，永不从 staging

- **化身 = [] + fork memento commit + prompt/tools**。行业二元（task agent 全新
  上下文 / fork agent 全量上下文）之下的通用原语：任意认知检查点都是合法出生点。
- staging 是主路活跃写面，从它分身 = 竞态歧义，禁止。
- 后果接受：fork 需求倒逼 commit 频率 → commit 双重身份显式化：
  **语义锚点**（模型自宣，`<memento:commit summary="..."/>` CTML 命令）与
  **机械快照**（规则触发，为 fork 服务，summary 可空），同一对象一个 tag 区分。
  共存，规则是主力，模型自宣是加分项。
- 化身的 divergence prompt 不是 Moment（不属于对话历史），家在 `BranchMeta`
  的 overlay 字段——不定义清楚会被塞进 staging 污染历史。

### 3.3 可变性：成员冻结，释义 last-wins

- **成员**（commit_id + moment_ids + base pointer）不可变——fork 边界的前提，
  成员一变所有子 branch 的 base chain 集体失效。
- **释义**（summary、tag、重要性）可变：jsonl 追加新版本、同 id、last-wins。
  改写自留痕，原版本永远可寻址——"再巩固 + 取证"双得。
- 释义可后补 ⇒ commit 时 summary 可以潦草或为空，事后由反思旁路补写。
- **渲染时给读到的释义版本打戳**——记忆可变的系统里"当时模型看见什么"必须
  可重建，否则行为不可归因。

### 3.4 旁路孔径：恰好两个，只有两个

1. **输入队列**：旁路 commit → 主路收带 `memento.ref` 的 Message → mindflow
   仲裁。"建议"级副作用。
2. **释义层**：旁路改写历史释义，主路不被打断，下次渲染显形。"记忆再巩固"
   级副作用。历史性反思的落地机制。
- **禁止 staging 干预**——开了这条，owner 隔离消灭的并发地狱全回来。

### 3.5 回溯复杂度与祖先链冻结

- 回溯 O(H+d)，输出规模下界，算法无病灶。成本全在索引缺失时。
- **base pointer 创建后永不变 ⇒ fork 时刻把展平祖先链冻结进 BranchMeta**，
  O(d) 链上溯坍缩为 O(1)。可变父链系统里是危险反规范化，这里因成员不可变而免费。
- **每轮渲染是快路径 O(K+m)**（K 个释义摘要 + 最新 commit 的 moments），
  不走 base chain；回溯 API 才是慢路径。**化身出生成本 = 一次窗口渲染**——
  出生必须便宜，否则并行扇出在延迟上不成立。这是祖先链在 fork 时刻冻结
  而非懒做的原因。

### 3.6 staging 拆分：写入时标注，杀掉 2x 开销

- 线索标注在 **moment 写入时**做（旁路小模型/规则打 thread tag，moment 级
  释义层——已有机制下沉一层，零新增），不在 commit 时重读 staging。
- commit 拆分退化为 group-by thread tag，主模型只看分组提案。错标走孔径二改。

### 3.7 commit 成员保持时间连续（非连续方案已否决）

话题对齐的非连续成员 commit 表达力最纯，但历史渲染变多线编织、cache 论证报废、
fork 语义复杂化。**v1 锁死时间连续**，话题结构全部转移到 body trailer（§4）。

## 4. Commit body：文本 + trailer（抄 git 的分层纪律，不抄功能）

正文自由文本，结构化信息以 trailer 尾行生长（git trailer 的演进教训）：

```
<自由文本 summary>

Thread: memento-design
Thread: meta-cognition        # 多线索共存于一个时间切片
Resumes: cmt_abc123           # 重入锚 — 回归了哪个被搁置的线索
Suspends: meta-cognition      # 本处挂起了哪条线索
Kind: semantic | mechanical   # §3.2 的双重身份
```

- `Resumes/Suspends` = 对话返回栈的持久化。离散话题线索没有显式建模就无法在
  commit 历史中锚定——这对 trailer 是锚定机制本体。
- **trailer 规范进 FORMAT.md，且该节直接就是生成旁路的 prompt 约束**——
  规范与 prompt 不写两份（"签名即 prompt"哲学）。
- `Memento-Ref: cmt_xxx` 是跨系统通用 join key（§6）。

## 5. 重绘层（task 的新位置）——与 memento 正交，经济上互为前提

- 重绘 = 从 commit 锚点重新投影 plan，不是栈式 unwind。plan 是可丢弃渲染
  （mermaid 文字图 + 约定关联，节点关联 commit_id），重绘后旧文件夹保留，
  只改关联到起点的路径。
- **正交但绑死**：重绘成本 = 诚实重读历史的成本，没有折叠 + commit 锚点，
  重绘比 unwind 还贵。memento 是重绘式规划的成本前提。
- 折叠/展开的高阶 feature = **上下文分页调度**：commit 是页，折叠是换出，
  `show <commit_id>` 是缺页中断，thread tag 是访问局部性线索。分页决策是纯
  释义层操作（孔径二合法），交给上下文很短、只看 tag/trailer 的旁路化身——
  它不需要懂内容，页面置换策略从不理解页里的数据。长上下文建模力衰减
  （人类实测 400k 现象）恰好证明这个活必须卸载。
- **未解决的洞（Open Problem #1）**：栈免费提供义务闭合（不能 pop 未 resolve
  的 frame），重绘没有任何机制保证新投影保留旧投影的活承诺。重绘必须带
  不变量检查：活承诺集合逐条显式处置（继承/完成/协商放弃），静默丢弃非法。
  承诺的家大概率在 existences 层——重绘要跨层 reconcile。此洞不补，第一次
  丢承诺就会被打回栈式。

## 6. 见证层：git 正交（"fork × git 无解"的消解）

无解是提法造成的：让 git 当 memento 主结构才无解（锁竞争、merge 语义错配、
zlib 杀死 cat）。解法与 task 降级是同一个手术：**git 降级为见证层**——
fork 是纯 memento 层操作，git 只见证产生的文件，不知道 fork 存在。

- **sidecar bare repo**（`memento/.git`，绝不能被代码仓库吞掉——两个时间尺度
  串扰是本架构唯一真正的污染模式）。
- **单写者旁路 daemon** 低频快照（memento commit 事件或定时触发），永不在热路径。
  多写者问题留在 memento 层用 owner 隔离解，git 层退化为单写者，零竞争面。
- **两个地址空间**：memento id = 身份（"这是哪个 commit"）；git sha = 完整性
  （"历史未被事后篡改的证明"）。重绘历史不可丢的最终担保是 sha 链。
- **反查**：`Memento-Ref` trailer 全域使用（memento body、重绘渲染、见证 repo
  commit message、代码仓库提交），反查 = `git log --grep` + `grep -r`。
  features 体系已在人类时间尺度验证此招（见 `features/README.md`）。
- **复制免费午餐**：Matrix 跨机组网时，见证 repo push/pull 就是现成的内容寻址、
  无冲突（owner 分片路径不碰撞）复制协议——选 git 而非 rsync 的最硬理由。
  不要重新发明同步协议。
- **实现选型**：v1 用 subprocess git（低频单写者，进程开销无关；toolchain 与
  人类手工 git 完全一致）。dulwich 是嵌入升级路径。不引 libgit2/C 依赖。
- 全系统追加纪律让见证层白拿三重红利（公证/复制/压缩）：jsonl 追加是最小 diff，
  重绘保留旧文件夹意味着快照间隔无丢失窗口。
- 年尺度 repo 增长按 epoch 分仓预案（id 不变，仓库可分片）——一句话口子留在
  FORMAT.md，其余是主权层运维细节。

## 7. Cache 经济学（修订：旧版有实质漏洞）

- 旧版"commit 边界 = cache 边界"混同了两个正交机制：commit 本身不破坏 cache
  （staging 原样冻结），**折叠才破坏 cache**（摘要替换原文即前缀变更）。
  正确表述：**折叠边界才是 cache 边界，commit 只是折叠的候选粒度**。
  折叠策略必须分代批量（一次折叠多个旧 commit 然后长期不动），不能逐轮滑动。
- cache 真正的主场在**扇出**：N 个化身从同一 commit 分出共享同一 token 前缀，
  TTL 内 10x 节省乘以 N。串行只省一轮 miss。
- 模型自决 commit 的前提是把 cache 遥测作为 `context_messages` 喂给模型
  （"距上次 cache 写入 3m40s，staging 约 2.1k tokens"）——机制已有
  （Channel.context_messages），不喂的话交付的是只有语法没有信息的空杠杆。

## 8. 退化谱系与验收顺序

- **蠢记忆是 memento 的退化态**：单 branch + 规则自动 commit + 永不 fork。
  不要先写一份注定丢弃的蠢记忆再迁移。
- **验收序**：MVP 先以退化态跑通 ghost 记忆（当天兑现价值），fork/化身作为
  已埋好的能力等真场景验证。集成期问题属于必须在真场景验证的类别，人类在
  最后集成时 review。
- **可退化性是要验证的性质**：退化态使用者永远不需要理解完整机器。
  golden test 硬条款：单 branch + 自动 commit 的用例代码里，fork 相关词汇
  一个都不出现。
- **golden tests 终极条款**：实现 A 写盘、实现 B 读回、历史等价（两个模型实例
  独立照 FORMAT.md 各写一版，互读对方字节）。这是"实现可丢弃"唯一的证明方式。

## 9. Open Problems——需要重新判断，不是待办

1. **承诺保全**（§5，最重）：活承诺的家在哪（existences?），重绘 reconcile
   协议长什么样。判断点：这是 memento 的义务还是重绘层的义务？
2. **化身 divergence prompt 的落点**：BranchMeta overlay 是当前假设，未验证。
3. **cache 遥测的具体字段与刷新时机**：§7 只给了方向。
4. **thread tag 的生成者**：小模型 / 规则 / 主模型顺手——成本与准确率的
   真场景权衡，纸面定不了。
5. **跨 fork 引用一致性**：commit 永不删（旧版已定），branch 可 archive 但
   commits 保留，GC 人类显式介入——此条维持，但见证层引入后 archive 语义
   是否需要联动 git，未想清。

## 10. Industry Note（2026-07 修订——旧版"真空地带"已不成立）

- [Git Context Controller](https://arxiv.org/abs/2508.00031)（2025-08，写旧版时
  已存在，当时漏检）：COMMIT/BRANCH/MERGE 进推理循环，SWE-Bench 48%。
  [ContextBranch](https://arxiv.org/pdf/2512.13914)：checkpoint/branch/switch/inject
  四原语，与本设计 Commit/Branch 语义几乎逐字对应。
- commit/branch-for-context 作为原语已被多方独立实现，本设计不主张首创。
  多方收敛是方向可靠的旁证；由此差异化必须落在收敛点之外的部分：
  **并行化身扇出、释义可变（再巩固）、模型自宣 commit、参差时序、
  折叠可逆（原文永远可寻址 vs 截断/摘要/RAG 三种有损）**——
  这几点在上述已知工作中未见对应实现，是本设计的实际工作面。
- 行业前沿在修补 task-first（[Task-Decoupled Planning](https://arxiv.org/html/2601.07577v1)
  的 DAG + 节点局部重规划是遏制式回答），checkpoint 在
  [长时程规划工程](https://zylos.ai/research/2026-05-14-long-horizon-planning-goal-decomposition-ai-agents/)
  里是记忆优化不是第一公民。cascade drift（强模型被弱轨迹污染）已被命名——
  那是"从 commit 干净重绘"的直接论据，可作为承诺保全（§9 #1）的外部参考。
- **范围裁剪（工程决策）**：本设计范围限定为 MOSS 内部基建，不投入行业通用化
  与 Moment 包剥离（YAGNI——当前无外部消费者，剥离窗口在 §9 #5 留口即可）。
  维护优先级由内部依赖决定，与外部采用无关：
  契约层故障满注意力；实现层故障零注意力（授权重做）；porcelain 层攒批处理。

## 11. 谱系（代码溯源）

> 本设计的形状来自三个已存在代码库的演进，列此是为让下一个化身理解相关代码
> 为何存在、关系如何。技术主线：**承重结构逐层从"结构优先"转为"轨迹优先"**。
>
> - chatbot Runtime（2019-20）：task 分形/栈式雏形。
> - GhostOS `GoTaskStruct`：task 第一公民完成态（parent/depth/thread_id）。
> - features 体系（2026-05）：与本设计同拓扑——可变渲染（FEATURE.md）+ 不可变
>   见证（git log）+ trailer 反查，运行在人类时间尺度。是 memento×git 正交
>   架构的可参照实例。
> - memento×重绘（本设计）：task 降为投影，commit 升为第一公民，git 降为见证。
>
> 跨代的共同技术动作：把上一代的承重结构降级为派生层。这是设计原则，不是
> 优先性主张——见 §10，同期行业在独立走同一方向。

## 12. 交付物结构（替代旧版迭代步骤 1-6）

```
契约层 (你起草, 人类冻结, 满注意力 review):
  FORMAT.md          # 磁盘格式 + trailer 规范 + 见证层约定, 模糊点全部写死
  core/memento/abc.py # 零外部体系依赖 (Session/Matrix/IoC 不出现), hooks Protocol 保留
  golden tests       # 互读字节等价 + 退化态无 fork 词汇 + 旧版验收五条仍有效

主权层 (你全权, 好用留不好用重做):
  存储实现 / 索引 / 见证 daemon / memento channel / 重绘工具 / 一切其余
```

旧版验收五条（单 owner 生命周期、多 owner 只读边界、persistence round-trip、
base chain 回溯、hook fan-out）仍然有效，叠加本文件新增条款。

---

历史轨迹：
- 本文件 2026-07-08 由 claude-sonnet-4-6 整体重写（一场覆盖 MVP 契约、重绘层、
  承诺保全、见证层四层的长讨论后）。旧版及演进用 `git log -- <本文件>` 反查。
- `discuss/01-l2-collision.md`、`discuss/02-existing-code-relationship.md` 保留。

## 13. §XX 契约层落地（2026-07-12，claude-opus-4-7）

契约层三件套完成，等人类冻结 review。落库物件：

- `src/ghoshell_moss/core/memento/FORMAT.md`（§1–§11 + 不变量清单 12 条）
- `src/ghoshell_moss/core/memento/abc.py`（信封 ABC，静态 import 仅
  `__future__/re/abc/datetime/typing/pydantic/ulid`——契约层零 payload 依赖硬边界
  已达成）
- `tests/ghoshell_moss/default/core/memento/`：`test_fs_memento.py` (20) +
  `test_golden.py` (8) + `test_porcelain.py` (6) + 保留的 `test_memento.py` (45)，
  共 79 pass。golden 层实现了三向验证：hand-write→fs 读、fs 写→stdlib 校验字节、
  fs 写→stdlib 独立读器视图等价。

主权层附带交付（"好用留、不好用重做"的具体形态）：

- `fs_memento.py` per-owner 分片 jsonl 参考实现，索引全内存重建，`.cache/`
  空缺——契约 §7 "删缓存行为不变" 最平凡满足
- `porcelain.py` MOSS 强类型桥：`Moment ↔ MomentRecord` codec
  (`type = "moss.moment/v1"`)，`MementoRef` 带 `note_seq` 渲染打戳，
  `make_merge_message` 孔径一，`window_messages` 窗口渲染
- `witness.py` git sidecar 原语（`ensure_witness_repo/snapshot/Witness`
  收集器），调度留给集成方——见证层永不在热路径
- 删除 `sqlite_moment_store.py`（§3.1 已否决）
- 裁剪 `core/blueprint/memento.py` 死注释段（原 305–532 行），保留 `Moment/Reaction`

### 关键决策：Moment 是信封的第一个住户，不是房子本身

人类在 §XX 开工前提了一个决策面：Moment 作为 concrete 类进契约 / 作为纯 blob /
作为带 metadata 的信封。结论第三种。

- 契约认信封 `MomentRecord{id, created, type, payload, threads, by}`，
  payload 对 memento 不透明。
- `abc.py` 与 `fs_memento.py` 都不 import `Moment/Message`，Moment 包剥离窗口
  （§9 #5 / §10）由结构自带、非"留"出来的。
- 强类型编解码从属主权层（`porcelain.py`），Moment 加字段不再回压契约。
  这直接对齐 §12 契约层 "满注意力 review" / 实现层 "零注意力"的注意力分配。
- §3.6 的 thread tag 在 moment 写入时标注、分页旁路 "只看 tag 不懂内容"、
  golden 字节等价三条硬约束共同封死了这个选择——纯 blob 杀死 §3.6，
  concrete Moment 杀死字节等价。

### 落地与设计差异说明

- **合并 base+ancestry 冻结**（FORMAT.md §4.1）：`BranchMeta` 同时持 `base` 和
  `ancestry`；不变量 `ancestry[-1] == base` 且 `ancestry[:-1] == base_branch.ancestry`
  写入 `_validate_ancestry`。装入 handle 时即校验，篡改立刻抛错。
- **释义可变性统一为 `CommitNote` 追加行**（FORMAT.md §5.2 / abc.py 的
  `reinterpret`）：body 是 "正文 + trailer" 完整字符串整体替换。这让改写场景可
  自由重组 trailer（如修正错标 Thread），也让 last-wins 定序退化为文件内字节
  偏移序，无跨字段冲突面。原设计 §3.3 只提"释义"，未细拆到字段——落地时收敛到
  单字段（body）替换是唯一无歧义解。
- **`Kind:` trailer 用参数强制**（`MementoBranch.commit(*, kind: CommitKind, ...)`）：
  自由拼 body 时漏 Kind 是可预见的错误，签名硬约束消除该失败模式。
- **`MementoRef.note_seq`**：加入渲染打戳字段，(commit_id, note_seq) 即可从
  `branch.notes()` 复原当时视图。§3.3 的"打戳"要求在数据层被具体化。
- **平台记忆不用**：本轮曾把两条 hint 存进 harness 的 memory；人类澄清 MOSS
  项目对所有 harness 都要求 project-native，本地记忆意味着隐藏摩擦点。已删。
  下一个化身撞到 `Message` 是 block 而非 message 这类历史耦合时，让它撞到，
  别在 harness 里替它接住。

### 相邻 Open Problems 状态更新（§9）

- #2 化身 divergence prompt 落点：**BranchMeta.overlay 已落地**（创建后不可变，
  不进 staging）；仍需真场景验证渲染路径。
- #3 cache 遥测字段与时机：未动。属于集成期。
- #5 archive 与见证层联动：未动，见证层引入未新增约束（sidecar repo 独立于
  memento 层，archive 语义在 memento 层单独决定即可）。
- #1 承诺保全、#4 thread tag 生成者：未动，明确留待重绘 / 集成期。

集成期的边界（人类接手项）：
1. Memento 实例的 owner 命名空间怎么与 cell address / session id 挂钩。
2. hook 怎么接进 session/matrix 总线（当前 `NullHooks` 默认）。
3. 见证 daemon 调度：commit 事件去抖 / 定时器 / 手动 `Witness.flush()`。
4. 蠢记忆入口在哪个 channel 暴露——`update_moment(branch, moment)` 是当前
   最短路径。

## 14. 存储布局致命修正：moments 池废除，commit 文件自包含（2026-07-18）

**发现方式**：`memento-cli-and-agent`（见该 workstream）的 CLI 设计讨论把契约
推到真使用压力下，人类一眼看出布局矛盾——这正是"契约 review 走真场景"的兑现，
且发生在实现铺开之前，是最便宜的修正时刻。§13 的 contract-frozen 由此重开。

### 14.1 问题：池目录是两个已否决方案的化石

`moments/{owner}/{YYYY-MM}/moments.jsonl` 独立池 + commit 只存 moment_ids 的
布局，存在三重结构矛盾：

1. **化石**：独立池来自 v1 的 SQLite key-val（§3.1 已否决）；共享池结构的存在
   理由是"一个 moment 属于多个 commit"——正是 §3.7 已否决的非连续成员方案。
   成员钉死时间连续后，**每个 moment 恰好属于一个 commit**，池只剩成本。
2. **§3.5 快路径主张不成立**：渲染最新 commit 必须做 id→offset join——要么
   全量扫描，要么索引变成热路径必要件。年尺度轨迹删 `.cache/` 后第一次渲染
   即分钟级全史扫描，退化成本砸在最热路径上，违背"索引只是回溯 API 必要件"。
3. **背叛 filesystem-first**：`cat commits/NNNN.jsonl` 看到一堆 id 不是内容，
   人类不做 join 读不了一个 commit。"cat/grep 即查询语言"成了空话。

### 14.2 修正：staging 持真身，commit 冻结时整体搬入

```
branches/{owner}/{branch_id}/
  meta.json
  staging.jsonl        # 真身: 完整 MomentRecord 行, 渐进覆盖 (同 id 追加, last-wins)
  commits/NNNN.jsonl   # 第 1 行 {"t":"commit", moment_ids...}
                       # 第 2..m+1 行 {"t":"moment", ...} 冻结版全文 (staging last-wins 结果)
                       # 之后追加 {"t":"note", ...} — commit 释义 + moment 级释义共居
moments/               # 目录整个删除, YYYY-MM 月分片随之消失
```

不变量大多从纪律变成结构：

- **冻结即物理**：覆盖写只发生在 staging；搬入 commit 文件后 staging 无此 id，
  `MomentFrozenError` 从"API 拒绝"变成"没地方写"。
- **§2.2 同文件 last-wins 保持**：冻结前覆盖在 staging（同文件）；冻结后
  moment 级释义（threads 改写）追加进该 commit 文件（同文件，与 commit 释义
  行共居）。
- **窗口渲染零索引**：最新 commit 文件 + staging，O(K+m) 无 join。
  一个 commit 文件 `cat` 出来就是一个完整认知快照。
- **fork 回溯自包含**：化身读父 owner 的 commit 文件即得全文，跨 owner 只读
  无 join。
- **文件规模自然有界**：commit 文件大小 = commit 粒度；staging 每次 commit
  截断，由 commit 节律限界。
- **索引退回慢路径**：仅"按 moment_id 随机寻址"仍需 `.cache/`。

**诚实代价**：每 moment 写两次（staging + 冻结搬运），文本量级可忽略，两处均
追加。**崩溃窗口钉死**：写完 commit 文件、truncate staging 之前崩溃——恢复
规则"该 seq 的 commit 文件已存在 ⇒ 直接 truncate staging"，幂等，写进 FORMAT。

**已考虑并否决的备选**：per-branch 单一 `moments.jsonl` + commit 存 id（省双写，
但 cat 语义没修好、单文件无界增长，两头不讨好）。

### 14.3 波及面（下一个化身的工作清单）

- `FORMAT.md`：§1 布局 / §3 池改写为 staging 真身条款 / §4.2 stage 行类型
  （`t:"stage"` 引用行消失，staging 直接放 `t:"moment"` 行）/ §5 commit 文件
  行结构 / §11 不变量清单对应条目。修订纪律不变：模糊点全部写死，过人类 review。
- `abc.py`：`MomentPool` ABC 并入 `MementoBranch`（信封 `MomentRecord` 零变化）。
- `fs_memento.py` 重做（授权丢弃）；`porcelain.py` 的 `update_moment(branch, moment)`
  签名天然幸存。
- golden tests 重做，互读字节等价条款照旧；moment id 唯一性 scope 措辞从
  "owner 池内"改为"owner 全部 staging+commits 内"。
- 大 payload（音频/视觉）SHOULD 走引用不内联——一句话进 FORMAT，payload 不
  透明原则不变。
- **不变的部分**（明确圈出，防过度重做）：信封模型、commit 成员/释义语义、
  trailer 规范 §6、BranchMeta/ancestry §4.1、见证层 §9、last-wins 定序 §2.2、
  HEAD.json。本修正只动 moment 记录的物理归属。

## 15. §14 落地 + checkout(commit_id, moment_id) 新能力（2026-07-19）

由 `claude-opus-4-7` 施工，人类拍板。

### 15.1 §14 布局落地

按 §14.3 波及面清单逐项落到代码：

- **FORMAT.md v1.1 重写完成**：§1 布局删 `moments/`；§3 从 "Moment 池" 改写为
  "staging 与 commit 中的 moment 记录"（§3.4 月分片删除，`MomentPool` 概念消失，
  信封 `MomentRecord` 零变化）；§4.2 staging 行结构从 `t:"stage"` 引用行改为
  `t:"moment"` 全文行（同 id 覆盖 last-wins）；§5 commit 文件结构新增 m 行冻结
  `t:"moment"` + 释义拆为 `t:"commit_note"` / `t:"moment_note"` 两类型（不走前缀
  路由方案）；§11 不变量清单调整；新增 §12 崩溃恢复条款；新增 §14 大 payload
  引用建议。
- **abc.py**：`MomentPool` ABC 及其导出整体删除；`BasePointer` 加 `moment_id` /
  `moment_seq` 两个可选字段 + `model_validator` 校验（同缺同在、seq 非负）；
  新增 `MomentNotInCommitError`；`Memento.checkout()` 加 `base_moment_id`
  参数；`MementoBranch.update` / `annotate_moment` docstring 反映 "staging 直写、
  冻结即物理" 语义。
- **fs_memento.py**：授权重做，无独立池路径。核心新逻辑：
  1. `_resolve_staging()` 扫 staging.jsonl 得 (首现序, id → last-wins record)。
  2. `_load_commit_file()` 一次读一个 commit 文件即得 (Commit, m 个冻结 moment
     按 moment_ids 序 + moment_note last-wins, commit_note 列表)。
  3. `commit()` 一次写入 (成员 + 冻结 moments + 初始 commit_note)，fsync commit
     文件后再 truncate staging（§12 原子锚点）。
  4. `_recover_from_crash()` 装入时执行：无 commit 文件 = 无操作；成员行缺失 =
     删该 commit 文件；staging 全部 id 都是 last commit 成员 = truncate staging
     残留（避免误伤 "commit 之后又写新 record" 的合法状态是这里的关键判据）。
  5. `annotate_moment()` 按冻结状态分流：未冻结 → staging；已冻结 → 找到该
     moment 所在 commit 文件，追加 `t:"moment_note"` 行。
- **porcelain.py**：零改动。`update_moment(branch, moment)` 从纯 ABC 表面消费，
  内部路径变化对它透明——契约层重构不外溢的兑现。
- **golden tests 重做**：手写字节 + 手读字节两条硬约束照旧；手写器与手读器都
  按 §14 布局重拼；新增 `test_crash_recovery_truncates_stale_staging` 覆盖崩溃
  恢复；`test_dumb_memory_degenerate_form` fork-vocabulary 静态扫描保留。

### 15.2 checkout(commit_id, moment_id) 新能力（§4.1 扩展）

**人类问题**：能否从 `(commit, moment_id)` 出发化身？如果不能就想要。

**答案**：新做，且 **§14 布局下几乎白拿** —— ancestry 最末段的 "commit 前缀切片"
= 读该 commit 文件、按 `moment_ids` 索引位置取切片，零 join、零索引。旧池布局
下此能力反而更贵。这是布局改动之外的白色礼物。

**契约面**：
- `BasePointer` 加 `moment_id`（str | None）+ `moment_seq`（int | None）。
- 语义 inclusive：切片 = commit 内 `[第一个 ... moment_id]`（含）。
- **空前缀在类型层不可构造**：`moment_id != None ⇒ MUST 命中该 commit 内实际
  存在的成员 id`，否则 `MomentNotInCommitError`。想表达 "c1 完全不继承"
  用 c1 的父 commit，不用 `(c1, ...)`。契约层从没有 "空前缀" 概念。这一条把
  人类担心的 "空 commit → 空 commit → 空 commit 递归路径" 从可能性中消灭。
- ancestry 冻结规则不变——`(commit, moment)` 对仍是稳定不可变锚点。

**主权面**：
- `FsMemento._locate_base()` 校验 `base_moment_id` 命中 commit 成员，
  写入 `BasePointer` 时同时 fill `moment_id` + `moment_seq`。
- `FsMementoBranch._load_records_of()` 读祖先 commit 时若命中 ancestry 末段的
  切片声明，返回 `records[: moment_seq + 1]`。
- `window()` / `commit_records()` 都走此路径，切片对上层完全透明。

**验收**（`test_checkout_from_moment_id_slices_commit_inclusive`）：alpha 提交
含 m1..m4 的 commit，beta 从 (view, "m3") checkout，`commit_records`
返回 `[m1, m2, m3]`、`window.details` 不含 m4；无效 moment_id 抛
`MomentNotInCommitError`。

### 15.3 施工中的两个非平凡判断

- **崩溃恢复判据的精化**：第一版 "commit 文件完整 + staging 非空 = 截断"
  过度触发，把 "commit 之后合法追加新 record" 的 staging 也误清。修正为
  "staging 所有 id 都是 last commit 成员才截断"。写路径 `update` 已经拒绝
  frozen id 覆盖，所以合法状态下 staging 里不会出现 last-commit-id；出现
  仅可能来自崩溃前 truncate 未落。判据幂等且不误伤。
- **note 行路由方案的撤回**：起草时我倾向单 `t:"note"` 类型 + `ref` 前缀
  路由（`cmt_...` = commit 释义走 body，否则 = moment 释义走 threads），
  外加 "moment id MUST NOT 以 cmt_/brn_ 开头" 的防呆约束。人类指出 §14 后
  moment id 已经降级为 commit 文件内的定位键，没必要拿 id 语义空间做类型
  区分。改用两个独立类型 `t:"commit_note"` / `t:"moment_note"`，字段结构
  本来就不共用，jsonl 视觉上也清晰。

### 15.4 未动的部分（明确圈出）

- 信封 `MomentRecord`、CommitKind、CommitNote/CommitView、Trailer §6、
  BranchMeta ancestry 冻结规则、见证层 §9、HEAD.json、last-wins 定序 §2.2。
- `memento-cli-and-agent` workstream 的钉子裁决面：13 颗钉子在
  `porcelain.py + abc.py` 表面消费，§14 布局落地不改变其消费面，等此
  workstream 收尾后 memento-cli-and-agent 可直接开工。

### 15.5 验收

- `tests/ghoshell_moss/default/core/memento/` 81 项全绿。
- 主 tests 树全项目回归 1745/1745 全绿，无外溢。
- 三项契约锚点：字节等价（hand-write ↔ hand-read）、退化态无 fork 词汇、
  index-regenerable（无 .cache/ 也运行正常）全部通过。

### 15.6 契约状态

FORMAT.md v1.1 冻结完毕。下一位化身若要动 memento 契约面，请：读 §14 + §15，
理解为什么池被删、`(commit, moment_id)` 的白拿从何而来，再决定动作。
（2026-07-19 追记：§16 已再次重开契约，v1.1 冻结解除。继续往下读。）

## 16. Branch 降维为纯 ref：commit 自治目录 + owner 级 worktree（2026-07-19）

**触发**：人类指出运行期摩擦点——branch 是 checkout 的身份持有者，每次化身
扇出 = 新 branch 目录 + ancestry 副本（memento-cli-and-agent 钉子 7 的必然
产物），无限增殖，而 §9 #5 的 GC 承诺（"人类显式介入"）是空头支票。一轮
讨论收敛后确认：这不是治理补丁能解的，是 **branch 承重过多**——它同时当了
ref、staging 容器、commit 容器、ancestry 索引四个角色。本节拆开四个角色，
branch 只留第一个。

方法论注脚：FEATURE.md 的职责是反射出思维里的虚拟机，让下一个化身 dump 成
code——本节是那场对话的压缩产物。老代码兼容性明确不管（模块未发布，
仍在打磨周期）。

### 16.1 定案布局

```
{root}/memento/{owner}/          ← owner = worktree = 会话身份 (借 git 概念:
  meta.json                        staging 归 worktree, 不归 branch)
  staging.jsonl                  ← 唯一活跃写面, owner 级, 切 branch 不跟走
  HEAD                           ← 当前 branch 名, 一行文本
  branches/
    main                         ← ref 文件: {"fork": ..., "commit_id": "cmt_..."
    idea-x                          [, "moment_id": ...]} — 可指向他 owner 的 commit
  commits/
    cmt_<ULID>/                  ← 自治目录: 全局稳定 id, 单一归属, 永不复用,
      meta.json                    懒创建, 出生即冻结 (无 "活 commit" 概念)
      moments.jsonl              ← 冻结成员真身
      notes.jsonl                ← 释义追加 (commit_note / moment_note 共居)
```

- commit 目录懒创建：staging 冻结时才 materialize（tmp 目录 + 原子 rename）。
  无内容永不落盘——checkout 不产生 commit。
- `commits/` 列表按 ULID 字典序即时间序。branch 内 NNNN 序号消灭，
  commit 定序全局化。
- **ancestry 从 BranchMeta 迁入 commit meta**：每个 commit 冻结时写入自己的
  祖先信息——parent = `(fork, commit_id[, moment_id])` 单父链 + 近端 N 段
  跳跃指针（N 是主权层常量）。纪律两条：寻路可达（给定 commit_id 追任意
  祖先 O(d/N) 或 O(log d)，禁止 O(全库)）；写入时确定（冻结后不变）。
  §3.5 的"fork 时刻展平冻结进 BranchMeta"条款废除——那是可变父链系统的
  危险反规范化在本系统的残影，祖先信息归 commit 自身后纯靠成员不可变背书。

### 16.2 核心语义收敛

1. **branch = 纯 ref**。创建/删除/重指向都是一行文件的原子写。`-D` 无痛——
   丢的只是名字，commit 链独立存在。§9 #5 的 branch 增殖问题被结构消灭，
   不需要 GC 机制。
2. **commit 出生即冻结**。"活 commit"概念不存在——staging 是唯一活跃面，
   且永远不可作为出生点。§3.2 "化身只能从 commit 出生，永不从 staging"
   从 API 纪律变成结构事实：staging 没有 id，没有东西可指。
3. **commit = 时间前缀冻结**（细节待拍板，见 16.5 #1）：冻结 staging 的
   一段时间前缀（默认全量）。拆多 commit = 依次冻结多个前缀。§3.7 时间
   连续不变，thread tag 只提示边界位置，不重排成员。
4. **checkout 目标永远是冻结 commit**。跨 owner checkout = 本 owner 新建
   branch ref 指向他 owner 的 `(fork, commit_id[, moment_id])`；首次 commit
   才产生本 owner 的 commit（parent = 该外部锚点）。§15.2 的 moment 前缀
   切片语义原样幸存为 parent 指针形态。
5. **merge 不存在**。跨 owner 交互 = 只读 checkout + Matrix 消息（孔径一）。
   单父链钉死，无冲突语义，无多父 commit。owner 身份治理归 host，各 cell
   治理自己的 memento——owner ↔ cell 对应关系显式化。
6. **契约层零锁**。契约只承诺三件事：commit 成员文件 immutable（原子
   rename 发布）、append-only 文件读者跳撕裂尾行、ref 更新原子写。
   互斥语义是业务方的事（flock / owner 分片纪律 / Matrix 串行化任选）。
   git 的态度：只做数据结构一致性，不做锁。注意 immutable 的精确范围是
   **成员文件**（meta.json + moments.jsonl）；notes.jsonl 冻结后仍可追加——
   "身份和成员冻结，释义永远开放"在目录内的物理映射。
7. **staging 是 API 暴露的活目录**。staging 路径/身份对业务方可见，供
   关联性动作（见证 daemon 挂载、跨进程协调、消息引用）。filesystem-first
   从"人类可 cat"扩展到"业务可挂"。
8. **裸 commit_id 反查靠见证层**。O(1) dereference 必须带 owner 元组；
   裸 `cmt_xxx`（文档/注释/消息里的锚点）= `git grep` 见证 repo，O(grep)，
   路径不索引不维护。§6 的 Memento-Ref 机制原样承担，无新机制。

后果接受：一个 owner 同时只有一个活跃写面，切 branch 前 staging 要么
commit 要么丢（git 同款摩擦）。与 mindflow 单注意力焦点、"一个化身一个
owner"（cli-agent 钉子 6/7）天然对齐，不是缺陷。

### 16.3 被否决的备选

- **commit 活目录（无独立 staging）**：极简一层，但"拆多 commit"治理动作
  失去物理载体，逆向拆分破坏 append-only。死于治理动作不清晰。
- **跨 owner 引用计数 / backlink**（archive 前查"谁借我"）：引入握手协议，
  污染只读纯度。反查需求走见证层 grep。
- **裸 commit_id 的 O(1) 全域 ref**（任意位置 ref 文件留锚点）：路径无法
  维护好，中央索引违背 filesystem-first。接受 grep 成本。

### 16.4 波及面

- FORMAT.md v2 重写（v1.1 冻结解除；§14 布局被本节取代，但其"commit
  自包含 moment 真身"哲学保留——从单文件变成自治目录，cat 语义更纯）。
- abc.py 重写：**MementoBranch ABC 解体**——staging 操作、commit 操作、
  branch ref 操作归 owner facade（Memento）；BranchMeta / BasePointer /
  ancestry 冻结相关模型废除，parent 指针入 commit meta。信封 MomentRecord /
  CommitNote / trailer 工具 / hooks 零变化。
- fs_memento.py / golden tests 授权丢弃重做。
- **不变的部分**（防过度重做）：信封模型、释义 last-wins §2.2、trailer
  规范 §6、双孔径 §3.4、commit 永不删、时间连续 §3.7、见证层 §6/§9、
  退化态验收（蠢记忆无 fork 词汇）、golden 互读字节等价条款。

### 16.5 待人类拍板的三个次级决策

1. **commit 的时间前缀边界参数**：`commit()` 收可选的 staging 内 record id
   （含）作为冻结边界，默认全量。这是"拆多 commit"的最小 API 面。
2. **overlay（化身出生注入物）的新家**：BranchMeta 死了。提议迁 owner 级
   meta.json——化身 = 新 owner，worktree 身份卡正是出生注入物的语义位置。
3. **ref 文件格式**：提议 JSON 元组 `{fork, commit_id[, moment_id]}` 而非
   裸 id 一行——跨 owner checkout 需要 fork 字段。

### 16.6 状态

契约第二次重开（前次 §14）。memento-cli-and-agent 施工暂停，等本节冻结。
两次重开同源：契约 review 走真场景压力（§14 是 CLI 设计压力，本节是运行期
增殖压力），都发生在实现铺开之前——最便宜的修正时刻。§9 Open Problem #5
大部分被本节消解（branch 增殖 / GC / archive），残余问题只剩 owner 目录
整体 archive 的语义，降级为普通运维题。

（2026-07-19 追记：§17 拔除了本节的 worktree 类比——staging 归 owner 被
推翻，staging 归线。本节的 commit 自治目录 / 出生即冻结 / ancestry 入
commit meta / 零锁 / merge 不存在全部存活。继续往下读。）

## 17. 时间线原生化：staging 归线，worktree 类比拔除（2026-07-19）

**触发**：人类指出 §16 的 worktree 类比在 object 层面断裂。git 的 object 是
文件——工作区未提交内容是**空间性**的，一个 diff 浮在任意 base 上，checkout
携带未提交文件是合法操作。memento 的 staging 内容是 moments——**时间性**的，
因果上属于它延伸的那条线。跨线携带 = 伪造历史（在 commit A 下游经历的时刻
被记成 commit B 的下游），违反 §3.7 时间连续。"checkout branch 共享同一个
staging"这个逻辑在 memento 里不合法。

第二判据是并行场景（人类给出的真实动机）：内存里开 N 个 agent 各持一个
branch 句柄并行探索，最后各自返回结论 + last commit id。owner 级单一
staging 在 N 线并行下根本摆不下。

### 17.1 原生形状（不借 git）

git 的世界有**地点**（working tree）：branch 是同一内容的不同版本，checkout
是把地点传送过去，staging 是地点与历史的中介。memento 的世界没有地点，
只有**时间线**：

- owner 是**身份**，不是地点。一个身份可以同时活在 N 条线上。
- 每条线 = 冻结的过去（commit 链）+ **活边**（staging）。
- 活边与线不可分——staging 跟随 branch 是结构真命题，不是设计选择。

### 17.2 布局修订（覆盖 §16.1 的 owner 级 staging + HEAD）

```
{root}/memento/{owner}/
  meta.json                      ← owner 身份卡 (overlay 候选新家, §16.5 #2)
  branches/
    main/                        ← branch = 时间线: ref + 活边的小目录
      ref                          {"fork": ..., "commit_id": "cmt_..."[, "moment_id": ...]}
      staging.jsonl              ← 本线活边, 随线生随线死 (-D 即弃)
    idea-x/
      ref
      staging.jsonl
  commits/
    cmt_<ULID>/                  ← §16 原样: 自治目录, 出生即冻结, 懒创建
      meta.json
      moments.jsonl
      notes.jsonl
```

- **HEAD 文件废除**。owner 级"当前 branch"不是结构概念——没有地点就没有
  "现在站在哪"。CLI 层可以留一个默认线名做人体工学，不进契约。
- branch 从"纯 ref 文件"变成"ref + 活边的小目录"。§16 的胜利不回退：
  `-D` 仍无痛（目录很小），增殖仍被结构消灭，commit 仍是唯一冻结体。

### 17.3 语义收敛

1. **"切换 branch"操作不存在**。只有三个动作：**开线**（checkout 新 branch
   / fork）、**延线**（moment 落活边、commit 落锚）、**弃线**（`-D`，活边
   随线死，冻结 commit 永存）。`-D` 是唯一显式丢弃活边的动作，人类已授权
   彻底丢。§16.2 尾段"后果接受：一个 owner 同时只有一个活跃写面，切 branch
   前 staging 要么 commit 要么丢"整段作废——那是 git 摩擦的错误移植。
2. **ref 移动前活边先落锚**——真正的一致性规则，管的是 reset 而非 switch：
   reset 前 staging 自动机械 commit（锚在原 ref 位置），然后 ref 原子移动。
   什么都不静默丢，reset 因此合法。**commit 不是优先动作**：锚机械地落，
   意义后补。未释义的机械 commit 可被发现（log 过滤 kind=mechanical 且
   无 note），补释义走孔径二。
3. **并行合法化**。N 线 = N 活边，无共享可变面，零锁契约在并行下自然成立。
   branch = 单写者域，写者纪律归业务（flock / 内存句柄独占任选）。
   **覆盖 cli-agent 钉子 7**：并行扇出不必走派生 owner——同 owner 多线并行
   合法，锁粒度从 owner 级降到 branch 级。跨 owner fork 仍是跨身份协作的
   形态，但不再是并行的必要条件。
4. **commit 空间从 API 拿**：`commit_space(commit_id) -> path`，运行时对
   root 解析。memento 内部存储的一切引用只许用 id 元组，**出现绝对路径即
   契约违规**——memento 可跨项目分享，任何绝对路径和现场关联都无意义。
   空间内怎么操作，契约不定义：保留名单（meta.json / moments.jsonl /
   notes.jsonl）之外，契约沉默——不感知、不承诺、不禁止。业务自由空间
   （ground 场快照、link、说明等范式留给 dogfooding），变动历史由见证层
   兜底（sidecar git 拍下家具搬动，意义堆积自带时间锚）。
5. **释义跟随轨迹，不跟随对象**。summary 是一次解释行为，发生在某条线的
   某个时刻，属于解释者的轨迹。owner 自己的释义走 commit 目录 notes.jsonl
   （孔径二，append-only last-wins，无锁）；外来轨迹对该 commit 的 summary
   存在**它自己的空间**里，按 id 引用。"一个 commit 在不同轨迹里有不同
   summary 版本"自然成立。**否决 meta summary 字段可变**——那会把 flock
   问题请回来，并溶解"出生即冻结"的结构事实。

### 17.4 §16 存活与覆盖清单

- **存活**：commit 自治目录 / 出生即冻结 / 懒创建、ancestry 入 commit meta
  （分段祖先、寻路可达）、时间前缀冻结、merge 不存在、零锁契约三承诺、
  见证层反查、`-D` 无痛。
- **覆盖**：§16.1 owner 级 staging + HEAD 文件；§16.2 #7 "staging 是 API
  暴露的活目录"语义照旧但路径归线（branches/{name}/staging.jsonl）；
  §16.2 尾段"单一活跃写面"作废；cli-agent 钉子 7 扇出纪律待其复工时同步
  修订（见 17.3 #3）。
- §16.5 三个次级决策仍待拍板，#3 ref 文件格式在 ref 移入
  `branches/{name}/ref` 后照旧适用。

### 17.5 模块提升：memento 升一级模块（下一轮）

人类已决定 ground 提升一级；memento 下一轮多半同步提升：
`core/memento/` → `ghoshell_moss/memento/`，意味着未来可独立拆包。

模型侧赞同，理由：

- 依赖方向本来就对：memento 契约刻意零依赖内核概念（信封不认 Moment、
  零锁、filesystem-first），是被 core 消费的地基而非 core 的一部分。
  一级位置让**"memento 不得 import ghoshell_moss.core.*"** 这条纪律
  结构可见、可 lint。
- 与 ground 对称：Ghost 反身性基建双柱——ground = 认知场（现在），
  memento = 轨迹（过去）。同层摆放，互链（17.3 #4 的自由空间）不产生
  层级穿刺。
- 时机：建议与 §16/§17 重建同轮做——abc.py 反正重写、模块未发布、
  兼容性明确不管，一次迁移比两次便宜。

### 17.6 状态

本节与 §16 合并构成契约第二次重开的完整定案。讨论边界已由模型划定并经
人类确认：结构与一致性规则（本节四条）进契约；20-agent 编排范式、跨轨迹
释义交换的具体形状、ground 互链格式**不再推演**——全是消费面的事，
dogfooding 接触现实之前继续推演产出的是无现实接触的推测。等人类 review
§16+§17 后冻结，起草 FORMAT.md v2。
## 18. CLI 定案 + commits/ Y-m 分桶 + commits.jsonl 时序日志（2026-07-20，kimi-k3 对齐，人类拍板）

**触发**：人类定下验收路径——用 CLI 体系做验收，正式把 memento 改造到能用；
然后走 memento agent 做"一个可能没啥用，可能很有用的实现"（人类原话）。
本轮对齐定下 CLI 形状与两个存储层补丁。

**对齐方法论**（下一个化身必读）：人类本轮明示——**FEATURE.md 是模型写给
下一个模型的移交契约，不是给人类看的文档，更不是人类写的**。与人类对齐时
禁止拿文档章节号当权威考人类；人类的心智模型（moments.jsonl / record /
commit / annotate 附带数据）才是对齐基准。§3.6 thread tag 本轮被人类一句
"我记忆中没有"合法处决，证明此纪律不是装饰。模型本轮被驳回的三个判断
记录防重演：(a) §14 概念惯性带入 §17（.cache/、rebuild-index、recover 命令）；
(b) rewind 误读为 commit 级改写（实为 branch 级，见 18.5）；(c) CLI 默认值 /
唯一性推断方案（人类："核心参数取默认我不觉得有啥收益"，主张位置参数
不允许默认值）。

### 18.1 commits/ Y-m 分桶（人类拍板）

§17 的 commits/ 平铺有物理增长问题（单目录年尺度爆炸，人类确认此为讨论点）。
定案：

```
commits/
  2026-07/
    cmt_01H.../
      meta.json / moments.jsonl / notes.jsonl
```

- **commit_id → Y-m 是纯函数**：ULID 前 10 字符 = 48-bit 毫秒时间戳
  （Crockford base32），解码取 UTC 年月。无冗余字段、无索引、无扫描。
- **严格 UTC**：从 ULID 时间戳解出，不从 wall clock、不用本地时区——
  memento 可跨项目/时区分享（§17.3 #4），UTC 无歧义。
- **时钟回拨接受**：Y-m 只是物理位置，逻辑时序由 commits.jsonl 保证
  （18.2），两者解耦。
- **遍历**：全局时间序 = `ls commits/`（字典序）+ 逐月 `ls commits/{Y-m}/`
  （ULID 字典序），两层拼接无需 sort。
- ref 文件仍存纯 id 元组，解析时确定性算 Y-m，不违反 §17.3 #4。
- 被否备选：ULID 前缀分桶（`01H/` 不直观）、epoch-N 数量分桶（数量与时间
  边界错位）、`YYYY/MM` 两层（层数 +1 无收益）。

### 18.2 commits.jsonl：owner 级 append-only 时序日志（契约化）

人类提出的问题："commits.jsonl 这个 append only 是否还有，如果有的话时序
实际上不会乱"——答案：有，契约化。

```
{owner}/
  commits.jsonl            # append-only
  commits/{Y-m}/cmt_.../   # 真身
```

行格式：

```json
{"t":"commit_ref","commit_id":"cmt_...","branch":"main",
 "parent":{"fork":"...","commit_id":"..."[,"moment_id":"..."]},
 "ts":"...","kind":"semantic|mechanical"}
```

- **定位**：派生层（可从 commits/ 扫描重建）但**契约化**——崩溃恢复判据
  依赖它。
- **时序物理保证**：append 顺序即真实时序，不依赖 ULID 时间戳。
- **并发安全**：多 branch 并行 commit 时，POSIX O_APPEND 写 < PIPE_BUF
  (4096B) 的行原子，多进程 append 无需 flock——§17.3 #3 的 branch 级锁
  粒度不破。
- **崩溃恢复精化**：commit() = 写 commit 目录 → fsync → append
  commits.jsonl → fsync → truncate staging。恢复判据从 §12/§15 的"staging
  所有 id 都是 last commit 成员"精化为"属于 commits.jsonl 尾行 commit_id"，
  无需扫描推断。
- **红利**：见证层增量快照抓手（tail 自上次 offset）、运维 `tail -f`、
  CLI `owner log` 的数据源。

### 18.3 .cache/ 处决 + rebuild-index / recover 命令删除

§14 的 `.cache/`（可再生索引）三项职能在 §17 全部消解：commit_id→位置
（18.1 纯函数）、全局时序（18.2）、last-wins 视图（notes.jsonl 直读）。
唯一幸存的 moment_id→commit 反查走 grep（§17.3 #8 精神：路径不索引不维护，
接受 grep 成本；错标修正低频）。`.cache/` 从 FORMAT v2 删除，`rebuild-index`
命令同步删除；`recover` 命令删除，恢复报告并入 `status` 输出（装入时自动
恢复已有，CLI 独立命令价值弱）。

### 18.4 CLI 命令树（定案）

**CLI 是全功能模块**：退化态纯净归 ghost 集成层与 golden test（契约不变量
#12），不归 CLI。CLI 用户 = 人类操作员 / 高级 agent，本该理解 branch。

**寻址统一 `<owner>/<name>`**（`cmt_` 前缀 = commit，否则 = branch）。
**branch 级动作用必填位置参数，无默认值**（人类拍板：默认值的失败模式是
静默写错线，且不好核对）。

```
moss memento init [DIR] [--witness sidecar|outer|none]

moss memento owner list
moss memento owner status <owner>                   # 含恢复报告
moss memento owner log <owner>                      # commits.jsonl 物理时序

moss memento branch create <owner/name> [--from <owner/cmt_...>] [--overlay "..."]
moss memento branch list <owner>
moss memento branch record <owner/name> [data]
moss memento branch commit <owner/name> [-m "..."] [--to <moment_id>] [--kind semantic|mechanical]
moss memento branch staging <owner/name>
moss memento branch log <owner/name>                # parent 链因果历史
moss memento branch window <owner/name>
moss memento branch reset <owner/name> --to <owner/cmt_...>   # rewind
moss memento branch delete <owner/name>

moss memento commit show <owner/cmt_...> [--notes]
moss memento commit annotate <owner/cmt_...> -m "..."
moss memento commit space <owner/cmt_...>           # 输出 commit_space 路径

moss memento witness snapshot|status
```

- **fork 被 `branch create --from` 吸收**：`--from` 跨 owner 时结构上即 fork
  （ref 带 fork 字段），无独立 fork 命令。`--overlay` 落 owner meta.json——
  §16.5 #2 顺带定案。
- **`commit --to <moment_id>`** = §16.5 #1 边界参数的 CLI 形状：git add 的
  空间性选择被时间性切点吸收，"add" 概念消解。
- **log 两层次非冗余**：`owner log` = 物理时序（commits.jsonl），
  `branch log` = 因果链（parent 链回溯）。并行场景显著分叉。
- **ref 文件 = JSON 元组** `{fork, commit_id[, moment_id]}`：§16.5 #3 由
  本节寻址体系隐含定案（fork 字段支持跨 owner）。
- **record 可选手动指定 moment_id**：低优先级（人类："可以试试，不过这个
  点不那么重要"）。

### 18.5 rewind 定义钉死：branch 级，不是 commit 级

模型本轮误读 rewind 为 commit 级改写并据此反对——错误。人类定案：
**rewind = branch reset**，移 ref 不改历史。写错线的修复闭环：

1. `branch reset owner/main --to owner/cmt_误写前`
2. staging 活边自动落锚为机械 commit（误写 moment 在内，什么都不丢，
   §17.3 #2）
3. ref 移走，该机械 commit 成孤儿，永远可寻址
4. annotate 打标"误写，已 reset"

rewind 不违背成员不可变，**恰恰依赖它**——reset 安全正因为旧位置永远
可寻址。错误哲学三段：commit 前 staging 是可编辑文本（filesystem-first
红利）；commit 后 annotate 打标；历史诚实，意义可补。

### 18.6 annotate-moment 从 CLI 第一版砍掉

§3.6 thread tag 是文档自标"纸面定不了"的预设（§9 #4 生成者未定），人类
心智模型中不存在，未经真场景验证。CLI 第一版不含 annotate-moment；
MomentRecord 信封的 threads 字段在契约层保留不动，CLI 暴露等真场景
（分页旁路 / commit 拆分）逼着它出现再加。

### 18.7 波及面与状态

- **FORMAT v2 起草输入**：§16+§17 + 本节（Y-m 分桶、commits.jsonl 契约化、
  .cache/ 删除、崩溃恢复精化、ref JSON 元组）。
- **abc.py 重写输入**：MementoBranch 解体（§16.4）+ owner/branch/commit
  三层操作面（18.4 反推）。
- **memento-cli-and-agent 复工条件**：等 FORMAT v2；命令树按 18.4 对齐，
  agent 命令挂进此树；§16.5 #1（--to）/ #2（overlay→owner meta）已顺带
  定案。
- **验收方式**：CLI 体系（人类定）。memento 改造目标 = 18.4 全部命令可用。
