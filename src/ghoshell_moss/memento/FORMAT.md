# Memento FORMAT v1

磁盘格式契约。本文件是 memento 的**契约层**交付物之一（另两件：`abc.py`、golden
tests），人类 review 后冻结。实现层（存储/索引/见证 daemon/channel）是可丢弃的；
本文件不是。**实现与本文件冲突时，以本文件为准；要改本文件，必须过人类 review。**

- MUST / MUST NOT：违反即实现 bug，golden tests 直接锚定这些条款。
- SHOULD：推荐约定，违反需在实现处注释理由。
- 设计动机不在此展开，见 `workstreams/2026/06/momento-mori/FEATURE.md`。

---

## 1. 总布局

```
{root}/memento/
  branches/{owner}/HEAD.json                  # 该 owner 的 current branch 指针
  branches/{owner}/{branch_id}/
    meta.json                                 # BranchMeta, 含冻结祖先链
    staging.jsonl                             # 活跃写面, moment 真身在此渐进构建
    commits/{NNNN}.jsonl                      # 每 commit 一个文件, 自包含 moment 全文
  renderings/                                 # 重绘投影, 自由格式, 只增不删
  .cache/                                     # 可再生索引, 删掉重扫, 见证层忽略
  .git/                                       # 见证层 sidecar repo (§9)
  .gitignore                                  # 至少包含: .cache/
```

- `{owner}` 是 owner 命名空间字符串，同时是目录名。MUST 匹配
  `[A-Za-z0-9._\-]{1,64}`。memento 不解释其语义（可以是 cell address 或任意约定）。
- **单写者纪律**：`branches/{owner}/` 下的所有文件，MUST 只由绑定该 owner 的单一
  Memento 实例写入。跨 owner 只读。释义改写（含旁路发起的）MUST 经由 owner 实例
  落盘——格式层不提供多写者协议。
- **无独立 moment 池**：moment 记录的物理归属 = staging 或某个 commit 文件，
  二者互斥。commit 冻结即把 staging 的 last-wins 视图整块搬入 commit 文件，
  staging truncate。fork 出的 branch 通过 `BranchMeta.ancestry` 引用祖先 commit
  文件中的 moment，永不复制。

## 2. 通用行格式

适用于所有 `*.jsonl` 文件：

1. 编码 UTF-8，无 BOM。`ensure_ascii=False`（非 ASCII 字符直接写原文，保 grep）。
2. 每行一个 JSON object，行终止符 LF（`\n`）。MUST NOT 使用 CRLF。
3. **换行转义**：正文中的换行以 JSON 标准转义（字符串内 `\n`）存储。
   一条逻辑记录永远恰好占一个物理行。
4. 每行首字段 MUST 是判别符 `"t"`。读者遇到未知 `t` 值 MUST 跳过该行（前向兼容）。
5. **撕裂尾行**：文件最后一行 JSON 解析失败时，读者 MUST 静默跳过（写入中断的
   合法残留）。**非最后一行**解析失败 = 数据损坏，MUST 抛错，MUST NOT 静默跳过。
6. 读者 MUST NOT 依赖 JSON key 顺序。写者 SHOULD 按 schema 声明序输出、紧凑分隔
   （`,` `:`），使见证层 diff 最小。
7. 时间字段一律 RFC 3339 字符串，MUST 带时区偏移（如
   `2026-07-11T12:34:56.789012+08:00`）。naive datetime 非法。
8. 可选字段缺省时 SHOULD 省略不写（不写 null）。

### 2.1 id 规则

| 对象 | 格式 | 生成方 |
|------|------|--------|
| moment record | 生产者自带 id 原样透传 | 生产者 |
| commit | `cmt_<ULID>` | memento |
| branch | `brn_<ULID>` | memento |

- ULID：26 字符 Crockford base32 大写。
- moment id MUST 非空、匹配 `[A-Za-z0-9._\-]{1,128}`、在同一 branch 的**可写范围**
  （staging + 该 branch 自己的 commits）内唯一。祖先 commit 中的 moment 不参与此
  唯一性——祖先 moment 只经 `(fork, branch_id, commit_id, moment_id)` 四元组
  引用，不会与本 branch 的 moment id 空间冲突。
- 前缀 `cmt_` / `brn_` 保留给 memento；grep 任意文本中的 `cmt_` 即得 commit 引用。

### 2.2 last-wins 定序（钉死）

释义可变、成员不可变。释义的多版本以追加行表达，读时取最新：

- **"last" = 同一文件内更大的字节偏移**（即更靠后的行）。不比时间戳，不比 id。
- 释义行 MUST 追加到**包含所释义对象的同一个文件**：
  - moment 在 staging 时，其覆盖行与 `moment_note` 都写 `staging.jsonl`；
  - moment 已冻结于某 commit 文件时，其 `moment_note` 追加到该 commit 文件；
  - commit 的 `commit_note` 一律追加到该 commit 自己的文件。
- 跨文件释义非法——这是 last-wins 无歧义的前提。
- 时间戳字段（`ts`）只作展示与诊断，MUST NOT 参与定序。

## 3. Moment 记录（信封）

### 3.1 信封模型（envelope）

memento 不理解 Moment 的内部结构。存储的是 **MomentRecord 信封**：

```json
{"t":"moment","id":"<生产者id>","created":"<RFC3339>","type":"moss.moment/v1","threads":["memento-design"],"payload":{...},"by":"ghost.main"}
```

| 字段 | 必选 | 语义 |
|------|:---:|------|
| `t` | ✓ | 恒为 `"moment"` |
| `id` | ✓ | 生产者 id，§2.1 |
| `created` | ✓ | 信封创建时间 |
| `type` | ✓ | payload schema 标识，如 `moss.moment/v1`。schema 归生产者所有 |
| `payload` | ✓ | 任意 JSON object，**memento 原样透传，MUST NOT 解析或改写** |
| `threads` | | 线索标签，写入时标注（可空可缺省），可经 `moment_note` 更新 |
| `by` | | 写入者标识（模型名/规则 id），诊断用 |

- payload 对 memento 不透明是硬边界：`abc.py` 与存储实现 MUST NOT import 任何
  payload schema（包括 `ghoshell_moss` 的 `Moment`/`Message`）。强类型编解码
  （codec）在信封之上、作为独立模块存在。
- 未来 Moment 包剥离的窗口即在此：信封层零依赖，天然可剥。
- **大 payload SHOULD 走引用不内联**：音频/视觉等 base64 编码超过 64 KiB 的 payload
  SHOULD 由生产者存到外部资源系统、payload 只放 URL 或资源 id。memento 不校验此
  条——payload 透明原则保持不变——但见证层 diff 与 grep 场景受益显著。

### 3.2 记录行的可变性：冻结前可覆盖，冻结后不可

Moment 在一轮交互内是渐进构建的（感知先到、logos 后到）。因此：

- 同一 `id` 的 `t:"moment"` 行 MAY 在 `staging.jsonl` 中出现多次，读者按 §2.2
  last-wins 取最新——每次覆盖写都是一个新对象共享同一 id（"更新即新对象"）。
- **一旦某 id 随 commit 搬入 commit 文件（§5），继续向 staging 追加其
  `t:"moment"` 行即契约违规**。写 API MUST 拒绝（`MomentFrozenError`）；
  语义上此时 staging 已无该 id 的槽位——冻结即物理，不是 API 层的软约束。
- commit 文件内的 `t:"moment"` 行 MUST NOT 出现同一 id 的第二条。commit 冻结
  的是 "冻结时刻该 id 的 staging last-wins 视图"，一次性、不再动。

### 3.3 moment 级释义（`moment_note`）

```json
{"t":"moment_note","ref":"<moment_id>","threads":["a","b"],"ts":"<RFC3339>","by":"tagger.rule"}
```

- `ref` 指向同文件内已有 `t:"moment"` 行的 id：
  - moment 未冻结时，写入 `staging.jsonl`；
  - moment 已冻结时，追加到该 moment 所在的 commit 文件。
- 语义：**整体替换**该 moment 的 `threads`（非增量合并）。
- moment 级释义 v1 只开放 `threads` 一个键。payload 永远不可经释义改写。

## 4. Branch 目录

### 4.1 meta.json

唯一的整文件重写（非追加）JSON 文件。释义可变性由见证层留痕（§9），
不在数据层追加。

```json
{
  "branch_id": "brn_01J...",
  "fork": "ghost.main",
  "name": "main",
  "title": "",
  "description": "",
  "base": {"fork":"...","branch_id":"brn_...","commit_id":"cmt_...","commit_seq":7,"moment_id":null,"moment_seq":null},
  "ancestry": [
    {"fork":"...","branch_id":"brn_...","commit_id":"cmt_...","commit_seq":3,"moment_id":null,"moment_seq":null},
    {"fork":"...","branch_id":"brn_...","commit_id":"cmt_...","commit_seq":7,"moment_id":"mmt_ab12","moment_seq":4}
  ],
  "overlay": {"divergence_prompt": "..."},
  "created": "<RFC3339>",
  "updated": "<RFC3339>"
}
```

| 字段 | 可变性 | 语义 |
|------|--------|------|
| `branch_id` `fork` `created` | 不可变 | 身份 |
| `base` | 不可变 | fork 起点。null/缺省 = root branch。见 §4.1.1 BasePointer |
| `ancestry` | 不可变 | **冻结的展平祖先链**，自最老祖先到直接 base，顺序排列。root branch 为空数组。fork 时刻一次性计算写入（= 父的 ancestry + 父的 base 条目），此后 MUST NOT 改写。回溯 O(d)→O(1) 的依据 |
| `overlay` | 创建后不可变 | 化身 divergence prompt 等出生注入物的家。**不属于对话历史，MUST NOT 进 staging** |
| `name` `title` `description` `updated` | 可变 | 释义性字段 |

- `ancestry` 的最后一项 MUST 等于 `base`（有 base 时）。
- 校验：读者发现 `ancestry` 与沿 `base` 链实际回溯结果不一致时 MUST 抛错
  （冻结链是反规范化，成员不可变保证其安全；不一致 = 数据被篡改或写入 bug）。

### 4.1.1 BasePointer 与 commit 内前缀切片

BasePointer 语义：从某 branch 的某 commit（可选：commit 内某 moment 为止）
出生。

```json
{"fork":"ghost.main","branch_id":"brn_...","commit_id":"cmt_...","commit_seq":7,"moment_id":"mmt_ab12","moment_seq":4}
```

| 字段 | 必选 | 语义 |
|------|:---:|------|
| `fork` | ✓ | 源 branch 的 owner 命名空间 |
| `branch_id` | ✓ | 源 branch id |
| `commit_id` | ✓ | fork 起点 commit id |
| `commit_seq` | ✓ | 起点 commit 在源 branch 的 seq，目录截断用 |
| `moment_id` | | 可选。给定时，该 commit 只贡献 `[第一个 moment ... moment_id]` 的前缀（含）给继承者 |
| `moment_seq` | | 可选。给定时 MUST 等于该 moment 在 commit `moment_ids` 中的 0-based 位置。缺省 = 无 moment 前缀切片 |

- `moment_id` 缺省 / null = 整个 commit 参与继承（`moment_seq` MUST 同时缺省）。
- `moment_id` 给定时：MUST 是该 commit 实际成员之一，MUST 与 `moment_seq` 位置
  一致；否则读取时 MUST 抛错。
- **空前缀在类型层不可构造**——`moment_id` 要么缺省，要么至少载 1 个 moment。
  "commit 完全不继承"的表达 = 直接用其父 commit，不用 `(commit, moment=?)`
  构造空切片。ancestry 中每个 BasePointer 都是非空的历史锚点。
- 切片是 **inclusive** 的：截止 moment 本身被继承者读到，"新故事从这个 moment
  之后开始"。
- fork 时刻切片语义即冻结进 ancestry，此后不再变——与整段 commit 同一档不变性。

### 4.2 staging.jsonl

活跃写面。**全格式中唯一允许清空（truncate）的文件**——它是投影，
冻结后的事实真身在 commit 文件里。

staging 直接容纳 moment 真身（不再是引用行）：

- `t:"moment"` 行：同 id 覆盖直接追加，读者按 §2.2 last-wins 取最新版本。
  同一 moment_id 在 staging 中出现次数无上限，但只有 last-wins 视图参与 commit。
- `t:"moment_note"` 行（§3.3）：追加到 staging，`ref` 指向 staging 内的 moment
  id。冻结后此行不再迁移——commit 文件冻结的是 staging last-wins 时刻的 threads
  最终值。
- 首次出现顺序即成员定序：commit 时，`moment_ids` 按每个 id 在 staging 中首次
  出现的行号升序排列（后续覆盖行不改变次序）。

commit 时的原子动作序列（详见 §5 与 §12）：

1. 计算 staging 的 last-wins 视图（同 id 保留最后一版全字段，threads 应用
   `moment_note` 最终替换值）；
2. 写 `commits/{NNNN}.jsonl`：第 1 行 `t:"commit"` 成员行 + 第 2..m+1 行
   `t:"moment"` 冻结版全文（按首次出现序）+ 第 m+2 行 `t:"commit_note"` 初始释义；
3. `fsync` commit 文件；
4. truncate `staging.jsonl`；

### 4.3 HEAD.json

```json
{"current": "brn_01J..."}
```

owner 的 current branch 指针。整文件重写。缺失时实现自动创建 root branch
（`name: "main"`）。

## 5. Commit 文件

路径 `commits/{NNNN}.jsonl`。`NNNN` = seq 左零填充至少 4 位（`0001.jsonl`）。
seq 从 1 起、branch 内连续递增。溢出 4 位自然加宽（`10000.jsonl`）——读者
MUST 按解析后的整数排序，MUST NOT 按文件名字典序。

commit 文件自包含：不依赖外部池，`cat commits/0007.jsonl` 即得该认知快照的
完整内容（成员表 + 每个 moment 全文 + 释义历史）。

### 5.1 成员行（第一行，不可变）

```json
{"t":"commit","id":"cmt_01J...","seq":12,"moment_ids":["a","b","c"],"created":"<RFC3339>"}
```

- MUST 是文件第一行。`moment_ids` 是 staging 冻结时刻的去重有序列表，MUST 非空
  （空 staging 禁止 commit）。
- 成员行写下后 MUST NOT 出现第二条 `t:"commit"` 行。成员不可变是 fork 边界的
  前提：动它，所有子 branch 的 ancestry 集体作废。

### 5.2 Moment 冻结行（第 2..m+1 行，不可变）

```json
{"t":"moment","id":"a","created":"<RFC3339>","type":"moss.moment/v1","threads":[...],"payload":{...},"by":"ghost.main"}
```

- 第 1 行 `t:"commit"` 之后紧跟的 m 行 MUST 是 `t:"moment"` 冻结版全文，
  顺序 MUST 与成员行的 `moment_ids` 逐个一致。
- 每个 `t:"moment"` 行的 id MUST 出现在同文件 `moment_ids` 中；反之亦然
  （一一对应，禁止漏行漏 id）。
- 这些行是 staging last-wins 视图的物理搬运结果，字段结构与 §3.1 完全一致。
- commit 文件内的 `t:"moment"` 行 **不参与 last-wins**——冻结即定型，同 id 不
  再出现。moment 级 threads 的后续更新走 §5.3 的 `moment_note`。

### 5.3 释义行（追加，last-wins）

commit 释义（commit 整体的语义摘要）：

```json
{"t":"commit_note","ref":"cmt_01J...","body":"<正文+trailer, §6>","ts":"<RFC3339>","by":"ghost.main"}
```

- `ref` MUST 等于本文件成员行的 id。
- `body` 整体替换语义：读者取最后一条 `commit_note` 的 body 为当前释义，
  **历史版本永远可寻址**（按行序枚举即全部版本）。
- commit 动作 MUST 在冻结的 `t:"moment"` 行之后紧跟写入一条初始 `commit_note`
  （机械 commit 的 body 可以只有 trailer 块，见 §6）。
- **渲染打戳**：任何把释义内容展示给模型的渲染，MUST 可追溯到所读的释义版本
  （实现记录 `(commit_id, 行号)` 或等价物）。"当时模型看见什么"必须可重建。

moment 级释义（冻结后追加，仅 threads）：

```json
{"t":"moment_note","ref":"<moment_id>","threads":["a","b"],"ts":"<RFC3339>","by":"tagger.rule"}
```

- `ref` MUST 是同文件 `t:"moment"` 行的 id（该 moment 冻结后 threads 变更的落点）。
- 语义同 §3.3：整体替换 `threads`，payload 永不可改。
- 跨 owner 只读：他 owner 的 branch 冻结的 moment，其 threads 只有该 branch 的
  owner 有权改写（走 owner 实例落盘，孔径二）。

### 5.4 note 行的两种类型：为何分开

commit 释义（`body`）和 moment 释义（`threads`）字段结构不同，两个 `t:` 类型
使解析路径彼此独立，无需按 `ref` 前缀区分。读者遇到未知 `t:` 值按 §2 跳过，
前向兼容自然满足。

## 6. Body 与 Trailer 规范

> **本节兼任释义生成旁路的 prompt 约束**——规范与 prompt 不写两份。
> 给模型的指令可直接引用本节。

body 结构 = 自由文本正文 + 空行 + trailer 块：

```
重构 staging 拆分逻辑，决定线索标注下沉到写入时。

Thread: memento-design
Thread: meta-cognition
Resumes: cmt_01JABC...
Suspends: meta-cognition
Kind: semantic
```

**解析规则**（git trailer 的简化子集）：

- trailer 块 = body **末尾**连续的、每行匹配 `^[A-Za-z][A-Za-z0-9-]*: .+$` 的行。
- trailer 块与正文之间 MUST 有至少一个空行。全文只有 trailer 无正文时，
  不需要空行。
- 同 key 可重复（如多条 `Thread:`）。key 大小写敏感，规范形式为首字母大写连字符
  风格（`Memento-Ref`）。
- 解析器遇到未知 key MUST 保留原样（前向兼容），MUST NOT 报错。

**v1 注册的 trailer key**：

| Key | 值 | 语义 |
|-----|----|------|
| `Thread:` | 线索名 | 本 commit 属于哪条话题线索，可多条 |
| `Resumes:` | commit id | 重入锚——本处回归了哪个被搁置线索（指向该线索最后的 commit） |
| `Suspends:` | 线索名 | 本处挂起了哪条线索 |
| `Kind:` | `semantic` \| `mechanical` | 语义锚点（模型自宣）/ 机械快照（规则触发，为 fork 服务）。初始释义行 MUST 含此 key |
| `Memento-Ref:` | commit id | 跨系统通用 join key。memento body、重绘渲染、见证 repo commit message、代码仓库提交里引用 commit 一律用它，反查 = `git log --grep` + `grep -r` |

**给生成旁路的写作约束**（prompt 直用）：

1. 正文写"发生了什么、为什么"，一段以内；没有信息就留空，不要凑字。
2. 线索归属写 `Thread:`，不要写进正文。
3. 回归旧话题时必须给 `Resumes:` 锚到该话题最后的 commit id；说"回到刚才"而
   不给锚点是丢信息。
4. 挂起话题时给 `Suspends:`。
5. 机械快照不要编造正文，`Kind: mechanical` + 需要的 trailer 即可。

## 7. 索引与缓存

- 索引（commit_id → branch+seq、moment_id → commit_id+行号、释义 last-wins
  视图等）MUST 全部放在 `{root}/memento/.cache/` 下。
- **jsonl 是唯一 truth，索引是可再生缓存**：删除 `.cache/` 后系统 MUST 能全量
  重扫重建，行为不变。
- 快路径（`window()` 渲染最新 commit）MUST NOT 依赖 `.cache/`：读最新 commit
  文件即得完整成员与 payload，零 join。索引仅在慢路径（按 commit_id / moment_id
  随机寻址、`all_commits()` 回溯）为必要件。
- `.cache/` 内部格式不属于本契约，实现自定（升级时直接作废重建）。
- `.cache/` MUST 被见证层忽略（§9）。

## 8. renderings/

重绘投影（plan 渲染等）的家。本契约 v1 只约束三条：

1. 子目录**只增不删**——旧投影永远保留，重绘 = 新建目录。
2. 投影内引用 commit 一律用 `Memento-Ref: cmt_...` 行（§6）。
3. 目录内格式自由（mermaid / markdown / 任意），不属于本契约。

## 9. 见证层（git sidecar）

git 不是 memento 的结构，是**见证**：fork 是纯 memento 层操作，git 只见证文件，
不知道 fork 存在。

- repo 位于 `{root}/memento/.git`，工作树即 `memento/`。
  **MUST NOT 被外层代码仓库吞掉**——memento root 必须被外层仓库 ignore，或位于
  仓库之外。两个时间尺度串扰是本架构唯一真正的污染模式。
- `memento/.gitignore` MUST 至少含 `.cache/`。
- 快照由**单写者旁路 daemon** 执行（memento commit 事件触发或定时），
  MUST NOT 出现在任何热路径。v1 实现用 subprocess git；不引 C 依赖。
- 快照 commit message 格式：

  ```
  snapshot: <RFC3339 时刻>

  Memento-Ref: cmt_01JAAA...
  Memento-Ref: cmt_01JBBB...
  ```

  正文 trailer 列出自上次快照以来新增的 memento commit id。
  由此 `git log --grep=cmt_xxx` 反查任意 commit 首次被见证的时刻。
- 两个地址空间：memento id = 身份（这是哪个 commit）；git sha = 完整性
  （历史未被事后篡改的证明）。
- Matrix 跨机复制直接复用见证 repo push/pull（owner 分片路径不碰撞，无冲突）。
  MUST NOT 为 memento 另造同步协议。
- 见证层是可选组件：不启用时其余格式条款全部照常成立。

## 10. Epoch 分仓（预留口子）

年尺度增长下，`{root}/memento/` MAY 按 epoch 分仓
（如 `memento-2027H1/` 平行目录）。id 全局稳定不变，跨 epoch 定位归 `.cache/`
索引层。本契约 v1 只保留此口子，分仓协议在启用时再冻结为 v2 条款。

## 11. 不变量清单（golden tests 锚点）

1. 成员行/记录行一经 commit 冻结即不可变（§3.2、§5.1、§5.2）。
2. 释义 last-wins = 同文件字节偏移序，且释义行与所释义对象的载体同文件（§2.2）：
   staging 内 moment 的释义在 staging；冻结 moment / commit 的释义在其 commit 文件。
3. payload 原样透传，字节不动；`abc.py` 与存储实现零 payload schema 依赖（§3.1）。
4. ancestry 冻结且与 base 链回溯一致（§4.1）。BasePointer 的 `moment_id/moment_seq`
   给定时 MUST 指向该 commit 实际成员，二者位置一致；`moment_id` 缺省 =
   `moment_seq` 缺省。空前缀在类型层不可构造（§4.1.1）。
5. staging 是唯一可 truncate 的文件（§4.2）。
6. 空 staging 禁止 commit；commit 原子动作 = 成员行 + m 个冻结 moment 行 +
   初始 `commit_note`（含 `Kind:`） + fsync commit 文件 + truncate staging（§5、§12）。
7. 撕裂尾行跳过，中段损坏抛错（§2）。
8. 删除 `.cache/` 后行为不变（§7）；快路径不依赖 `.cache/`。
9. renderings 与旧投影永不删除（§8）。
10. 见证 repo 独立于代码仓库，快照 message 携带 `Memento-Ref:`（§9）。
11. **互读等价**：两个独立实现照本文件各写一份历史，互读对方字节，重建出的
    历史（commit 序列、成员、每个 commit 内的 moment 全文序列、当前释义、
    ancestry）等价。
12. **退化态纯净**：单 branch + 自动 commit 的用例代码中，fork 相关词汇
    （fork/checkout/ancestry/overlay/base/moment_id 切片）一个不出现。
13. **冻结即物理**（§3.2）：某 moment id 被搬入 commit 文件后，staging 中 MUST
    无同 id 的可写槽位——冻结检查是 API 拒绝之上的结构约束。
14. **恢复幂等**（§12）：任何时刻中断，按 §12 恢复规则重放后系统状态与"无中断
    完成"等价，不产生重复 commit 或悬空 staging。

## 12. 崩溃恢复

commit 的原子动作序列（§4.2 复述）：
写 commit 文件 → fsync → truncate staging。中间任意点崩溃，恢复规则：

- **该 seq 的 commit 文件不存在**：staging 未动，等同没 commit 过；无需任何操作。
- **该 seq 的 commit 文件存在但成员行残缺或撕裂**：MUST 删除该文件重试
  （成员行是 commit 的物理身份，缺失即无 commit）。
- **该 seq 的 commit 文件存在且成员行完整**：commit 已成立（fsync 之后崩溃），
  MUST 直接 truncate `staging.jsonl`；本次已完成，不重跑。

规则幂等：多次触发 = 一次触发。恢复 MUST 在 Memento 装入 branch handle 时
（或首次访问 staging / head 时）执行。

---

历史：
- 2026-07-11 由 claude-fable-5 起草 v1。
- 2026-07-18 由 claude-opus-4-7 修订：§14 存储布局落地——`moments/` 池整体废除，
  staging 持真身、commit 文件自包含；新增 §4.1.1 BasePointer moment 前缀切片、
  §5.2 冻结 moment 行、§5.3 note 类型二分、§12 崩溃恢复；相应更新 §11 不变量清单。
  依据 workstreams/2026/06/momento-mori/FEATURE.md §14。
