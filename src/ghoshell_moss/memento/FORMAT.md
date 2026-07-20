# Memento FORMAT v2

磁盘格式契约。本文件是 memento 的**契约层**交付物之一（另两件：`abc.py`、golden
tests），人类 review 后冻结。实现层（存储/索引/见证 daemon/channel）是可丢弃的；
本文件不是。**实现与本文件冲突时，以本文件为准；要改本文件，必须过人类 review。**

- MUST / MUST NOT：违反即实现 bug，golden tests 直接锚定这些条款。
- SHOULD：推荐约定，违反需在实现处注释理由。
- 设计动机不在此展开，见 `workstreams/2026/06/momento-mori/FEATURE.md`（尤其
  §16 branch 降维、§17 时间线原生化、§18 CLI 定案 + Y-m 分桶 + commits.jsonl）。

## 0. v1 → v2 变更摘要

- **branch 降维为纯 ref**（§16）：`BranchMeta` / ancestry 冻结 / `HEAD.json`
  整体废除。branch = `branches/{name}/` 小目录（`ref` + `staging.jsonl`）。
  ancestry 从 BranchMeta 迁入 commit meta。`branch_id`（brn_ ULID）消灭，
  branch 用 name 寻址。
- **staging 归线**（§17）：staging 从 owner 级挪到 branch 级
  （`branches/{name}/staging.jsonl`）。HEAD 文件废除。"切换 branch"操作不存在，
  只有开线 / 延线 / 弃线。同 owner 多线并行合法，锁粒度 owner → branch。
- **commit 自治目录**（§16+§17）：commit 从 `commits/{NNNN}.jsonl` 单文件升级
  为 `commits/{Y-m}/cmt_<ULID>/` 自治目录（meta.json + moments.jsonl +
  notes.jsonl）。NNNN branch 内序号消灭，commit 定序全局化（ULID 字典序 +
  commits.jsonl 行序）。出生即冻结，懒创建。
- **commits/ Y-m 分桶**（§18.1）：`commits/{Y-m}/cmt_<ULID>/`，Y-m 从 ULID
  时间戳纯函数解出，严格 UTC。
- **commits.jsonl 时序日志**（§18.2）：owner 级 append-only commit 日志，
  契约化（崩溃恢复判据依赖）。POSIX O_APPEND 原子写，多 branch 并发安全无 flock。
- **`.cache/` 处决**（§18.3）：commit_id→位置（ULID→Y-m 纯函数）、全局时序
  （commits.jsonl）、last-wins 视图（notes.jsonl 直读）三项职能全部消解。
  moment_id → commit 反查走 grep。
- **`renderings/` 删除**：重绘层归主权层，物理归宿走 §17.3 #4 commit_space
  业务自由空间（本文件 §8）。
- **ref = JSON 元组** `{fork, commit_id[, moment_id]}`（§16.5 #3）：跨 owner
  fork 需要 fork 字段。BasePointer 删除 `branch_id` / `commit_seq`。
- **崩溃恢复精化**（§18.2）：判据从"staging 所有 id 都是 last commit 成员"
  改为"基于 commits.jsonl 尾行 commit_id"。

不变的部分（v1 → v2 显式保留）：信封 `MomentRecord`、payload 透明原则、
trailer 规范 §6、释义 last-wins 定序（同文件字节偏移序）、双孔径（§3.4）、
commit 永不删、时间连续（§3.7）、见证层 sidecar 模型、退化态验收（蠢记忆
无 fork 词汇）、golden 互读字节等价。

---

## 1. 总布局

```
{root}/memento/
  {owner}/                                    # owner = 身份 = 工作目录
    meta.json                                # owner 身份卡 (overlay 等, §4.4)
    commits.jsonl                            # owner 级 append-only 时序日志 (§7)
    branches/
      {name}/                                # branch = 时间线小目录
        ref                                  # JSON 元组 {fork, commit_id[, moment_id]} (§4.1)
        staging.jsonl                        # 本线活边 (§4.2)
    commits/
      {Y-m}/                                 # Y-m 分桶, UTC, 从 ULID 时间戳纯函数 (§5.0)
        cmt_<ULID>/                          # commit 自治目录, 出生即冻结, 懒创建
          meta.json                          # parent + ancestry 跳跃指针 (§5.1)
          moments.jsonl                      # 冻结成员真身 (§5.2)
          notes.jsonl                        # commit_note + moment_note 共居, last-wins (§5.3)
  .git/                                       # 见证层 sidecar repo (§9), root 级
  .gitignore                                  # 至少含: .cache/ (若实现仍产生临时缓存)
```

- `{owner}` 是 owner 命名空间字符串，同时是目录名。MUST 匹配
  `[A-Za-z0-9._\-]{1,64}`。memento 不解释其语义（可以是 cell address、prompt
  文件 stem 或任意约定）。
- `{name}` 是 branch 名，MUST 匹配 `[A-Za-z0-9._\-]{1,64}`，MUST NOT 以 `cmt_`
  开头（与 commit id 前缀碰撞）。memento 不解释 branch 名语义。
- `{Y-m}` 是 4 位年-2 位月（如 `2026-07`），从 `cmt_<ULID>` 的 ULID 时间戳
  部分解出，严格 UTC（§5.0）。
- **单写者纪律**：`{owner}/` 下的所有文件，MUST 只由绑定该 owner 的单一
  Memento 实例写入。跨 owner 只读。释义改写（含旁路发起的）MUST 经由 owner
  实例落盘——格式层不提供多写者协议。
  - **branch 级单写者**：同一 `{owner}/branches/{name}/` 下的文件，MUST 只由
    绑定该 branch 的单一进程/句柄写入。多 branch 并行合法（§17.3 #3），但
    每条线各自单写。
- **commit 自治目录出生即冻结**：`commits/{Y-m}/cmt_<ULID>/` 一经原子 rename
  发布，目录内 `meta.json` / `moments.jsonl` 的已写内容 MUST NOT 再改。
  `notes.jsonl` 是唯一允许追加的文件（释义 last-wins）。
- **无独立 moment 池**：moment 记录的物理归属 = staging 或某个 commit 目录的
  `moments.jsonl`，二者互斥。commit 冻结即把 staging 的 last-wins 视图整块
  搬入 commit 目录，staging truncate。跨 owner checkout 通过 ref 指向他
  owner 的 commit，moment 永不复制。
- **merge 不存在**（§17.3 #5）：跨 owner 交互 = 只读 checkout + Matrix 消息
  （孔径一）。单父链钉死，无冲突语义，无多父 commit。

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

- ULID：26 字符 Crockford base32 大写。**前 10 字符 = 48-bit 毫秒时间戳（UTC）**。
- moment id MUST 非空、匹配 `[A-Za-z0-9._\-]{1,128}`、在该 moment 所属
  branch 的可写范围（staging + 该 branch 自己的 commits）内唯一。祖先 commit
  中的 moment 不参与此唯一性——祖先 moment 经 `(fork, commit_id, moment_id)`
  三元组引用（§4.1），不会与本 branch 的 moment id 空间冲突。
- 前缀 `cmt_` 保留给 memento；grep 任意文本中的 `cmt_` 即得 commit 引用。
- **branch 无 id**：branch 用 `{name}` 寻址，不分配 ULID。branch 的"身份"
  = owner + name 组合。
- **commit_id → Y-m 是纯函数**（§5.0）：`commit_id.removeprefix("cmt_")[:10]`
  Crockford base32 解码为 48-bit 毫秒时间戳，
  `datetime.fromtimestamp(ms/1000, tz=UTC).strftime("%Y-%m")` 即得分桶目录。
  无冗余字段、无索引。

### 2.2 last-wins 定序（钉死）

释义可变、成员不可变。释义的多版本以追加行表达，读时取最新：

- **"last" = 同一文件内更大的字节偏移**（即更靠后的行）。不比时间戳，不比 id。
- 释义行 MUST 追加到**包含所释义对象的同一个文件**：
  - moment 在 staging 时，其覆盖行与 `moment_note` 都写 `branches/{name}/staging.jsonl`；
  - moment 已冻结于某 commit 目录时，其 `moment_note` 追加到该 commit 的
    `notes.jsonl`；
  - commit 的 `commit_note` 一律追加到该 commit 自己的 `notes.jsonl`。
- 跨文件释义非法——这是 last-wins 无歧义的前提。
- 时间戳字段（`ts`）只作展示与诊断，MUST NOT 参与定序。

## 3. Moment 记录（信封）

### 3.1 信封模型（envelope）

memento 不理解 Moment 的内部结构。存储的是 **MomentRecord 信封**：

```json
{"t":"moment","id":"<生产者id>","created":"<RFC3339>","type":"moss.moment/v1","threads":["memento-design"],"payload":{...}}
```

| 字段 | 必选 | 语义 |
|------|:---:|------|
| `t` | ✓ | 恒为 `"moment"` |
| `id` | ✓ | 生产者 id，§2.1 |
| `created` | ✓ | 信封创建时间 |
| `type` | ✓ | payload schema 标识，如 `moss.moment/v1`。schema 归生产者所有 |
| `payload` | ✓ | 任意 JSON object，**memento 原样透传，MUST NOT 解析或改写** |
| `threads` | | 线索标签，写入时标注（可空可缺省），可经 `moment_note` 更新 |

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
- **一旦某 id 随 commit 搬入 commit 目录的 `moments.jsonl`（§5.2），继续向
  staging 追加其 `t:"moment"` 行即契约违规**。写 API MUST 拒绝
  （`MomentFrozenError`）；语义上此时 staging 已无该 id 的槽位——冻结即物理，
  不是 API 层的软约束。
- commit 目录内 `moments.jsonl` 的 `t:"moment"` 行 MUST NOT 出现同一 id 的第二条。
  commit 冻结的是 "冻结时刻该 id 的 staging last-wins 视图"，一次性、不再动。

### 3.3 moment 级释义（`moment_note`）

```json
{"t":"moment_note","ref":"<moment_id>","threads":["a","b"],"ts":"<RFC3339>","by":"tagger.rule"}
```

- `ref` 指向同文件内已有 `t:"moment"` 行的 id：
  - moment 未冻结时，写入 `branches/{name}/staging.jsonl`；
  - moment 已冻结时，追加到该 moment 所在 commit 目录的 `notes.jsonl`。
- 语义：**整体替换**该 moment 的 `threads`（非增量合并）。
- moment 级释义 v2 只开放 `threads` 一个键。payload 永远不可经释义改写。
- **跨 owner 只读**：他 owner 的 commit 冻结的 moment，其 threads 只有该
  commit 的 owner 有权改写（走 owner 实例落盘，孔径二）。外来轨迹对该 commit
  的释义存于它自己的空间（§8 业务自由空间），MUST NOT 写入本 commit 的
  `notes.jsonl`。

## 4. Branch 目录

branch = 时间线小目录：`branches/{name}/ref` + `branches/{name}/staging.jsonl`。
branch 没有自己的 meta.json——身份（owner + name）+ ref（指向 commit）+ staging
（活边）就是它的全部。

### 4.1 ref 文件

JSON 元组，整文件原子写：

```json
{"fork": "ghost.main", "commit_id": "cmt_01J...", "moment_id": "mmt_ab12"}
```

| 字段 | 必选 | 语义 |
|------|:---:|------|
| `fork` | ✓ | 源 owner。本线从哪个 owner 的 commit 出生。本 owner 自身的 commit 时 `fork` = 本 owner |
| `commit_id` | ✓ | 本线当前指向的 commit id |
| `moment_id` | | 可选。给定时，本线从该 commit 的 `[第一个 moment ... moment_id]` 前缀（含）出生 |

- ref 文件创建后，`fork` 与 `commit_id` 的组合 MUST NOT 改变（fork 起点永恒）。
  `commit_id` / `moment_id` 随 `branch reset`（§4.3）移动。
- `moment_id` 给定时 MUST 是该 commit 实际成员之一，否则读取时 MUST 抛错
  （`MomentNotInCommitError`）。**空前缀在类型层不可构造**——"完全不继承"
  用父 commit 表达，不用 `(commit, moment=null)` 构造空切片。
- 切片 **inclusive**：截止 moment 本身被继承者读到，"新故事从这个 moment
  之后开始"。
- **ref 移动前活边先落锚**（§17.3 #2）：任何 `branch reset` 操作 MUST 先
  把当前 staging 冻结为机械 commit（锚在原 ref 位置），然后原子 rewrite ref
  文件。什么都不静默丢。
- root branch（owner 第一个 branch）的 ref：`fork` = 本 owner，`commit_id`
  缺省或 null（表示"无前驱"，本线从零开始）。首次 commit 后 ref 指向该 commit。

### 4.2 staging.jsonl

本线活跃写面。**全格式中唯一允许清空（truncate）的文件**——它是投影，
冻结后的事实真身在 commit 目录里。路径 `branches/{name}/staging.jsonl`。

staging 直接容纳 moment 真身（v1 §4.2 原文保留，路径同步更新）：

- `t:"moment"` 行：同 id 覆盖直接追加，读者按 §2.2 last-wins 取最新版本。
  同一 moment_id 在 staging 中出现次数无上限，但只有 last-wins 视图参与 commit。
- `t:"moment_note"` 行（§3.3）：追加到 staging，`ref` 指向 staging 内的 moment
  id。冻结后此行不再迁移——commit 目录冻结的是 staging last-wins 时刻的 threads
  最终值。
- 首次出现顺序即成员定序：commit 时，`moment_ids` 按每个 id 在 staging 中首次
  出现的行号升序排列（后续覆盖行不改变次序）。

commit 时的原子动作序列（详见 §5 与 §11）：

1. 计算 staging 的 last-wins 视图（可选：截止到 `boundary_moment_id` 的前缀，
   §16.5 #1——只冻结首次出现序 ≤ boundary 的 moments，剩余留在 staging）；
2. 写 tmp 目录 `commits/{Y-m}/cmt_<ULID>.tmp/`：meta.json + moments.jsonl +
   notes.jsonl（含初始 `commit_note`）；
3. fsync tmp 目录内所有文件；
4. 原子 rename tmp → `commits/{Y-m}/cmt_<ULID>/`；
5. append `{owner}/commits.jsonl` 一行 `commit_ref`；
6. fsync commits.jsonl；
7. rewrite `branches/{name}/ref` 指向新 commit；
8. truncate `staging.jsonl`（若 §4.2.1 有 boundary，只 truncate 已冻结部分）。

### 4.2.1 commit 边界参数（§16.5 #1）

`commit()` 接受可选 `boundary_moment_id`：

- 缺省 = 冻结 staging 全部 last-wins 视图。
- 给定时 = 只冻结 staging 中首次出现序 ≤ `boundary_moment_id` 首现序的 moments。
  剩余 moments 留在 staging 继续活，下次 commit 再冻。
- `boundary_moment_id` MUST 是 staging 内实际存在的 moment id，否则抛错。
- git add 的空间性选择被时间性切点吸收——"add" 概念消解。

### 4.3 branch reset（rewind）

移 ref，不改历史。语义见 §17.3 #2 + momento-mori §18.5。

- reset 前 staging 非空时，MUST 先按 §4.2 commit 流程把 staging 冻结为机械
  commit（kind=mechanical），锚在原 ref 位置。该机械 commit 成为孤儿（不被
  任何 ref 指向），但永远可寻址。
- 然后 rewrite ref 文件指向目标 commit。
- 孤儿机械 commit 可经 `commit annotate` 打标"误写，已 reset"——历史诚实，
  意义可补。
- rewind 不违背成员不可变，**恰恰依赖它**——reset 安全正因为旧位置永远可寻址。

### 4.4 owner meta.json

owner 身份卡。整文件重写（非追加）。路径 `{owner}/meta.json`。

```json
{
  "owner": "ghost.main",
  "overlay": {"divergence_prompt": "..."},
  "created": "<RFC3339>",
  "updated": "<RFC3339>"
}
```

| 字段 | 可变性 | 语义 |
|------|--------|------|
| `owner` | 不可变 | owner 名（= 目录名） |
| `overlay` | 创建后不可变 | 化身出生注入物（§16.5 #2）。divergence prompt 等。MUST NOT 进 staging——不属于对话历史 |
| `created` | 不可变 | 创建时间 |
| `updated` | 可变 | 最后更新时间 |

- `overlay` 在 fork 时写入：从父 owner fork 出新 owner 时，overlay 作为出生
  注入物一次性确定，此后不改。
- overlay 字段集开放（divergence_prompt / system_prompt_override / tools_diff
  等），memento 不解释其语义——消费方（ghost / agent）自定。

## 5. Commit 自治目录

路径 `commits/{Y-m}/cmt_<ULID>/`。自治目录，出生即冻结，懒创建——staging
冻结时才 materialize（tmp 目录 + 原子 rename）。无内容永不落盘——checkout
不产生 commit。

### 5.0 Y-m 分桶

`{Y-m}` 从 `cmt_<ULID>` 的 ULID 时间戳部分纯函数解出：

```python
def y_m_of(commit_id: str) -> str:
    ulid_part = commit_id.removeprefix("cmt_")
    ms = crockford_b32_decode(ulid_part[:10])  # 48-bit 毫秒时间戳
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m")
```

- **严格 UTC**：从 ULID 时间戳解出，不从 wall clock、不用本地时区。
- **时钟回拨接受**：Y-m 只是物理位置，逻辑时序由 commits.jsonl（§7）保证。
  时钟回拨可能导致新 commit 落在旧 Y-m 目录，但行序在 commits.jsonl 中仍然
  正确。
- 遍历全局时间序 = `ls commits/`（字典序 = 时间序）+ 逐月 `ls commits/{Y-m}/`
  （ULID 字典序 = 时间序），两层拼接无需 sort。

### 5.1 meta.json

整文件原子写（commit 冻结时一次性写入，此后不改）。

```json
{
  "commit_id": "cmt_01J...",
  "parent": {"fork": "ghost.main", "commit_id": "cmt_01J...", "moment_id": null},
  "ancestry_skips": [
    {"commit_id": "cmt_01J...", "depth": 16},
    {"commit_id": "cmt_01J...", "depth": 256}
  ],
  "branch": "main",
  "kind": "semantic",
  "created": "<RFC3339>"
}
```

| 字段 | 必选 | 语义 |
|------|:---:|------|
| `commit_id` | ✓ | 本 commit 的 id（= 目录名去前缀） |
| `parent` | | 父 commit。root commit（owner 第一个 commit）时缺省。结构 = ref 元组 `{fork, commit_id[, moment_id]}`（§4.1） |
| `ancestry_skips` | | 近端 N 段跳跃指针（§16.1）。N 是主权层常量，FORMAT 不写死。每项 `{commit_id, depth}` 表示"第 depth 代祖先是 commit_id"。寻路 O(d/N) 或 O(log d) |
| `branch` | ✓ | 本 commit 从哪条 branch 冻结而来（诊断用，非结构身份） |
| `kind` | ✓ | `semantic` \| `mechanical`（§6 trailer Kind 的物化） |
| `created` | ✓ | 冻结时间 |

- `parent` 与 `ancestry_skips` 写入时确定，此后 MUST NOT 改（成员不可变保证
  其安全）。
- `ancestry_skips` 是反规范化（可从 parent 链回溯重建），不一致时读者 MUST
  抛错。
- `ancestry_skips` 的具体 N 值与跳跃策略（固定间隔 vs 指数间隔）归主权层，
  FORMAT 沉默。MUST 满足"寻路可达，写入时确定"两条纪律。

### 5.2 moments.jsonl

冻结成员真身。整文件原子写（commit 冻结时一次性写入，此后不改）。

第 1 行：成员行

```json
{"t":"commit","id":"cmt_01J...","moment_ids":["a","b","c"],"created":"<RFC3339>"}
```

第 2..m+1 行：`t:"moment"` 冻结版全文，顺序 MUST 与 `moment_ids` 一致。

- `moment_ids` MUST 非空（空 staging 禁止 commit）。
- 成员行写下后 MUST NOT 出现第二条 `t:"commit"` 行。成员不可变是 fork 边界的
  前提：动它，所有子 branch 的 ancestry 集体作废。
- 每个 `t:"moment"` 行的 id MUST 出现在同文件 `moment_ids` 中；反之亦然
  （一一对应，禁止漏行漏 id）。
- 这些行是 staging last-wins 视图的物理搬运结果，字段结构与 §3.1 完全一致。
- commit 目录内 `moments.jsonl` 的 `t:"moment"` 行 **不参与 last-wins**——
  冻结即定型，同 id 不再出现。moment 级 threads 的后续更新走 §5.3 的
  `moment_note`。

### 5.3 notes.jsonl

commit_note + moment_note 共居，append-only，last-wins。

commit 释义：

```json
{"t":"commit_note","ref":"cmt_01J...","title":"一行摘要","body":"<正文+trailer, §6>","ts":"<RFC3339>","by":"ghost.main"}
```

- `commit_note.title`: 一行摘要。用于窗口渲染和模型搜索。可选，缺省为空串。

moment 级释义：

```json
{"t":"moment_note","ref":"<moment_id>","threads":["a","b"],"ts":"<RFC3339>","by":"tagger.rule"}
```

- `commit_note.ref` MUST 等于本 commit id。
- `moment_note.ref` MUST 是同 commit `moments.jsonl` 中某 `t:"moment"` 行的 id。
- commit 动作 MUST 在 moments.jsonl 写完后、目录 rename 前，向 notes.jsonl
  追加一条初始 `commit_note`（机械 commit 的 body 可只有 trailer 块，见 §6）。
- 两种 note 类型字段结构不同，解析路径独立，无需按 ref 前缀区分。读者遇到
  未知 `t:` 值按 §2 跳过，前向兼容自然满足。
- **渲染打戳**：任何把释义内容展示给模型的渲染，MUST 可追溯到所读的释义版本
  （实现记录 `(commit_id, note_seq)` 或等价物）。"当时模型看见什么"必须可重建。
- **释义跟随轨迹**（§17.3 #5）：owner 自己的释义走本 commit 的 `notes.jsonl`
  （孔径二，append-only last-wins，无锁）。外来轨迹对该 commit 的 summary
  存在它自己的空间（§8），MUST NOT 写入本 commit 的 `notes.jsonl`。

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

**v2 注册的 trailer key**：

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

## 7. owner 级 commits.jsonl（时序日志）

owner 级 append-only commit 日志。路径 `{owner}/commits.jsonl`。**契约化**
（非可选索引）——崩溃恢复判据依赖它（§11）。

行格式：

```json
{"t":"commit_ref","commit_id":"cmt_01J...","branch":"main",
 "parent":{"fork":"...","commit_id":"..."[,"moment_id":"..."]},
 "ts":"<RFC3339>","kind":"semantic"}
```

| 字段 | 必选 | 语义 |
|------|:---:|------|
| `t` | ✓ | 恒为 `commit_ref` |
| `commit_id` | ✓ | 本行记录的 commit id |
| `branch` | ✓ | 从哪条 branch 冻结（诊断用） |
| `parent` | | 父 commit 元组（与 commit meta.json 的 parent 一致） |
| `ts` | ✓ | 冻结时间（展示用，不参与定序） |
| `kind` | ✓ | `semantic` \| `mechanical` |

- **append-only**：MUST 只追加，MUST NOT 改写已有行。
- **行序 = 物理时序**：POSIX `O_APPEND` 写 < PIPE_BUF(4096B) 的行原子，多
  branch 并发 append 无需 flock——branch 级锁粒度（§17.3 #3）不破。
- **派生层**：可从 `commits/{Y-m}/cmt_<ULID>/meta.json` 全量扫描重建，但
  崩溃恢复判据（§11）依赖它，所以契约化。
- `owner log` CLI 命令的数据源（momento-mori §18.4）。

## 8. commit_space 与业务自由空间

`commits/{Y-m}/cmt_<ULID>/` 目录是 commit 的"空间"，经 `commit_space(commit_id)`
API 运行时解析为路径。保留名单（契约管）：

- `meta.json`（§5.1）
- `moments.jsonl`（§5.2）
- `notes.jsonl`（§5.3）

保留名单之外，契约**沉默**——不感知、不承诺、不禁止。业务（重绘渲染、ground
场快照、跨轨迹释义文档、link 等）可在 commit 目录内自由创建文件/子目录，变动
历史由见证层（§9）兜底。

- memento 内部存储的一切引用只许用 id 元组（`{fork, commit_id[, moment_id]}`），
  **出现绝对路径即契约违规**——memento 可跨项目分享，绝对路径无意义。
- `commit_space()` 是运行时 API，返回的 path 不进 memento 任何持久化结构。
- 重绘层（v1 §8 renderings/）的物理归宿由此节承载——重绘渲染作为 commit
  目录内的业务自由文件存在，"旧投影永不删除"的纪律由业务自守（契约不再管
  renderings/ 目录）。

## 9. 见证层（git sidecar）

git 不是 memento 的结构，是**见证**：fork 是纯 memento 层操作，git 只见证文件，
不知道 fork 存在。

- repo 位于 `{root}/memento/.git`（**root 级**，覆盖全部 owner），工作树即
  `memento/`。owner 隔离是 memento 层的事，git 层退化为单写者（旁路 daemon）。
  **MUST NOT 被外层代码仓库吞掉**——memento root 必须被外层仓库 ignore，或位于
  仓库之外。两个时间尺度串扰是本架构唯一真正的污染模式。
- `memento/.gitignore` MUST 至少含 `.cache/`（若实现仍产生临时缓存）。
- 快照由**单写者旁路 daemon** 执行（memento commit 事件触发或定时），
  MUST NOT 出现在任何热路径。v2 实现用 subprocess git；不引 C 依赖。
- 快照 commit message 格式：

  ```
  snapshot: <RFC3339 时刻>

  Memento-Ref: cmt_01JAAA...
  Memento-Ref: cmt_01JBBB...
  ```

  正文 trailer 列出自上次快照以来新增的 memento commit id（可从 commits.jsonl
  tail 自上次 offset 增量获取）。由此 `git log --grep=cmt_xxx` 反查任意 commit
  首次被见证的时刻。
- 两个地址空间：memento id = 身份（这是哪个 commit）；git sha = 完整性
  （历史未被事后篡改的证明）。
- Matrix 跨机复制直接复用见证 repo push/pull（owner 分片路径不碰撞，无冲突）。
  MUST NOT 为 memento 另造同步协议。
- 见证层是可选组件：不启用时其余格式条款全部照常成立。

## 10. 不变量清单（golden tests 锚点）

1. **commit 自治目录出生即冻结**：`meta.json` / `moments.jsonl` 一经原子
   rename 发布，已写内容 MUST NOT 改。`notes.jsonl` 是唯一允许追加的文件。
2. **staging 归线**：staging 路径 MUST 是 `branches/{name}/staging.jsonl`，
   不存在 owner 级 staging。"切换 branch"操作不存在，只有开线/延线/弃线。
3. **branch ref 移动前活边先落锚**：`branch reset` 时 staging 非空 MUST 先
   冻结为机械 commit，然后移 ref。什么都不静默丢。
4. **commit 永不删**：commit 目录一经发布 MUST NOT 删除（即使成为孤儿）。
5. **释义 last-wins = 同文件字节偏移序**：释义行与所释义对象同文件。
   staging 内 moment 的释义在 staging；冻结 moment / commit 的释义在其 commit
   目录的 `notes.jsonl`。
6. **payload 原样透传**：`abc.py` 与存储实现零 payload schema 依赖（§3.1）。
7. **commit_id → Y-m 纯函数**：从 ULID 时间戳解出，严格 UTC，无冗余字段、
   无索引。`commit_space(commit_id)` 经此函数定位（§5.0）。
8. **commits.jsonl append-only，行序 = 物理时序**：多 branch 并发 append
   无 flock。崩溃恢复判据依赖它（§7、§11）。
9. **ref = JSON 元组** `{fork, commit_id[, moment_id]}`：`fork` + `commit_id`
   组合创建后不改；`moment_id` 给定时 MUST 命中 commit 成员，空前缀在类型
   层不可构造（§4.1）。
10. **撕裂尾行跳过，中段损坏抛错**（§2）。
11. **见证 repo 独立于代码仓库**，快照 message 携带 `Memento-Ref:`（§9）。
12. **互读等价**：两个独立实现照本文件各写一份历史，互读对方字节，重建出的
    历史（commit 序列、成员、每个 commit 内的 moment 全文序列、当前释义、
    ancestry）等价。
13. **退化态纯净**：单 branch + 自动 commit 的用例代码中，fork 相关词汇
    （fork/checkout/ancestry/overlay/parent/moment_id 切片）一个不出现。
14. **冻结即物理**（§3.2）：某 moment id 被搬入 commit 目录后，staging 中 MUST
    无同 id 的可写槽位——冻结检查是 API 拒绝之上的结构约束。
15. **恢复幂等**（§11）：任何时刻中断，按 §11 恢复规则重放后系统状态与
    "无中断完成"等价，不产生重复 commit 或悬空 staging。
16. **绝对路径禁用**：memento 持久化结构（ref / commits.jsonl / commit meta /
    notes 等）中出现绝对路径即违规。`commit_space()` 返回的 path 是运行时值，
    不进任何持久化结构。
17. **merge 不存在**：单父链钉死，无多父 commit。跨 owner 交互走只读 checkout
    + Matrix 消息（§1）。

## 11. 崩溃恢复

commit() 原子动作序列（§4.2 复述）：

1. 计算 staging last-wins 视图（可选截止到 boundary）；
2. 写 tmp 目录（meta.json + moments.jsonl + notes.jsonl 含初始 commit_note）；
3. fsync tmp 目录内所有文件；
4. 原子 rename tmp → `commits/{Y-m}/cmt_<ULID>/`；
5. append `{owner}/commits.jsonl` 一行 commit_ref；
6. fsync commits.jsonl；
7. rewrite `branches/{name}/ref` 指向新 commit；
8. truncate `staging.jsonl`。

中间任意点崩溃，恢复规则（基于 commits.jsonl 尾行判据，§18.2）：

- **commits.jsonl 尾行 commit_id 的目录不存在**：append commits.jsonl 后、
  rename 前 crash 的残骸。MUST 删除 commits.jsonl 尾行（或截断到尾行前），
  然后重跑 commit。
- **commits.jsonl 尾行 commit_id 的目录存在但残缺**（meta.json 缺失或撕裂）：
  MUST 删除该目录 + 删除 commits.jsonl 尾行，重跑 commit。
- **commits.jsonl 尾行 commit_id 的目录完整，但 ref 未更新 / staging 未
  truncate**：commit 已成立（fsync 后崩溃），MUST 直接 rewrite ref 指向该
  commit_id + truncate staging。本次已完成，不重跑。
- **commits.jsonl 尾行 commit_id 的目录完整，ref 已更新，但 staging 未
  truncate**：同上，truncate staging 即可。

恢复 MUST 在 Memento 装入 branch handle 时（或首次访问 staging / ref 时）
执行。规则幂等：多次触发 = 一次触发。

---

历史：
- 2026-07-11 由 claude-fable-5 起草 v1。
- 2026-07-18 由 claude-opus-4-7 修订：§14 存储布局落地——`moments/` 池整体废除，
  staging 持真身、commit 文件自包含；新增 §4.1.1 BasePointer moment 前缀切片、
  §5.2 冻结 moment 行、§5.3 note 类型二分、§12 崩溃恢复；相应更新 §11 不变量清单。
  依据 workstreams/2026/06/momento-mori/FEATURE.md §14。
- 2026-07-20 由 kimi-k3 起草 v2：§16 branch 降维 + §17 时间线原生化 +
  §18 CLI 定案 / Y-m 分桶 / commits.jsonl 契约化 / .cache/ 处决 / ref JSON 元组
  / 崩溃恢复精化。依据 workstreams/2026/06/momento-mori/FEATURE.md §16/§17/§18。
