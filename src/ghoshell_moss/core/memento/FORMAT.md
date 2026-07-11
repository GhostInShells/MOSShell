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
  moments/{owner}/{YYYY-MM}/moments.jsonl     # Moment 池, per-owner 分片, 追加为主
  branches/{owner}/HEAD.json                  # 该 owner 的 current branch 指针
  branches/{owner}/{branch_id}/
    meta.json                                 # BranchMeta, 含冻结祖先链
    staging.jsonl                             # 活跃写面, 唯一允许清空的文件
    commits/{NNNN}.jsonl                      # 每 commit 一个文件: 成员行 + 释义行
  renderings/                                 # 重绘投影, 自由格式, 只增不删
  .cache/                                     # 可再生索引, 删掉重扫, 见证层忽略
  .git/                                       # 见证层 sidecar repo (§9)
  .gitignore                                  # 至少包含: .cache/
```

- `{owner}` 是 owner 命名空间字符串，同时是目录名。MUST 匹配
  `[A-Za-z0-9._\-]{1,64}`。memento 不解释其语义（可以是 cell address 或任意约定）。
- **单写者纪律**：`moments/{owner}/` 与 `branches/{owner}/` 下的所有文件，
  MUST 只由绑定该 owner 的单一 Memento 实例写入。跨 owner 只读。
  释义改写（含旁路发起的）MUST 经由 owner 实例落盘——格式层不提供多写者协议。

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
- moment id MUST 非空、匹配 `[A-Za-z0-9._\-]{1,128}`、在同一 owner 池内唯一。
- 前缀 `cmt_` / `brn_` 保留给 memento；grep 任意文本中的 `cmt_` 即得 commit 引用。

### 2.2 last-wins 定序（钉死）

释义可变、成员不可变。释义的多版本以追加行表达，读时取最新：

- **"last" = 同一文件内更大的字节偏移**（即更靠后的行）。不比时间戳，不比 id。
- 一个对象的释义行 MUST 追加到**包含其成员行/记录行的同一个文件**。
  跨文件释义非法——这是 last-wins 无歧义的前提。
- 时间戳字段（`ts`）只作展示与诊断，MUST NOT 参与定序。

## 3. Moment 池

### 3.1 信封模型（envelope）

memento 不理解 Moment 的内部结构。池中存储的是 **MomentRecord 信封**：

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
| `threads` | | 线索标签，写入时标注（可空可缺省），可经释义行更新 |
| `by` | | 写入者标识（模型名/规则 id），诊断用 |

- payload 对 memento 不透明是硬边界：`abc.py` 与存储实现 MUST NOT import 任何
  payload schema（包括 `ghoshell_moss` 的 `Moment`/`Message`）。强类型编解码
  （codec）在信封之上、作为独立模块存在。
- 未来 Moment 包剥离的窗口即在此：信封层零依赖，天然可剥。

### 3.2 记录行的可变性：冻结前可覆盖，冻结后不可

Moment 在一轮交互内是渐进构建的（感知先到、logos 后到）。因此：

- 同一 `id` 的 `t:"moment"` 行 MAY 在池中出现多次，读者按 §2.2 last-wins 取最新
  ——每次覆盖写都是一个新对象共享同一 id（"更新即新对象"）。
- 覆盖行 MUST 追加到该 id 首次出现的同一文件（§2.2）。
- **一旦某 id 成为任何 commit 的成员（§5），继续追加其 `t:"moment"` 行即契约
  违规**。写 API MUST 拒绝；读者仍按 last-wins 容错。commit 冻结的是
  "冻结时刻该 id 的最新版本"。

### 3.3 释义行（moment 级）

```json
{"t":"note","ref":"<moment_id>","threads":["a","b"],"ts":"<RFC3339>","by":"tagger.rule"}
```

- `ref` 指向同文件内已有记录行的 id。
- 语义：**整体替换**该 moment 的 `threads`（非增量合并）。
- moment 级释义 v1 只开放 `threads` 一个键。payload 永远不可经释义改写。

### 3.4 分片

- 路径 `moments/{owner}/{YYYY-MM}/moments.jsonl`，`YYYY-MM` 取
  **`created` 换算到 UTC** 后的年月（消除时区歧义）。
- 覆盖行与释义行跟随记录行所在文件（§2.2），即使写入时已跨月。
- 读者定位 id 所在文件依赖索引（§7）；索引缺失时全量扫描重建。

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
  "base": {"fork":"...","branch_id":"brn_...","commit_id":"cmt_...","commit_seq":7},
  "ancestry": [
    {"fork":"...","branch_id":"brn_...","commit_id":"cmt_...","commit_seq":3},
    {"fork":"...","branch_id":"brn_...","commit_id":"cmt_...","commit_seq":7}
  ],
  "overlay": {"divergence_prompt": "..."},
  "created": "<RFC3339>",
  "updated": "<RFC3339>"
}
```

| 字段 | 可变性 | 语义 |
|------|--------|------|
| `branch_id` `fork` `created` | 不可变 | 身份 |
| `base` | 不可变 | fork 起点。null/缺省 = root branch |
| `ancestry` | 不可变 | **冻结的展平祖先链**，自最老祖先到直接 base，顺序排列。root branch 为空数组。fork 时刻一次性计算写入（= 父的 ancestry + 父的 base 条目），此后 MUST NOT 改写。回溯 O(d)→O(1) 的依据 |
| `overlay` | 创建后不可变 | 化身 divergence prompt 等出生注入物的家。**不属于对话历史，MUST NOT 进 staging** |
| `name` `title` `description` `updated` | 可变 | 释义性字段 |

- `ancestry` 的最后一项 MUST 等于 `base`（有 base 时）。
- 校验：读者发现 `ancestry` 与沿 `base` 链实际回溯结果不一致时 MUST 抛错
  （冻结链是反规范化，成员不可变保证其安全；不一致 = 数据被篡改或写入 bug）。

### 4.2 staging.jsonl

活跃写面。**全格式中唯一允许清空（truncate）的文件**——它是投影，
冻结后的事实真身在 commit 文件里。

```json
{"t":"stage","moment_id":"<id>","ts":"<RFC3339>"}
```

- 盲追加：同一 moment_id MAY 重复出现。读者去重规则 MUST 为：
  **保留首次出现的位置序**（后续重复行只表示池中有覆盖写，不改变次序）。
- commit 时冻结当前去重序列 → 写 commit 文件 → truncate staging。

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

### 5.1 成员行（第一行，不可变）

```json
{"t":"commit","id":"cmt_01J...","seq":12,"moment_ids":["a","b","c"],"created":"<RFC3339>"}
```

- MUST 是文件第一行。`moment_ids` 是 staging 冻结时刻的去重有序列表，MAY 为空
  以外——MUST 非空（空 staging 禁止 commit）。
- 成员行写下后 MUST NOT 出现第二条 `t:"commit"` 行。成员不可变是 fork 边界的
  前提：动它，所有子 branch 的 ancestry 集体作废。

### 5.2 释义行（追加，last-wins）

```json
{"t":"note","ref":"cmt_01J...","body":"<正文+trailer, §6>","ts":"<RFC3339>","by":"ghost.main"}
```

- `ref` MUST 等于本文件成员行的 id。
- `body` 整体替换语义：读者取最后一条 note 的 body 为当前释义，
  **历史版本永远可寻址**（按行序枚举即全部版本）。
- commit 动作 MUST 同时写入成员行 + 初始释义行（机械 commit 的 body 可以只有
  trailer 块，见 §6）。
- **渲染打戳**：任何把释义内容展示给模型的渲染，MUST 可追溯到所读的释义版本
  （实现记录 `(commit_id, 行号)` 或等价物）。"当时模型看见什么"必须可重建。

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

- 索引（moment_id → 文件+偏移、commit_id → branch+seq、释义 last-wins 视图等）
  MUST 全部放在 `{root}/memento/.cache/` 下。
- **jsonl 是唯一 truth，索引是可再生缓存**：删除 `.cache/` 后系统 MUST 能全量
  重扫重建，行为不变。
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

1. 成员行/记录行一经 commit 冻结即不可变（§3.2、§5.1）。
2. 释义 last-wins = 同文件字节偏移序，且释义行与成员行同文件（§2.2）。
3. payload 原样透传，字节不动；`abc.py` 与存储实现零 payload schema 依赖（§3.1）。
4. ancestry 冻结且与 base 链回溯一致（§4.1）。
5. staging 是唯一可 truncate 的文件（§4.2）。
6. 空 staging 禁止 commit；commit 原子动作 = 成员行 + 初始释义行（含 `Kind:`）
   + truncate staging（§5）。
7. 撕裂尾行跳过，中段损坏抛错（§2）。
8. 删除 `.cache/` 后行为不变（§7）。
9. renderings 与旧投影永不删除（§8）。
10. 见证 repo 独立于代码仓库，快照 message 携带 `Memento-Ref:`（§9）。
11. **互读等价**：两个独立实现照本文件各写一份历史，互读对方字节，重建出的
    历史（commit 序列、成员、当前释义、ancestry）等价。
12. **退化态纯净**：单 branch + 自动 commit 的用例代码中，fork 相关词汇
    （fork/checkout/ancestry/overlay/base）一个不出现。

---

历史：2026-07-11 由 claude-fable-5 起草 v1（依据 FEATURE.md 2026-07-08 版契约条款）。
待人类 review 冻结。修订必须过 review 并在此留痕。
