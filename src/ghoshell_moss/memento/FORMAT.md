# Memento FORMAT v3

磁盘格式契约。本文件是 memento 的**契约层**交付物之一（另两件：`abc.py`、golden
tests），人类 review 后冻结。实现层（存储 / 索引 / 见证 daemon）是可丢弃的；
本文件不是。**实现与本文件冲突时，以本文件为准；要改本文件，必须过人类 review。**

- MUST / MUST NOT：违反即实现 bug，golden tests 直接锚定这些条款。
- SHOULD：推荐约定，违反需在实现处注释理由。
- 设计动机不在此展开，见 feature discuss 目录。

## 0. v2 → v3 变更摘要

- **uid 与 name 分离**（§1）：branch 新增稳定标识 brn_ ULID。name 是 movable
  pointer（heads/ 文件），uid 拥有工作区与轨迹。rewind 退位给 fork — 向后看的
  唯一合法动作是"读旧锚点"或"从旧锚点分叉"。
- **动态工作区 ws/{uid}/**（§4.3）：branch 的动态状态（ref / staging / status /
  task 产物）从 branches/{name}/ 迁移到 ws/{uid}/。name 仅保留为 heads 指针。
- **owner 级 append-only 索引文件**（§7–§9）：新增 branches.jsonl（全量 branch
  索引）、checkouts.jsonl（fork 事件）、confluents.jsonl（引用式 confluent 融汇）。
- **MomentRecord +content**（§2.2, §4.2）：moment 行新增 `content` 字段（纯文本投影）。
- **merge 不存在 → confluent 融汇**：原 §1 的"merge 不存在"条款废除，替换为
  confluents.jsonl — 引用式 confluent，commit parent 链不动，独立 append-only 事件。
- **rewind 废除**：CLI branch reset 删除或降级为 `branch create --from` 的别名。
  ref 移动前的自动机械 commit（原 §4.1 #3 条款）随之废除 — fork 是诚实的动作。
- **Line / Memento API 对齐**：新增 BranchMeta、CheckoutRecord、ConfluentRecord
  模型；Line 暴露 `branch_identifier` 属性；Memento 新增 `list_all_branches()`、
  `checkouts()`、`confluents()` 方法；`reset_line()` 删除。

不变的部分（v2 → v3 显式保留）：commit 自治目录 + 出生即冻结 + Y-m 分桶、
commits.jsonl 契约化、payload 透明原则、trailer 规范 §6、释义 last-wins 定序、
双孔径、commit 永不删、时间连续、见证层 sidecar 模型、退化态验收、
golden 互读字节等价、通用行格式（§2）、ref = JSON 元组（§4.1）。
`.cache/` 及其相关机制不恢复。

---

## 1. 总布局

```
{root}/
  {owner}/                                    # owner = 身份 = 工作目录
    meta.json                                # owner 身份卡 (§5.0)
    commits.jsonl                            # owner 级 append-only 时序日志 (§6)
    branches.jsonl                           # owner 级 branch 索引, append-only (§7)
    checkouts.jsonl                          # fork 事件记录, append-only (§8)
    confluents.jsonl                         # 引用式 confluent 事件, append-only (§9)
    heads/
      {name}                                 # 纯文本: branch_uid, 一行无换行 (§4.1)
    ws/
      {branch_uid}/                          # branch 动态工作区 (§4.3)
        ref                                  # JSON 元组 {fork, commit_id[, moment_id]} (§4.2)
        staging.jsonl                        # 活边: 未冻结 moments (§4.4)
        status.json                          # 生命周期 + task 状态 (§4.5)
    commits/
      {Y-m}/                                 # Y-m 分桶, UTC, ULID 时间戳纯函数 (§5.1)
        cmt_<ULID>/                          # commit 自治目录, 出生即冻结, 懒创建
          meta.json                          # parent + 跳跃指针 (§5.2)
          moments.jsonl                      # 冻结成员真身 (§5.3)
          notes.jsonl                        # commit_note + moment_note 共居 (§5.4)
  .git/                                       # 见证层 sidecar repo (§10), root 级
  .gitignore                                  # 至少含临时缓存目录 (若实现产生)
```

### 1.1 命名与约束

- `{owner}`: owner 命名空间字符串，同时是目录名。MUST 匹配 `[A-Za-z0-9._\-]{1,64}`。
  memento 不解释其语义。
- `{name}`: branch head 名，MUST 匹配 `[A-Za-z0-9._\-]{1,64}`。MUST NOT 以 `cmt_`
  或 `brn_` 开头（与 commit / branch id 前缀碰撞）。name 是 movable pointer —
  可 rename、可抢占（指向另一个 uid）、可 `-D`（不删 workspace）。
- `{branch_uid}`: branch 稳定标识，MUST = `brn_<ULID>`。终生不变。workspace 目录
  名即 uid，O(1) 寻址（纯函数拼接，无需索引）。
- `{Y-m}`: 4 位年-2 位月（如 `2026-08`），从 `cmt_<ULID>` 的 ULID 时间戳部分解出，
  严格 UTC（§5.1）。

### 1.2 写者纪律

- **owner 级单写者**：`{owner}/` 下的所有文件，MUST 只由绑定该 owner 的单一
  Memento 实例写入。跨 owner 只读。释义改写 MUST 经由 owner 实例落盘。
- **branch 级单写者**：同一 `ws/{branch_uid}/` 下的文件，MUST 只由绑定该 branch
  的单一进程/句柄写入。多 branch 并行合法，但每条线各自单写。
- **heads/ 写者**：head 文件的读写与 ws/{branch_uid}/ 使用同一锁粒度 — 更新 head
  前持有对应 branch 的写锁。
- **commit 自治目录出生即冻结**：一经原子 rename 发布，`meta.json` / `moments.jsonl`
  的已写内容 MUST NOT 再改。`notes.jsonl` 是唯一允许追加的文件。

---

## 2. 通用行格式

（同 v2，重申）

1. 编码 UTF-8，无 BOM。`ensure_ascii=False`。
2. 每行一个 JSON object，行终止符 LF（`\n`）。MUST NOT 使用 CRLF。
3. 正文换行以 JSON 标准转义（`\n`）。一条逻辑记录 = 一个物理行。
4. 每行首字段 MUST 是判别符 `"t"`。读者遇到未知 `t` 值 MUST 跳过该行（前向兼容）。
5. **撕裂尾行**：文件最后一行 JSON 解析失败时，读者 MUST 静默跳过。非最后一行
   解析失败 = 数据损坏，MUST 抛错，MUST NOT 静默跳过。
6. 读者 MUST NOT 依赖 JSON key 顺序。写者 SHOULD 按 schema 声明序输出。
7. 时间字段一律 RFC 3339 字符串，MUST 带时区偏移。naive datetime 非法。
8. 可选字段缺省时 SHOULD 省略不写。

### 2.1 id 规则

| 对象 | 格式 | 生成方 |
|------|------|--------|
| moment record | 生产者自带 id 原样透传 | 生产者 |
| commit | `cmt_<ULID>` | memento |
| branch | `brn_<ULID>` | memento |

- ULID：26 字符 Crockford base32 大写。**前 10 字符 = 48-bit 毫秒时间戳（UTC）**。
- moment id MUST 非空、匹配 `[A-Za-z0-9._\-]{1,128}`、在该 branch 的可写范围内唯一。
- 前缀命名空间：`cmt_` = commit、`brn_` = branch、`mmt_` = moment（推荐，非强制）。
  跨类型前缀碰撞在设计上接受 — 寻址语境中 type prefix disambiguates。

### 2.2 moment 行 schema（staging.jsonl 与 moments.jsonl 共用）

```json
{"t":"moment", "id":"mmt_<ULID>", "created":"<RFC3339>", "type":"<payload_schema>",
 "content":"<plain_text_projection>", "payload":{...}, "threads":[...]}
```

| 字段 | 类型 | 必须 | 描述 |
|------|------|------|------|
| `t` | string | MUST | 判别符 `"moment"` |
| `id` | string | MUST | moment id |
| `created` | string | MUST | RFC 3339 时间戳 |
| `type` | string | MUST | payload schema 标识 |
| `content` | string | MUST | 纯文本投影。v3 新增。可为空串。由记录方填入。 |
| `payload` | object | MUST | 不透明 JSON object。memento 不解析。 |
| `threads` | [string] | MUST | 线索标签。可空列表。 |

`content` 字段是契约字段（非软约定）。动机：使 CLI `branch window`、commit show
等结构视图能渲染人类可读输出而不依赖 payload codec。空 content 合法 — 有些 moment
没有天然文本表示（如纯工具调用帧）。

---

## 3. owner 级文件

### 3.0 meta.json

owner 身份卡。JSON object。已知字段：
- `overlay`: 化身出生注入物，`create_line(from_ref=cross_owner, overlay=...)` 时写入。

MUST tolerate unknown fields（前向兼容）。不在本文件定义完整 schema — 由
`abc.py` 的对应模型承载。

### 3.1 commits.jsonl — owner 级时序日志

（v2 §7，保留。行 schema 不变，branch 字段语义更新为 uid。）

```json
{"t":"commit_ref", "commit_id":"cmt_<ULID>", "branch_uid":"brn_<ULID>",
 "parent":{"fork":"...", "commit_id":"cmt_...", "moment_id":"mmt_..."|null},
 "ts":"<RFC3339>", "kind":"semantic|mechanical"}
```

- `branch_uid`: v3 从 name 字符串改为 branch uid（brn_ 前缀）。v2 兼容：
  读者对不含 brn_ 前缀的历史行，应以 name 查找当前 heads/ 解析；若 name 已删，
  记 warning 并保留原字符串作诊断。
- 时序物理保证：POSIX O_APPEND 写 < PIPE_BUF (4096B) 的行原子。
- **崩溃恢复判据依赖此文件**：commit() 原子写 = commit 目录 fsync → append
  commits.jsonl → fsync → truncate staging。恢复判据：属于 commits.jsonl
  尾行 commit_id 的 staging 残留 → truncate；否则保留。

## 4. branch 体系

### 4.1 heads/{name} — head pointer

纯文本文件，一行无换行符，内容 = branch_uid（`brn_<ULID>`）。无 JSON wrapper。

- glob `heads/*` = 活跃 branch 列表（O(1) 发现，无需索引）。
- 删除 head 文件 = `-D`（删 name，不删 workspace）。
- head 文件内容是纯 text 而非 JSON 是刻意设计 — `cat heads/main` 直读，
  script 中 `$(cat heads/main)` 即得 uid。
- 写者 MUST 以原子 rename 写入（写 tmp → fsync → rename），保证读者不会读到半截 uid。

### 4.2 ws/{branch_uid}/ref — branch pointer

单行 JSON object，schema = BranchRef：

```json
{"fork":"<owner>", "commit_id":"cmt_<ULID>", "moment_id":"mmt_<ULID>"|null}
```

| 字段 | 类型 | 必须 | 描述 |
|------|------|------|------|
| `fork` | string | MUST | 目标 commit 所属 owner。同 owner 时可为空串 `""`。 |
| `commit_id` | string | MUST | 目标 commit id。 |
| `moment_id` | string | MAY | 切片截止 moment id（含）。null = 整个 commit。 |

- fork 为 `""` 时表示同 owner — 消费者以当前 owner 解析。
- 原子 rename 写入（同 head 文件）。
- 首次 commit 后 ref 更新指向新 commit（开线时 ref = from_ref；第一次 commit 后
  ref.commit_id = 新 commit）。

### 4.3 ws/{branch_uid}/ — 动态工作区

branch 的动态状态全部在此目录。保留名单（ref / staging.jsonl / status.json）之外，
契约不感知、不承诺、不禁止 — 业务自由放置 PLAN.md / todo 文件 / ground 快照 / link
等产物。变动历史由见证层兜底（git 拍下家具搬动）。这条"契约沉默自由空间"条款
与 §8 commit_space 的自由空间条款同构。

### 4.4 ws/{branch_uid}/staging.jsonl — 活边

行类型：`t:"moment"`（§2.2 schema）。同 id 覆盖直接追加，读者取 last-wins（同文件
内后出现者胜）。

- commit() 冻结时：读 staging last-wins、写入 commit 目录的 moments.jsonl、
  fsync commit 目录、append commits.jsonl、**再** truncate staging（§11 原子锚点）。
- staging truncate 后同 id 的 moment 重新可写（旧版本已冻结在 commit 目录）。
- **staging 永远不可作为出生点**：化身只能从 commit 出生, staging 无稳定 id。

### 4.5 ws/{branch_uid}/status.json — 生命周期与 task 状态

```json
{"status":"active|frozen|abandoned", "title":"...", "description":"...",
 "updated":"<RFC3339>"}
```

| 字段 | 类型 | 必须 | 描述 |
|------|------|------|------|
| `status` | string | MUST | 生命周期：active / frozen / abandoned |
| `title` | string | SHOULD | 人类可读标题，一行 |
| `description` | string | MAY | task 描述或当前 plan 摘要 |
| `updated` | string | MUST | RFC 3339 时间戳 |

- 此文件是**原地覆写**（非 append-only）— branch workspace 是动态态。
- status 变更时需同步追加一行到 branches.jsonl（§7）。
- 含义：active = 使用中；frozen = 完成/闭合，只读；abandoned = 丢弃但保留轨迹。

---

## 5. commit 体系

（v2 §5 保留，少量修改。）

### 5.1 commits/{Y-m}/ — Y-m 分桶

commit_id → Y-m 是纯函数：ULID 前 10 字符 = 48-bit 毫秒时间戳（Crockford base32），
解码取 UTC 年月。严格 UTC — 跨时区/跨项目分享无歧义。时钟回拨接受：Y-m 只是物理
位置，逻辑时序由 commits.jsonl 保证，两者解耦。

### 5.2 commits/{Y-m}/cmt_<ULID>/meta.json

```json
{"commit_id":"cmt_<ULID>", "created":"<RFC3339>",
 "parent":{"fork":"...", "commit_id":"cmt_...", "moment_id":"mmt_..."|null},
 "kind":"semantic|mechanical"}
```

parent = BranchRef JSON 元组（同 §4.2）。None (JSON null) = root commit。
**单父链钉死** — 写入后永不修改。

### 5.3 commits/{Y-m}/cmt_<ULID>/moments.jsonl

冻结的 moment 行，schema 同 §2.2。写者从 staging last-wins 视图整体搬运。
行序 = staging 首次出现序。此文件出生即冻结，MUST NOT 追加或修改。

### 5.4 commits/{Y-m}/cmt_<ULID>/notes.jsonl

混合行类型，apply-only，last-wins by ref：

```json
{"t":"commit_note", "ref":"cmt_<ULID>", "title":"...", "body":"...",
 "ts":"<RFC3339>", "by":"..."}

{"t":"moment_note", "ref":"mmt_<ULID>", "threads":[...],
 "ts":"<RFC3339>", "by":"..."}
```

- 同 ref 多条 = 后出现者胜（last-wins）。
- commit_note 和 moment_note 独立 last-wins（不同 ref 类型，不互相覆盖）。
- 此文件是 commit 目录内唯一允许追加的文件。

---

## 6. commits.jsonl — owner 时序日志

（同 v2 §7 / 本文件 §3.1。本节保留作交叉引用锚点，schema 见 §3.1。）

---

## 7. branches.jsonl — owner 级 branch 索引

append-only。Appended on branch creation and on every status change。

```json
{"t":"branch_meta", "uid":"brn_<ULID>", "name":"<current_head_name>",
 "status":"active|frozen|abandoned",
 "fork_ref":{"fork":"...", "commit_id":"cmt_...", "moment_id":"mmt_..."|null},
 "created":"<RFC3339>", "updated":"<RFC3339>"}
```

- `uid`: 稳定 branch 标识。主键。
- `name`: 本次 append 时的 head name。因 name 可变更（rename / 抢占），历史 name
  必须从 append log 重建。
- `status`: 本次 append 后的状态。
- `fork_ref`: BranchRef JSON 元组 — checkout origin。
- 全量搜索 API（低频）读此文件。活跃 branch 发现请用 `heads/*` glob（§4.1）。

---

## 8. checkouts.jsonl — fork 事件记录

append-only。由**派生方本地追加**（零协调，无跨 owner 写）。

```json
{"t":"checkout", "branch_uid":"brn_<ULID>",
 "from_ref":{"fork":"...", "commit_id":"cmt_...", "moment_id":"mmt_..."|null},
 "owner":"<local_owner>", "created":"<RFC3339>"}
```

- Appended at branch creation time（`create_line`）。
- 正向读（"我从哪些 commit fork 了"）：本地顺读，O(1) 发现。
- 反向读（"谁从我的 commit fork 了"）：跨 owner 查询，走见证层 grep 或
  branches.jsonl 全量扫描 — 低频操作，接受 O(n) cost。
- 派生方追加是刻意设计 — "写入负担与获益方同一"消除 backlink 的跨 owner 协调问题。

---

## 9. confluents.jsonl — 引用式 confluent 事件记录

append-only。由**接收方本地追加**。

```json
{"t":"confluent", "from_branch_uid":"brn_<ULID>", "from_owner":"<owner>",
 "to_branch_uid":"brn_<ULID>", "to_owner":"<owner>",
 "kind":"reference", "created":"<RFC3339>"}
```

- 引用式 confluent（融汇）：目标 branch 接收源 branch 的引用提交。commit 的
  parent 链不动 — confluent 是独立关联事件。"提交引用而非内容，消灭冲突解决
  问题域"。
- `kind`: v3 仅 `"reference"`。未来可能扩展其他融汇形态。
- v1 不做多父 commit — confluents.jsonl 承载了原"merge"需求但没有 merge 的冲突
  语义负担。

---

## 10. commit_space 自由空间

每个 commit 目录中，下列文件是保留名单，契约 MUST 定义：

```
meta.json
moments.jsonl
notes.jsonl
```

保留名单之外的一切文件，契约不感知、不承诺、不禁止 — 业务自由空间。渲染缓存、
ground 快照、external link 等都可放置。变动历史由见证层兜底。

此条款与本文件 §4.3（workspace 自由空间条款）同构。

---

## 11. 崩溃恢复

commit() 原子写序列：

```
1. 读 staging last-wins → 写入 commits/{Y-m}/cmt_<ULID>/  (tmp 目录)
2. fsync commit 目录
3. atomic rename tmp → 正式目录
4. fsync commits/ 父目录
5. append commits.jsonl
6. fsync commits.jsonl
7. truncate staging
```

恢复判据（Memento 实例装入时执行，幂等）：

| 条件 | 动作 | 依据 |
|------|------|------|
| commit 目录存在且完整，staging 该 commit 的 moment ids 仍残留 | truncate staging（幂等） | commits.jsonl 尾行 commit_id |
| commit 目录存在但成员行缺 | 删除该 commit 目录 + commits.jsonl 尾行回滚 | 原子 rename 未完成 |
| heads/{name} → uid 但 ws/{uid}/ 不存在 | 删除该 head 文件，branches.jsonl 追加 status=abandoned 行 | head 写入后 workspace 创建前崩溃 |
| ws/{uid}/ 存在但 heads/ 无对应指针 | 保留 workspace（不丢轨迹），branches.jsonl 状态反映 | name 删除后、branches.jsonl 更新前崩溃 |
| confluents.jsonl 尾行撕裂 | 截断尾行（通用 §2 第5条） | 通用撕裂尾行恢复 |
| branches.jsonl 尾行撕裂 | 截断尾行 | 同上 |
| checkouts.jsonl 尾行撕裂 | 截断尾行 | 同上 |

恢复后 MUST 保证不变量：workspace 存在 ⇒ branches.jsonl 有该 uid 的行；
head 文件存在 ⇒ ws/{uid}/ 存在。不变量违反时恢复 MUST 以 workspace 为
authority（轨迹 > 名字），自动补写 branches.jsonl。

---

## 12. 见证层

git sidecar repo（`.memento/.git/` 或外层 repo），正交于 memento 逻辑层。memento
id = 身份，git sha = 完整性。裸 commit_id / branch_uid 反查走 `git grep`，O(grep)
成本接受 — 路径不索引不维护。

init 时选择 witness mode：`sidecar`（.memento/.git）、`outer`（外层 repo 直接见证）、
`none`。sidecar 模式下 init 负责把 `.memento/` 写进外层 .gitignore。

---

## 13. 退化态

退化态 = 单 line main + record → commit → log 循环。golden test 硬条款：
退化态用例代码中 fork / branch / confluent / checkout 词汇一个不出现。

退化态不要求 heads/ 文件 — get_line("main") 在无 heads/main 时隐式创建 uid
并写 head 文件（首次 use 时初始化，不显式 `create_line`）。branches.jsonl 同步追加。

---

## 14. 从 v2 迁移

v2 → v3 是不可逆升级（模块未发布，无向后兼容承诺）。若检测到 v2 布局
（branches/{name}/ 存在），MUST 拒绝启动并报错指引手动迁移。迁移脚本不属于
本契约 — 它是一个一次性工具，读 v2 → 写 v3，契约只保证两个格式各自自洽。
