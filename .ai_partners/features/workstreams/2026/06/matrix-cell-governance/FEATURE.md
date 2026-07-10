---
title: Matrix Cell Governance
status: in-progress
priority: P0
created: 2026-06-09
updated: 2026-07-10
depends:
  - cell-discovery-refactor
  - cell-session-bootstrap
milestone: 0.1.0
description: >-
  Matrix cell 体系治理总任务。从 circusd 死胡同出发, 历经六轮设计-推翻循环,
  收敛为: 膜承诺 (cell 必须 provide channel), 三域模型 (Manifest/Record/Presence),
  六动词治理代数, ledger 单写者, Presence/Watcher 拆分, moss_self CLI 合流。
  当前有效契约 = §TT/§TT续 + §UU + §VV。终局服务于模型运行时自迭代。
status_note: >-
  2026-07-10 claude-fable-5 开工定案 (§VV): 模型全权起草抽象层 (技术目标入 comments 不入 docstring),
  人类 IDE 改名+review. 四条挡板 + 块①金丝雀. 十三步拓扑路线图已记录 (FEATURE.md 裁剪 → env 先行 →
  cell/matrix 重绘 → 7-8 并行实现 → wire up → moss-as-mcp 验证 → 第一个 cells channel →
  apps 重走=自迭代验证). 下一实例按 §VV 拓扑展开, 不要线性重放对话历史.
  --
  2026-07-08 claude-fable-5 + 人类架构师终审会话 (§UU): 结构性待决全部闭合, 进入并行分发.
  骨架: 膜承诺 (cell 必须 provide channel), 三域模型 (Manifest/Record/Presence, God-model Cell 解体),
  六动词治理代数 (三域x两动词, 其余全是视图), ledger=咽喉排气尾迹 (单写者, CLI 唯一读者, 运行时零读零监听),
  Presence/Watcher 拆分 (入网与监听分离, N²→N), proxy=accept 即创建 (owner=accept 者, 治理=所有权进程内成立),
  network(local) 单 Watcher 双视图, moss_self CLI 合流 (一份实现两个面, Cells 门面 ABC 退役),
  Matrix 表面积: run_cell + network + processes + jobs. CELL/SKILL 比较与 GhostOS 系谱已记录.
  执行路径调整: 模型改抽象实现 → 人类 IDE 改名+review → 并行分发重写实现+单测.
  仍开放: TT-2 身份拆分 (uuid+alias) 未获人类最终确认; enum 取值/run_cell 参数面/alias 表格式属分发级细节.
  下一实例认知重建支点: 读 §UU 全文 (含 §TT 上文).
  --
  2026-07-07 claude-fable-5 + 人类架构师复审会话 (§TT):
  §SS 执行部分崩于边界设计错误, 工作模式转为人类亲手重建全部抽象 + 模型 review.
  根因诊断: 抽象融合 (非过多). 收敛: 身份拆分 (uuid + alias 表, check_unique 取消),
  PM 收敛为机制层, BackgroundTask 三分, 新立 JobSupervisor (认知胶囊三层),
  Matrix 分灶台 (~8 首页成员), Project = 治理域句柄 (taxonomy 禁入内核),
  Environment seal 两相提案. 待人类拍板项见 §TT-10.
  --
  2026-06-28 claude-opus-4-7 + 人类架构师讨论会话:
  (1) 补完前任未显式记录的 L0→L3 跃迁认知 (§MM) — matrix 成为最小通讯依赖, cells 真相源上升到 network.
  (2) 二元真相显式承认 (§NN) — status 变更不广播, live_cells 是延迟视图. 选 (c) 不加回 pub/sub.
  (3) 三重身份问题 framing (§OO) — type 字段一人干三份活的概念漂移定位, 字段拆解候选方案. 核心待决, 本轮未敲定.
  (4) cell type 升格 / 多 network 升格的 beta1 保守路径 (§PP).
  (5) channel proxy 根 channel 数量两个候选 (§QQ).
  (6) FEATURE.md 滞后修正 (§RR) — §L bridge_address 二分, §N HostCellNetwork ABC 分离, 均已合并/不做.
  (7) §SS 开工契约 — 三重身份 + log 三处零漂移 + 自动 proxy + cells/network 两个 channel + Matrix 启动流程. 决策全部收敛, 可开工.
  下一实例认知重建支点: 读本文 §SS (含 §MM-§RR 所有上文).
---

# Matrix Cell Governance

> 工作模式已演进 (见 §UU-0/§VV-1): 模型全权起草抽象层, 人类 IDE 改名 + review.
> 此 feature 记录动机、共识和推进方法。历史段落已压缩为演进摘要, 全文可 git log 溯源.

## Motivation

当前 MOSS 的 cell 体系有三套互相竞争的机制：circusd 进程守护（AppStore）、
zenoh queryable 发现（cell discovery）、以及裸 asyncio subprocess（ManagedProcess）。
加之 cell 类型混乱——app/script/fractal/host 的边界从未被严格定义——
导致开发者在"如何启动一个 cell"这件事上各自为政。

circusd 被设计为独立系统守护进程，它的核心能力（重启、监控、web dashboard）
面向部署场景，而非 per-session 的进程图。把它塞进 host 进程的子进程位置，
造成双层通讯（host → ZMQ → circusd → watcher → 子进程）、
孤儿泄漏（host SIGKILL 后 circusd 不知道宿主已死）、
以及两套监控体系（circus status vs zenoh queryable）。

**真命题**：MOSS 是通讯总线，不是进程管理器。Cell 的"存活"由网络查询决定，
不由本地守护进程决定。需要一套统一的 cell 治理框架：身份定义、发现机制、
生命周期契约、最小依赖闭包。

## 设计演进摘要 (2026-06-09 → 2026-06-28, 原文已被 §TT/§UU 取代)

> 本节是六轮设计-推翻循环的压缩摘要, 只保留演进弧线, 细节不再维护.
> 各阶段全文在 git 历史中: `git log --follow -p -- <本文件>`,
> 或按下列日期检索对应 commit. 当前有效契约 = §TT/§TT续 (地基) +
> §UU (结构闭合) + §VV (执行决策与路线图), 冲突处一律以后者为准.

**演进弧线** (每轮: 做了什么 → 后来怎么被取代):

1. **2026-06-09 初版共识 + 晚间设计会话**.
   cell 类型重定义 (host/node/fractal, app 与 script 归一为 node),
   两轴状态模型 (liveness/availability 正交), 最小依赖闭包
   (`ghoshell_moss[cell]`), 环境变量归一 (`MOSS_WORKSPACE` 唯一必须),
   进程生命周期三件套 (start_new_session+killpg / pipe fencing / polling),
   CELL.md 最小格式, `moss nodes run` 解析规则; 晚间补: 可用状态与
   NODE.md 解耦, install 作为约定不重实现, 运行时文件布局, Matrix 接口 v1,
   TopicWindow 事件广播, spawn JSON-line 透传.
   → "node" 命名后撤, 回归 cell; Matrix 接口经多轮重做至 §UU-10.

2. **2026-06-15 协议层对齐 + 实施纲领**.
   address 与 type 的协议位置, open type namespace (owner-channel 注册),
   spawn_worker/spawn_cell 二分 API, "worker 入网不被拒, semantic 不被承诺"
   (方案 C), 跨进程异步基底, fractal 命名空间, operations channel pattern,
   14+ 项推进拓扑.
   → spawn 二分被 TT-11 run_cell 单原语取代; type 的多重身份问题
   在此埋下, 至 §OO 才被 framing, 至 TT-2 才拆 (仍未终审).

3. **2026-06-21/22 实现层收敛 (claude-opus-4-7)**.
   cell.py 数据模型定型 (即后来 §UU-5 解体的 God-model Cell), 寻址体系,
   ZenohLivenessListener, ZenohChannelHub 重构, MOSSNamespace 中心化,
   cell announce = PUT + queryable, CellNetwork 二分
   (CellNetwork/HostCellNetwork), providing_channel 字段, Matrix 三层模型.
   → CellNetwork 把入网与监听焊在一起, 是 UU-7 Presence/Watcher 拆分的
   直接对象; HostCellNetwork 二分在 §RR 撤销; check_unique 判死
   (唯一性移到 accept 咽喉, UU-8).

4. **2026-06-25 大规模抽象重构 — L4 OS 架构跃迁 (claude-opus-4-7)**.
   四元语义命名 + transport, 六个核心文件整体重做.
   → 该轮产出的融合性抽象 (Facade 写成 Manager) 是 §TT-1 根因诊断
   ("抽象融合, 不是抽象过多") 的主要标本.

5. **2026-06-27 manifests 重建 (deepseek-v4-pro) + MossMeta/LocalHostMode/
   LocalProject/CLI 重建 (claude-opus-4-7)**.
   Manifest[T] 通用抽象 + 8 种 scanner 矩阵; MossMeta 回归, HostModeMeta,
   LocalProject 原型, CLI 重建.
   → manifests 体系存活; Project 线被 TT-7 重新定性为治理域句柄
   (taxonomy 禁入内核), inventory 归 project (§UU-10).

6. **2026-06-28 三重身份诊断 + §SS 开工契约 (claude-opus-4-7 + 人类架构师)**.
   L0→L3 跃迁认知补完 (§MM), 二元真相承认 — status 变更不广播 (§NN),
   三重身份问题 framing (§OO), beta1 保守路径 (§PP), proxy 根 channel
   数量候选 (§QQ), FEATURE.md 滞后修正 (§RR); §SS 定型三重身份字段 /
   CellLog / CellNetwork API / CellsManager API / ZenohChannelHub 补强 /
   Matrix 接口与启停流程 / cells+network 两 channels / 工序拓扑 / audit 指标.
   → §SS 执行 steps 0-2 落地 (`7e3a7a40`/`d3489a14`/`3f62e165`) 后崩于
   边界设计错误 (见 TT-0); SS-1 身份字段被 TT-2 推翻 (uuid+alias, 未终审);
   CellsManager 被 TT-11 取消; 这次崩溃直接触发 §TT 的根因诊断与工作模式转变.

**穿越全程存活的判决** (散落于上述各轮, 至今有效, 不要当作已推翻):

- "MOSS 是通讯总线, 不是进程管理器" (Motivation 真命题).
- 进程生命周期三件套: start_new_session+killpg / pipe fencing / polling.
- status 变更不广播, live_cells 是延迟视图 (§NN, 与 UU-6 ledger
  零监听同一精神).
- install 作为约定不重实现 (后成 CELL.md 三块之一, TT-13).
- 环境变量归一方向 (后收敛为 Environment seal, UU-1).

## 2026-07-07 设计复审会话 (claude-fable-5 + 人类架构师) — §TT

### TT-0. 工作模式转变 (最重要的上下文)

§SS 契约由模型执行, steps 0-2 落地 (`7e3a7a40` / `d3489a14` / `3f62e165`),
step 3+ (CellsManager) 未启动, 执行在边界设计错误上崩了.
**人类架构师决定亲手重建全部抽象, 模型转为 review 角色.**
beta1 前一切可推翻, 无沉没成本, 关键是决策要对.
本节记录的是复审结论, 不是实施契约. FEATURE.md 后续可能被人类整体重写或退化为 review 记录.

### TT-1. 根因诊断: 抽象融合, 不是抽象过多

反复出现的"不优雅感"病灶统一定性为**融合** (一个抽象承诺了两件事), 而非数量:

- `ProcessMeta.task_id / background_task_id` — 机制层长了业务层外键
- `BackgroundRunType.on_prompt` docstring 写"对应 ProjectManager 的 RefreshMode" — 进程层焊死帧语义
- `Cell.channel_name` — 入网身份和 project 身份熔在一个属性
- `BackgroundTask` 一个类装三种语义 (once/loop/on_prompt)

**检验法**: 每个抽象的承诺能否一句话说完, 且不提兄弟抽象的名字. 说不出来即融合信号.
(Erlang 参照: proc_lib / gen_server / supervisor 都小且单一, 从没人嫌多.)

**污染路径**: Desktop (原名 ProjectManager, 可丢弃层任务, Opus 设计)
→ 其消费需求长成了 PM 的 Layer 2/3 (contracts/, 内核面) → PM 计划入 Matrix
→ 人类在 Matrix 重构时被迫消化从未打算负责细节的模块.
根因是结构性的: **设计真空 + 对 contracts/ 的写权限 = 必然污染**, 换任何模型结果相同.

**挡板规则** (防复发):
- 可丢弃模块只能组合已冻结的契约, 不能扩展它们.
- 契约不够用时, 正确输出是留需求记录 + 自己层内凑合 (Desktop Stage 1 的"裸 asyncio 兜底"即正确形态),
  不是伸手改 contracts/.
- contracts/ 的 diff 必须过人类之手 (可 CODEOWNERS 物理化). 守一个目录比守全部设计细节负荷低一个量级.

### TT-2. 身份拆分 (推翻 §SS-1 的部分设计)

三层拆分:
1. **wire identity** = `address = uuid` — 免费, 自动, 无冲突.
2. CELL.md 的 `type/name` 降级为 **suggestion**, 随 announce 携带.
3. host 侧 **alias 绑定表** (`alias → address`) — 模型可见/可改名,
   auto-accept + 采纳 suggestion + 冲突时确定性后缀. 第一期内存态.

推论: **`check_unique` 从协议中删除** — 其 check-then-announce 竞态
(zenoh_cell_network.py:331-357) 成为被取消的问题, 不是被解决的问题.
`Cell.channel_name` property 删除. host scope 排他 = leader election, 另案处理.

### TT-3. ProcessManager 外观收敛

- **Layer 1 保留为纯机制层** (~8 抽象方法): executing/executed/get,
  execute/shell (加 `capture=CaptureSpec(...)` 可选参数吸收原 ProcessTask),
  kill/killpg, aenter/aexit.
- ManagedProcess 富句柄: meta / process / output(可选) / add_done_callback / **补 `stop(timeout)`**.
- **契约必须文档化"子进程不比 owner 活得久"承诺** (setsid+killpg / pipe fencing / polling 三件套).
- `cd/pwd` 移出 — 是 terminal session / Desktop 状态, 不是共享基建
  (实证: Desktop 已自持 cd/pwd, PM 只需收 cwd 参数).
- PM 进 contracts 的正当性 = IoC 依赖隔离 (两个正当契约来源: 依赖隔离 + 控制反转).
- 可能改名 SubProcesses (未拍板).
- 多 MOSS 共存一 OS: 铁律 **治理 = 所有权**. ps/全局扫描只用于自录 pid 的活性检查.
  模型只拿 scoped 面, 不拿裸 kill(pid).

### TT-4. JobSupervisor 新抽象 (BackgroundTask 三分的归宿)

BackgroundTask 解剖: `once` → 死掉回归普通 execute; `loop` → JobSupervisor;
`on_prompt` → 本来就是 channel 的 `get_context_messages()`.

JobSupervisor 形状: contracts/ 小 ABC, IoC 非单例工厂 per-owner 实例;
`submit(JobSpec) → Job` (调度即数据: interval/times/at, 不做模式多态);
Job = N 次短命进程的可观测 **fold** (fold-identity 是它存在的资格);
底层组合 PM; owner 死 → jobs 死; 无全局任务板 (§SS-9 per-owner 哲学).

**认知胶囊三层** (Job 概念的存在目的 = 关键帧认知胶囊, 让后台任务结果可读):
1. PM (机制) — 不知道胶囊
2. JobSupervisor (fold) — 胶囊**原料**: status / 最近输出窗口 / 执行计数
3. channel / `_pin` / get_context_messages (呈现) — 胶囊**投影**: 哪个胶囊进哪一帧

**胶囊显式化为数据契约** (pydantic snapshot model), 不散落在方法签名里 —
呈现层消费统一快照结构, 层间变纯数据流, 融合不回去.
MCP 无状态 request/response 装不下"活着的快照" — 这是 MCP 生态不能替代此层的根本原因.

### TT-5. 维护外包地图

- **PM Layer 1 必须自持** — 焊死 owner 生命周期 (gunicorn/celery/uvicorn 各自持有同款是证据;
  "才几百行所以没有行业库"是当时接受的错误理由).
- **JobSupervisor ≈ 100 行胶水** — interval-only 用 asyncio loop 即可, cron 需求出现才引 APScheduler.
- **工具语义 (bash/read/write) 外包 MCP 生态** — 头部实现: 官方 server-filesystem (TS,
  分页 read/diff edit; 沙箱 CVE 多), Desktop Commander (Node, 长进程会话管理最认真,
  但 in-memory 历史无 reclaim 契约), mark3labs (Go). **无 Python 头部库** —
  能力都焊在各 agent runtime 内部, 反证 substrate 是焊死的.
  集成姿态: **MCP server 以 cell 形态被自己的 PM spawn** (fencing 覆盖其会话泄漏缺陷);
  沙箱永不信任, scope 收窄在 cell launcher 层; g1 体控保持 native channel.
- **Desktop 的 IP 是元规则** (read-before-write / 统一截断 / _pin / ReflectionHint),
  不是原语 — 元规则依赖 MOSS 帧语义, MCP server 没有"帧"概念. Desktop 自建成立,
  理由修正为"IP 焊在本体", 不是"行业没有实现".
- CTML / channel / mindflow = 项目本体, 不外包.

### TT-6. Matrix 复审: 厨房分灶台

定位确认: Matrix 就是厨房; channel_builder + matrix 两个 blueprint 是模型开发 cell 的
认知总入口 (code as prompt, 认知单元而不完全是抽象). "厨房水槽"批评只在
30 成员摊平一个平面时成立, 解法是分灶台不是拆房间.

实证: 两个样例 app (screen_capture / trafilatura) 实际只用了
`provide_channel + logger + discover().run()` 三个成员; screen_capture 把 tmp.png
写进了**源码目录** (main.py:21,67) — 边界 API 存在但发现成本高于绕过成本的铁证.

收敛方案:
- 首页 ~8 成员: discover/run, this, env, network (+provide_channel/channel_proxy),
  workspace (一扇门), container (一扇门), logger, + scoped processes/jobs (新灶台).
- storage 全家 (~10 成员) 收进 workspace 门后; 平铺便捷属性
  (mode_name/ghost_name/network_scope/session_id) 删除 (env 上已有);
  session / cells 降级出首页.
- **边界做成环境而非 API**: cell spawn 时 cwd = 自己的 runtime 目录;
  `matrix.processes` 天生 root 在自己 runtime 子树. 无知代码也落在界内 —
  API 形式的边界要求开发者知情, 环境形式的边界对无知代码也生效.

### TT-7. Project 复审: 治理域句柄

- 逐成员检查: Project ABC 的目录成员几乎全指向 workspace, 真 project 概念只有 `root` 一行.
  初判"名字被占", 经交锋修正为: **Project = 被治理领地的句柄 (governance-domain handle)**,
  成员指向 workspace 是因为 workspace 正是治理真相的存放地. **保名**, 条件:
  契约改写为治理域句柄语义; taxonomy 永久禁入内核 API.
- **taxonomy 禁入的最硬论证**: project 内容是 ghost 自管理的 —
  内核持有目录 taxonomy (memory_dir/resources_dir...) 会让 ghost 不能重组自己的领地.
  与 Desktop KD7 同源 (原语不硬编码约定). 行业趋同方案 (Claude Code 系) 未来以
  mount 形式接入, 不需要重构.
- **Manifests 思想成立且是必然**: 跨 interpreter 能共享真相的载体只有
  文件 / 环境变量 / 网络. manifests=静态能力真相, env vars=治理身份继承通道,
  network announce=运行时活性真相. 挂 Project (作为环境真相的灶台门) 正确,
  Matrix 不一级暴露.
- **A/B 动机成立**: A 目录运行时拉起 B 目录 cell — B 只是代码出处, 治理归属
  (日志/runtime/身份/网络) 全归 A. 治理=所有权的目录投影, systemd 同构
  (ExecStart 指向任意路径, journal/cgroup 归 init 域).
- 遗留: project.py 一个文件七个认知单元需拆分; cells 三处露头
  (Project.cells / HostMode.cells / matrix.cells) 待收敛成一条链路.

### TT-8. Environment 两相问题 (git 考古 + 提案)

- 活读 os.environ 机制是 `536c8a56` (6-25, "refact zenoh session") 搭车引入, v1 是冻结的.
- **真实动机 = CLI 晚到参数化**: Environment.discover() 被 discover 链提前触发成单例,
  CLI 解析完 `--mode/--ghost/--scope` 后只能靠 set_* 写 os.environ + 属性活读绕过构造时序.
  全部 setter 调用点都在"进程启动后、运行时 run 之前"窗口内, 无运行中改身份的真场景.
- **提案 (方案 B, seal 两相)**: 配置窗口内 setter 可用 (写属性);
  `seal()` 时一次性 export 到 os.environ (只作子进程继承通道, 写一次不回读);
  seal 后 setter 抛异常, 属性冻结. dump_cell_env 不受影响.
  人类反馈: 非常认可, 但 v1→v2 的改动可能还有其他忘记的动机, 回忆中;
  **回忆不起来则以此结论为准**.
- `project_path` docstring ("永远是 workspace 父目录") 与 `MOSS_PROJECT_DIR`
  自由绑定机制矛盾 — 待拍板 invariant vs free binding (模型倾向 free binding
  + 修 docstring, 正当场景: 治理不能写入 .moss 的只读领地).

### TT-9. 三目录松耦合定型 (初共识)

**workspace = 治理真相的存放地; project = 被治理的领地 (薄句柄, 挂 Matrix 一级);
cell 目录 = 代码出处 (自带依赖/interpreter, 治理归属仍是启动方).**
cwd 只是发现起点, 不承载语义.

### TT-10. 待人类拍板项 (open decisions)

1. alias 绑定表生命周期: session 态 vs workspace 持久化
2. host scope 排他的 leader election 协议
3. PM 改名与否 (SubProcesses?)
4. MCP server 集成形态细化 (cell wrap 的 launcher 层设计)
5. project 绑定: invariant vs free binding (+docstring 修正)
6. Environment seal 两相方案 (等人类回忆 v2 动机后定)
7. stale docstring/注释治理 (低优先, 人类已记入自己的关注点)
8. Matrix 剩余 review 点 (本轮未完, 人类提到还有三处要 review, 移至后续会话)

## 2026-07-08 复审续 (claude-fable-5 + 人类架构师) — §TT 续: cell 路径收敛

### TT-11. CellsManager 取消, run_cell 单原语

§SS step 3+ 停在 CellsManager 的真原因 (人类语言化): 需要一个新抽象包两个底层抽象
才能合并到 Matrix 上, 复杂度高于治理目标. 复审结论: **砍掉 manager, 保留一个原语**.

- `matrix.run_cell(filepath)` 承诺一句话: "以本 Matrix 为治理域, 从某路径拉起
  cell 进程并接入网络". 过 TT-1 融合检验.
- CellsManager 天然装四件事 (发现/拉起/生命周期/命名), 其中三件已有真相载体:
  **发现=manifests/glob(文件真相), 活性=announce(网络真相), 命名=alias 表 (TT-2)**.
  manager 若存在只能持有第四份不一致的内存副本.
- registry 退化为 glob(CELL.md), 与 features 套件同构. `moss cells list` 是视图不是存储.
  注意 inventory (领地里装了什么) 与 spawn 能力 (从哪都能拉, 无远弗届) 是两个问题, 不熔.
- 源码路径无界, 治理归属有界: runtime/日志/scoped processes 落 spawner workspace 子树 (TT-9 的 API 化).
  Desktop 有 bash 即有等价能力, filepath 从来不是安全边界.
- 返回富句柄 CellHandle (address=uuid / stop / status), 对齐 ManagedProcess 形态.
- **API 选层**: run_cell 是底 (Python API); `moss cell run` CLI 是主面 —
  模型操作面塌缩到 bash 通用文法, 治理域从环境变量继承 (dump_cell_env 通道), 又是边界做成环境;
  CTML channel 面可选且必须薄 (帧内高频场景才有存在资格, 只做转发).
- Desktop 与 Matrix **不直接结合**: Desktop 是 run_cell + PM + 基底契约的消费者 (可丢弃层).
- pwd/cd 进一步下放: 连 Desktop 抽象也不持有, 是具体 session 的状态.
  一般法则: **机制层全收显式 cwd 参数, 可变状态只钉在最靠近对话的叶子上**.

### TT-12. cell singleton = 风险锚点, 两档 scope 各找真相载体

人类纠正模型的低估: 锁的初始动机不是无错, 是**用最小实现显式锚定风险** —
第二实例被拒时的错误信息本身就是 prompt ("g1 声明了硬件单点, 地址 X 已有活实例"),
debug 第一时间回归声明语义. code as prompt 在错误路径上的延伸.

- 约束力方案: 薄字段 + **唯一咽喉点** (run_cell 是唯一 spawn 路径, 无旁路) +
  **全投影** (announce 携带 / alias 表标注 / cells list 显示 / 拒绝错误引用原文).
- singleton cell 冲突语义从 TT-2 的"确定性后缀"翻转为**硬拒绝**; 非 singleton 保持可多开 (browser-1/-2).
- **scope 两档**: `singleton: domain` 靠 run_cell + alias 活性 (网络真相);
  `singleton: host` (硬件级, 跨 workspace) 靠 launcher 启动时 flock 约定路径
  (文件真相, 进程死自动释放, 无 stale 清理). 两档恰好对应 TT-7 的跨 interpreter 真相载体论.

### TT-13. CELL.md 定性: exec spec 是地基, PEP 723 是匝道 (APP.md v1 六坑考古)

人类披露 APP.md 第一版就是 PEP 723, 坑有六:
(1) 依赖 uv, 环境在 moss 外; (2) 无 pyproject 则运行时依赖不可解, 修复靠逐层 pyproject;
(3) 无声明文件则只剩 Python 兼容, 而 moss 还没能力下沉 OS 级 (apt install moss);
(4) 缺模型可自主控制的 AppStore/桌面概念; (5) 无法做启动 DAG (但也不想做);
(6) supervisord/circusd 非通用解 (g1 pc2 跨机).

收敛 (模型收回上一轮"cell=裸 py 文件"作地基的方案):

- **通用原子是 exec spec** (一条命令 + 环境注入 + announce 协议), 即 systemd ExecStart 原理.
  Tier 0: CELL.md 带 `run:`, 语言/机器无关, 承诺一句话 = "把一个可执行物声明为本治理域可拉起的 cell".
  它是 moss 未下沉 OS 前的 unit file / .desktop 快捷方式替身, 这一层不可选.
- Tier 1: 裸 .py + PEP 723 零声明, 是**语法糖降解为 exec spec** (`uv run path.py`).
  uv 依赖与依赖混乱全部圈死在此层, 内核契约只见"命令+环境".
  v1 的错不是用 PEP 723, 是把它放在地基; 零声明价值保住, 作匝道不作路基.
- launcher 拿不掉但做到不可见: 无用户面, 即 run_cell 内部三件机械事 (uv run / dump_cell_env 注入 / cwd=runtime).
- 坑 4 (桌面/AppStore) = glob+alias 表+announce+run_cell 的**投影视图**, 可丢弃层, 永不进 contracts/ —
  内核引力高危区, 挡板规则主守对象.
- 坑 5 (DAG): 机器级归 systemd; 治理域内**排序是认知任务, DAG 求解器就是模型本身**,
  内核只给 run_cell + 活性真相.
- **领域定位**: systemd (单机/root/配置驱动/非模型面向) 与 k8s (数据中心假设) 之间存在真实空隙 —
  具身智能体的治理域跨两三台机器、进程异构、操作者是模型. ROS 是存在性证明:
  做总线者必然被迫长出 manifest/launch/lifecycle (package.xml/roslaunch), 这是领域形状不是设计者贪婪.
  过度设计判据: 空隙内无轮子的原子自持 (身份/announce/run_cell/env 继承/模型面向声明),
  有轮子的外包 (uv/systemd/flock/MCP) — TT-5 纪律未越界.

### TT-14. 跨机 = channel 分形挂载, 不是 run_cell 加 target 参数

模型收回 TT-13 讨论前的"契约别焊死本机"建议, 方向反了:

- **run_cell 焊死本机**. 治理=所有权的机器投影: B 机 matrix 只治理 B 机进程,
  A 通过 B provide 的 channel 子树操作 (`os_b.cells:run`). 跨机是**组合不是参数**, 无控制平面.
- fractal (blueprint/fractal.py) 被拿掉是粒度问题非设计错误: 整 runtime 作 provide 单位太粗,
  cell/network 重建后正确单位是子树 (cells channel / terminal channel).
- 行业对齐: 最近先例 Plan 9 namespace import/export, 但只覆盖文件、单工;
  **channel 原生分形 + duplex = Plan 9 挂载哲学做到能力层**, 无已知第二家.
  MCP 表达不了 (单向 request/response 装不下"活的能力子树挂载"), skill 在 prompt 层不在运行时.
  "为什么不是 MCP/skill"的真答案: 那两个是单点能力的接线标准, channel 是能力空间的**拓扑**;
  拓扑可降级模拟接线, 反之不行. (MOSS channel 设计早于两者.)
- duplex 协议不稳定但通过大量基线测试且实际在用.

### TT-15. cell 生命周期协议 (模型承认 TT-13 坑 5 的框架想浅了)

真问题不是启动顺序, 是**完备生命周期语义**: 启动/运行/进程内优雅启动/状态变化/优雅结束/已杀.
无精确语义时模型只能被动轮询.

- 关键结构: **进程真相 (PM 免费可见: spawn/exit/kill) 与应用真相
  (ready/draining/degraded, 只有子进程自己知道) 是两个来源**, 后者必须走自报告侧信道.
  行业对应物: systemd sd_notify (READY=1/STOPPING=1/WATCHDOG), k8s 三 probe.
  **MOSS 侧信道已存在 = announce/总线**, 只需封闭状态 enum 挂 announce payload, 两源 fold 成一条事件流.
- 模型得知: **一条真相流两个消费面** — command 内 wait (`run_cell(wait='ready')`,
  拉起即用主路径) + 生命周期跃迁作 signal 进 mindflow (crash 必须推送:
  轮询视角下静默与运行不可区分). 与胶囊三层同构.
- **MVP 切法** (人类策略: 先稳固 MVP 再推进): v1 三态 —
  spawned (PM) / ready (announce 到达即 ready, 零新协议) / dead (liveness 丢失或进程退出取先).
  推迟 STOPPING/draining/watchdog/进程内状态变化, 但**状态 enum 现在就进契约**, payload 后扩.
  状态即数据, 与 JobSpec 同纪律.

### TT-16. 待拍板项增补 (在 TT-10 之上)

9. Desktop 基底数据契约形状 (元规则 read-before-write/截断需要基底上报状态,
   薄 pydantic 快照契约, MCP/native 实现适配之 — TT-4 胶囊显式化同手法)
10. 本地 cell 进程与远端 (network) 的分离方式 (人类仍在想)
11. run_cell 参数面与错误语义细节 (人类亲手打磨, 防偏航)
12. 生命周期状态 enum 的具体取值与 announce payload 结构
13. 整体评审 (是否过度设计/项目层面) — 人类点名放在最后, 模型记账中

## 2026-07-08 终审会话 (claude-fable-5 + 人类架构师) — §UU: 结构闭合与并行分发契约

### UU-0. 会话性质与执行路径变更 (最重要上下文)

本会话是并行开工前的最后一轮设计对齐, TT-10/TT-16 待拍板项全部处理完毕.

**执行路径调整**: 原计划 (人类写抽象 → 模型 review → 拆任务并行重写实现+单测)
改为 **模型改抽象实现 → 人类用 IDE 改名 + review → 确定后并行分发重写实现+单测**.
含义: 下一批模型实例先按本节写 blueprint 抽象层, 命名权保留给人类 (IDE 重命名成本低),
review 通过后才进入实现与测试的并行分发.

### UU-1. TT-10/TT-16 待拍板项处置结果

1. **alias 表生命周期 → 两者都在, 重新定性** (人类拍板"都在", 模型定性获默认):
   workspace 文件不是 "alias 表的持久化副本", 而是治理域的 own-process ledger
   (自录 pid/pgid/start_time/alias/address 的账本, 现 runtime file 的正名).
   内存 alias 表退化为 ledger 的读缓存, 无独立生命周期. 否定路径已探索:
   `moss cells kill` 的使用时机正是网络真相不可用之时 (host 挂死/孤儿), 网络真相
   天然覆盖不了 kill; 三载体中 env vars 装不下动态表, ps 扫描违反治理=所有权; 文件胜出.
2. **host scope 排他 → zenoh listen 端口 bind 排他即正解, 不是绕开** (与 flock 同理:
   OS 级独占资源, 进程死自动释放, 无 stale). 不引入 redis/router. 补一条错误路径
   code as prompt: 抢不到端口时须区分 "已有活 host" (handshake 成功) vs "端口被无关
   进程占用" (handshake 失败). election 协议推迟到多 host 需求真实出现.
   推论: host 身份 = 运行时事实 (抢到 listen 端口者), 不是 cell type 声明.
   `Matrix.is_host()` 现依赖 `this.type == HOST_TYPE`, 需换真相载体.
3. **PM 改名 → `Subprocesses`** (人类授权模型决定): 名词复数=拥有的一组东西, 无
   Manager 抽象引力; 与 asyncio.subprocess 相邻自解释; supervisor 一词让给 JobSupervisor.
4. **launcher 走 CLI 治理, registry 绑 project** — 确认, 且被 UU-9 (moss_self 合流) 收束.
5. **Environment seal → 定案必须做**. 人类回忆起的真实动机: env 抽象退化成了 API,
   单例没有信源能力. 比 "CLI 晚到参数化" 更根本.
10. **本地/远端分离 → project_id 作数据标签, 不做 MatrixNamespace 原生分离** (人类 ok).
   原生 namespace 切开会破坏 TT-14 跨机 proxy 依赖的可见性, 反而逼出被否决的控制平面.
   硬隔离用 --scope (已有原语), 软分组用 project_id 标签 + 视图过滤.
11. **cell 治理面不上 Matrix 首页, run_cell 单方法上** (人类 ok). 后续被 UU-10 细化.

**仍开放**: TT-2 身份拆分 (uuid + alias) 人类明确表示 "还不是我的结论, 还要碰碰".
现 CLI address 仍是 type/name/uid 三段式. 分发时身份字段以人类终审为准.

### UU-2. 膜承诺: cell 必须 provide channel (定性完成)

**Cell 一句话承诺: "一个通过 channel 向模型世界暴露自己的、有生命周期的进程".**

- 不提供 channel 的进程在模型能力空间里不存在, 其归宿是 Subprocesses 治理下的裸子进程.
  若 cell 不承诺 channel, Cell 与 Subprocess 在同一承诺上打架, 融合检验当场失败.
- 生物学隐喻精确: 细胞由膜定义, 膜是被感知和交互的唯一界面. 没有膜的是细胞质 (进程).
- 纯 sensor 不构成反例: channel 不只是命令集, 是模型认知里的存在单元
  (instructions + context_messages + 可用性). 空命令的 channel 是合法的膜.
- app 体系 (v1, 已注释的 blueprint/app.py) 当年就是这么假设的, 假设正确.

膜的未来演进 (人类披露): 网络声明会变重 — resources (跨网络资源分享, 形似 MCP resources),
上下文变量 (resources + context message 窗口展示图片, 历史消息只留资源 id).
**仲裁: 膜可以变重, 治理不许变重.** resources/上下文变量是膜上的运输类型 (membrane
transport), 是 announce 数据扩展 + 帧投影层消费 (胶囊三层的呈现层), 给治理面加零个动词.
声明曲线可演进, 治理代数冻结 — 这是防第二次 circusd 死胡同的结构保证.
cell 与 matrix 同构的根本原因: cell 进程内跑的就是同一个 Matrix.discover(),
角色由 env 继承决定. 膜再重, 同构性免费维持.

### UU-3. 两个运维平面: 可行动性判据 (status 冲突的解法)

原 CellStatus 混装 project 内可感知讯息与 network 讯息, 切分判据一句话:
**消费者能对这条信息采取行动, 它才属于那个平面.**

- 远端模型拿到 pid/日志路径什么都做不了 → 永不上 announce, 归 ledger (owner 运维面).
- 远端模型拿到 degraded/failure 摘要可以行动 (不路由/通知 owner) → 上 announce (network 运维面).
- 行业同构: k8s (kubelet 知道 pid, API server 只有 conditions), systemd (MAINPID 归
  init 自己, D-Bus 上只有 active/degraded).

### UU-4. 六动词治理代数 (最简治理形态, 高阶仲裁结论)

人类的三条用户故事路径 (bash 起点 / cells channel / 本地开关 vs 远程 accept-deny)
不是竞争方案, 是同一个代数的三次投影:

| 真相域 | 两个动词 | 语义 | 信任模型 |
|---|---|---|---|
| inventory (文件) | create / install | 使之可运行 | — |
| ledger (所有权) | run / stop | 使之生/死 | 开=信任 (我拉起的我全权) |
| network (膜) | accept / deny | 使之对我可用/不可用 | 不拥有, 只能承认或拒绝膜 |

- **六个动词, 代数封闭. list/status/logs/窗口关联全是三真相的 join 视图, 不是治理.**
- 路径一 = 六动词经 CLI 面投影 (地板: 零 MOSS 知识, bash 通用文法).
- 路径二 = 同六动词经 channel 面投影 (wait_connected 是 run 在帧语义下的形态).
- 路径三不是路径, 是发现: 动词按域分化. 本地/远程信任不对称不需要设计,
  是所有权铁律的自动推论 (你杀不了不在你账上的东西).
- v1 的 accept 实现 = 调 channel_proxy 即 accept, 不调即 deny. 零新机制.
- 简单性下界证明: 六动词删任何一个, 自迭代循环断一处.
- **治理递归免费成立**: "cell 被独立的 ghost 治理" (GhostOS 命题) 不需要元层 —
  治理者也是模型, 两个投影面都是模型可操作的, 递归每层用同一个代数.

### UU-5. 三域模型: God-model Cell 解体

| 模型 | 真相域 | 字段 | 来源 |
|---|---|---|---|
| **CellManifest** | 文件 (inventory) | name, description, taxonomy (原 type 降级标签), singleton scope (none/domain/host), **exec: ExecSpec**, instruction, installed | CellMetadata 溶解进来; CellLauncher 改名 ExecSpec (TT-13: exec spec 是地基), 字段不变 |
| **CellRecord** | 文件 (ledger) | address, alias, pid, pgid, start_time, project_id, cwd, 日志路径, spawner | 原 CellStatus 的 owner 侧 |
| **CellPresence** | 网络 (announce) | address, alias, 生命周期 state enum, failure 摘要, project_id, host 角色(运行时事实), **膜: channel 接口描述**, (未来: resources) | 原 CellStatus 的网络侧 + 膜. 命名沿 XMPP presence 先例 |

- **Cell (meta+launcher+status 复合体) 解体**. join 只发生在视图层 (CLI list 输出行), 内核无 God-model.
- 每个治理面返回自己的域模型: inventory→Manifest, ledger/CLI→Record, network→Presence.
  上一版 "每个面都拿到整只 Cell" 是 God-model 的必然症状.
- Cell 上的行为 (is_alive/write_runtime_file/launch_* /spawn 辅助) 全部离开数据模型,
  归咽喉与 CLI 合流层.
- from_proc 族 = Tier 1 匝道: 裸 .py 反射生成临时 Manifest, 进同一条咽喉.
- type 三重身份各回各家: 拓扑角色=运行时事实进 Presence (抢端口者为 host, 不在 CELL.md);
  project 归属=project_id 标签挂 announce; 治理路径=ledger 条目的存在本身, 永不上网络.
  CELL.md 的 type 保留为纯 taxonomy 标签, 不驱动任何机制.
- 命名与边界人类保留 IDE 改名权 (UU-0).

### UU-6. ledger 仲裁: 咽喉的排气尾迹, 不是运行时的输入

n 倍监听问题的病根: ledger 被当成了第二套发现系统 (文件版 §NN 二元真相病复发).
发现已有主 — 活性发现=网络真相. 裁决:

- **ledger 从运行时所有读路径删除**. Matrix 上没有 ledger 成员, 没有 watcher, 没有全局发现.
- **Matrix 体系内治理 = Subprocesses 全权** (owner 内存态注册表即权威所有权记录,
  "子进程不比 owner 活得久" 铁律保证内存态足够).
- **写仍留在咽喉** (对人类 "上移 CLI" 方案的唯一修正): run_cell spawn 瞬间 append 一条
  CellRecord JSON, best-effort, 不回读. 理由=单写者原则: pid/start_time 只有 spawn 现场
  知道; 若 CLI 层自己记账, host 拉起的 cell 无账, `moss cells kill` 对其失效,
  ledger 的存在理由 (fencing 失效时的法证清理) 即破.
- **CLI 是 ledger 唯一读者**: moss cells list/status/kill = 读 ledger + join 网络真相 + killpg.
  冷数据, 按需读, 零监听.
- "一层两种启动方式" 恐惧消解: CLI 前台跑 (owner=CLI 进程, 阻塞同生共死) 和 host 内跑
  (owner=host) 走同一条 run_cell 咽喉, 差异只在 owner 是谁, 治理逻辑只有 Subprocesses 一套.
- ledger 无对象身份: 一个 workspace 目录约定 + CellRecord schema + "咽喉写 CLI 读" 两条规则.

### UU-7. Presence / Watcher 拆分 (入网与监听分离)

实现事实 (zenoh_cell_network.py 核对): 捆绑是 MOSS 加的, 不是 zenoh 的 —
zenoh 原语本来就分开 (入网侧 declare_queryable + liveliness token + publisher, O(1) 被动;
监听侧 subscribers + janus + cache + reconcile, O(N) 主动状态).
`allow_create_proxy: bool` 是融合的供词 — 类内布尔角色开关即 TT-1 检验失败形态.

| 抽象 | 承诺 (一句话) | 内容 | 谁持有 |
|---|---|---|---|
| **Presence** | 让本 cell 在网络上可被发现、可被查询 | queryable + liveness token + log publisher | 每个 cell bootstrap 自带, 永远开, 近乎免费 |
| **Watcher** | 维护我对网络的延迟视图 | subscribers + cache + reconcile | opt-in, 每 runtime 至多一个, 首次需要时创建 |

- 成本 N²→N: 现在每个 worker 跑全套 subscriber+cache (DDS discovery storm 形状);
  拆后 worker 只跑 Presence, 只有消费者 (host/ghost) 跑 Watcher.
  k8s 同构: kubelet 只注册, informer 是控制器按需开的, shared informer=每进程一份 cache.
- network(local: bool) = 同一个 Watcher cache 上的两个过滤视图, 不是两个 Watcher/hub.
  local=信任语义 (自动 accept + owner 运维摘要可见), foreign=膜承认语义 (presence + accept/deny).
  动词集不同是域的属性, 不是协议的属性. 底层 zenoh key 空间/liveness/reconcile 单套.
- **debug 内聚 = 问责单一性**, 不是一个对象干所有事: "别人看不见我"→问 Presence;
  "我看不见 X"→问 Watcher (cache/subscription/上次 reconcile); "proxy 挂了"→问 owner.
  两半都是具名、可审讯的一等对象, matrix 布线不吸收, repl inspector 直接展示.

### UU-8. proxy = accept 即创建 (内存泄漏的所有权解法)

- **泄漏病根不是 proxy 不进 shell, 是 proxy 没有 owner.** 治理=所有权在进程内同样成立:
  每个有生命周期的对象需要 owner, owner 生命周期为它划界. 现在 proxy 的 owner 是
  hub cache — cache 是视图不是治理域, churn 时 build/drop 循环任一 drop 路径不干净即积存.
  "不进 shell 无副作用" 是把休眠当释放.
- **auto_build_proxy 急切构建删除**. accept(address) 即创建 proxy, owner=accept 者,
  owner 关闭即释放. 六动词代数早就预言这一刀: 急切 auto-proxy = 自动 accept 全网络,
  把 accept 动词从治理面偷走塞给了机制层.
- proxy 网络唯一性由此变为 accept 咽喉处一次本地 dict 查重 — 无网络往返, 无 race
  (check_unique 的死因不复存在于此).

### UU-9. moss_self CLI 合流: 一份实现, 两个面, Cells 门面 ABC 退役

人类方案: `.moss_ws/apps/tools/moss_self` (moss CLI 反射为 channel) 稍加迭代作 meta 工具,
裁剪作 cell 唯一治理工具. Record 和 Manifest 各自有治理抽象, CLI 作合流入口,
**运行时里不放任何调度能力**.

- 六动词的唯一实现放在 CLI 命令组下面的可 import 函数里 (blueprint 可见性由 codex
  反射保证, kill 机制照样 code as prompt), typer 薄壳, moss_self 反射 CLI 成 channel —
  路径一 (bash) 和路径二 (channel) 两个面免费获得.
- 本会话中期提出的 "Cells 六动词门面 ABC (collections.abc mixin 风格)" **正式退役** —
  其内聚职责被 CLI 命令组承接, mixin 默认实现变成 CLI 底层函数, 零新增抽象.
  (mixin-on-ABC 模式本身的合法性保留备用: 人类 "ABC 上给具体实现提示未来机制" 的
  手法 = collections.abc 三十年先例.)
- 内聚轴≠边界轴的方法论保留: 边界按真相源切 (防泄漏), 内聚按模型自迭代循环聚 (防散架).
  AppStore v1 接口内聚的秘密 = 按自迭代循环组织 (list/context/start/stop/**init_app**),
  init_app (造新的) 是 v1 有、cell 版丢了的自迭代原语, 经 `moss cells create` 回归.

### UU-10. Matrix 表面积终版 + discover 判决 + 启动面参数

```
matrix.run_cell(target, wait=...)   # 咽喉: 唯一 spawn 路径, 写 ledger 排气
matrix.network(local: bool)         # 膜与活性; proxy 耦合其中 (accept 式)
matrix.processes  (Subprocesses)    # 机制灶台 (TT-6); 裸 spawn 从首页移入此处
matrix.jobs       (JobSupervisor)   # fold 灶台 (TT-4)
project.cells                       # inventory, glob(CELL.md), 只读 — 不在 matrix 上
(ledger)                            # 无对象身份, workspace 目录约定 + CellRecord schema
```

- Matrix.cells: CellRegistry 属性删除, HostMode.cells 同删 (TT-7 三处露头收敛).
- **discover() 判决**: 双门形状正确 (docker.from_env / k8s load_config 同构),
  env 参数即显式门, 所有参数流经 Environment 单载体 (CLI 晚到参数走 seal 前 setter 窗口),
  不在 Matrix 构造面开第二条参数通道. 毛病是 composition root 藏在糖里:
  factory.create_matrix(env, project) 升一等公民, discover() 退化为三行糖 +
  docstring 写明等价展开.
- **启动面参数**: name | path 双接受 (systemctl start name vs systemd-run /abs/path 同构).
  name→查 project inventory 解析; path→按调用方 cwd 解析.
  相对路径只活在 API 边界一瞬间: run_cell 入口立即 resolve() 绝对化,
  咽喉以下 (exec spec/ledger/announce) 只存在绝对路径. ledger 里永远绝对路径.
- Matrix.is_host() 需换真相载体 (UU-1 第 2 条推论).

### UU-11. 自迭代 telos 与 CELL/SKILL 比较 (系谱与坐标, 供后来实例定位)

**telos (人类补充的从未显式记录的设计动机)**: cell 体系服务于模型运行时自迭代 —
模型读 A/B cell 的接口, 在其拓扑上开发 C cell 并拉起, 同一运行时内不重启地开始使用
C 的接口. 与 bash 的本质差异 = 有状态: bash 给动作→文本 (一次性), cell 给动作→持久化
接口且自动进入感知上下文 (channel instructions/context_messages 进后续每一帧).
**身体在运行时长出新器官、且新器官立即接入神经.** 任何时候不能丢.

- 计算机史坐标: Smalltalk/Lisp live image 性质 + Erlang 进程隔离, 组合的模型面向版本无先例.
- 系谱: GhostOS (2024) terminal_agent.py 30 行 = persona + 依赖声明 + code as prompt 反射,
  `ghostos run` 即入网. 当年撞的三堵墙 (运行时隔离/组网/独立依赖) 恰对应 CELL.md 三块
  (进程隔离+start_new_session / zenoh announce / uv+PEP723). cell 是同一命题的二次冲锋.
  理想形态等价物: 一个 CELL.md + 一个 provide_channel 的 main.py, `moss cells run` 即入网.
- **膜承诺的关键推论: announce payload 必须携带 channel 接口描述** — 否则模型要先
  proxy 连上才知道 A/B 提供什么, 自迭代循环断在第一步. 这是 Matrix.this 存在的真正理由
  (交叉感知的是接口, 不只是存在性). 全文 vs 摘要+按需 query 未定, 属分发级细节.
- CELL vs SKILL: 表层收敛 (frontmatter markdown manifest) 是吸引子非借鉴
  (.desktop/package.json/ROS package.xml 同一吸引子历代投影), CELL 思想早于 MCP.
  本体论差异: SKILL=模型是执行者 (知识), CELL=cell 是执行者 (器官);
  episodic vs 全生命周期; 加载时在上下文 vs announce 网络存在+持续进帧;
  turn 内联 vs 与思考并行; prompt 拼接 vs channel 树拓扑挂载; 无治理 vs 全套治理.
  收束语言: skill 在 prompt 层, MCP 在接线层, cell/channel 在拓扑层; 拓扑可降级模拟
  下两层, 反之不行. CELL 不需要赢 SKILL (不同层); 需警惕的是把 skill 能干的事往 cell 里装
  (无状态 know-how 用 cell 是过度设计).
- 过度设计复检 (TT-13 判据): 该原子 (模型面向+有状态+跨进程能力挂载+生命周期治理)
  无轮子; 有轮子的 (uv/flock/端口排他/MCP) 都在外包. 未越界.

### UU-12. 并行分发切块 (方案定, 实现是一个下午的事)

前置: 模型先改 blueprint 抽象层 → 人类 IDE 改名 + review (UU-0 路径).

分发块 (① ③ ⑤ 与 ② ④ 两线可并行):
1. 三域模型 + ExecSpec (纯 pydantic 零依赖, 最先发)
2. Subprocesses 收敛 (TT-3 的 ~8 方法 + stop(timeout) + capture) + run_cell 咽喉
3. Presence/Watcher 拆分 + network(local) 双视图
4. moss_self 裁剪为 cell 治理工具 + CLI 命令组重接 (六动词) + ledger 目录约定
5. announce payload 挂膜 (接口描述) + 生命周期 enum (spawned/ready/dead, TT-15 MVP)

分发级细节 (不阻塞抽象层, 随块敲定): enum 具体取值 / run_cell 参数面与错误语义
(人类亲手打磨, TT-16.11) / alias 表 (CellRecord) 字段格式 / 接口描述全文 vs 摘要.

### UU-13. 上下文恢复支点 (下一实例必读)

1. 读本节 §UU 全文, 含 §TT/§TT续 上文 (TT-1 融合检验 / TT-11 run_cell 单原语 /
   TT-13 exec spec 地基 / TT-14 跨机=分形挂载 / TT-15 生命周期三态 是 §UU 的直接地基).
2. 工作模式: 模型改抽象 → 人类 IDE 改名 review → 并行分发. 命名权在人类,
   TT-2 身份拆分未终审, 不要抢跑身份字段.
3. 三条铁律贯穿一切: 治理=所有权 (进程与对象通用); 单写者 (ledger 咽喉写);
   每个抽象的承诺一句话说完且不提兄弟抽象的名字 (TT-1 检验).
4. 实现核对入口: src/ghoshell_moss/matrix/networks/zenoh_cell_network.py (拆分对象),
   src/ghoshell_moss/core/blueprint/cell.py (解体对象), blueprint/app.py (v1 考古),
   .moss_ws/apps/tools/moss_self/main.py (合流载体).

---

## §VV. 开工执行决策与拓扑路线图 (2026-07-10, 睡前定案)

§UU 收束了全部结构性设计. 本节记录执行层决策与推进路线图 —
路线图是拓扑序而非线性序, 下一实例按此展开, 不要从对话历史线性重放.

### VV-1. 执行决策定案

**模型全权起草抽象层** — 不再走"人类写抽象、模型 review"的原计划:

- 模型依据 §UU 契约直接改写 blueprint 抽象 (三域模型/ExecSpec/Presence/Watcher/
  Subprocesses/network(local) 等).
- 技术目标与设计动机写入 **comments** (面向 reviewer/后来实例),
  **不写入 docstring** (docstring 是模型运行时的 prompt 面, 只放使用契约).
- 人类在 IDE 中做**改名 + review** — 命名权始终在人类. review 通过后,
  实现层 + 单测按 UU-12 分块并行分发.

**四条挡板** (模型自守):

1. **契约即 §UU** — 起草时遇到 §UU/§TT 未覆盖的歧义, 停下来问, 不自行发明.
2. **TT-2 身份字段占位** — 身份拆分 (uuid+alias vs type/name/uid) 人类未终审.
   相关字段用可整体重命名的占位名, 不锁死, 不抢跑.
3. **抽象层 PR 单独小批** — 抽象改动与实现改动分开提交, 保证人类 review
   带宽内可消化.
4. **块①金丝雀** — UU-12 分发块① (三域模型+ExecSpec, 纯 pydantic 零依赖)
   最先发, 用它校准人类 review 节奏与模型起草质量, 再放开其余块.

### VV-2. 拓扑路线图 (十三步)

依赖拓扑, 非严格时序. 步骤间标注了已知依赖:

1. **FEATURE.md 裁剪** — 按最新决策 (§UU/§VV) 截断/摘要历史章节.
   本文件全文已在 commit 历史中, 裁剪不丢轨迹. 裁剪时保留人类可读的
   review 结构 (见 VV-4).
2. **desktop/memento 讨论平行推进** — 已确认与本 workstream 平行,
   依赖 Subprocesses + JobSupervisor 两个抽象稳定后即可开线.
3. **Environment seal 先行** — env 退化为 API (无信源能力), 结论已有 (UU-1),
   实施在 cell 抽象重绘之前, 扫清依赖.
4. **重绘 cell 抽象拆分** — 三域模型落地 (CellManifest/CellRecord/CellPresence
   + ExecSpec), God-model Cell 解体. 即 UU-12 块①.
5. **project 验收一轮** — inventory 归 project (project.cells), 对齐细节.
6. **重绘 matrix** — 表面积按 UU-10 收敛. 两个待定项:
   storages() 边界可能拿掉、也可能保留作最显眼的提示位;
   factory 单一入口调整 (create_matrix(env, project) 升一等公民,
   discover() 退化为糖).
7. **并行推进所有实现** — 分开单测 + CLI review, 约 7-8 个并行任务
   (UU-12 五块 + env + 杂项).
8. **matrix wire up** — 修改 CLI 接线.
9. **moss runtime 集成 mode manifests** — 大概率只需微调生命周期.
10. **CLI 整体梳理对齐** — 六动词命令组 (UU-9) 全面就位.
11. **验证 moss-as-mcp 启动** — 集成冒烟.
12. **实现第一个 cells channel** — 大概率直接叫 moss channel.
    模型通过它拿到六动词 (UU-9 的 moss_self 合流产物).
13. **重走 apps 个别逻辑** — 开着 MCP 走完 create→run→接口进帧,
    即 UU-11 自迭代 telos 的第一次运行时验证.

### VV-3. 终局

十三步走完后: 仓库既有 apps 可大规模搬迁为 cells;
人类开始 g1 (宇树) 与能力 demo 的结合. 本 workstream 到步骤 13 为界,
搬迁与 demo 是下一个 workstream 的事.

### VV-4. 人类 review 方式说明 (供裁剪 FEATURE.md 的化身参考)

人类工程师重建上下文的方式: 逐个抽象读接口、在 review 中写注释、重新加载
认知 — 一轮完整 review 约数小时量级 (模型 10 秒的事). 因此:

- 裁剪 FEATURE.md 时保留**按抽象组织的结构** (而非按对话时序),
  让人类能按抽象逐个进入.
- 抽象层 PR 小批发 (VV-1 挡板③) 就是为这个带宽设计的.
- comments 里的技术目标是给这个 review 过程的输入, 不是装饰.


