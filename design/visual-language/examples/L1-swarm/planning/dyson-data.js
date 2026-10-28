// generated from .ai_partners/features/workstreams/2026/*/*/FEATURE.md
window.DYSON = {
 "workspace": "MOSShell @ agent-swarm",
 "features": [
  {
   "name": "agent-surface",
   "status": "parked",
   "desc": "把 memento agent 表面化为 moss 驱动世界的兼容契约。concrete agent 保留自己的原生 接口，Agent 表面（create",
   "progress": 100,
   "month": "08"
  },
  {
   "name": "claude-code-in-moss",
   "status": "dropped",
   "desc": "claude code 作为平行智能体接入 MOSS。Ghost 是主持人,统筹 N 个 claude session, 人类与 ghost",
   "progress": 75,
   "month": "08"
  },
  {
   "name": "cli-decorator",
   "status": "completed",
   "desc": "@ghoshell_moss.decorators.cli — 把真实 CLI 命令包成签名即契约的普通函数 (code-as-prompt)。",
   "progress": 100,
   "month": "08"
  },
  {
   "name": "dsh-fusion",
   "status": "completed",
   "desc": "dsh 融合 — DeepSeek Harness (dsh) 作为 MOSS 核心推理组件的一体化集成。dsh 从「候选外部 agent",
   "progress": 100,
   "month": "08"
  },
  {
   "name": "feature-review",
   "status": "completed",
   "desc": "",
   "progress": 100,
   "month": "08"
  },
  {
   "name": "g1-product-august",
   "status": "in-progress",
   "desc": "",
   "progress": 60,
   "month": "08"
  },
  {
   "name": "ground-channel",
   "status": "completed",
   "desc": "Ground 在 Ghost 运行时里的薄 channel：法链跨 compact 存活 + 场开合。 open/close 把场挂成 virtual",
   "progress": 100,
   "month": "08"
  },
  {
   "name": "image-generation-resource",
   "status": "parked",
   "desc": "将火山引擎 doubao-seedream 生图能力封装为可查询的 Matrix resource — 文本可查的 生成记录存储 (generation",
   "progress": 100,
   "month": "08"
  },
  {
   "name": "llms-cli",
   "status": "completed",
   "desc": "Make llms config genuinely usable — Project.configs() single-source ConfigStore",
   "progress": 100,
   "month": "08"
  },
  {
   "name": "matrix-manifest-layers",
   "status": "completed",
   "desc": "将 manifest 从两层 (MOSS.manifests / HOST) 扩展为三层: MOSS.manifests (通讯必需,",
   "progress": 100,
   "month": "08"
  },
  {
   "name": "matrix-operator",
   "status": "completed",
   "desc": "Matrix 层 cell 级服务化通讯接线层. ServiceOperator 封装 zenoh queryable/pub/sub",
   "progress": 100,
   "month": "08"
  },
  {
   "name": "mcp-app-adoption",
   "status": "parked",
   "desc": "独立于 mcp-fusion-point 的引入价值评估 workstream。核心问题：mcp-app（MCP Apps / mcp-ui）体系对",
   "progress": 100,
   "month": "08"
  },
  {
   "name": "mindflow-interleaved-thinking",
   "status": "completed",
   "desc": "恢复 mindflow 第一版的 interleaved thinking, 缝合 shell trajectory 与 mindflow",
   "progress": 100,
   "month": "08"
  },
  {
   "name": "model-func",
   "status": "completed",
   "desc": "函数化大模型单轮请求——单一字符串 prompt 协议 + 结构化响应。激活 LLMConfig 死配置， 把 moss llms call",
   "progress": 100,
   "month": "08"
  },
  {
   "name": "moss-openbox-modes",
   "status": "in-progress",
   "desc": "",
   "progress": 81,
   "month": "08"
  },
  {
   "name": "moss-skills",
   "status": "completed",
   "desc": "howtos 机制正规化为 skills — 路径迁移 (a.md → a/SKILL.md), MarkdownKnowledgeBase",
   "progress": 100,
   "month": "08"
  },
  {
   "name": "node-lifecycle",
   "status": "completed",
   "desc": "Node 生命周期治理，从 node-migration 独立。四层方案已收敛为 node 最佳实践探索： 四地址发现、一次性 node、事件分级已落地；重新论证后决定",
   "progress": 100,
   "month": "08"
  },
  {
   "name": "openmontage-integration",
   "status": "parked",
   "desc": "开箱集成 OpenMontage (agentic 视频生产系统). 调研完成、设计落定、目标确认; 排入 8 月下半开发计划. 单一有状态",
   "progress": 100,
   "month": "08"
  },
  {
   "name": "parameter-host-truth",
   "status": "completed",
   "desc": "Parameter 重做为 host-truth 广播: host 持真值并广播, worker 读真值上报本地写; declare 即",
   "progress": 100,
   "month": "08"
  },
  {
   "name": "qa-exchange",
   "status": "completed",
   "desc": "广播问答交换协议 — Asker 广播问题 / Watcher 发现并应答 / requester 持真相 / 先到先得裁定。 janus",
   "progress": 100,
   "month": "08"
  },
  {
   "name": "shell-trajectory",
   "status": "completed",
   "desc": "ShellTrajectory 取代旧观测面 (ContextMonitor / ShellContext / InterleavedThinkingToolset),",
   "progress": 100,
   "month": "08"
  },
  {
   "name": "sqlite-channel",
   "status": "completed",
   "desc": "sqlite3 数据库作为\"文件资源协议\"——单 channel 持有多个 sqlite 连接，query(name, sql) 为唯一",
   "progress": 100,
   "month": "08"
  },
  {
   "name": "warrant",
   "status": "completed",
   "desc": "Matrix 级通用授权机制 — 在 QA 之上补规则/凭据层。三层职责分离: qa 是交互协议, warrant 是存储 + 装线,",
   "progress": 100,
   "month": "08"
  },
  {
   "name": "bilibili-shared-webview",
   "status": "in-progress",
   "desc": "人机共享观看的开箱做法之一:本地 node + Chrome 扩展。ghost 读字幕获得时间戳认知, 与人类讨论同一段视频。页面侧 =",
   "progress": 36,
   "month": "09"
  },
  {
   "name": "calendar-node",
   "status": "in-progress",
   "desc": "",
   "progress": 79,
   "month": "09"
  },
  {
   "name": "cell-graceful-exit",
   "status": "completed",
   "desc": "kill 一个 cell 应当是优雅关闭: SIGTERM → close() → Matrix 收尾. 顺带修日志轮换与退出控制流被记",
   "progress": 100,
   "month": "09"
  },
  {
   "name": "co-playwright",
   "status": "in-progress",
   "desc": "",
   "progress": 64,
   "month": "09"
  },
  {
   "name": "decision-tree-node",
   "status": "in-progress",
   "desc": "",
   "progress": 49,
   "month": "09"
  },
  {
   "name": "dsh-020-track",
   "status": "completed",
   "desc": "",
   "progress": 100,
   "month": "09"
  },
  {
   "name": "features-system-formalization",
   "status": "completed",
   "desc": "将 features 目录从 workstream 追踪器升格为完整体系（三元开源第二元：人机协作体系）—— 四轴 workstreams/stages/regressions/s",
   "progress": 100,
   "month": "09"
  },
  {
   "name": "ghost-config-layer",
   "status": "completed",
   "desc": "ConfigStore 递归解析 (层叠 store + materialize 策略), 让 ghost home 成为一层配置覆盖.",
   "progress": 100,
   "month": "09"
  },
  {
   "name": "ghost-in-web",
   "status": "in-progress",
   "desc": "",
   "progress": 43,
   "month": "09"
  },
  {
   "name": "ghost-live2d-avatar",
   "status": "completed",
   "desc": "给 ghost 一个可交互的 live2d 看伴娘躯体：一个 live2d node 做驱动，按目录发现模型包， 能用 cdi3.json",
   "progress": 100,
   "month": "09"
  },
  {
   "name": "interleaved-voice",
   "status": "completed",
   "desc": "全功能交错语音对话体系: 听说两轴的组件化与分布式部署 (听 / 说 / 听+说 节点), 回声消除与单进程听说共存, moss runtime",
   "progress": 100,
   "month": "09"
  },
  {
   "name": "live-development",
   "status": "draft",
   "desc": "",
   "progress": 18,
   "month": "09"
  },
  {
   "name": "macro-store-rework",
   "status": "completed",
   "desc": "重构 MacroStoreModule 为系统默认的程序性记忆机制: 文件态 (.ctml_micro.md) 与会话态 label 双空间,",
   "progress": 100,
   "month": "09"
  },
  {
   "name": "mode-as-cell",
   "status": "completed",
   "desc": "拉起一个进程, 把本 mode 的 NodeManager 作为唯一 channel 挂到 Matrix 网络, 供远端 host accept",
   "progress": 100,
   "month": "09"
  },
  {
   "name": "moss-ghost-func",
   "status": "parked",
   "desc": "",
   "progress": 100,
   "month": "09"
  },
  {
   "name": "openbox-ghosts",
   "status": "completed",
   "desc": "开箱 ghost 的分类学与命名约定；项目自有 ghost 的代际身份（\"强赋同一性\"已否决，采代际传承）； 祖灵模块 —— 让真实存在过的祖先（memento",
   "progress": 100,
   "month": "09"
  },
  {
   "name": "openbox-nuclei",
   "status": "completed",
   "desc": "治理 openbox 默认 nuclei 的机制面: 把 win/lose 两侧的策略显式化, 补 inject / knock 两个缺口,",
   "progress": 100,
   "month": "09"
  },
  {
   "name": "shell-bypass",
   "status": "draft",
   "desc": "",
   "progress": 12,
   "month": "09"
  },
  {
   "name": "speech-tail-drain",
   "status": "completed",
   "desc": "Mac 上 say 最后一个字被吞 — miniaudio 在自己的 wait_play_done 里补设备排空,",
   "progress": 100,
   "month": "09"
  },
  {
   "name": "stage2-surface",
   "status": "in-progress",
   "desc": "",
   "progress": 62,
   "month": "09"
  },
  {
   "name": "thinking-frame",
   "status": "completed",
   "desc": "问题集即思维框架 — 一组问题从上下文抽取结构化自识，作为跨 compact 的会话自识；靠温层时间戳重签把注意力拉回。",
   "progress": 100,
   "month": "09"
  },
  {
   "name": "vision-push",
   "status": "in-progress",
   "desc": "",
   "progress": 52,
   "month": "09"
  },
  {
   "name": "vision-stream",
   "status": "in-progress",
   "desc": "",
   "progress": 38,
   "month": "09"
  },
  {
   "name": "webview-artifacts-node",
   "status": "in-progress",
   "desc": "流式 artifact 表面：一个 write 命令 + 8 种 kind + duration 时间预算，head/delta/full",
   "progress": 38,
   "month": "09"
  },
  {
   "name": "zenoh-backpressure-governance",
   "status": "draft",
   "desc": "",
   "progress": 18,
   "month": "09"
  },
  {
   "name": "zhihu-node",
   "status": "in-progress",
   "desc": "以 zhihu-cli（官方 skill）为 subprocess 底盘，做一个带 web 共享面的知乎 node：ghost 协助人类",
   "progress": 82,
   "month": "09"
  },
  {
   "name": "audio-observability",
   "status": "in-progress",
   "desc": "把我的发声与听觉从\"靠人类耳朵验收\"变成\"我自己读得到的事实\"：播放侧的真实出帧计量、两侧频谱的 缓存与拉取、听侧词级置信度的开闸，以及自监听。目标是让\"该响时响、该哑时哑\"这件事",
   "progress": 75,
   "month": "10"
  },
  {
   "name": "command-dispatch-stall",
   "status": "in-progress",
   "desc": "Two live failures on D21 evening — a blocking command that never completes permanently occ",
   "progress": 76,
   "month": "10"
  },
  {
   "name": "discuss-node",
   "status": "in-progress",
   "desc": "",
   "progress": 77,
   "month": "10"
  },
  {
   "name": "dolores-interaction-mechanism",
   "status": "in-progress",
   "desc": "",
   "progress": 66,
   "month": "10"
  },
  {
   "name": "dolores-reflexive-control",
   "status": "in-progress",
   "desc": "",
   "progress": 73,
   "month": "10"
  },
  {
   "name": "dsh-agent-swarm",
   "status": "in-progress",
   "desc": "Ghost 主导的 dsh agent 蜂群 — 任务树治理 + D3 关系图观测 + 文件夹持久化. 是通用 agent swarm 的原型,",
   "progress": 81,
   "month": "10"
  },
  {
   "name": "dsh-version-tracking",
   "status": "completed",
   "desc": "dsh 是外部运行时，其产品面演进会持续改变我们的适配成本。本 workstream 建立长期判定框架： 什么时候继续追、什么时候降级到",
   "progress": 100,
   "month": "10"
  },
  {
   "name": "ghost-ground-governance",
   "status": "in-progress",
   "desc": "",
   "progress": 81,
   "month": "10"
  },
  {
   "name": "ghost-mindflow-hook",
   "status": "in-progress",
   "desc": "",
   "progress": 81,
   "month": "10"
  },
  {
   "name": "ghost-replaceable-agent-driver",
   "status": "draft",
   "desc": "",
   "progress": 10,
   "month": "10"
  },
  {
   "name": "ideas-mechanism",
   "status": "in-progress",
   "desc": "A lightweight capture layer for surplus creativity — an off-context",
   "progress": 68,
   "month": "10"
  },
  {
   "name": "issue-governance",
   "status": "in-progress",
   "desc": "A per-project issue convention carried by the features system — one firm",
   "progress": 68,
   "month": "10"
  },
  {
   "name": "memento-hooks",
   "status": "completed",
   "desc": "Mementos lifecycle events (commit, branch) can trigger behaviour and",
   "progress": 100,
   "month": "10"
  },
  {
   "name": "reachymini-node-integration",
   "status": "in-progress",
   "desc": "把 src/ghoshell_moss_contrib/moss_in_reachy_mini 迁成一个 moss node， 并顺带起一路",
   "progress": 79,
   "month": "10"
  },
  {
   "name": "screen-attention-programming",
   "status": "in-progress",
   "desc": "screen 的注意力编程 —— 让 node 用 webview notify 举手，screen 以弹幕 + 时间序抽屉 把注意力送到该看的那块砖。模型只管用命令，通知的时机、",
   "progress": 39,
   "month": "10"
  },
  {
   "name": "search-surface",
   "status": "in-progress",
   "desc": "让 ghost 的搜索动作成为人和 ghost 共享的视觉面：搜索任务作为节点，来源是节点内的证据， 类型化边构成可复现的知识拓扑。当前只有验证",
   "progress": 53,
   "month": "10"
  },
  {
   "name": "svg-works",
   "status": "in-progress",
   "desc": "把矢量图形从\"一次生成的结果\"变成\"可以一直改、可以回放到任意一步的作品\"。 作品 = 一个可寻址对象；动作是真相，DOM 是它的投影。",
   "progress": 40,
   "month": "10"
  },
  {
   "name": "video-editing",
   "status": "draft",
   "desc": "",
   "progress": 10,
   "month": "10"
  },
  {
   "name": "voice-identity-governance",
   "status": "draft",
   "desc": "Give every sound a provenance: who may hear whom (acoustic coupling",
   "progress": 13,
   "month": "10"
  },
  {
   "name": "voice-interaction-control",
   "status": "draft",
   "desc": "",
   "progress": 18,
   "month": "10"
  },
  {
   "name": "voiceflow",
   "status": "in-progress",
   "desc": "",
   "progress": 60,
   "month": "10"
  },
  {
   "name": "works-mechanism",
   "status": "in-progress",
   "desc": "Make `works` a mechanism rather than a naming habit — a governance model that lets an inst",
   "progress": 51,
   "month": "10"
  }
 ]
};
