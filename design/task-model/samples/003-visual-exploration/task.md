---
id: "003"
name: 视觉探索：四种隐喻的原型（本轮）
swarm: moss
workspace: .ai_partners/features/workstreams/2026/10/dsh-agent-swarm/prototypes
parent: null
opened: D28 20:46
closed: D28 20:50
state: done
---

# 003 · 视觉探索（本轮）

## 1. 目的

在 `prototypes/` 里做一轮广度优先的视觉概念探索：四种互不相同的隐喻各一个可运行原型，
再回头把戴森球方案做深，最后给对比与推荐。**产物可以停在中间，但每一步都要能看。**

## 2. 组件清单

| 类型 | 名字 | 职责 |
|---|---|---|
| loop | 自驱探索 loop | 立 goal → 排步数 → 做 → 截图自评 → 改 → 收口 |
| agent-preset | `deepseek-flash / max` | 用户把思考深度提到 max |
| skill | 前端可视化（SVG 投影 / CSS 动效 / 性能预算） | — |
| tool | `co_browser` | 自己看自己的图（截图 → 读图 → 判） |
| tool | `terminal` / bash | 落盘 |
| tool | `webview_screen` | 投屏给人看 |
| boundary | `prototypes/` | 只动原型，不碰 src / nodes / 配置 |

## 3. 图（父 task）

```
op-1 立 GOAL（目标/步数/判据）
  ├─ sub-003a 05 领地图      ─ 2 轮（固定 blob → 内容生成凸包）
  ├─ sub-003b 06 工程图      ─ 2 轮（正交走线 → 径向 + 引出标注）
  ├─ sub-003c 07 流水线      ─ 1 轮（浅色底一次成立）
  └─ sub-003d 04 戴森加深    ─ 3 轮（细分 320 面 → 六边形面板 → 时间铺法）
        └─ 子任务都只在自己的目录里写，互不跨边界
op-2 出 INDEX（自评表 + 结论）
op-3 投屏（8 扇窗，3×3）
```

## 4. 边界与授权

- 父 task 的边界是 `prototypes/`；四个子任务**都在同一个 workspace 里**签发，
  所以只共享这份边界，不需要新的授权层。
- 用户的授权是一次性的："我给你开完全权限，改动尽量不要出原型探索界限。"

## 5. 出口与事实

`done`（这一轮收口，探索还在继续）：

- `design/visual-language/decision-log.md`（目标与决策记录）、`design/visual-language/README.md`（自评与推荐）
- `design/visual-language/prototypes/04…07.html`、`.../shots/*.png`
- 四条结论：主题=语法；形状由内容生成；纬度=时间最强；静息与审计必须分开

## 6. 视图对照

- L1 任务：四个子任务并行，边界同一个 —— 这是"一个 workspace 内多任务"的样本。
- L2 设计：上面那棵树。
- L2 运行：每个子任务里的"改主意"节点（05 从 blob 改凸包、04 从 80 面改 320 面）
  才是真实的运行图，不是教科书里的直线。
