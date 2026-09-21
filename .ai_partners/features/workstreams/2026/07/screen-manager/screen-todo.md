# Screen Manager TODO

> screen-manager 的问题清单 —— **单一事实源**。dogfood 只负责发现与记录，状态在此维护。
> 状态: `open`(待修) / `uncertain`(不确定) / `fixed`(已修, 带 commit) / `verified`(下轮 dogfood 验证) / `invalid`(判定非 bug) / `superseded`(被重写取代)。
> 由 `screen-manager` FEATURE.md 关联索引。dogfood 发现新问题在此登记，修复/验证在此改状态。
> 编号在本 workstream 内唯一；跨 workstream 引用时写 `screen D3`。

> **2026-09-21 全量重写**：store / projection / channel / surface / index.html 按新语义
> （桌面为一级公民、至多一组、per-group layout+fullscreen、自动物化、dismiss/destroy 分家）
> 重写。D1–D4 是旧结构的缺陷，随重写**消失**，标 `superseded`。新结构本身尚未 dogfood，
> 其行为是**判据候选**，不是已复现的事实 —— 见下方「待 dogfood」。

## 缺陷（旧结构，已随重写消失）

| # | 状态 | Pri | 问题 |
|---|------|-----|------|
| D1 | superseded | P0 | 切组不撤全屏 —— 旧 `activate` 清全局 `_fullscreen` 但帧不带；新结构 fullscreen 是 per-group，切组天然切换 |
| D2 | superseded | P0 | 关掉全屏 item 不撤全屏 —— 旧 close 不补帧；新结构 dismiss/destroy 走 `_detach` 清 fullscreen + `state` 帧全量下发 |
| D3 | superseded | P1 | snapshot 不去重 → 重复 slot —— 旧 makeSlot 无幂等；新 snapshot 带全量 `items` + reconcile |
| D4 | superseded | P1 | family/dir 模型级 —— 新结构 layout 随 group 存（`Group` 对象），组删随组忘 |

## 待 dogfood（新结构的判据，尚未在浏览器复现）

| # | 判据 |
|---|------|
| V1 | `open` 落桌面 → 代理圆盘出现，iframe 隐藏；`arrange` 拉入组 → 圆盘消失、iframe 分格 |
| V2 | 切组 / 桌面↔组 有过渡，不闪屏（当前 `.slot` 有 CSS `grid-row/column` transition，是否够用待验） |
| V3 | 全屏某 item → 切组 → 回来仍全屏（per-group fullscreen 记忆） |
| V4 | 桌面代理圆盘物理（花束/布朗/环绕）不抖不漂出，点击命中准确 |
| V5 | `mock_scene` 连调两次 → 不叠 iframe（snapshot reconcile） |
| V6 | 字符雨字符集：CJK 均匀采样 + 假名点缀，不全是日文；图集渲染正常 |
| V7 | 桌面代理点击 → aside 到达模型（`tap` uplink） |

## 未接能力

| # | 状态 | 能力 | 依赖 |
|---|------|------|------|
| W1 | open | **background 真实音频数据面** — 唯一生产者仍是 `MockAudioSource`；真实生产者 = 订阅 `types/topics/audio.py` 的 `AudioSampleTopic` | audio topic 发布方 |
| W2 | open | **veil 视觉坐标桥** — 页面有 `geometry()`，模型侧无取几何/截图通道 | vision 通道 |
| W3 | open | **chrome 交互面** — 侧边 `#chrome` 只占位，输入框（配合语音输入法）、通知、历史未落地 | O1 输入协议 |
| W4 | open | **被嵌入页面的 resize 自适应契约** — 合成器只改 iframe viewport，不负责内部重排；avatar/terminal/file_editor 不监听 resize | 各节点自查 |
| W5 | open | **playwright 高阶面** — 真外部 web 内容（绕 X-Frame-Options）与接管浏览器 | playwright node |

## 设计问题

| # | 状态 | 问题 |
|---|------|------|
| O1 | open | **输入框未接线** — `#input` 只有外观，无上行帧类型；chrome 交互面（输入/通知/历史）整套未定 |
| O2 | open | **三档命名** — node 已定 `webview_screen`（档位进名字），但 qt/os 档命名、channel 命名是否跟档位绑定仍未定 |
| O3 | open | **qt 档 node 落点** — `qt_compositor.md` 说另起目录，新目录名待定 |
