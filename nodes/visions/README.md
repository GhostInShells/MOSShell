# Visions — MOSS 视觉感知 node 家族

`nodes/visions/` 是 vision 感知族的共享 venv 父目录（仿 `nodes/tools/` 共享组模式）。
家族内每个子 node 提供一路视觉感知面，共用环境，无 per-node venv。

## 家族契约（每个 vision node 遵循）

1. **可配置化暴露**：`config` 在安全边界内暴露可调项（相机 index/分辨率/fps、监视器），
   ghost 自行配置，取值校验，持久化到 node home。
2. **watch toggle**：`watch(on|off)` 门控主动感知，暴露给模型。ON = 后台持续采帧进
   滚动缓存（+ 可选分析/推流），OFF = 完全空闲。
3. **状态走 help + shell-trajectory**：`help` 反映实时状态（设备、watch 开关、最近采帧
   ts、缓存大小），变更落观测轨迹，ghost 感知增量。
4. **极简图形化**：本地 HTTP 推流（MJPEG）+ 单个网页，人类同步看到 ghost 的视野。
5. **时序 capture 走缓存**：滚动 `(ts, frame)` 缓存是时序脊柱，`capture` 返回调用瞬间
   的快照，非阻塞。

## 授权（知情同意）种子

摄像头、截屏是隐私敏感感知。vision node 应该用 `qa`（答案来自用户）+ `event` 通知
ghost 节点在等待授权或授权失败。q 是交互，不是审查——让双方都知道发生了什么。

> 当前的授权是**轻量种子**（`authorize()` 命令 + 启动 announce），不阻断感知。
> 完整的 per-node 知情同意 / warrant 授权机制（P2 warrant workstream）可在此之上
> 逐步堆叠——这是已知扩展点，而非最终形态。

## 子 node

| node | 路径 | 感知面 |
|---|---|---|
| camera | `nodes/visions/camera` | 相机视觉（cv2）+ 人脸 FaceTopic + MJPEG 推流 |
| screen_capture | `nodes/visions/screen_capture` | 屏幕截图（mss）— 尚未迁入 |

## 依赖分组备注

有意偏离 node-migration 的"vision 独立 venv（cv2 重依赖）"共识：vision 感知族是内聚
能力，共用家族 venv 是合理取舍。轻依赖 screen_capture 原本可能进 tools 共享组，此处
为了体系内聚并进来。
