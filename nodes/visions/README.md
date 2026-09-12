# Visions — MOSS 视觉感知 node 家族

`nodes/visions/` 是 vision 感知族的共享 venv 父目录（仿 `nodes/tools/` 共享组模式）。
家族内每个子 node 提供一路视觉感知面，共用环境，无 per-node venv。

vision 是一级开箱能力：主力模型能直接消费图像 —— 图像是**模型的感知输入**，不是给人看的
调试产物。

## 家族契约（每个 vision node 遵循）

1. **设备归属在 node 生命周期**：一个 node = 一个设备，多设备 = 多 node 实例。设备 open 于
   node start、close 于 node stop，不惰性开关。占用指示（如摄像头绿灯）因此是诚实的。

2. **config 三类分开**：node config（可调参数，ghost 在安全边界内自配、校验；走 node 级 env，
   dotenv 原生加载）、persistent config（授权 / 知情同意，落在 warrant 的存储，不进 node env）、
   argument（启动时可变量，用于身份 / 本地绑定，如设备 index）。不混存，也不混在同一段说明里。

3. **watch 门控每轮感知，不门控设备**：watch 只决定每轮是否携带当前图像进 context，默认
   OFF；设备仍由 node 持有，单次感知动作仍可用。常驻的代价是每轮一张图 —— 约定写进 channel
   的 instruction，`watch on` 的返回值给出当前每轮图像预算。

4. **像素走命令**：要能被记住的看，必须随命令返回图像数据（命令结果进历史；dynamic context
   瞬态）。任何可能被持久化的消息，必须在创建瞬间就真实 —— 图像带 ts / age / watch 状态。

5. **感知面是协议面**：跨 node 输出用强类型 topic（topic 模型即协议声明，暂无消费者也合法）。
   几何输出（人脸坐标一类）面向**关联设备**，不面向模型 —— 模型对时变坐标没有稳定语义，
   要向模型报告就只给质的判断。

6. **图形化是人类面**：本地推流 + 单页，与模型面共用同一采集源，不建第二套管线。

## 授权（知情同意）

摄像头、截屏是隐私敏感感知。授权状态是 **persistent config**（契约 2），持久化但不可由
ghost 自行调。当前的 `authorize` 命令与启动 announce 是轻量种子，**不阻断感知** —— 作用是让
双方都知道发生了什么（`qa` 交互 + 事件通知），不是审查。完整机制挂在 warrant 抽象上
（`moss codex blueprint warrant`），是已知扩展点。

## 子 node

| node | 路径 | 感知面 |
|---|---|---|
| camera | `nodes/visions/camera` | 相机视觉（cv2）+ 人脸 FaceTopic + MJPEG 推流 |
| screen_capture | `nodes/visions/screen_capture` | 屏幕截图（mss）— 尚未迁入 |

## 依赖分组备注

有意偏离 node-migration 的"vision 独立 venv（cv2 重依赖）"共识：vision 感知族是内聚
能力，共用家族 venv 是合理取舍。轻依赖 screen_capture 原本可能进 tools 共享组，此处
为了体系内聚并进来。
