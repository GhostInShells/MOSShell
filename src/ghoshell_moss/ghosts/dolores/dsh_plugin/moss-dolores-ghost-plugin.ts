import type { Context } from '@deepseek-ai/cordis'

export const name = 'moss-dolores-ghost-plugin'

export const inject: string[] = ['webServer']

/*
 * ═══════════════════════════════════════════════════════════════════════
 * Dolores ghost — dsh 内核特权桥插件 (设计文档, 逐步落地)
 * ═══════════════════════════════════════════════════════════════════════
 *
 * 定位: MOSS 侧 ghost (Dolores) 与 dsh 推理中枢之间的窄桥, 与 apiproxy 平级,
 *       经 ctx.webServer.register 挂 HTTP 路由. 接口刻意收窄, 不做"任意 append
 *       任意事件"的裸口 (见 FEATURE: 内核特权桥接).
 *
 * ── 1. Module 级状态 ────────────────────────────────────────────────────
 *   doloresEgoSessionId: SessionId | null   // ego session 唯一 id, 由 ghost 侧同步
 *   articulating: boolean                    // articulation 进行中标记 (语义待定)
 *
 * ── 2. Tool 表面 (4 个, 预定义但不默认注册) ─────────────────────────────
 *   full_facade()              — 拉全量 channel 操作面
 *   get_channel_facade(path)   — 拉单个 channel facade
 *   moss_observe(budget?)      — 读观测轨迹 (只读)
 *   ctml_interrupt()           — 紧急停止
 *
 *   保留这 4 个的理由: CTML 是模型的「文本输出」, 不是 JSON tool call —
 *   ctml_append/exec/replan 这类"发 CTML"的动词被文本流取代. 剩下的都是
 *   pull/control 旁路 (拉世界模型 / 拉结果 / 紧急停).
 *
 * ── 3. Ego session guard ───────────────────────────────────────────────
 *   每个 ego tool 运行前校验: 当前 session id === doloresEgoSessionId.
 *   不匹配立刻返回 "当前会话已经不是主任务" — 防 fork/subagent 会话 (共享
 *   preset 组合从而带着 ego tool schema) 越权使用 ego 特权.
 *
 * ── 4. Ego session id 同步 (双机制) ─────────────────────────────────────
 *   a) create ego session RPC: ghost 显式调用, 建 ego session + 注册 tool
 *      + 设 doloresEgoSessionId + 挂 preStep 拦截.
 *   b) 后台同步轮询: ghost 周期性重申/校准 ego id (防漂移/重启后失同步).
 *   TS 单线程事件循环, 无线程锁问题.
 *
 * ── 5. Tool 执行桥 (锁/解锁) ────────────────────────────────────────────
 *   真实逻辑: 模型调 ego tool → dsh 发 tool/call event → 上游 (MOSS ghost)
 *   监听 → 执行真实逻辑 → 经 RPC 回调 → 解锁 tool → 返回结果.
 *   dsh 里的 tool handler 是"锁住"的占位: 不等同进程执行, 而是等外部 (MOSS)
 *   经 RPC 投递结果才解锁. 关联键 = tool/call 的 callId.
 *
 *   锁原语: Promise + Map<CallId, {resolve, reject}>.
 *   TS 单线程事件循环下 map 操作在单个回合内原子, 无需锁 (不同于 Go channel /
 *   Python 线程模型). 每个 pending 挂 timeout, 防 MOSS 永久不应答时挂死.
 *
 * ── 6. 竞态 + 双 map 防御 ──────────────────────────────────────────────
 *   竞态: dsh 先广播 tool/call event, 才调用 tool handler — ghost 侧可能比
 *   plugin 侧更早拿到请求并完成 RPC 回调, 此时 handler 尚未注册 pending.
 *   双 map (early-arrival 模式, 两侧互查对方):
 *     pendingCalls:  Map<CallId, {resolve, reject}>  // handler 注册
 *     arrivedResults: Map<CallId, ToolResult>         // RPC 早到则先存
 *   handler: arrivedResults 命中 → 立即 resolve; 否则 pendingCalls 注册 + await.
 *   RPC:     pendingCalls 命中 → 立即 resolve; 否则 arrivedResults 存早到结果.
 *
 * ── 7. RPC 面 (ctx.webServer.register) ──────────────────────────────────
 *   POST /plugin-api/ego/create    — 建 ego session (tool 注册 + preStep + 设 id)
 *   POST /plugin-api/tool-result   — {callId, result} 解锁 pending tool
 *   (未来) 非 ego 观测 tool 面     — 下一步
 *
 * ── 待确认 ──────────────────────────────────────────────────────────────
 *   - dsh tool Consumer 是否把 tool/call 的 callId 透传给 handler (关联键来源).
 *   - preStep 拦截入参/出参 (PreStepDecision) 与 ego 语义的结合点.
 */

export function apply(ctx: Context) {
  // 工具面 / RPC 面按上述设计逐个落地.
}
