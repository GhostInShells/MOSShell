import { randomUUID } from 'node:crypto'
import type { IncomingMessage, ServerResponse } from 'node:http'

import type { Agent, ModelSelection, ModelSelectionRef } from '@deepseek-ai/dsh-agent'
import { installModelSelection } from '@deepseek-ai/dsh-agent'
import type { AgentDefaultModelConfig } from '@deepseek-ai/dsh-agent-default-model'
import { admitEncodedImages } from '@deepseek-ai/dsh-attachment'
import type { ImageAttachmentRef, ImageMediaType } from '@deepseek-ai/dsh-attachment'
import type { Context } from '@deepseek-ai/cordis'
import { createUserMessage, ReasoningEffortId, type UserMessage } from '@deepseek-ai/dsh-llm'
import type { ContentBlock } from '@deepseek-ai/dsh-llm'
import { foldSurface, isSurfaceEligibleType, SessionSeq } from '@deepseek-ai/dsh-session'
import type { JsonValue, Session, SessionEvent, SessionId, SurfaceEventType } from '@deepseek-ai/dsh-session'
import { PERSONA_PREFIX_SECTION, PERSONA_SUFFIX_SECTION, renderPrompt } from '@deepseek-ai/dsh-system-prompt'
import { defineTool } from '@deepseek-ai/dsh-tools'
import type { WorkspaceId } from '@deepseek-ai/dsh-workspace'

/*
 * ═══════════════════════════════════════════════════════════════════════
 * Dolores ghost — dsh 内核特权桥插件 (plugin 表面: 函数 + 注释, 实现逐步落地)
 * ═══════════════════════════════════════════════════════════════════════
 *
 * ── 设计定位 (2026-08-28 收敛, 见 .discuss 与 _ego.py 模块 docstring) ──
 * B 范式: MOSS mindflow 是 dsh 每个 turn 的「上下文服务方」, pre-step enter
 * 是服务接口. 每个 turn 自包含: enter-inject → model 跑 → turn/end 收线.
 * 帧按状态分叉: mindflow 活跃 → live moment; mindflow idle → 静态状态快照.
 *
 * 对比 A (dsh 解耦, 模型自己调 wait-moss tool 交接): B 让 MOSS 持有思考运行时
 * (mindflow 三循环), dsh 每 turn 是一次 MOSS 服务的无状态请求. 锁由 thinking
 * signal 提供 (TS 原生 async — 一个 pre-step await 的 promise, thinking/enter
 * resolve 它), 不需要 cancel/接管/双向阻塞的舞蹈, 也不需要 python 式伪 async.
 *
 * ── 表面 (8 点) ────────────────────────────────────────────────────────
 * 1. ego/create       — instruction + messages (ghost.memory: 压缩/快照/ground).
 * 2. thinking/enter   — context + inputs 两个 message 槽位 + epoch 槽位 + effort, 阻塞执行完.
 * 3. thinking/exit    — 反转 thinking 状态; agent 非 idle 则显式 cancel (interrupt).
 * 4. perStep 锁       — ego session 非 thinking → 阻塞等 thinking/enter 反转;
 *                       非主 ego (旁路) → 降级 (sandbox read-only) + 插入旁路
 *                       instruction 跑单轮, tools 全拒, turn/end 折叠 (旁路无残留).
 * 5. moment/epoch 映射 — python 侧组装, plugin 只收现成 content blocks (dumb transport,
 *                       不 parse xml-like). context (echoes/dynamic/executing → <moment>,
 *                       inject) + inputs (percepts + hint → <inputs>, steer) + epoch
 *                       (<epoch index=N> 容器: <recap> + <baseline>, inject, 变更时).
 * 6. moment 投放      — perStep 挂载点: enter 把 context/epoch 缓冲到 pendingMoments, pre-step
 *                       在 next() 后插到本步历史最前 (背景, 不驱动 turn); inputs → steer (输入, 驱动 turn).
 *                       enter 不 inbox inject — claim 已在 pre-step 顶部穿越, 晚到的 inject 落下一轮.
 *                       (修正: 早期 docstring 称 context → inject 是本轮, 那是误读; 见 agent-loop claim 时序.)
 * 7. tool 面          — observe (主动观测, approach a 内联返回 moment content blocks) +
 *                       moss_reasoning(effort) (纯声明默认思考档; ego 记下 → 下一轮 enter 携带
 *                       reasoning_effort → thinking/enter 应用到 per-agent selection, turn 边界生效);
 *                       interleaved_logos / switch_model deferred (落文档不实现).
 * 8. 时序图           — 见下方 ASCII.
 *
 * ── 时序: MOSS 驱动路径 (thinking = turn) ──────────────────────────────
 * [MOSS mindflow]      [plugin]                     [dsh agent loop]
 *      │ thinking start   │                              │
 *      │── thinking/enter │  {context, inputs, epoch, effort} │
 *      │                  │── openThinking()                │  (release pre-step gate)
 *      │                  │── inject(context) ─────────────▶│
 *      │                  │── steer(inputs) (若 idle) ─────▶│
 *      │                  │                              │── turn/start
 *      │                  │◀── agent/pre-step ───────────┤
 *      │                  │── enter (frame in surface) ──▶│
 *      │                  │                              │── model 跑 → logos 流
 *      │                  │◀── turn/end ─────────────────┤
 *      │ thinking exit    │                              │
 *      │── thinking/exit ─┤                              │
 *      │                  │── closeThinking()            │
 *      │                  │── cancel (非 idle) ─────────▶│
 *
 * ── 时序: 外部唤醒路径 (dsh UI 输入, mindflow idle) ────────────────────
 * [dsh UI]           [plugin]                    [dsh agent loop]      [MOSS]
 *      │ 输入 ──────────┤ steer ─────────────────────▶│
 *      │               │                              │── turn/start
 *      │               │◀── agent/pre-step ───────────┤
 *      │               │── thinking? false → await gate (阻塞)
 *      │               │── notifyExternalWake() ─────────────────────▶│  (seam)
 *      │               │                              │                 │ mindflow 处理
 *      │               │◀── thinking/enter {context, inputs} ──────────┤
 *      │               │── inject(context) + steer(inputs) + openThinking()
 *      │               │── release gate ─────────────▶│
 *      │               │── enter (frame in surface) ──▶│
 *      │               │                              │── model 跑 → 回应
 *
 * 接缝 (外部唤醒): 通知与背压分工 —
 *   通知 = turn/start 广播 (ego 侧 _on_turn_start 监听 → 自醒 signal, 已存在),
 *   不另发显式讯号. pre-step 阻塞只是背压 (hold 住模型等 thinking/enter 注入帧).
 *
 * ── observe tool (approach a) ────────────────────────────────────────
 * 主动观测, 与 yield 互补 (yield 被动让出, observe 主动观测). tool execute 挂
 * pendingCalls[callId] 阻塞 → MOSS 侧 thinking.observe() 生产 moment → /tool-result
 * RPC 按 callId 解锁, 内联返回 moment content blocks (context + inputs 拼接, 保留图片).
 * 不 break turn — 模型在 tool result 到达后继续思考 (interleaved thinking).
 *
 * ── thinking/exit 结算契约 (打断不再等于报错) ──────────────────────────
 * tool execute 与 MOSS 侧的结果回话是两个方向, 永远可能错位. thinking 结束时仍未结算的
 * call 已经拿不到这一轮的 moment, 于是**由 exit 直接结算**: 回一个普通结果 (interrupted),
 * 并登记墓碑 (settledCallIds) 让随后迟到的 /tool-result 被安静吞掉 — 而不是撞上"号已注销"
 * 报 400 打穿 MOSS 侧那一轮 (旧行为: 一次打断 = 一句 aborted + 一次 articulate 报错).
 * 契约细节见 settlePendingCallsOnExit / settledCallIds. MOSS 侧同样兜底吸收 RPC 失败.
 *
 * ── 遗留问题 ────────────────────────────────────────────────────────
 * 1. **epoch 周期接线**: epoch 槽位已实现 (<epoch> 容器), 但触发周期 (compact 压上下文)
 *    尚未装线 — recap/baseline 的生产接在 compact 上.
 * 2. **on_event 内部逻辑**: token 记账 / tool 桥 / seq 跟踪 (deferred).
 * 3. **command_logos 提示**: command_logos (<executing>) 是「感知」不是「输入」, 需在
 *    instruction 里 prompt 模型不要重复它 (待接).
 */

export const name = 'moss-dolores-ghost-plugin'

export const inject: string[] = ['webServer', 'workspaceRegistry', 'agents', 'systemPrompt', 'attachments', 'agentPresets', 'agentDefaultModel']

// 强相关路径命名空间: /moss-api/ghost/<ghost 名> — 体现 moss + ghost 类型 + dolores 实例, 不用通用 /plugin-api 弱命名.
const DOLORES_API_ROOT = '/moss-api/ghost/dolores'

// ego 专属 preset id (目录名即 id) — plugin 侧的单一权威. 内容由 launcher 从 repo 自持
// 的 dolores-ego preset 复制进 <DSH_HOME>/.agent-presets (不再从 shipped standard 逐字再生).
const DOLORES_EGO_PRESET = 'dolores-ego'

const DOLORES_EGO_CREATE = `${DOLORES_API_ROOT}/ego/create`
// 通用 session 观测面: 任意 live session 的 instruction / surface 读取 (sessionId 收在 body).
const DOLORES_SESSION_INSTRUCTION = `${DOLORES_API_ROOT}/session/instruction`
const DOLORES_SESSION_SURFACE = `${DOLORES_API_ROOT}/session/surface`
// thinking 事务面 (取代旧的 articulate/enter|exit): 帧注入 + 锁反转 + 退出 cancel.
const DOLORES_THINKING_ENTER = `${DOLORES_API_ROOT}/thinking/enter`
const DOLORES_THINKING_EXIT = `${DOLORES_API_ROOT}/thinking/exit`
const DOLORES_TOOL_RESULT = `${DOLORES_API_ROOT}/tool-result`
// 旁路原语: 走**身份旁路**单轮 — seed 成 dolores-ego preset 的旁路 session (id ≠
// doloresEgoSessionId → pre-step 自动降级 + tools 全拒 + turn/end 折叠), 不走 subagent,
// 不 attach ego workspace (attach 到 ghost_home 上的 home workspace 归组). 必须复用 ego
// preset 而非另建瘦 preset: 旁路 prompt 要与主路逐字节同前缀, 才能吃到 LLM 前缀缓存
// (见 plugin 顶注 / memento-plan #7).
// prompt 由调用方给 (note / chat 都是它的调用方) —— 语义留在 MOSS 侧, 这里只跑一轮.
const DOLORES_BYPASS_RUN = `${DOLORES_API_ROOT}/bypass/run`
// read: ref 的 turn 区间 → 源 log 的原始事件切片 (live-or-cold). 折叠成文本归 MOSS 侧.
const DOLORES_READ = `${DOLORES_API_ROOT}/read`
const HARNESS_IDENTITY_TEXT = ''

// ego workspace: project_home 上的 workspace, ego session 归组用, 模块级共享.
let doloresEgoWorkspaceId: WorkspaceId | null = null

// ghost home workspace: ghost_home 上的 workspace, 旁路 (note/chat) session 归组用, 模块级共享.
let doloresHomeWorkspaceId: WorkspaceId | null = null
let doloresHomePath: string | null = null

// ego session id + thinking 状态. id 由 ego/create 设.
let doloresEgoSessionId: SessionId | null = null

// 防旁路 token (点 4): ego/create 生成返回, thinking/enter|exit 校验 — 拒绝非 ego 发起的调用.
let doloresThinkingToken: string | null = null

// ego 的 persona 文本 (instruction) — ego/create 写入, session/start 时由 apply_ego_agent 注入 persona 段.
let doloresInstruction = ''

// cancel_turn 标记: 某个 tool 的结果协议带了 cancel → 该 turn 的下一 step 直接 cut.
// activeTurn 由 pre-step 更新 (当前正在跑的 turn); awaitToolResult 在 execute 时把 turn 存进
// pending, /tool-result 见到 cancel 时把它记成 cancelTurn. 带 turn 才不会误 cancel 下一轮.
let activeTurn: number | null = null
let cancelTurn: number | null = null

// ── 旁路 (bypass): 非主 ego session 的记账与注入 ─────────────────────
// 旁路 = ego-class (dolores-ego preset) 但 id ≠ doloresEgoSessionId 的 session. 它不跑
// thinking 事务、不消费 pendingMoments、工具全拒; 每次只跑一轮. bypassTurns 记每个旁路
// session 当前开的 turn 号 — 下一轮 pre-step 时据此折叠上一轮的 surface (前置清理).
const bypassTurns = new Map<SessionId, number>()

// 旁路 instruction 正文 — pre-step 前置注入, 让模型知道这是降级后的单轮旁路会话.
const BYPASS_INSTRUCTION = 'You are in a bypass (non-primary) ego session. This session has been superseded and is read-only: tools are disabled, so answer in a single turn without taking any action. Respond to the input below directly.'

// ── thinking 锁 (B 范式核心): pre-step await 的 gate, thinking/enter open ──
// TS 单线程事件循环, gate = asyncio.Event 等价物 (可反复 open/close, wait 阻塞到 open).
// 注意: Promise 是一次性的, resolve 后不能重臂 — 不能表达"当前是否 thinking"的持续状态.
// wait 返回三态 outcome: open (正常释放) / aborted (exit cancel 打断) / timeout (仅显式传超时).

type GateOutcome = 'open' | 'aborted' | 'timeout'

class ThinkingGate {
  private _open = false
  private _waiters: Array<() => void> = []

  get isOpen(): boolean {
    return this._open
  }

  open(): void {
    this._open = true
    const waiters = this._waiters
    this._waiters = []
    for (const resolve of waiters) resolve()
  }

  close(): void {
    this._open = false
  }

  async wait(timeoutMs?: number, signal?: AbortSignal): Promise<GateOutcome> {
    if (this._open) return 'open'
    if (signal?.aborted) return 'aborted'

    return await new Promise<GateOutcome>((resolve) => {
      let waiter: (() => void) | undefined
      const remove = (): void => {
        if (waiter !== undefined) {
          const i = this._waiters.indexOf(waiter)
          if (i >= 0) this._waiters.splice(i, 1)
          waiter = undefined
        }
      }
      waiter = () => { remove(); resolve('open') }
      this._waiters.push(waiter)
      signal?.addEventListener('abort', () => { remove(); resolve('aborted') }, { once: true })
      if (timeoutMs !== undefined) {
        setTimeout(() => { remove(); resolve('timeout') }, timeoutMs)
      }
    })
  }
}

const thinkingGate = new ThinkingGate()

function openThinking(): void {
  thinkingGate.open()
}

function closeThinking(): void {
  thinkingGate.close()
}

// tool 回调桥 (approach a): 需要 MOSS 侧 round-trip 的 tool (observe 等) execute 挂 pending
// promise, 由 /tool-result RPC 按 callId 解锁. Map keyed by callId — 多 tool 各自 pending
// 互不干扰. resolve 载荷 = 各 tool 的返回值 (observe = moment 文本 str).
// turn 在 execute 时刻捕获: 结果协议里的 cancel flag 只作用于调用它的那一轮.
const pendingCalls = new Map<string, { resolve: (value: unknown) => void; reject: (error: Error) => void; turn: number | null }>()

/**
 * 挂一个 tool 的 pending promise — execute 阶段调用, /tool-result RPC 按 callId 解锁.
 *
 * turn 在**此刻**捕获 (不是 resolve 时刻): cancel 只作用于调用它的那一轮. 结果协议里的
 * cancel flag 由 /tool-result 应用到 captured turn —— 迟到的回话 (已被 thinking/exit 结算)
 * 不会误 cancel 下一轮 (见 dolores-tool-surface.md 的 cancel flag 段).
 * abort 时 reject 并登记墓碑, 使随后迟到的 /tool-result 被安静吞掉.
 */
function awaitToolResult(toolName: string, callId: string, signal: AbortSignal): Promise<JsonValue> {
  return new Promise<unknown>((resolve, reject) => {
    pendingCalls.set(callId, { resolve: resolve as (value: unknown) => void, reject, turn: activeTurn })
    signal.addEventListener('abort', () => {
      if (pendingCalls.delete(callId)) {
        rememberSettled(callId)
        reject(new Error(`${toolName} aborted`))
      }
    }, { once: true })
  }) as unknown as Promise<JsonValue>
}

/**
 * 已结算但 MOSS 侧仍会回话的 callId 墓碑 (callId → 结算时刻毫秒).
 *
 * 时序契约: tool 的 execute 与 MOSS 侧的结果 RPC 是两个方向, 永远可能错位 —— thinking/exit
 * 先把 pending 结算掉, MOSS 的 /tool-result 随后才到. 没有墓碑时后者会撞上"号已注销" →
 * 400 → 异常穿回 _dispatch_tool_result → 整轮 articulate 报错. 墓碑让这次回话**被安静吞掉**,
 * 已结算的工具结果 (模型已看到) 不被改写.
 *
 * 一帧一份, 正常路径 (RPC 先到) 不产生墓碑 — 只有结算时该 callId 仍在 pending 才登记,
 * 所以墓碑表天然是小的; 再加 TTL 兜底, 极端情况下也不会长留.
 */
const settledCallIds = new Map<string, number>()

/** 墓碑 TTL — 超过即视为 MOSS 侧永不再回话, 丢弃 (兜底, 非功能路径). */
const SETTLED_CALL_TTL_MS = 60_000

function rememberSettled(callId: string): void {
  const now = Date.now()
  for (const [id, at] of settledCallIds) {
    if (now - at > SETTLED_CALL_TTL_MS) settledCallIds.delete(id)
  }
  settledCallIds.set(callId, now)
}

/**
 * thinking 退出时结算所有 pending tool — 把"被打断"变成一次**正常的工具返回**, 而不是异常.
 *
 * MOSS 侧宣布这一轮思考结束时, 还在等结果的 tool 已经不可能拿到这一轮的 moment 了; 与其
 * 让它以 rejected (aborted) 收场, 不如回一个模型能直接读懂的普通结果: 这一轮被中断了、
 * 这一帧不会被注入. 模型据此自行决定下一步 (重拉 / 直接回答), 整轮不再被一次迟到回话打穿.
 *
 * 返回值 = 本次结算掉的 callId 数 (观测用).
 */
function settlePendingCallsOnExit(): number {
  if (pendingCalls.size === 0) return 0
  const entries = [...pendingCalls.entries()]
  pendingCalls.clear()
  for (const [callId, pending] of entries) {
    rememberSettled(callId)
    pending.resolve({
      interrupted: true,
      message: 'the thinking turn ended before this call was answered — the moment was not injected; re-issue the call if you still need it',
    })
  }
  return entries.length
}

/** moment 的 wire content 段 — text 直传, image 为 base64 (dsh EncodedImageAttachment 形状). */
type MomentContentPart =
  | { type: 'text'; text: string }
  | { type: 'image'; mediaType: string; data: string }

/** perStep 挂载点的 moment 缓冲帧 — thinking/enter 写入, pre-step 消费 (插到本步历史最前). */
interface MomentFrame {
  /** 按注入顺序排好的 user message (epoch → context), 插到本步历史最前. */
  messages: UserMessage[]
}

/**
 * 缓冲队列 (FIFO): thinking/enter 写, agent/pre-step 排空. 消费后才清, **绝不丢消息** — 若
 * 此刻没有 pre-step 在消费 (无 turn 运行/门未开), 帧留守等下一次真实 pre-step. 缓冲到这里的
 * moment 不 inbox inject (claim 已在 pre-step 顶部穿越), 由挂载点插进本步历史.
 */
const pendingMoments: MomentFrame[] = []

/** thinking/enter 入参 (点 3). 阻塞执行完: handler 完成注入 + 开锁才返回. */
interface ThinkingEnterPayload {
  /** moment 拆两条 (python 侧映射): context (inject) + inputs (steer). */
  moment?: {
    /** context — echoes/dynamic/executing 折叠的 <moment> 容器 content blocks (inject). */
    context: MomentContentPart[]
    /** inputs — percepts + hint 的 <inputs> 容器 content blocks (steer, 允许空). */
    inputs: MomentContentPart[]
    moment_id?: string
  }
  /** epoch 变更时才携带 (python 侧比较 epoch.id): <epoch> 容器 content blocks (inject, 稳定背景). */
  epoch?: MomentContentPart[]
  /** memento notice (commit 提醒等) — 纯文本, 与 moment 同级注入, 只告知不驱动 turn. */
  notices?: string[]
  effort: string
  /** default effort (moss_reasoning 延迟生效): ego 记录 → 下一轮 enter 携带 → 应用到 selection. */
  reasoning_effort?: string
  /**
   * observe 续帧标记 (python 侧判定): 这一帧是「上一轮的回声要求再看一眼」产生的自我延续,
   * 而不是外部输入. 这种帧常常 inputs 为空, 但它**必须开一轮** —— 见 thinking/enter 的 turn 驱动规则.
   */
  needsObserve?: boolean
  /** 防旁路 (点 4): ego/create 返回的 token, 校验失败直接拒绝. */
  thinkingToken?: string
}

// ── ego tools (module-level, 构建一次) ──────────────────────────────────
// defineTool 在 profile 文件里能 import @deepseek-ai/dsh-tools (profiles/node_modules
// 链接农场在向上解析路径上), 所以这里直接定义; 注册动作在 apply_ego_agent (session/start
// 时) 完成, 不经过 agent preset (那样 defineTool 解析不到).
const egoTools = [
  defineTool({
    name: 'moss_ctml_append',
    // 唯一流式 tool: 参数只有 ctml, 经 tool-call-delta 逐字进 articulator (见 _ctml_stream.py).
    // 模型写一个大的 ctml 时, 不用等一次输出完再编译.
    description: 'Append CTML mid-thought so the world can see your ongoing thinking as you generate it. Your ctml is streamed into its own action and compiled as you write; the tool returns once it compiles (or "ctml syntax error" if it does not).',
    parameters: {
      ctml: { type: 'string', required: true, description: 'The CTML to append.' },
    },
    output: {
      schema: { type: 'json' },
      render: (_args, value) => [{ type: 'text', text: typeof value === 'string' ? value : JSON.stringify(value) }],
    },
    execute: async (_args, exec) => awaitToolResult('moss_ctml_append', String(exec.callId), exec.signal),
  }),
  defineTool({
    name: 'moss_wait_next_moment',
    // 结果协议带 cancel=true: MOSS 侧 wait_actions_done 后回结果, plugin 立刻 cut 本 turn,
    // 下一 step 不再跑 — 用它代替没人看的 final answer. turn 由 awaitToolResult 在 execute 时捕获.
    description: 'Wait for all your actions to finish, then yield the turn so the next moment wakes you. Use this in a voice/body interaction instead of emitting empty text nobody will read.',
    parameters: {},
    output: {
      schema: { type: 'json' },
      render: (_args, value) => [{ type: 'text', text: String(value) }],
    },
    execute: async (_args, exec) => awaitToolResult('moss_wait_next_moment', String(exec.callId), exec.signal),
  }),
  defineTool({
    name: 'moss_wait_action_done',
    // replan 非空 (含 "") 先发 replan action (clear interpreter + 这段 ctml) 替换当前 plan,
    // 再等全部结束 + observe 最新 moment; replan 缺省/null 表示只 wait 不 replan. timeout=-1 无限等.
    description: 'Wait for all your actions to finish, then observe the freshest moment (returned as a moment_ref). Pass replan as a CTML string to replace the current plan first (an empty string replans with nothing); omit replan to just wait. timeout is seconds to wait, or -1 to wait without bound.',
    parameters: {
      replan: { type: 'string', description: 'CTML to replan with before waiting; omit (null) to just wait. An empty string replans with nothing.' },
      timeout: { type: 'number', default: -1, description: 'Seconds to wait; -1 waits without bound.' },
    },
    output: {
      schema: { type: 'json' },
      render: (_args, value) => [{ type: 'text', text: typeof value === 'string' ? value : JSON.stringify(value) }],
    },
    execute: async (_args, exec) => awaitToolResult('moss_wait_action_done', String(exec.callId), exec.signal),
  }),
  defineTool({
    name: 'moss_reasoning',
    description: 'Set your default thinking depth (off / low / high / max). The ego records it and applies it from the next round; it stays until you change it again.',
    parameters: {
      effort: { type: 'string', required: true, enum: ['off', 'low', 'high', 'max'], description: 'How deeply to think.' },
    },
    output: {
      schema: { type: 'json' },
      render: (_args, value) => [{ type: 'text', text: String(value) }],
    },
    execute(args, _exec) {
      // 纯声明: 只返回 effort, 不立刻改 selection.current (perStep 会让下一 LLM call 在上下文未重编时
      // 换思考模式 → DeepSeek 报错). ego (MOSS) 监听 tool/call 记下 default effort, 下一轮
      // enter_thinking 携带 reasoning_effort, 由 thinking/enter 应用到 selection — turn 边界生效.
      return mapThinkingEffort(args.effort)
    },
  }),
  defineTool({
    name: 'moss_shell_status',
    description: 'Check what your Shell is doing right now, for the thinking that precedes acting. Returns a status description; no new moment comes with it.',
    parameters: {},
    output: {
      schema: { type: 'json' },
      render: (_args, value) => [{ type: 'text', text: String(value) }],
    },
    execute: async (_args, exec) => awaitToolResult('moss_shell_status', String(exec.callId), exec.signal),
  }),
  defineTool({
    name: 'moss_channel_facade',
    // 自省工具: recursive=true 列 path 前缀下的 channel (取代 moss_channels), false 读单个完整操作面.
    description: 'Read your channel operating surface. With recursive=true (default) list every channel under channel_path (empty = all); with recursive=false read one channel\'s full surface (instruction, commands, notices, state). For self-inspection — normally you already have this in context.',
    parameters: {
      channel_path: { type: 'string', required: true, description: 'The channel path (a prefix when recursive), e.g. "ghost.frame".' },
      recursive: { type: 'boolean', default: true, description: 'true lists channels under the path prefix; false reads one channel\'s full surface.' },
    },
    output: {
      schema: { type: 'json' },
      render: (_args, value) => [{ type: 'text', text: String(value) }],
    },
    execute: async (_args, exec) => awaitToolResult('moss_channel_facade', String(exec.callId), exec.signal),
  }),
]

// ── per-agent model selection (thinking/enter 应用 default effort 的目标) ──────
// 每个 ego agent 一份 selection, 装在它自己的 ctx 上 (installModelSelection), 不是模块单例.
// moss_reasoning 是纯声明 (不改 selection), ego 记下 default effort → 下一轮 enter 携带
// reasoning_effort → thinking/enter 咬 selection.current 的 reasoningEffort (turn 边界生效).
// provider/model 的权威始终是 canonical 链: picked → request/header(持久) → settings 默认;
// 改 effort 由 agent/request 应用并落 request/header 日志, 界面自然同步.
const egoSelections = new WeakMap<Agent, ModelSelectionRef>()

function ensureEgoSelection(agent: Agent, ctx: Context): ModelSelectionRef {
  const existing = egoSelections.get(agent)
  if (existing !== undefined) return existing
  const defaults: AgentDefaultModelConfig = ctx.agentDefaultModel
  let picked: ModelSelection | undefined
  const selection: ModelSelectionRef = {
    get current() {
      if (picked !== undefined) return picked
      const logged = agent.session.requestHeader()?.config
      if (logged !== undefined) {
        return {
          provider: logged.provider,
          model: logged.model,
          ...(logged.reasoningEffort === undefined ? {} : { reasoningEffort: logged.reasoningEffort }),
        }
      }
      return defaults.currentSelection()
    },
    set current(next) {
      picked = next
    },
    assembled: undefined,
  }
  egoSelections.set(agent, selection)
  return selection
}

/**
 * ego agent 的模型面装配 — session/start 时对 agent.ctx 做全套注册.
 *
 * identity/persona 段 + ego tools + perStep 锁都挂在 agent 自己的 scope ctx 上 (per-agent),
 * 替代「全局 perStep + setup 里注册」的旧形状. 由 agent/session-start 在 create 和 resume
 * 各触发一次 (fresh ctx), 保证 resume 后仍还原 — 不经过 agent preset (那样 defineTool 解析
 * 不到), 也不依赖 create-only 的 setup.
 */
function apply_ego_agent(agent: Agent, ctx: Context): void {
  const agentCtx = agent.ctx
  // shadow 全局 harness:identity 成 GIS/MOSS 身份.
  agentCtx.effect(() => agentCtx.systemPrompt.section({
    name: 'harness:identity',
    order: agentCtx.systemPrompt.getSectionOrder('HARNESS_IDENTITY'),
    text: HARNESS_IDENTITY_TEXT,
  }), 'dolores-ego-identity.section()')
  // shadow 全局 persona prefix 成 ghost instruction (ego/create 已写入 doloresInstruction).
  agentCtx.effect(() => agentCtx.systemPrompt.section({
    name: PERSONA_PREFIX_SECTION,
    order: agentCtx.systemPrompt.getSectionOrder('DEPLOYMENT_PERSONA_PREFIX'),
    text: doloresInstruction,
  }), 'dolores-ego-persona.section()')
  // shadow away standard 的 persona suffix (working directory 那句).
  agentCtx.effect(() => agentCtx.systemPrompt.section({
    name: PERSONA_SUFFIX_SECTION,
    order: agentCtx.systemPrompt.getSectionOrder('DEPLOYMENT_PERSONA_SUFFIX'),
    text: '',
  }), 'dolores-ego-persona-suffix.section()')
  // shadow away host 平面的 web-surface (dsh web 不是第一公民); harness:source 保留.
  agentCtx.effect(() => agentCtx.systemPrompt.section({
    name: 'app:web-surface',
    order: agentCtx.systemPrompt.getSectionOrder('WEB_SURFACE'),
    text: '',
  }), 'dolores-ego-web-surface.section()')
  // per-agent model selection — canonical 链读 provider/model, thinking/enter 应用 default effort.
  installModelSelection(agentCtx, ensureEgoSelection(agent, ctx))
  // ego tools 注册到 agent scope (scoped) — 只有 ego agent 可见.
  for (const tool of egoTools) {
    agentCtx.tools.register(tool)
  }
  // 旁路 tools 全拒 (动态判定): agent 一旦不再是主 (id ≠ doloresEgoSessionId), 其工具调用在
  // dispatch 前被 guard 拒掉 (guard 经 agent.ctx 注册只对该 agent 生效). 主路 agent 不受影响.
  agentCtx.tools.guard((exec) => {
    if (exec.agent !== undefined && exec.agent.id !== doloresEgoSessionId) {
      return 'This is a bypass session; tools are unavailable. Answer in a single turn only.'
    }
    return undefined
  })
  // perStep 锁 (per-agent): 只有当前 ego 放行 (背压等 thinking/enter), 非主 ego 走旁路分支.
  // 挂在 agent 自己的 ctx 上 — 只拦这个 agent 的 pre-step, 不用全局预设过滤.
  agentCtx.on('agent/pre-step', async ({ agent: stepAgent, turn, signal }, next) => {
    // 旁路分支 (纯身份判定, 不看 gate/token): 非主 ego session.
    if (stepAgent.id !== doloresEgoSessionId) {
      // 前置清理 (map 思路): 下一轮 pre-step 时折叠上一轮旁路的 surface, 取代 session/event
      // 后置清理 — 后者经 session carrier 的 scope 分发, plugin ctx 收不到事件. 上一轮的
      // instruction + 回答在下轮 pre-step 被 replace 掉, 模型看不到上一轮副作用.
      const prevTurn = bypassTurns.get(stepAgent.id)
      if (prevTurn !== undefined && prevTurn !== turn) {
        try {
          collapseTurn(stepAgent.session, prevTurn)
          ctx.logger.info('dolores: collapsed bypass turn %d in session %s', prevTurn, stepAgent.id)
        } catch (error) {
          ctx.logger.warn('dolores: bypass turn collapse failed: %s', String(error))
        }
      }
      bypassTurns.set(stepAgent.id, turn)
      // 降级 (先降级再放行): sandbox 改 read-only (sandbox/mode 是 last-wins, 追加即切换). 旁路只读、单轮.
      // effort 不再硬编码 low — 归 UI/约定驱动 (note 旁路走 bypass/run 显式注入).
      stepAgent.session.append('sandbox/mode', { mode: 'read-only' })
      const decision = await next()
      if (decision.kind === 'reject') return decision
      // 前置旁路 instruction → 成为该 turn 的 surface 节点, 下轮 pre-step 时被 collapseTurn 折叠.
      return { kind: 'enter', messages: [bypassInstruction(), ...decision.messages] }
    }
    // cancel_turn: 本 turn 的某个 tool 结果已带 cancel (wait_next_moment / react), 直接 cut.
    // 这个 turn/end (aborted) 就是 MOSS 侧 thinking 退出的信号 — MOSS 自己不主动 cancel.
    activeTurn = turn
    if (cancelTurn !== null && cancelTurn === turn) {
      cancelTurn = null
      stepAgent.cancel({ kind: 'hook', reason: 'moss cancel_turn' })
      return { kind: 'reject' }
    }
    await thinkingGate.wait(undefined, signal)
    const decision = await next()
    if (decision.kind === 'reject') return decision
    // perStep 挂载点: 把 enter 缓冲的 moment (epoch + context) 插到本步历史最前.
    if (pendingMoments.length > 0) {
      const queued = pendingMoments.splice(0, pendingMoments.length)
      const prefix = queued.flatMap(frame => frame.messages)
      return { kind: 'enter', messages: [...prefix, ...decision.messages] }
    }
    return decision
  })
}

/**
 * agent start 语义: 每个 ego agent 实例启动时装配一次.
 *
 * 载体随 dsh 版本漂移 (调研结论 2026-09-19, 对齐 0.1.5-rc.2):
 * - 0.1.5-rc.2: `agent/session-start` {agent, source}, 同步通知, 抛错无害.
 * - 0.1.6-alpha.2: 该事件被移除, 语义并进 `agent/created` {agent, source, signal},
 *   且 mode 由 emit 变 serial — 监听器抛错会 veto 掉 agent 创建. 追到那一版时这个
 *   handler 必须改成不抛错; 还要判 source 新增的 'clear'/'compact' 是否会新建 agent
 *   实例 (若是, 过滤条件要放开, 否则新实例拿不到 ego tools).
 *
 * 两个载体都挂, 靠 payload 形状区分: 只有带 source 的才承载 start 语义 (0.1.5 的
 * agent/created 只有 {agent}), 天然去重, 不会双跑.
 */
function installEgoAgentStart(ctx: Context, assemble: (agent: Agent) => void): void {
  const dispatch = (payload: unknown): void => {
    if (typeof payload !== 'object' || payload === null) return
    const { agent, source } = payload as { agent?: unknown; source?: unknown }
    if (agent === undefined || typeof source !== 'string') return
    if (source !== 'startup' && source !== 'resume') return
    assemble(agent as Agent)
  }
  ctx.on('agent/session-start', dispatch)
  ctx.on('agent/created', dispatch)
}

export function apply(ctx: Context) {
  // ── 0. agent start: 每个 ego agent 实例装配一次 ────────────────────────
  // startup 和 resume 都装配, 各自 fresh ctx — 这里调 apply_ego_agent 做 tools +
  // identity/persona + perStep 的全套注册, 替代「全局 perStep + setup 里注册」.
  // 非主 ego session (界面误建 / fork 出的旁路) 也走这里装配, 由 perStep 的旁路分支降级.
  installEgoAgentStart(ctx, agent => {
    if (agent.session.header.agentPreset !== DOLORES_EGO_PRESET) return
    apply_ego_agent(agent, ctx)
  })

  // ── 1. ego agent 创建 (点 1) ──────────────────────────────────────────
  // 入参: instruction (system prompt: baseline + identity + persona) +
  //       messages (ghost.memory: 压缩/快照/ground, ghost 侧组装后塞入).
  //       ping/pong 预热 (可选): 创建后验证 session 可服务.
  ctx.webServer.register({
    kind: 'exact',
    path: DOLORES_EGO_CREATE,
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (req.method !== 'POST') {
        res.writeHead(405, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify({ error: 'method not allowed' }))
        return
      }
      try {
        const body = await readJson(req)
        const {
          project_home: projectHome,
          project_name: projectName,
          title: sessionTitle,
          instruction,
          messages,
          permission,
          ghost_home: ghostHome,
          home_title: homeTitle,
        } = body
        if (typeof projectHome !== 'string' || projectHome === '') {
          throw new Error('project_home must be a non-empty string')
        }
        if (typeof projectName !== 'string' || projectName === '') {
          throw new Error('project_name must be a non-empty string')
        }
        if (typeof sessionTitle !== 'string' || sessionTitle === '') {
          throw new Error('title must be a non-empty string')
        }
        if (typeof instruction !== 'string') {
          throw new Error('instruction must be a string')
        }
        if (!Array.isArray(messages)) {
          throw new Error('messages must be an array of {text} context messages')
        }
        if (typeof permission !== 'string' || permission === '') {
          throw new Error('permission must be a non-empty string')
        }
        // 0. ego 专属 preset 必须已就位 (由 launcher 复制 repo 自持文件). create 挂载它的 id,
        //    否则 UnknownPresetError.
        // 1. ensure workspace over project_home, title = project_name.
        let workspace = await ctx.workspaceRegistry.resolveByPath(projectHome)
        if (workspace === undefined) {
          workspace = await ctx.workspaceRegistry.create(projectHome)
        }
        if (workspace.title !== projectName) {
          await workspace.setTitle(projectName)
        }
        doloresEgoWorkspaceId = workspace.id
        // 1b. ghost home workspace: ghost_home 上的 workspace, 旁路 note/chat session 归组用.
        //     title = home_title (如 "deepseek @ home"). 缺省 (无 home) 时跳过 —— 旁路回落 process.cwd().
        if (typeof ghostHome === 'string' && ghostHome !== '') {
          let homeWorkspace = await ctx.workspaceRegistry.resolveByPath(ghostHome)
          if (homeWorkspace === undefined) {
            homeWorkspace = await ctx.workspaceRegistry.create(ghostHome)
          }
          if (typeof homeTitle === 'string' && homeTitle !== '' && homeWorkspace.title !== homeTitle) {
            await homeWorkspace.setTitle(homeTitle)
          }
          doloresHomeWorkspaceId = homeWorkspace.id
          doloresHomePath = ghostHome
        }
        // persona 文本落到模块级, 供 apply_ego_agent 在 session/start 时注入 persona 段.
        doloresInstruction = instruction
        // 2. ref 存在时才走构造器 seed: seed = memory + 切点之后的 surface 尾巴 —— memory 必须排在
        //    尾巴之前 (前情提要在前, "刚刚发生"在后), 所以只有带尾巴这一路要把 memory 也放进 seed.
        //    无 ref 时保持原路 (create 空 session, 之后逐条 append memory), 不动既有主路行为.
        const ref = readRef(body.ref)
        const seed = ref === undefined ? [] : await buildEgoSeed(ctx, messages, ref)
        // 3. create ego session: 专属 preset (standard 的工具面) + overridden identity/persona.
        const sessionId = randomUUID()
        const handle = await ctx.agents.create({
          sessionId,
          ...(seed.length > 0 ? { seed } : {}),
          meta: { cwd: projectHome, agentPreset: DOLORES_EGO_PRESET },
          setup: async (agentCtx: Context) => {
            await agentCtx.get('agentPresets').mount(agentCtx, DOLORES_EGO_PRESET)
          },
        })
        doloresEgoSessionId = handle.agent.id
        doloresThinkingToken = randomUUID()
        // 4. title + sandbox mode + workspace membership (log-only events + account).
        handle.agent.session.append('session/title', { title: sessionTitle, messageSeqs: [], source: { kind: 'user' } })
        handle.agent.session.append('sandbox/mode', { mode: permission })
        await workspace.attachSession(handle.agent.id)
        // 5. 无 ref: 注入 ghost.memory 上下文 (点 1) — messages → user/message (surfaceOp append).
        //    这是初见上下文 = 建立模型首轮可见的表面.
        if (ref === undefined) {
          for (const msg of messages) {
            if (typeof msg?.text === 'string' && msg.text.length > 0) {
              handle.agent.session.append('user/message',
                createUserMessage({
                  content: [{ type: 'text', text: msg.text }],
                  source: { kind: 'plugin', plugin: name },
                }),
                { surfaceOp: 'append' })
            }
          }
        }
        // todo: ping/pong 预热 (可选) — 创建后验证 session 可服务, 失败返回错误.
        res.writeHead(200, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify({ sessionId: handle.agent.id, thinkingToken: doloresThinkingToken }))
      } catch (error) {
        res.writeHead(400, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify({ error: String(error) }))
      }
    },
  })

  // ── 通用 session 观测面: 读任意 live agent 的 instruction / surface (只读, 零副作用) ──

  ctx.webServer.register({
    kind: 'exact',
    path: DOLORES_SESSION_INSTRUCTION,
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (req.method !== 'POST') {
        res.writeHead(405, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify({ error: 'method not allowed' }))
        return
      }
      try {
        const agent = resolveLiveAgent(ctx, await readJson(req))
        // 现场组装当前指令: 与 request/header.system 同源 (agent-loop 用 renderPrompt(assembly) 生成 system).
        const assembly = await ctx.systemPrompt.assemble({ agent, scope: agent })
        const instruction = renderPrompt(assembly)
        res.writeHead(200, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify({ instruction }))
      } catch (error) {
        res.writeHead(400, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify({ error: String(error) }))
      }
    },
  })

  ctx.webServer.register({
    kind: 'exact',
    path: DOLORES_SESSION_SURFACE,
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (req.method !== 'POST') {
        res.writeHead(405, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify({ error: 'method not allowed' }))
        return
      }
      try {
        const agent = resolveLiveAgent(ctx, await readJson(req))
        // surface 投影: 只含 user/assistant/tool-result, 模型可见序, 尊重 compact replace.
        const messages = agent.session.deriveMessages()
        res.writeHead(200, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify({ messages }))
      } catch (error) {
        res.writeHead(400, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify({ error: String(error) }))
      }
    },
  })

  // ── 2. thinking/enter (点 2/3/5/6) ────────────────────────────────────
  // 入参 = moment 一条 user message + epoch + effort. handler 阻塞执行完才返回:
  //   1. moment 投放 — idle → steer (turn 输入); 非 idle → append (注入已在跑的 turn).
  //   2. openThinking — 释放 pre-step gate (外部唤醒路径的阻塞解除).
  ctx.webServer.register({
    kind: 'exact',
    path: DOLORES_THINKING_ENTER,
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (req.method !== 'POST') {
        res.writeHead(405, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify({ error: 'method not allowed' }))
        return
      }
      try {
        const body = await readJson(req) as unknown as ThinkingEnterPayload
        if (doloresEgoSessionId === null) {
          throw new Error('no ego session — call ego/create first')
        }
        if (body.thinkingToken !== doloresThinkingToken) {
          throw new Error('invalid thinkingToken — rejected (non-ego caller)')
        }
        const agent = resolveLiveAgent(ctx, { sessionId: doloresEgoSessionId })
        // default effort (moss_reasoning 延迟生效): ego 记录 → 下一轮 enter 携带 → 应用到 selection.
        // turn 边界生效 (上下文重编), 不在 perStep 改 — 避免思考模式中途切换报错.
        if (body.reasoning_effort === 'off' || body.reasoning_effort === 'low' || body.reasoning_effort === 'high' || body.reasoning_effort === 'max') {
          const selection = ensureEgoSelection(agent, ctx)
          const current = selection.current
          if (current !== undefined) {
            selection.current = { ...current, reasoningEffort: ReasoningEffortId(body.reasoning_effort) }
          }
        }
        // moment 拆两条 (python 侧映射): context (inject, 背景) + inputs (steer, 输入).
        const context = await durableMomentContent(ctx, body.moment?.context ?? [])
        const inputs = await durableMomentContent(ctx, body.moment?.inputs ?? [])
        // notices (memento commit 提醒等) — 纯文本, 与 moment 同级注入; 只告知不驱动 turn.
        const notices = Array.isArray(body.notices)
          ? body.notices.filter((notice): notice is string => typeof notice === 'string' && notice !== '')
          : []
        // none: 只吸收背景不驱动 turn — epoch/context 缓冲到 perStep 挂载点, 不 openThinking 不
        // steer, 等下一次真实 pre-step 消费. inputs 属 turn 驱动, none 帧不送.
        if (body.effort === 'none') {
          const epoch = Array.isArray(body.epoch) && body.epoch.length > 0
            ? await durableMomentContent(ctx, body.epoch)
            : undefined
          const frame = buildMomentFrame(epoch, context, notices)
          if (frame.length > 0) {
            pendingMoments.push({ messages: frame })
          }
          res.writeHead(200, { 'Content-Type': 'application/json' })
          res.end(JSON.stringify({ thinking: false }))
          return
        }
        // perStep 挂载点: epoch/context 不 inbox inject (claim 在 pre-step 顶部已穿越), 而是缓冲
        // 到 pendingMoments, 由 pre-step 在 next() 后插到本步历史最前 — 本 turn 生效.
        // 必须先 push 再 openThinking — gate 的 resolve 会同步调度 pre-step 续体为 microtask,
        // 若在 openThinking 后又 await, 续体可能在 push 落地前就 fold pendingMoments (丢帧).
        const epoch = Array.isArray(body.epoch) && body.epoch.length > 0
          ? await durableMomentContent(ctx, body.epoch)
          : undefined
        const frame = buildMomentFrame(epoch, context, notices)
        if (frame.length > 0) {
          pendingMoments.push({ messages: frame })
        }
        openThinking()
        if (inputs.length > 0) {
          // 正常 enter: steer inputs 驱动 turn. inputs 为空则不起 turn — turn 由真实输入驱动,
          // 不再为「无 percepts」造 'thinking' 占位.
          agent.steer(createUserMessage({
            content: inputs,
            source: { kind: 'user' },
          }))
        } else if (body.needsObserve === true) {
          // observe 续帧 (python 侧 needsObserve): 这一帧是「上一轮的回声要求再看一眼」的自我延续,
          // 不是外部输入 — 它往往没有 percepts, 于是 inputs 为空. 但**它必须自己开一轮**:
          // 早先"inputs 为空不起 turn"的规则本意是不为「无 percepts」造占位, 对自我延续不成立 ——
          // 少了这一支, 续帧只能滞留 pendingMoments 等下一次真实输入捎带, 于是 need_observe 亮着
          // 却不思考, 而那一帧又会在下一轮开头迟到落地 (fetch 的 moment 同样被这条缓冲拖着).
          // 用 steer 而非 inject: inject 只投递不唤醒, 此刻没有在跑的 step, 帧会一直躺着.
          // 空 content 只唤醒、不塞字 —— 回声内容照旧走 pendingMoments 注入. 若塞 "observe
          // continuation: N" 这类字, 会被模型当成用户输入.
          agent.steer(createUserMessage({
            content: [],
            source: { kind: 'plugin', plugin: name },
          }))
        }
        res.writeHead(200, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify({ thinking: true }))
      } catch (error) {
        res.writeHead(400, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify({ error: String(error) }))
      }
    },
  })

  // ── 3. thinking/exit (点 4) ───────────────────────────────────────────
  // 反转 thinking 状态; agent 非 idle 时显式 cancel, 不让 dsh 空跑失速.
  ctx.webServer.register({
    kind: 'exact',
    path: DOLORES_THINKING_EXIT,
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (req.method !== 'POST') {
        res.writeHead(405, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify({ error: 'method not allowed' }))
        return
      }
      try {
        const body = await readJson(req)
        if (doloresEgoSessionId === null) {
          throw new Error('no ego session — call ego/create first')
        }
        if (body.thinkingToken !== doloresThinkingToken) {
          throw new Error('invalid thinkingToken — rejected (non-ego caller)')
        }
        const agent = resolveLiveAgent(ctx, { sessionId: doloresEgoSessionId })
        closeThinking()
        // 先结算 (再 cancel). 这一轮思考已经结束, 仍在等 MOSS 侧回话的 tool 不会再拿到这一轮的
        // moment. 与其等 cancel 的 abort 把它变成 rejected, 不如现在给一个**普通结果** (interrupted),
        // 让模型自己决定下一步. 结算同时登记墓碑, 使 MOSS 侧随后到的 /tool-result 被安静吞掉.
        const settled = settlePendingCallsOnExit()
        if (settled > 0) {
          ctx.logger.info('dolores: settled %d pending tool call(s) on thinking/exit', settled)
        }
        // agent 非 idle → 显式 cancel (MOSS 已宣布 thinking 结束, 不让 dsh 空跑失速).
        if (agent.status !== 'idle') {
          agent.cancel({ kind: 'hook', reason: 'moss thinking/exit' })
        }
        // 兜底投递: 本交易残余的缓冲帧 (none/空 inputs 补帧没触发 turn, 没有 pre-step 消费) 直接
        // 投进 inbox (next-step), 由下一次 pre-step 认领 — 绝不丢. 必须在 cancel 之后: cancel 默认
        // 清空 inbox, 先 flush 会被抹掉.
        flushPendingMoments(agent)
        res.writeHead(200, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify({ thinking: false }))
      } catch (error) {
        res.writeHead(400, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify({ error: String(error) }))
      }
    },
  })

  // ── tool-result 桥 (approach a): MOSS 侧 /tool-result 按 callId 解锁 pending tool ──
  ctx.webServer.register({
    kind: 'exact',
    path: DOLORES_TOOL_RESULT,
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (req.method !== 'POST') {
        res.writeHead(405, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify({ error: 'method not allowed' }))
        return
      }
      try {
        const body = await readJson(req)
        const callId = String(body.callId ?? '')
        const pending = pendingCalls.get(callId)
        if (pending === undefined) {
          // 墓碑路径: 该调用已在 thinking/exit 被结算 (模型已收到普通结果), 这次回话是迟到的.
          // 静默吞掉 — 已结算的结果不被改写, 也绝不把"号已注销"变成 400 打穿 MOSS 侧这一轮.
          if (settledCallIds.has(callId)) {
            res.writeHead(200, { 'Content-Type': 'application/json' })
            res.end(JSON.stringify({ ok: true, dropped: 'already settled' }))
            return
          }
          throw new Error(`no pending tool call for ${callId}`)
        }
        if (doloresEgoSessionId === null) {
          throw new Error('no ego session — call ego/create first')
        }
        const agent = resolveLiveAgent(ctx, { sessionId: doloresEgoSessionId })
        pendingCalls.delete(callId)
        // moment 注入 (先 inject 后 resolve): 观察到的 moment 进下一个 step 的上下文.
        // 无 moment (缺省/空数组) 则不注入 — result 单独回给模型.
        if (Array.isArray(body.moment) && body.moment.length > 0) {
          const blocks = await durableMomentContent(ctx, body.moment as MomentContentPart[])
          agent.inject(createUserMessage({
            content: blocks,
            source: { kind: 'plugin', plugin: `${name}:moment` },
          }))
        }
        // result = tool 给模型的返回值 (observe 为 "{epoch}-{moment}" 短 id).
        pending.resolve(body.result)
        // cancel flag: 本调用所属那一轮的下一 step 立刻 cut. 只认 captured turn —— 迟到的回话
        // (已被 thinking/exit 结算) 不会误 cancel 下一轮. 见 awaitToolResult 的注释.
        if (body.cancel === true && pending.turn !== null) {
          cancelTurn = pending.turn
        }
        res.writeHead(200, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify({ ok: true }))
      } catch (error) {
        res.writeHead(400, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify({ error: String(error) }))
      }
    },
  })

  // ── 旁路原语 (单轮) — note / chat / 任何"用某段上下文跑一轮"的调用方 ─────────
  // 入参 {ref, prompt}. 冷读源 session 的 log → 尾部截断成 verbatim seed → 新建一个
  // 身份旁路 agent (tools 全拒 + turn/end 折叠) → 跑一轮 → 读最后一条 assistant text → dispose.
  // 不走 subagent (child 结果回流会驱动父 turn = 红线); 不 attach workspace (不污染工作区).
  ctx.webServer.register({
    kind: 'exact',
    path: DOLORES_BYPASS_RUN,
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (req.method !== 'POST') {
        res.writeHead(405, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify({ error: 'method not allowed' }))
        return
      }
      try {
        const body = await readJson(req)
        const ref = parseSessionRangeRef(body.ref)
        const prompt = typeof body.prompt === 'string' ? body.prompt : ''
        if (prompt === '') throw new Error('prompt must be a non-empty string')
        // 旁路约束参数面 (MOSS 按用途传值): 显式低思考 + 输出上限. 缺省 = 不覆盖 (走 dsh 默认).
        const reasoningEffort = typeof body.reasoning_effort === 'string' && body.reasoning_effort !== ''
          ? body.reasoning_effort
          : undefined
        const maxTokens = typeof body.max_tokens === 'number' && Number.isInteger(body.max_tokens) && body.max_tokens > 0
          ? body.max_tokens
          : undefined
        // seed = 源 session 的逐字节前缀 (seq 0..切点). live 走 snapshotEvents (未 flush 的事件也在),
        // 源已不在本进程 (历史 commit / 上次运行) 时回落持久化层冷读 — 两者契约同为 seq 0 连续.
        const events = await loadSourceEvents(ctx, ref.session_id)
        const seed = seedPrefix(events, ref.end_turn, ref.end_seq)
        const handle = await ctx.agents.create({
          sessionId: randomUUID(),
          seed,
          // 必须复用 ego preset: 旁路 prompt 要与主路逐字节同前缀, 否则 LLM 前缀缓存整条失效.
          // cwd = ghost_home (home workspace 归组); 未设 home workspace 时回落 process.cwd().
          meta: { cwd: doloresHomePath ?? process.cwd(), agentPreset: DOLORES_EGO_PRESET, seedLength: seed.length },
          setup: async (agentCtx: Context) => {
            await agentCtx.get('agentPresets').mount(agentCtx, DOLORES_EGO_PRESET)
            // 显式约束 (不靠 pre-step 身份分支): 单轮请求侧注入 effort + maxTokens.
            if (reasoningEffort !== undefined || maxTokens !== undefined) {
              agentCtx.on('agent/request', async (_payload, next) => {
                const resolved = await next()
                return {
                  ...resolved,
                  ...(reasoningEffort !== undefined ? { reasoningEffort: ReasoningEffortId(reasoningEffort) } : {}),
                  ...(maxTokens !== undefined ? { maxTokens } : {}),
                }
              })
            }
          },
        })
        // 归组进 home workspace (用完 dispose, log 仍留; 旧数据由用户删 ghost_home 目录重建).
        // best-effort: 归组是 UI 语义, cwd 不匹配 / workspace 已删时 note 仍照常产出, 只落不进 home 分组.
        if (doloresHomeWorkspaceId !== null) {
          const homeWorkspace = ctx.workspaceRegistry.get(doloresHomeWorkspaceId)
          if (homeWorkspace !== undefined) {
            await homeWorkspace.attachSession(handle.agent.id).catch((error) => {
              ctx.logger.warn('dolores: bypass attach to home workspace failed: %s', String(error))
            })
          }
        }
        try {
          handle.agent.followup(createUserMessage({
            content: [{ type: 'text', text: prompt }],
            source: { kind: 'plugin', plugin: name },
          }))
          await handle.agent.whenIdle()
          const message = lastAssistantText(handle.agent)
          res.writeHead(200, { 'Content-Type': 'application/json' })
          res.end(JSON.stringify({ message }))
        } finally {
          // dispose 必须 (dsh 活 session 无 LRU, 不销毁会泄漏); dispose 不删 log.
          await handle.dispose()
        }
      } catch (error) {
        res.writeHead(400, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify({ error: String(error) }))
      }
    },
  })

  // ── read: commit 区间 → 源 log 的原始事件切片 ─────────────────────────────
  // 回原始事件而不是 surface 投影 —— transcript 要 tool/call 这类只在 log 里的记录; 折叠成文本
  // 是 MOSS 侧 render_transcript 的事 (plugin 不解析事件语义). live-or-cold, 与 seed 同一源.
  ctx.webServer.register({
    kind: 'exact',
    path: DOLORES_READ,
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (req.method !== 'POST') {
        res.writeHead(405, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify({ error: 'method not allowed' }))
        return
      }
      try {
        const body = await readJson(req)
        const ref = parseSessionRangeRef(body.ref)
        const events = await loadSourceEvents(ctx, ref.session_id)
        const slice = sliceRange(events, ref)
        res.writeHead(200, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify({ events: slice }))
      } catch (error) {
        res.writeHead(400, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify({ error: String(error) }))
      }
    },
  })
}

/**
 * seedPrefix — ref → verbatim seed (源 log 的 seq 前缀). 镜像 apiproxy 的 fork cut:
 * 切在 end_seq (缺省时按 end_turn 反查 turn/end), 再向后吞 trailing standalone
 * (session/title / injection 等) 到下一个 turn/start. 切在 turn/end 天然 balanced.
 *
 * 区间是半开的 ``(start_turn, end_turn]``: ``end_turn`` **含端**进 seed, 它之后的原文归下一个区间 ——
 * 所以尾巴从 turn ``end_turn + 1`` 的 ``turn/start`` 起, 那个 +1 是区间定义, 不是读取侧的推测.
 */
function seedPrefix(
  events: readonly SessionEvent[],
  endTurn: number,
  endSeq: number | undefined,
): SessionEvent[] {
  let cut = -1
  if (endSeq !== undefined) {
    cut = events.findIndex(event => event.seq === endSeq)
    if (cut < 0) throw new Error(`end_seq ${endSeq} out of log range`)
  } else {
    for (let i = 0; i < events.length; i++) {
      const event = events[i]
      if (event.type === 'turn/end' && (event.data as { turn?: number }).turn === endTurn) cut = i
    }
    if (cut < 0) throw new Error(`no turn/end for turn ${endTurn}`)
  }
  let end = cut
  for (let i = cut + 1; i < events.length; i++) {
    if (events[i].type === 'turn/start') break
    end = i
  }
  return events.slice(0, end + 1)
}

/** 持久化读句柄的最小结构面 (不 import dsh-session-persistence, 保持包依赖面不变). */
type SessionReadHandle = {
  read(offset?: number, length?: number): Promise<{ events: readonly SessionEvent[] }>
  close(): Promise<void>
}

/**
 * 源 session 的完整事件 log — live 优先, 源已不在本进程时回落持久化层冷读.
 *
 * 冷读 (ctx.sessionPersistence) 对任何**已落盘**的 session 恒成立, 与是否 live 无关 ——
 * 它让重启后的 note 补漏 (源 session 已死) 也能成立. live 优先的理由: 运行中的 session
 * 尾部事件可能尚未落盘, 冷读会读到被截断的前缀, 而 seed 必须覆盖到切点.
 */
async function loadSourceEvents(ctx: Context, sourceId: string): Promise<readonly SessionEvent[]> {
  const live = ctx.agents.get(sourceId)
  if (live !== undefined) return live.session.snapshotEvents()
  const persistence = ctx.get('sessionPersistence') as
    | { open(id: string, access: 'read'): Promise<SessionReadHandle> }
    | undefined
  if (persistence === undefined) {
    throw new Error(`no live session for ${sourceId}, and sessionPersistence service is unavailable`)
  }
  const handle = await persistence.open(sourceId, 'read')
  try {
    return (await handle.read()).events
  } finally {
    await handle.close()
  }
}

/** memento commit 的坐标 (``DshSessionRef`` 的 wire 形): 源 session + 半开 turn 区间 ``(start, end]``. */
type SessionRangeRef = {
  session_id: string
  start_turn: number
  end_turn: number
  start_seq?: number
  end_seq?: number
}

/**
 * 组装一条 seed 事件 (契约: seq 0 连续, surface 类型必须带 surfaceOp).
 *
 * 需要一次断言: `SessionEvent` 是按 type 判别的联合, 而这里的 type 是动态取的, 编译器收窄不了.
 * 断言是安全的 —— 这些事件进构造器时会被逐条校验信封 (assertSessionEventEnvelope /
 * assertCurrentLlmShape) 与 surface 计划 (SurfaceManager.validateNext), 不合格直接抛.
 */
function seedEvent(type: SurfaceEventType, time: number, data: unknown): SessionEvent {
  return { type, seq: SessionSeq(0), time, data, surfaceOp: 'append' } as SessionEvent
}

/**
 * ego 带 ref 时的初始表面 = memory 事件 + 切点之后的 surface 尾巴.
 *
 * memory 必须排在最前: 它是前情提要 (ground + memento view), 尾巴是"刚刚发生". 两者合成一个
 * constructor seed, 所以整段是一次性表面. ref 的源 session 常常已经不在本进程 (上次运行留下的
 * commit) —— 走 loadSourceEvents 的冷读兜底, 冷态下 surface 节点由 foldSurface 从 log 折出,
 * 与 live 的 session.surface.nodes 同源同义.
 *
 * 代价要注意: 冷读 + 全量 fold 是 O(源 session log), 所以 ego 重建的开销由上一次 session 的
 * 体积决定 —— 这也正是它的上界 (最终由 session 切换的阈值兜住).
 */
async function buildEgoSeed(
  ctx: Context,
  messages: readonly { text?: unknown }[],
  ref: SessionRangeRef,
): Promise<SessionEvent[]> {
  const texts = messages
    .map(message => message?.text)
    .filter((text): text is string => typeof text === 'string' && text !== '')
  const seed = memorySeed(texts)
  const events = await loadSourceEvents(ctx, ref.session_id)
  const cut = resolveCut(events, ref)
  seed.push(...surfaceTailSeed(events, cut))
  return withSeq(seed)
}

/** memory 文本 → user/message 事件, 建立模型首轮可见的表面. */
function memorySeed(texts: readonly string[]): SessionEvent[] {
  return texts.map(text => seedEvent('user/message', Date.now(), createUserMessage({
    content: [{ type: 'text', text }],
    source: { kind: 'plugin', plugin: name },
  })))
}

/** body.ref 可空: 缺省 → undefined (调用方走无 ref 的路); 给了但形状不对 → 抛. */
function readRef(raw: unknown): SessionRangeRef | undefined {
  if (raw === undefined || raw === null) return undefined
  return parseSessionRangeRef(raw)
}

/** 解析 body.ref; 形状不对直接抛 (路由是外部入口, 不静默吞). */
function parseSessionRangeRef(raw: unknown): SessionRangeRef {
  const record = (raw ?? {}) as Record<string, unknown>
  const sessionId = record['session_id']
  const startTurn = record['start_turn']
  const endTurn = record['end_turn']
  if (typeof sessionId !== 'string' || sessionId === '') throw new Error('ref.session_id must be a non-empty string')
  if (!isTurn(startTurn)) throw new Error('ref.start_turn must be a non-negative integer')
  if (!isTurn(endTurn)) throw new Error('ref.end_turn must be a non-negative integer')
  const startSeq = asSeq(record['start_seq'])
  const endSeq = asSeq(record['end_seq'])
  return {
    session_id: sessionId,
    start_turn: startTurn,
    end_turn: endTurn,
    ...(startSeq === undefined ? {} : { start_seq: startSeq }),
    ...(endSeq === undefined ? {} : { end_seq: endSeq }),
  }
}

function isTurn(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0
}

function asSeq(value: unknown): number | undefined {
  return isTurn(value) ? value : undefined
}

/**
 * ref 的 turn 区间 → 源 log 的原始事件切片. 区间是半开的 ``(start_turn, end_turn]``.
 *
 * 左端取**第一个 turn 严格大于 start_turn 的 turn/start** —— turn ``start_turn`` 归上一个 commit
 * (它就是上一个的 ``end_turn``), 本区间不重复覆盖; 首个 commit 的 ``start_turn`` 是 0, dsh 的 turn 从
 * 1 起, 严格取第一个 > 0 的 turn 自然就是 turn 1, 不需要任何退让.
 * 右端取 ``end_turn`` 的 ``turn/end``, **含端**且必须存在 (它就是 commit 的边界).
 */
function sliceRange(events: readonly SessionEvent[], ref: SessionRangeRef): SessionEvent[] {
  const start = events.findIndex(event => event.type === 'turn/start' && turnOf(event) > ref.start_turn)
  const end = events.findLastIndex(event => event.type === 'turn/end' && turnOf(event) === ref.end_turn)
  if (start < 0) throw new Error(`no turn/start after turn ${ref.start_turn}`)
  if (end < 0) throw new Error(`no turn/end for turn ${ref.end_turn}`)
  if (end < start) throw new Error(`turn range (${ref.start_turn}, ${ref.end_turn}] is empty`)
  return events.slice(start, end + 1)
}

function turnOf(event: SessionEvent): number {
  return (event.data as { turn?: number }).turn ?? -1
}

/**
 * ref → 切点 seq, 镜像官方 fork 的 cut (`dsh-api-session-controller` 的 session/fork):
 * 先校正到 end_turn 的 `turn/end`, 再向后吞掉非 `turn/start` 的杂事件, 使切点落在 turn 边界上.
 *
 * 区间是半开的 ``(start_turn, end_turn]``: 被切的这一轮 (``end_turn``) **含端进摘要**, 原文尾巴从
 * 下一个 ``turn/start`` 起 —— 归属清晰, 与相邻区间不重叠.
 */
function resolveCut(events: readonly SessionEvent[], ref: SessionRangeRef): number {
  let boundary = -1
  if (ref.end_seq !== undefined) {
    boundary = events.findIndex(event => event.seq === ref.end_seq)
    if (boundary < 0) throw new Error(`ref.end_seq ${ref.end_seq} out of log range`)
  } else {
    for (let i = 0; i < events.length; i++) {
      const event = events[i]
      if (event.type === 'turn/end' && (event.data as { turn?: number }).turn === ref.end_turn) boundary = i
    }
    if (boundary < 0) throw new Error(`no turn/end for turn ${ref.end_turn}`)
  }
  let cut = boundary + 1
  while (cut < events.length && events[cut].type !== 'turn/start') cut += 1
  return cut
}

/**
 * 切点之后的 surface 节点 → seed 事件 (逐条以 append 重发), 即"模型真正看见过的那些消息".
 *
 * 三条依据: ① 只取 surface 节点, 被 replace 遮蔽过的内容不会复活; ② 一律改写成 append, seed 里
 * 没有 replace op, 不触发 surface 的 range/provenance 校验; ③ `tool/call` 本就不进 surface ——
 * 工具调用装在 assistant/message 的 content 块里 (实机验证过), 所以不丢。
 *
 * `events` 是完整 log 且 seq 连续从 0 (live 的 snapshotEvents / 冷读 read 同契约), 故可直接按
 * 下标取事件; surface 节点由 `foldSurface` 折出 (冷态唯一可用, 与 live 同源).
 */
function surfaceTailSeed(events: readonly SessionEvent[], cutSeq: number): SessionEvent[] {
  const seed: SessionEvent[] = []
  for (const seq of foldSurface(events).nodes) {
    if (seq < cutSeq) continue
    const event = events[seq]
    if (event === undefined || !isSurfaceEligibleType(event.type)) continue
    seed.push(seedEventFromTail(event))
  }
  return seed
}

/**
 * 尾部 surface 事件 → seed 事件: 只留 {type,time,data} 并以 append 重发.
 *
 * 刻意丢的: replace 专属的 `sourceEventSeqs` (我们只发 append); assistant/message 的 `stream`
 * (逐字重放数据 —— seed 校验只要求它是数组, 见 dsh-session 的 assertAssistantSettlementShape)。
 * `usage` 保留: 它是真实记账, 不是重放数据; `data` 其余字段原样透传.
 */
function seedEventFromTail(event: SessionEvent): SessionEvent {
  const data = event.type === 'assistant/message'
    ? { ...(event.data as Record<string, unknown>), stream: [] }
    : event.data
  return seedEvent(event.type, event.time, data)
}

/** 重排 seq 为 0 连续 (构造器 seed 契约: `snapshot.seq === index`). */
function withSeq(seed: SessionEvent[]): SessionEvent[] {
  return seed.map((event, index) => ({ ...event, seq: SessionSeq(index) }) as SessionEvent)
}

/** 最后一条非空 assistant text (镜像 python _final_response: assistant/message 的 text 块). */
function lastAssistantText(agent: Agent): string {
  const events = agent.session.snapshotEvents()
  for (let i = events.length - 1; i >= 0; i--) {
    const event = events[i]
    if (event.type !== 'assistant/message') continue
    const content = (event.data as { message?: { content?: Array<{ type?: string; text?: string }> } })
      ?.message?.content ?? []
    const text = content.filter(block => block.type === 'text').map(block => block.text ?? '').join('')
    if (text !== '') return text
  }
  return ''
}

/** moment contents (wire PromptContentPart) → durable ContentBlock[], admit base64 image 成 ref. */
async function durableMomentContent(ctx: Context, contents: readonly MomentContentPart[]): Promise<ContentBlock[]> {
  if (contents.every(part => part.type === 'text')) {
    return contents.map(part => ({ type: 'text', text: part.text }))
  }
  const images = contents.filter((part): part is Extract<MomentContentPart, { type: 'image' }> => part.type === 'image')
  const refs = await admitEncodedImages(ctx.attachments, images.map(({ mediaType, data }) => ({
    mediaType: mediaType as ImageMediaType,
    data,
  })))
  let next = 0
  return contents.map(part => part.type === 'text'
    ? { type: 'text', text: part.text }
    : { type: 'image', attachment: refs[next++] as ImageAttachmentRef })
}

/**
 * Enter 缓冲 — 组装 moment 帧消息 (按注入顺序): epoch <容器> 是底座消息, context <容器> 是
 * 坐落在其上的帧消息. 只做组装, 不 inbox inject; 由 pre-step 挂载点插到本步历史最前.
 */
function buildMomentFrame(
  epoch: ContentBlock[] | undefined,
  context: ContentBlock[],
  notices: string[] = [],
): UserMessage[] {
  const messages: UserMessage[] = []
  if (epoch !== undefined && epoch.length > 0) {
    messages.push(createUserMessage({
      content: epoch,
      source: { kind: 'plugin', plugin: `${name}:epoch` },
    }))
  }
  // notices (memento commit 提醒) — 纯文本, 与 moment 同级注入, 只告知不驱动 turn.
  if (notices.length > 0) {
    messages.push(createUserMessage({
      content: notices.map(text => ({ type: 'text' as const, text })),
      source: { kind: 'plugin', plugin: `${name}:notice` },
    }))
  }
  if (context.length > 0) {
    messages.push(createUserMessage({
      content: context,
      source: { kind: 'plugin', plugin: `${name}:moment` },
    }))
  }
  return messages
}

/** Exit 兜底 — 本交易残余缓冲帧直接投进 inbox (next-step), 由下一次 pre-step 认领, 绝不丢. */
function flushPendingMoments(agent: Agent): void {
  if (pendingMoments.length === 0) return
  // splice 原地排空并返回帧 — pendingMoments 是 const, 不能重绑 (会抛 "Assignment to constant variable").
  const queued = pendingMoments.splice(0, pendingMoments.length)
  for (const frame of queued) {
    for (const message of frame.messages) {
      agent.inject(message)
    }
  }
}

/** python ThinkingEffort → dsh reasoningEffort 档位. none 由调用方特判 (no turn);
 *  off / '' (default) → off; low/high/max 直传 (DeepSeek 不支持 medium). */
function mapThinkingEffort(effort: string | undefined): string {
  switch (effort) {
    case 'low':
    case 'high':
    case 'max':
      return effort
    default:
      return 'off'
  }
}

// ── helper: 旁路 instruction + turn 折叠 ────────────────────────────────

/** 旁路 instruction — 作为旁路 turn 的第一条 user/message 注入 (pre-step 前置). */
function bypassInstruction(): UserMessage {
  return createUserMessage({
    content: [{ type: 'text', text: BYPASS_INSTRUCTION }],
    source: { kind: 'plugin', plugin: name },
  })
}

/**
 * collapseTurn — 把旁路 turn 的 surface 节点折叠成一条空 user/message.
 *
 * 取该 turn 的 turn/start~turn/end seq 窗口内、当前仍挂在 surface 上的节点, 用一条空 content
 * 的 user/message replace 掉整段 (compaction 同款 replace 语义). 为何不是空 assistant/message:
 * deriveEventMessage 对空 assistant 返 null 才是"真零残迹", 但 session invariant 要求
 * assistant/message 命名当前 open step — 折叠发生在下一轮 pre-step (openStep 尚为 null, 未发
 * step/start), 空 assistant 仍过不了 requireOpenStep. 唯一合法的 replace 节点是 user/message
 * (invariant 对 user/message 无约束, 见 dsh-session/invariant).
 * 在 pre-step 前置清理路径里执行, 调用方已 try/catch.
 */
function collapseTurn(session: Session, turn: number): void {
  let startSeq: number | undefined
  let endSeq: number | undefined
  // 0.1.5: log 走 session.snapshotEvents() (Session 无 `events` 访问器).
  for (const event of session.snapshotEvents()) {
    if (event.type === 'turn/start' && event.data.turn === turn && startSeq === undefined) {
      startSeq = event.seq
    }
    if (event.type === 'turn/end' && event.data.turn === turn) {
      endSeq = event.seq
    }
  }
  const start = startSeq
  const end = endSeq
  if (start === undefined || end === undefined) return
  const shadowed = session.surface.nodes.filter(seq => seq >= start && seq <= end)
  if (shadowed.length === 0) return
  session.append('user/message', createUserMessage({
    content: [],
    source: { kind: 'plugin', plugin: name },
  }), {
    surfaceOp: { op: 'replace', start: shadowed[0], end: shadowed[shadowed.length - 1] },
    sourceEventSeqs: [...shadowed],
  })
}

function resolveLiveAgent(ctx: Context, body: Record<string, unknown>) {
  const sessionId = body.sessionId
  if (typeof sessionId !== 'string' || sessionId === '') {
    throw new Error('sessionId must be a non-empty string')
  }
  const agent = ctx.agents.get(sessionId)
  if (agent === undefined) {
    throw new Error(`no live agent for sessionId ${sessionId}`)
  }
  return agent
}

function readJson(req: IncomingMessage): Promise<Record<string, unknown>> {
  return new Promise((resolve, reject) => {
    let data = ''
    req.on('data', (chunk: Buffer) => { data += chunk })
    req.on('end', () => {
      try {
        resolve(JSON.parse(data) as Record<string, unknown>)
      } catch (error) {
        reject(error)
      }
    })
    req.on('error', reject)
  })
}
