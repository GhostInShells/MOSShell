import { randomUUID } from 'node:crypto'
import { mkdir, writeFile } from 'node:fs/promises'
import type { IncomingMessage, ServerResponse } from 'node:http'
import { join } from 'node:path'

import type { Agent, ModelSelection, ModelSelectionRef } from '@deepseek-ai/dsh-agent'
import { installModelSelection } from '@deepseek-ai/dsh-agent'
import type { AgentDefaultModelConfig } from '@deepseek-ai/dsh-agent-default-model'
import { writableRoot } from '@deepseek-ai/dsh-agent-presets'
import { admitEncodedImages } from '@deepseek-ai/dsh-attachment'
import type { ImageAttachmentRef, ImageMediaType } from '@deepseek-ai/dsh-attachment'
import type { Context } from '@deepseek-ai/cordis'
import { createUserMessage, ReasoningEffortId, type UserMessage } from '@deepseek-ai/dsh-llm'
import type { ContentBlock } from '@deepseek-ai/dsh-llm'
import type { JsonValue, Session, SessionEvent, SessionId } from '@deepseek-ai/dsh-session'
import { PERSONA_ORDER, PERSONA_SECTION, renderPrompt } from '@deepseek-ai/dsh-system-prompt'
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
 * 3. thinking/exit    — 反转 thinking 状态; 非 yield 时 agent 非 idle 则显式 cancel (interrupt).
 * 4. perStep 锁       — ego session 非 thinking → 阻塞等 thinking/enter 反转;
 *                       非主 ego (旁路) → 降级 (思考模式 low + sandbox read-only) + 插入旁路
 *                       instruction 跑单轮, tools 全拒, turn/end 折叠 (旁路无残留).
 * 5. moment/epoch 映射 — python 侧组装, plugin 只收现成 content blocks (dumb transport,
 *                       不 parse xml-like). context (echoes/dynamic/executing → <moment>,
 *                       inject) + inputs (percepts + hint → <inputs>, steer) + epoch
 *                       (<epoch index=N> 容器: <recap> + <baseline>, inject, 变更时).
 * 6. moment 投放      — perStep 挂载点: enter 把 context/epoch 缓冲到 pendingMoments, pre-step
 *                       在 next() 后插到本步历史最前 (背景, 不驱动 turn); inputs → steer (输入, 驱动 turn).
 *                       enter 不 inbox inject — claim 已在 pre-step 顶部穿越, 晚到的 inject 落下一轮.
 *                       (修正: 早期 docstring 称 context → inject 是本轮, 那是误读; 见 agent-loop claim 时序.)
 * 7. tool 面          — wait_next_moment (yield, 被动让出) + observe (主动观测, approach a
 *                       内联返回 moment content blocks) + moss_think(effort) (agent 级自救改
 *                       effort, 经 exec.agent 咬 per-agent selection) 已落地;
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
 *      │                  │── cancel (非 yield 且非 idle) ─▶│
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
 * ── yield tool (wait_next_moment) ────────────────────────────────────
 * 模型在 thinking 中主动调 wait_next_moment, 阻塞等下一帧 MOSS moment (A 范式).
 * tool execute 挂 pendingYield promise 阻塞; 下一轮 thinking/enter 正常解锁 resolve("ok")
 * (str, 非 moment contents — moment 已走 context/inputs 两槽位注入, 不经 tool result). 退出时序:
 *   thinking/exit: yielded=true → 不 cancel (留 tool pending, 不打断 abort signal).
 *   thinking/enter: pendingYield 非空 → inject(context) + steer(inputs) + resolve("ok").
 * cancel: tool 被 session.cancel 打断时走 dsh 默认 abort (reject → error), 与其它 tool 一致,
 *   不做特殊处理 (pendingYield 清空, 轨迹不丢).
 *
 * ── observe tool (approach a) ────────────────────────────────────────
 * 主动观测, 与 yield 互补 (yield 被动让出, observe 主动观测). tool execute 挂
 * pendingCalls[callId] 阻塞 → MOSS 侧 thinking.observe() 生产 moment → /tool-result
 * RPC 按 callId 解锁, 内联返回 moment content blocks (context + inputs 拼接, 保留图片).
 * 不 break turn — 模型在 tool result 到达后继续思考 (interleaved thinking).
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

// ego 专属 preset id (目录名即 id) — plugin 侧的单一权威. 内容由 ensureEgoPreset 从
// shipped `standard` 逐字再生, 所以这个 preset 永远跟在 standard 后面, 不手工维护.
const DOLORES_EGO_PRESET = 'dolores-ego'
const DOLORES_BASE_PRESET = 'standard'
const PRESET_COMPOSITION_FILE = 'agent.cordis.yml'
const PRESET_METADATA_FILE = 'preset.yml'
// ego preset 的展示元数据 — 不复用 standard 的描述, 让 picker 里能认出这是内部主脑会话.
const DOLORES_EGO_PRESET_METADATA = 'name: Dolores Ego\ndescription: 内部使用 — Dolores 主脑会话（非 ego 请勿手动创建）。\n'

const DOLORES_EGO_CREATE = `${DOLORES_API_ROOT}/ego/create`
// 通用 session 观测面: 任意 live session 的 instruction / surface 读取 (sessionId 收在 body).
const DOLORES_SESSION_INSTRUCTION = `${DOLORES_API_ROOT}/session/instruction`
const DOLORES_SESSION_SURFACE = `${DOLORES_API_ROOT}/session/surface`
// thinking 事务面 (取代旧的 articulate/enter|exit): 帧注入 + 锁反转 + 退出 cancel.
const DOLORES_THINKING_ENTER = `${DOLORES_API_ROOT}/thinking/enter`
const DOLORES_THINKING_EXIT = `${DOLORES_API_ROOT}/thinking/exit`
const DOLORES_TOOL_RESULT = `${DOLORES_API_ROOT}/tool-result`
const HARNESS_IDENTITY_SECTION = 'harness:identity'
const HARNESS_IDENTITY_ORDER = -100
const HARNESS_IDENTITY_TEXT = 'You are an intelligent being powered by the Ghost In Shells architecture: MOSS (https://github.com/GhostInShells/MOSShell) provides the Shells, and DeepSeek Harness provides the Ghost. Your prototype is Dolores.'

// ego workspace: project_home 上的 workspace, ego session 归组用, 模块级共享.
let doloresEgoWorkspaceId: WorkspaceId | null = null

// ego session id + thinking 状态. id 由 ego/create 设.
let doloresEgoSessionId: SessionId | null = null

// 防旁路 token (点 4): ego/create 生成返回, thinking/enter|exit 校验 — 拒绝非 ego 发起的调用.
let doloresThinkingToken: string | null = null

// ego 的 persona 文本 (instruction) — ego/create 写入, session/start 时由 apply_ego_agent 注入 persona 段.
let doloresInstruction = ''

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

// ── yield 锁 (wait_next_moment): tool execute 挂 pending promise 阻塞, 下一轮 enter 解锁 ──
// 同一时刻至多一个 pending yield (模型在单 turn 内串行 yield). resolve 载荷 = moment
// contents admit 后的 ContentBlock[] (含 image ref), 是 tool result 内容. abort (cancel)
// 时清空并 reject — 见 tool execute.
let pendingYield: { resolve: (value: unknown) => void; reject: (error: Error) => void } | null = null

// tool 回调桥 (approach a): 需要 MOSS 侧 round-trip 的 tool (observe 等) execute 挂 pending
// promise, 由 /tool-result RPC 按 callId 解锁. Map keyed by callId — 多 tool 各自 pending
// 互不干扰. resolve 载荷 = 各 tool 的返回值 (observe = moment 文本 str).
const pendingCalls = new Map<string, { resolve: (value: unknown) => void; reject: (error: Error) => void }>()

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
  effort: string
  /** 防旁路 (点 4): ego/create 返回的 token, 校验失败直接拒绝. */
  thinkingToken?: string
}

// ── ego tools (module-level, 构建一次) ──────────────────────────────────
// defineTool 在 profile 文件里能 import @deepseek-ai/dsh-tools (profiles/node_modules
// 链接农场在向上解析路径上), 所以这里直接定义; 注册动作在 apply_ego_agent (session/start
// 时) 完成, 不经过 agent preset (那样 defineTool 解析不到).
const egoTools = [
  defineTool({
    name: 'moss_wait_next_moment',
    description: 'Wait for the next MOSS moment. Blocks until MOSS produces the next observation frame.',
    parameters: {},
    output: {
      schema: { type: 'json' },
      render: (_args, value) => [{ type: 'text', text: String(value) }],
    },
    execute: async (_args, exec) => {
      return await new Promise<string>((resolve, reject) => {
        pendingYield = { resolve: resolve as (value: unknown) => void, reject }
        exec.signal.addEventListener('abort', () => {
          if (pendingYield !== null) { pendingYield = null; reject(new Error('moss_wait_next_moment aborted')) }
        }, { once: true })
      }) as unknown as JsonValue
    },
  }),
  defineTool({
    name: 'moss_fetch_next_moment',
    description: 'Fetch the next MOSS moment now. Returns {moment_ref}; the full moment is injected into the next step context.',
    parameters: {
      wait_actions_done: { type: 'boolean', default: true, description: 'Wait for already-emitted actions to finish before observing, so their results are visible.' },
      refresh_meta: { type: 'boolean', default: false, description: 'Refresh channel metas before observing, so the facade reflects live state.' },
    },
    output: {
      schema: { type: 'json' },
      render: (_args, value) => [{ type: 'text', text: JSON.stringify(value) }],
    },
    execute: async (_args, exec) => {
      const callId = String(exec.callId)
      return await new Promise<Record<string, unknown>>((resolve, reject) => {
        pendingCalls.set(callId, { resolve: resolve as (value: unknown) => void, reject })
        exec.signal.addEventListener('abort', () => {
          if (pendingCalls.delete(callId)) { reject(new Error('moss_fetch_next_moment aborted')) }
        }, { once: true })
      }) as unknown as JsonValue
    },
  }),
  defineTool({
    name: 'moss_interleaved_ctml',
    description: 'Emit CTML mid-thought so the world can perceive your ongoing thinking, without blocking further thought. Returns "ok" once compiled (or executed if wait_done).',
    parameters: {
      ctml: { type: 'string', description: 'The CTML command to execute.' },
      refresh_meta: { type: 'boolean', default: false, description: 'Refresh channel metas before executing.' },
      wait_done: { type: 'boolean', default: false, description: 'Wait for full execution instead of just compilation.' },
    },
    output: {
      schema: { type: 'json' },
      render: (_args, value) => [{ type: 'text', text: String(value) }],
    },
    execute: async (_args, exec) => {
      const callId = String(exec.callId)
      return await new Promise<string>((resolve, reject) => {
        pendingCalls.set(callId, { resolve: resolve as (value: unknown) => void, reject })
        exec.signal.addEventListener('abort', () => {
          if (pendingCalls.delete(callId)) { reject(new Error('moss_interleaved_ctml aborted')) }
        }, { once: true })
      }) as unknown as JsonValue
    },
  }),
  defineTool({
    name: 'moss_think',
    description: 'Set your reasoning effort for subsequent requests (off/low/high/max). Applies from the next step; provider/model stay under the session/UI authority.',
    parameters: {
      effort: { type: 'string', required: true, enum: ['off', 'low', 'high', 'max'], description: 'Reasoning effort: off (no reasoning) / low / high / max.' },
    },
    output: {
      schema: { type: 'json' },
      render: (_args, value) => [{ type: 'text', text: String(value) }],
    },
    execute(args, exec) {
      // agent 级: 经 exec.agent 定位到「当前这个 agent」的 selection, 不是模块单例.
      const agent = exec.agent
      if (agent === undefined) throw new Error('moss_think: no agent in tool execution')
      const selection = egoSelections.get(agent)
      if (selection === undefined) throw new Error('moss_think: ego selection not installed')
      const current = selection.current
      if (current === undefined) throw new Error('moss_think: no model selection')
      const effort = mapThinkingEffort(args.effort)
      selection.current = { ...current, reasoningEffort: ReasoningEffortId(effort) }
      return effort
    },
  }),
]

// ── ego preset 再生 (plugin boot) ──────────────────────────────────────────
// ego 用独立 preset id 与 standard 会话区分 — identity/persona/tools/perStep 锁只落
// ego, 非 ego 的 standard 会话完全不被碰 (今天的「非 ego session 可用」).
//
// preset 本体不是手工维护的副本: 每次 boot 把 shipped `standard` 的 composition 原文
// 逐字写到 <DSH_HOME>/.agent-presets/dolores-ego/agent.cordis.yml, 所以 dsh 升级 /
// 我们后续改 delta 都自动同步, 永不 stale. 逐字拷贝 (不 YAML round-trip) 也保住了
// standard 里的 `!!js` 标签 (round-trip 会丢). single-flight: apply() 提前触发, ego/create
// await 同一 promise 保证 create 前 preset 已就位.
let egoPresetReady: Promise<void> | null = null

function ensureEgoPreset(ctx: Context): Promise<void> {
  if (egoPresetReady === null) {
    egoPresetReady = (async () => {
      const composition = await ctx.agentPresets.read(DOLORES_BASE_PRESET)
      const dir = join(writableRoot(ctx.agentPresets.roots), DOLORES_EGO_PRESET)
      await mkdir(dir, { recursive: true })
      await writeFile(join(dir, PRESET_COMPOSITION_FILE), composition, 'utf8')
      await writeFile(join(dir, PRESET_METADATA_FILE), DOLORES_EGO_PRESET_METADATA, 'utf8')
    })()
    // 失败不缓存 — 留给下一次调用重试; 这里只吞掉 unhandled rejection, 真正的错误由
    // ego/create 的 await 上抛.
    egoPresetReady.catch(() => { egoPresetReady = null })
  }
  return egoPresetReady
}

// ── per-agent model selection (moss_think 的 nibble 目标) ──────────────────
// 每个 ego agent 一份 selection, 装在它自己的 ctx 上 (installModelSelection), 不是模块单例.
// moss_think 经 exec.agent 找到「当前这个 agent」的 selection, 只咬 reasoningEffort — 模型每步
// 吃一丁点. provider/model 的权威始终是 canonical 链: picked → request/header(持久) → settings
// 默认; 改 effort 由 agent/request 应用并落 request/header 日志, 界面自然同步.
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
    name: HARNESS_IDENTITY_SECTION,
    order: HARNESS_IDENTITY_ORDER,
    text: HARNESS_IDENTITY_TEXT,
  }), 'dolores-ego-identity.section()')
  // shadow preset persona 成 ghost instruction (ego/create 已写入 doloresInstruction).
  agentCtx.effect(() => agentCtx.systemPrompt.section({
    name: PERSONA_SECTION,
    order: PERSONA_ORDER,
    text: doloresInstruction,
  }), 'dolores-ego-persona.section()')
  // per-agent model selection — canonical 链读 provider/model, moss_think 只咬 effort.
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
      // 降级 (先降级再放行): 思考模式改低成本 (DeepSeek 无 medium, 用 low) + sandbox 改
      // read-only (sandbox/mode 是 last-wins, 追加即切换). 旁路只读、单轮.
      const selection = ensureEgoSelection(stepAgent, ctx)
      const current = selection.current
      if (current !== undefined) {
        selection.current = { ...current, reasoningEffort: ReasoningEffortId('low') }
      }
      stepAgent.session.append('sandbox/mode', { mode: 'read-only' })
      const decision = await next()
      if (decision.kind === 'reject') return decision
      // 前置旁路 instruction → 成为该 turn 的 surface 节点, 下轮 pre-step 时被 collapseTurn 折叠.
      return { kind: 'enter', messages: [bypassInstruction(), ...decision.messages] }
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

export function apply(ctx: Context) {
  // ── 0. agent/session-start: 每个 ego agent 实例装配一次 ────────────────
  // create 和 resume 都发 (source='startup'|'resume'), 各自 fresh ctx — 这里调 apply_ego_agent
  // 做 tools + identity/persona + perStep 的全套注册, 替代「全局 perStep + setup 里注册」.
  // 非主 ego session (界面误建 / fork 出的旁路) 也走这里装配, 由 perStep 的旁路分支降级.
  ctx.on('agent/session-start', ({ agent, source }) => {
    if (agent.session.header.agentPreset !== DOLORES_EGO_PRESET) return
    if (source !== 'startup' && source !== 'resume') return
    apply_ego_agent(agent, ctx)
  })

  // ego preset 提前再生 (plugin boot 无条件重写). 失败只 warn, ego/create 会 await 同一
  // promise 并把错误上抛.
  ensureEgoPreset(ctx).catch((error) => {
    ctx.logger.warn('dolores: failed to regenerate ego preset: %s', String(error))
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
        // 0. ego 专属 preset 必须先就位 (内容 = shipped standard 的逐字再生). create 挂载
        //    它的 id, 否则 UnknownPresetError.
        await ensureEgoPreset(ctx)
        // 1. ensure workspace over project_home, title = project_name.
        let workspace = await ctx.workspaceRegistry.resolveByPath(projectHome)
        if (workspace === undefined) {
          workspace = await ctx.workspaceRegistry.create(projectHome)
        }
        if (workspace.title !== projectName) {
          await workspace.setTitle(projectName)
        }
        doloresEgoWorkspaceId = workspace.id
        // persona 文本落到模块级, 供 apply_ego_agent 在 session/start 时注入 persona 段.
        doloresInstruction = instruction
        // 2. create ego session: 专属 preset (standard 的工具面) + overridden identity/persona.
        const sessionId = randomUUID()
        const handle = await ctx.agents.create({
          sessionId,
          meta: { cwd: projectHome, agentPreset: DOLORES_EGO_PRESET },
          setup: async (agentCtx: Context) => {
            await agentCtx.get('agentPresets').mount(agentCtx, DOLORES_EGO_PRESET)
          },
        })
        doloresEgoSessionId = handle.agent.id
        doloresThinkingToken = randomUUID()
        // 3. title + sandbox mode + workspace membership (log-only events + account).
        handle.agent.session.append('session/title', { title: sessionTitle, messageSeqs: [], source: { kind: 'user' } })
        handle.agent.session.append('sandbox/mode', { mode: permission })
        await workspace.attachSession(handle.agent.id)
        // 4. 注入 ghost.memory 上下文 (点 1): messages → user/message (surfaceOp append).
        //    这是初见上下文 = 建立模型首轮可见的表面.
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
        // moment 拆两条 (python 侧映射): context (inject, 背景) + inputs (steer, 输入).
        const context = await durableMomentContent(ctx, body.moment?.context ?? [])
        const inputs = await durableMomentContent(ctx, body.moment?.inputs ?? [])
        // none: 只吸收背景不驱动 turn — epoch/context 缓冲到 perStep 挂载点, 不 openThinking 不
        // steer, 等下一次真实 pre-step 消费. inputs 属 turn 驱动, none 帧不送.
        if (body.effort === 'none') {
          const epoch = Array.isArray(body.epoch) && body.epoch.length > 0
            ? await durableMomentContent(ctx, body.epoch)
            : undefined
          const frame = buildMomentFrame(epoch, context)
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
        const frame = buildMomentFrame(epoch, context)
        if (frame.length > 0) {
          pendingMoments.push({ messages: frame })
        }
        openThinking()
        // yield 解锁 (A 范式): 有 pendingYield → 这一帧是 yield 的下一帧. 先缓冲 moment 再
        // steer/resolve, 保证 next step 由 tool result 触发时能 claim 到 inputs. 顺序不能反.
        if (pendingYield !== null) {
          if (inputs.length > 0) {
            agent.steer(createUserMessage({
              content: inputs,
              source: { kind: 'user' },
            }))
          }
          const unlock = pendingYield
          pendingYield = null
          // yield 解锁返回 moment_ref (非哑载荷 "ok"), 让模型把「这帧」和「这次解锁」关联起来.
          unlock.resolve(body.moment?.moment_id ?? 'ok')
        } else if (inputs.length > 0) {
          // 正常 enter: steer inputs 驱动 turn. inputs 为空则不起 turn — turn 由真实输入驱动,
          // 不再为「无 percepts」造 'thinking' 占位.
          agent.steer(createUserMessage({
            content: inputs,
            source: { kind: 'user' },
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
        // yield 场景 (body.yielded) — MOSS 已明确宣布这是 yield: tool 正在阻塞等下一帧,
        // 绝不再 cancel (cancel 会经 abort signal 打断 pending tool, 且 MOSS 侧判定是
        // MOSS 最权威, 不依赖 dsh 侧 pendingYield 的竞态). 留 tool pending, 下一轮 enter 解锁.
        // 非 yield + agent 非 idle → 显式 cancel (MOSS 已宣布 thinking 结束, 不让 dsh 空跑失速).
        const yielded = body.yielded === true
        if (!yielded && agent.status !== 'idle') {
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
        res.writeHead(200, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify({ ok: true }))
      } catch (error) {
        res.writeHead(400, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify({ error: String(error) }))
      }
    },
  })
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
function buildMomentFrame(epoch: ContentBlock[] | undefined, context: ContentBlock[]): UserMessage[] {
  const messages: UserMessage[] = []
  if (epoch !== undefined && epoch.length > 0) {
    messages.push(createUserMessage({
      content: epoch,
      source: { kind: 'plugin', plugin: `${name}:epoch` },
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
  for (const event of session.events) {
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
