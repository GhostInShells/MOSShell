import { randomUUID } from 'node:crypto'
import type { IncomingMessage, ServerResponse } from 'node:http'

import { admitEncodedImages } from '@deepseek-ai/dsh-attachment'
import type { ImageAttachmentRef, ImageMediaType } from '@deepseek-ai/dsh-attachment'
import type { Context } from '@deepseek-ai/cordis'
import { createUserMessage } from '@deepseek-ai/dsh-llm'
import type { ContentBlock } from '@deepseek-ai/dsh-llm'
import type { JsonValue, SessionId } from '@deepseek-ai/dsh-session'
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
 * 2. thinking/enter   — context + inputs 两个 message 槽位 + epoch 槽位 + effort + model config, 阻塞执行完.
 * 3. thinking/exit    — 反转 thinking 状态; 非 yield 时 agent 非 idle 则显式 cancel (interrupt).
 * 4. perStep 锁       — foreign session → reject + mux 提示冻结;
 *                       ego session 非 thinking → 阻塞等 thinking/enter 反转.
 * 5. moment/epoch 映射 — python 侧组装, plugin 只收现成 content blocks (dumb transport,
 *                       不 parse xml-like). context (echoes/dynamic/executing → <moment>,
 *                       inject) + inputs (percepts + hint → <inputs>, steer) + epoch
 *                       (<epoch index=N> 容器: <recap> + <baseline>, inject, 变更时).
 * 6. moment 投放      — context/epoch → inject (背景, 不驱动 turn); inputs → steer (输入, 驱动 turn).
 * 7. tool 面          — wait_next_moment (yield, 被动让出) + observe (主动观测, approach a
 *                       内联返回 moment content blocks) 已落地; interleaved_logos /
 *                       switch_model deferred (落文档不实现).
 * 8. 时序图           — 见下方 ASCII.
 *
 * ── 时序: MOSS 驱动路径 (thinking = turn) ──────────────────────────────
 * [MOSS mindflow]      [plugin]                     [dsh agent loop]
 *      │ thinking start   │                              │
 *      │── thinking/enter │  {context, inputs, epoch, effort, model} │
 *      │                  │── applyModelConfig()            │
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
 * 1. **applyModelConfig todo**: thinking/enter 的 provider/model/reasoningEffort
 *    未应用到下个 request (agent/request waterfall / session.selectModel).
 * 2. **epoch 周期接线**: epoch 槽位已实现 (<epoch> 容器), 但触发周期 (compact 压上下文)
 *    尚未装线 — recap/baseline 的生产接在 compact 上.
 * 3. **on_event 内部逻辑**: token 记账 / tool 桥 / seq 跟踪 (deferred).
 * 4. **command_logos 提示**: command_logos (<executing>) 是「感知」不是「输入」, 需在
 *    instruction 里 prompt 模型不要重复它 (待接).
 */

export const name = 'moss-dolores-ghost-plugin'

export const inject: string[] = ['webServer', 'workspaceRegistry', 'agents', 'systemPrompt', 'tools', 'attachments']

// 强相关路径命名空间: /moss-api/ghost/<ghost 名> — 体现 moss + ghost 类型 + dolores 实例, 不用通用 /plugin-api 弱命名.
const DOLORES_API_ROOT = '/moss-api/ghost/dolores'

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
  model: { provider: string; model: string; reasoningEffort?: string }
  /** 防旁路 (点 4): ego/create 返回的 token, 校验失败直接拒绝. */
  thinkingToken?: string
}

export function apply(ctx: Context) {
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
          agent_preset: agentPreset,
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
        if (typeof agentPreset !== 'string' || agentPreset === '') {
          throw new Error('agent_preset must be a non-empty string')
        }
        if (typeof permission !== 'string' || permission === '') {
          throw new Error('permission must be a non-empty string')
        }
        // 1. ensure workspace over project_home, title = project_name.
        let workspace = await ctx.workspaceRegistry.resolveByPath(projectHome)
        if (workspace === undefined) {
          workspace = await ctx.workspaceRegistry.create(projectHome)
        }
        if (workspace.title !== projectName) {
          await workspace.setTitle(projectName)
        }
        doloresEgoWorkspaceId = workspace.id
        // 2. create ego session: standard preset (tools) + overridden identity/persona.
        const sessionId = randomUUID()
        const handle = await ctx.agents.create({
          sessionId,
          meta: { cwd: projectHome, agentPreset },
          setup: async (agentCtx: Context) => {
            await agentCtx.get('agentPresets').mount(agentCtx, agentPreset)
            // shadow global harness:identity with the GIS/MOSS identity.
            agentCtx.effect(() => agentCtx.systemPrompt.section({
              name: HARNESS_IDENTITY_SECTION,
              order: HARNESS_IDENTITY_ORDER,
              text: HARNESS_IDENTITY_TEXT,
            }), 'dolores-ego-identity.section()')
            // shadow preset persona with the ghost instruction.
            agentCtx.effect(() => agentCtx.systemPrompt.section({
              name: PERSONA_SECTION,
              order: PERSONA_ORDER,
              text: instruction,
            }), 'dolores-ego-persona.section()')
            // perStep 锁 (点 4): 两分支 — foreign reject + mux 提示 / ego 帧背压等反转.
            agentCtx.on('agent/pre-step', async ({ agent, signal }, next) => {
              if (agent.id !== doloresEgoSessionId) {
                // foreign session (fork/subagent 共享 preset 而带 ego tool schema) →
                // 直接 reject + mux 提示「session 已冻结」. seam: mux 提示形态待定
                // (ask-user 对话框 / log-only 事件 / plugin-source user message).
                notifySessionFrozen(ctx, agent)
                return { kind: 'reject' }
              }
              // ego session: 帧背压 — 锁到 MOSS thinking/enter 开闸 (open) 才放行, 不设超时.
              // 注意: pre-step 进入时 driver 已 claim 消息 (移出 inbox). 任何 reject 都会让
              // driver 把 turn 记 blocked 并丢弃已 claim 的消息 (吞). 故 ego 分支绝不 reject.
              // aborted (被 cancel 打断) 时 next() 走 signal.throwIfAborted 收成 aborted turn.
              await thinkingGate.wait(undefined, signal)
              return next()
            })
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
  // 入参 = moment 一条 user message + epoch + effort + model config. handler 阻塞执行完才返回:
  //   1. applyModelConfig — provider/model/reasoningEffort 应用到下个 request.
  //   2. moment 投放 — idle → steer (turn 输入); 非 idle → append (注入已在跑的 turn).
  //   3. openThinking — 释放 pre-step gate (外部唤醒路径的阻塞解除).
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
        openThinking()
        // epoch 变更时: <epoch> 容器作为 epoch 级稳定背景 (inject, 不驱动 turn). 在 moment
        // context 之前注入 — epoch 是底座, moment 是坐落在其上的帧.
        if (body.epoch !== undefined && body.epoch.length > 0) {
          const epochBlocks = await durableMomentContent(ctx, body.epoch)
          agent.inject(createUserMessage({
            content: epochBlocks,
            source: { kind: 'plugin', plugin: `${name}:epoch` },
          }))
        }
        // inject context — 背景上下文, 进 inbox 不 wake (不驱动 turn).
        if (context.length > 0) {
          agent.inject(createUserMessage({
            content: context,
            source: { kind: 'plugin', plugin: `${name}:moment` },
          }))
        }
        // yield 解锁 (A 范式): 有 pendingYield → 这一帧是 yield 的下一帧. 先 inject/steer
        // 再 resolve("ok"), 保证 next step 由 tool result 触发时能 claim 到 inputs. 顺序不能反.
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
        } else {
          // todo: applyModelConfig — provider/model/reasoningEffort 应用到下个 request
          //   (agent/request waterfall 提案 / session.selectModel).
          // 正常 enter: steer inputs (输入, 驱动 turn). 无 percepts 时 idle 需占位起 turn.
          if (agent.status === 'idle') {
            agent.steer(createUserMessage({
              content: inputs.length > 0 ? inputs : [{ type: 'text', text: 'thinking' }],
              source: { kind: 'user' },
            }))
          } else if (inputs.length > 0) {
            agent.steer(createUserMessage({
              content: inputs,
              source: { kind: 'user' },
            }))
          }
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
        res.writeHead(200, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify({ thinking: false }))
      } catch (error) {
        res.writeHead(400, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify({ error: String(error) }))
      }
    },
  })

  // ── 5. yield tool (moss_wait_next_moment) ─────────────────────────────
  // 模型在 thinking 中主动调 wait, 阻塞等下一帧 moment. execute 挂 pendingYield 阻塞;
  // 下一轮 thinking/enter 解锁 (resolve moment_ref). cancel 时清空 pendingYield 并 reject.
  ctx.tools.register(defineTool({
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
  }))

  // ── moss_fetch_next_moment tool ───────────────────────────────────────
  // 主动 fetch: 模型调 fetch → execute 挂 pendingCalls[callId] 阻塞 → MOSS 侧 thinking.observe()
  // 生产 moment → /tool-result RPC 按 callId 解锁 (resolve {moment_ref}) 并注入 moment context.
  ctx.tools.register(defineTool({
    name: 'moss_fetch_next_moment',
    description: 'Fetch the next MOSS moment now. Returns {moment_ref}; the full moment is injected into the next step context.',
    parameters: {},
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
  }))

  // ── moss_append_ctml tool (interleaved thinking) ──────────────────────
  // 追加 ctml 到执行, 思维超前于行为: MOSS 侧 articulator.send(ctml) + wait (compiled/done).
  ctx.tools.register(defineTool({
    name: 'moss_append_ctml',
    description: 'Append a CTML command to execution and keep thinking. Returns "ok" once compiled (or executed if wait_done).',
    parameters: {
      ctml: { type: 'string', description: 'The CTML command to execute.' },
      refresh_meta: { type: 'boolean', default: false, description: 'Refresh shell meta before executing (deferred — no-op for now).' },
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
          if (pendingCalls.delete(callId)) { reject(new Error('moss_append_ctml aborted')) }
        }, { once: true })
      }) as unknown as JsonValue
    },
  }))

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

// ── helper: model config 应用 (点 3) ───────────────────────────────────
/**
 * applyModelConfig — 把 thinking/enter 的 model 配置 (provider/model/reasoningEffort)
 * 应用到下个 request. 候选: session.selectModel / agent/request waterfall 提案.
 */
// function applyModelConfig(agent: Agent, model: { provider: string; model: string; reasoningEffort?: string }): void { ... }

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

// ── helper: foreign session 冻结提示 (点 4) ─────────────────────────────
/**
 * notifySessionFrozen — foreign session 尝试运行 ego tool 时, 发 mux 提示
 * 「session 已冻结」. seam: 形态待定 — ask-user 对话框 (question/requested) /
 * log-only session event / plugin-source user message.
 */
function notifySessionFrozen(ctx: Context, agent: { id: string }): void {
  // todo: seam — 见 SKILL 讨论 (mux 提示形态), 表面阶段只记录.
  ctx.logger.info('dolores: foreign session %s blocked — ego session frozen', agent.id)
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
