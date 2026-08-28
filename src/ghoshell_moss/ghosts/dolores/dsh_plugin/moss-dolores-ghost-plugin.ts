import { randomUUID } from 'node:crypto'
import type { IncomingMessage, ServerResponse } from 'node:http'

import type { Agent } from '@deepseek-ai/dsh-agent'
import type { Context } from '@deepseek-ai/cordis'
import { createUserMessage } from '@deepseek-ai/dsh-llm'
import type { SessionId } from '@deepseek-ai/dsh-session'
import { PERSONA_ORDER, PERSONA_SECTION, renderPrompt } from '@deepseek-ai/dsh-system-prompt'
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
 * 2. thinking/enter   — moment(三块) + effort + model config, 阻塞执行完.
 * 3. thinking/exit    — 反转 thinking 状态; agent 非 idle 时显式 cancel.
 * 4. perStep 锁       — foreign session → reject + mux 提示冻结;
 *                       ego session 非 thinking → 阻塞等 thinking/enter 反转.
 * 5. moment 三块      — results (moss output 上一轮) / percepts (inputs) /
 *                       dynamic (dynamic_context + hint).
 * 6. dynamic 处理     — 注入后记 seq; 每次重注入, 历史 dynamic 节点 surface remove.
 * 7. tool 面暂缓      — 4 个 ego tool + tool-result 桥不做, 先暴露表面.
 * 8. 时序图           — 见下方 ASCII.
 *
 * ── 时序: MOSS 驱动路径 (thinking = turn) ──────────────────────────────
 * [MOSS mindflow]      [plugin]                     [dsh agent loop]
 *      │ thinking start   │                              │
 *      │── thinking/enter │  {moment, effort, model}     │
 *      │                  │── applyModelConfig()         │
 *      │                  │── injectMoment()             │  (results/percepts/dynamic → surface)
 *      │                  │── openThinking()             │  (release pre-step gate)
 *      │                  │── steer (若 idle) ──────────▶│
 *      │                  │                              │── turn/start
 *      │                  │◀── agent/pre-step ───────────┤
 *      │                  │── enter (frame in surface) ──▶│
 *      │                  │                              │── model 跑 → logos 流
 *      │                  │◀── turn/end ─────────────────┤
 *      │ thinking exit    │                              │
 *      │── thinking/exit ─┤                              │
 *      │                  │── closeThinking()            │
 *      │                  │── cancel (若非 idle) ────────▶│
 *
 * ── 时序: 外部唤醒路径 (dsh UI 输入, mindflow idle) ────────────────────
 * [dsh UI]           [plugin]                    [dsh agent loop]      [MOSS]
 *      │ 输入 ──────────┤ steer ─────────────────────▶│
 *      │               │                              │── turn/start
 *      │               │◀── agent/pre-step ───────────┤
 *      │               │── thinking? false → await gate (阻塞)
 *      │               │── notifyExternalWake() ─────────────────────▶│  (seam)
 *      │               │                              │                 │ mindflow 处理
 *      │               │◀── thinking/enter {frame} ────────────────────┤
 *      │               │── injectMoment + openThinking()
 *      │               │── release gate ─────────────▶│
 *      │               │── enter (frame in surface) ──▶│
 *      │               │                              │── model 跑 → 回应
 *
 * 接缝 (外部唤醒): 通知与背压分工 —
 *   通知 = turn/start 广播 (ego 侧 _on_turn_start 监听 → 自醒 signal, 已存在),
 *   不另发显式讯号. pre-step 阻塞只是背压 (hold 住模型等 thinking/enter 注入帧).
 *
 * ── 遗留问题 (2026-08-28) ────────────────────────────────────────────
 * 1. **外部唤醒链路未接通 (perStep 帧背压已启用, 靠超时兜底)**: turn/start → ego
 *    自醒 signal → mindflow → Thinking → thinking/enter 的链路未接通 (ego._signal_broadcast
 *    未绑 mindflow 路由). perStep 帧背压已启用 — ego 非 thinking 时 await thinkingGate.wait,
 *    超时 5s reject; 外部 turn 在链路接通前会阻塞到超时 reject. 需验证:
 *    a) 绑定 ego.bind_signal_broadcast(session.add_signal) 是否让链路通;
 *    b) 链路通后外部 turn 的帧注入时序 (thinking/enter open 释放 pre-step).
 * 2. **applyModelConfig todo**: thinking/enter 的 provider/model/reasoningEffort
 *    未应用到下个 request (agent/request waterfall / session.selectModel).
 * 3. **epoch 设计**: moment 携带 epoch id, 对比后决定 ghost.memory 尾部更新 (下一轮).
 * 4. **injectMoment dynamic replace**: surface-replace 机制可能过度设计 — 按 cache
 *    洞察 (尾部操作才 cache 友好), dynamic 应简化为尾部 append/替换.
 * 5. **on_event 内部逻辑**: token 记账 / tool 桥 / seq 跟踪 (deferred).
 */

export const name = 'moss-dolores-ghost-plugin'

export const inject: string[] = ['webServer', 'workspaceRegistry', 'agents', 'systemPrompt']

// 强相关路径命名空间: /moss-api/ghost/<ghost 名> — 体现 moss + ghost 类型 + dolores 实例, 不用通用 /plugin-api 弱命名.
const DOLORES_API_ROOT = '/moss-api/ghost/dolores'

const DOLORES_EGO_CREATE = `${DOLORES_API_ROOT}/ego/create`
// 通用 session 观测面: 任意 live session 的 instruction / surface 读取 (sessionId 收在 body).
const DOLORES_SESSION_INSTRUCTION = `${DOLORES_API_ROOT}/session/instruction`
const DOLORES_SESSION_SURFACE = `${DOLORES_API_ROOT}/session/surface`
// thinking 事务面 (取代旧的 articulate/enter|exit): 帧注入 + 锁反转 + 退出 cancel.
const DOLORES_THINKING_ENTER = `${DOLORES_API_ROOT}/thinking/enter`
const DOLORES_THINKING_EXIT = `${DOLORES_API_ROOT}/thinking/exit`
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
// wait 返回三态 outcome: open (正常释放) / aborted (exit cancel 打断) / timeout (fail-safe).

// pre-step 帧背压超时 (fail-safe): MOSS 未及时 thinking/enter 时 reject, 不空跑失速.
const THINKING_GATE_TIMEOUT_MS = 5000

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

// ── dynamic 上下文注入的 seq 记录 (点 6) ──
// 每次重注入, 把上一轮注入的 dynamic 节点从 surface remove (surface replace).
let dynamicSeqs: number[] = []

/**
 * thinking/enter 的 moment 三块 (点 5/6). MOSS 侧序列化 Moment 后经 HTTP 传入.
 *
 *   results  — moss output: 上一轮结果 (previous.messages 投影). 注入为上下文.
 *   percepts — inputs: 本轮新输入 (source → 文本). 作为唤醒内容注入.
 *   dynamic  — dynamic_context + hint: hot 帧. 注入后记 seq, 重注入 surface remove.
 */
interface ThinkingMomentPayload {
  results: { text: string }[]
  percepts: { source: string; text: string }[]
  dynamic: { context: { source: string; text: string }[]; hint: string }
}

/** thinking/enter 入参 (点 3). 阻塞执行完: handler 完成注入 + 开锁才返回. */
interface ThinkingEnterPayload {
  moment: ThinkingMomentPayload
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
              // ego session: 帧背压 — 非 thinking 时阻塞等 thinking/enter open.
              // 三态: open 放行 / aborted 走 next() 让 throwIfAborted 收成 aborted turn
              // (exit 的 cancel 打断卡在 pre-step 的 step) / timeout reject 停住 (fail-safe).
              const outcome = await thinkingGate.wait(THINKING_GATE_TIMEOUT_MS, signal)
              if (outcome === 'timeout') {
                ctx.logger.warn(
                  `dolores: ego pre-step blocked ${THINKING_GATE_TIMEOUT_MS}ms — MOSS thinking/enter not arrived, rejecting`,
                )
                return { kind: 'reject' }
              }
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
  // 入参 = moment 三块 + effort + model config. handler 阻塞执行完才返回:
  //   1. applyModelConfig — provider/model/reasoningEffort 应用到下个 request.
  //   2. injectMoment — results/percepts 注入 surface; dynamic 注入 + 记 seq,
  //      重注入时 surface remove 历史 dynamic 节点.
  //   3. openThinking — 释放 pre-step gate (外部唤醒路径的阻塞解除).
  //   4. steer (若 agent idle) — MOSS 驱动路径: 唤醒 turn.
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
        // todo: applyModelConfig — provider/model/reasoningEffort 应用到下个 request
        //   (agent/request waterfall 提案 / session.selectModel).
        injectMoment(agent, body.moment)
        openThinking()
        // MOSS 驱动路径: agent idle → steer 唤醒 turn, content = percepts (真实输入).
        // 模式提示 (CTML 环境) 已在 ghost 基础 instruction, 不进 steer (cache 稳定).
        // 外部唤醒路径: turn 已在 pre-step 阻塞, openThinking 放行, 无需再 steer.
        if (agent.status === 'idle') {
          const perceptText = (body.moment?.percepts ?? [])
            .map(p => p?.text).filter(Boolean).join('\n')
          agent.steer(createUserMessage({
            content: [{ type: 'text', text: perceptText || 'thinking' }],
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
        // agent 仍在跑 → 显式 cancel (MOSS 已宣布 thinking 结束, 不能让 dsh 空跑失速).
        // cancel 是同步发出 (在进程内, 走 abort signal), 轮次异步收线; 无需等待 — 下一
        // 个 thinking/enter 自会重新 openThinking + steer.
        if (agent.status !== 'idle') {
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
}

// ── helper: moment 注入 (点 5/6) ───────────────────────────────────────
/**
 * injectMoment — 把 thinking moment 三块注入 ego session 的 surface.
 *
 *   results: 上一轮 moss output → user/message (普通 append, 进 surface).
 *   percepts: 本轮输入 → user/message (唤醒内容, append).
 *   dynamic:  hot 帧 → 注入后把节点 seq 记进 dynamicSeqs; 下次重注入时,
 *            用 surface {op:'replace'} 把历史 dynamic 节点 shadow 掉 (点 6).
 *
 * surface replace 规则 (dsh): 新节点替换 [start, end] surface 区间, sourceEventSeqs
 * 必须包含所有被 shadow 的节点 seq. 这是 compaction 原语 — hot 帧退役的落地.
 */
// function injectMoment(agent: Agent, moment: ThinkingMomentPayload): void { ... }

// ── helper: model config 应用 (点 3) ───────────────────────────────────
/**
 * applyModelConfig — 把 thinking/enter 的 model 配置 (provider/model/reasoningEffort)
 * 应用到下个 request. 候选: session.selectModel / agent/request waterfall 提案.
 */
// function applyModelConfig(agent: Agent, model: { provider: string; model: string; reasoningEffort?: string }): void { ... }

// ── helper: moment 注入 (点 5/6) ───────────────────────────────────────
/**
 * injectMoment — 把 thinking moment 三块注入 ego session 的 surface.
 *
 *   results: 上一轮 moss output → user/message (append, 进 surface).
 *   percepts: 本轮输入 → user/message (append).
 *   dynamic:  hot 帧 → 注入后把节点 seq 记进 dynamicSeqs; 下次重注入时用 surface
 *             {op:'replace'} 把历史 dynamic 节点 shadow 掉 (点 6, hot 帧退役).
 */
function injectMoment(agent: Agent, moment: ThinkingMomentPayload | undefined): void {
  if (moment === undefined) return
  const session = agent.session
  // percepts 经 steer 送达 (steer content = percepts), 不在此注入 — 避免重复.
  for (const item of moment.results ?? []) {
    if (item?.text) appendUserMessage(session, item.text, 'results')
  }
  const dynamicText = [
    ...(moment.dynamic?.context ?? []).map(c => c?.text).filter(Boolean),
    moment.dynamic?.hint,
  ].filter(Boolean).join('\n')
  if (dynamicText.length > 0) {
    const message = createUserMessage({
      content: [{ type: 'text', text: dynamicText }],
      source: { kind: 'plugin', plugin: name },
    })
    if (dynamicSeqs.length > 0) {
      // surface replace: 新 dynamic 节点替换旧 dynamic 区间, sourceEventSeqs 必须含被 shadow 节点.
      const start = dynamicSeqs[0]
      const end = dynamicSeqs[dynamicSeqs.length - 1]
      const evt = session.append('user/message', message,
        { surfaceOp: { op: 'replace', start, end }, sourceEventSeqs: dynamicSeqs })
      dynamicSeqs = [evt.seq]
    } else {
      const evt = session.append('user/message', message, { surfaceOp: 'append' })
      dynamicSeqs = [evt.seq]
    }
  }
}

function appendUserMessage(session: Agent['session'], text: string, tag: string): void {
  // createUserMessage 生成唯一 id — 否则多条注入消息 id 全是 undefined, UI 撞 "more than one start Match".
  session.append('user/message',
    createUserMessage({
      content: [{ type: 'text', text }],
      source: { kind: 'plugin', plugin: `${name}:${tag}` },
    }),
    { surfaceOp: 'append' })
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
