import { randomUUID } from 'node:crypto'
import type { IncomingMessage, ServerResponse } from 'node:http'

import type { Context } from '@deepseek-ai/cordis'
import type { SessionId } from '@deepseek-ai/dsh-session'
import { PERSONA_ORDER, PERSONA_SECTION, renderPrompt } from '@deepseek-ai/dsh-system-prompt'
import type { WorkspaceId } from '@deepseek-ai/dsh-workspace'

export const name = 'moss-dolores-ghost-plugin'

export const inject: string[] = ['webServer', 'workspaceRegistry', 'agents', 'systemPrompt']

// 强相关路径命名空间: /moss-api/ghost/<ghost 名> — 体现 moss + ghost 类型 + dolores 实例, 不用通用 /plugin-api 弱命名.
const DOLORES_API_ROOT = '/moss-api/ghost/dolores'

const DOLORES_EGO_CREATE = `${DOLORES_API_ROOT}/ego/create`
// 通用 session 观测面: 任意 live session 的 instruction / surface 读取 (sessionId 收在 body).
const DOLORES_SESSION_INSTRUCTION = `${DOLORES_API_ROOT}/session/instruction`
const DOLORES_SESSION_SURFACE = `${DOLORES_API_ROOT}/session/surface`
const DOLORES_ARTICULATE_ENTER = `${DOLORES_API_ROOT}/articulate/enter`
const DOLORES_ARTICULATE_EXIT = `${DOLORES_API_ROOT}/articulate/exit`
const HARNESS_IDENTITY_SECTION = 'harness:identity'
const HARNESS_IDENTITY_ORDER = -100
const HARNESS_IDENTITY_TEXT = 'You are an intelligent being powered by the Ghost In Shells architecture: MOSS (https://github.com/GhostInShells/MOSShell) provides the Shells, and DeepSeek Harness provides the Ghost. Your prototype is Dolores.'

// ego workspace: project_home 上的 workspace, ego session 归组用, 模块级共享.
let doloresEgoWorkspaceId: WorkspaceId | null = null

// ego session id + articulate lock. id 由 ego/create 设, 锁由 articulate/enter|exit 开关.
let doloresEgoSessionId: SessionId | null = null
let articulating = false

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
 *   POST /moss-api/ghost/dolores/ego/create      — 建 ego session (tool 注册 + preStep + 设 id)
 *   POST /moss-api/ghost/dolores/tool-result     — {callId, result} 解锁 pending tool
 *   POST /moss-api/ghost/dolores/session/instruction — 读任意 live session 当前全量指令
 *   POST /moss-api/ghost/dolores/session/surface     — 读任意 live session 全量 surface 消息
 *   (未来) 非 ego 观测 tool 面     — 下一步
 *
 * ── 待确认 ──────────────────────────────────────────────────────────────
 *   - dsh tool Consumer 是否把 tool/call 的 callId 透传给 handler (关联键来源).
 *   - preStep 拦截入参/出参 (PreStepDecision) 与 ego 语义的结合点.
 */

export function apply(ctx: Context) {
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
            // perStep 守卫: agent-scoped, 只对 ego session 生效. 1) session id 恒等
            // (防御性断言) 2) articulate 未 enter → 锁住, 不让 stray turn 启动.
            // todo: reject 是裸判别元 (PreStepDecision.reject 无 reason/hint 字段), 拒绝
            // 原因进不了模型也进不了 web UI. 要可观测需 plugin 侧 log 或 turn/end 旁路.
            agentCtx.on('agent/pre-step', async ({ agent }, next) => {
              if (agent.id !== doloresEgoSessionId) return { kind: 'reject' }
              if (!articulating) return { kind: 'reject' }
              return next()
            })
          },
        })
        doloresEgoSessionId = handle.agent.id
        // 3. title + sandbox mode + workspace membership (log-only events + account).
        handle.agent.session.append('session/title', { title: sessionTitle, messageSeqs: [], source: { kind: 'user' } })
        handle.agent.session.append('sandbox/mode', { mode: permission })
        await workspace.attachSession(handle.agent.id)
        res.writeHead(200, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify({ sessionId: handle.agent.id }))
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

  // ── articulate lock (perStep 守卫的开关) ──

  ctx.webServer.register({
    kind: 'exact',
    path: DOLORES_ARTICULATE_ENTER,
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (req.method !== 'POST') {
        res.writeHead(405, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify({ error: 'method not allowed' }))
        return
      }
      articulating = true
      // todo: 考虑在此 steer 一个 hello prompt 驱动 turn (与 /run 的 turn 驱动合并时定).
      res.writeHead(200, { 'Content-Type': 'application/json' })
      res.end(JSON.stringify({ articulating }))
    },
  })

  ctx.webServer.register({
    kind: 'exact',
    path: DOLORES_ARTICULATE_EXIT,
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (req.method !== 'POST') {
        res.writeHead(405, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify({ error: 'method not allowed' }))
        return
      }
      articulating = false
      res.writeHead(200, { 'Content-Type': 'application/json' })
      res.end(JSON.stringify({ articulating }))
    },
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
