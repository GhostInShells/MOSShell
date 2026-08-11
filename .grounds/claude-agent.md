---
name: claude-agent
description: Claude Code agent 集合 ground — 发现 agent 定义, 索引能力面.
pins:
- label: agents
  verb: frontmatter
  arguments:
    path: "*.md"
    keys: ["name", "description", "tools", "model", "permissionMode"]
    limit: 30
  description: agent 定义索引 — name/description/tools/model, 一层扫描
---

# Claude Code Agents

This directory contains Claude Code sub-agent definitions. Each `.md`
file (except GROUND.md) is an agent — a specialized Claude instance with
its own system prompt, tool set, and permissions.

## How to invoke

In Claude Code, spawn an agent via the Agent tool:

```
Agent(subagent_type="<name>", description="<what to do>",
      prompt="<full task description>")
```

Or in natural language: "Use the <name> agent to ..."

## Agent anatomy

Each agent file has YAML frontmatter (see pin below) and a markdown body
that serves as the agent's system prompt. Key frontmatter fields:

| Field | Purpose |
|-------|---------|
| `name` | Unique identifier — used in `subagent_type` |
| `description` | Routing signal — when Claude should delegate to this agent |
| `tools` | Tool allowlist (Read, Grep, Glob, Bash, Write, etc.) |
| `model` | Model override: sonnet, opus, haiku, or inherit |
| `permissionMode` | Permission level: default, acceptEdits, auto, bypassPermissions |
| `maxTurns` | Maximum turns before auto-stop |
| `background` | Run as background task (true/false) |
| `isolation` | `worktree` for filesystem-level sandbox |

## Key rules

- **Context isolation**: Each agent runs in a fresh context — no inherited
  conversation, reasoning, or sibling outputs.
- **Tool restriction**: Give each agent only the tools it needs. A reviewer
  shouldn't have Write access.
- **Parallel**: Multiple agents run concurrently since they share nothing.
- **Nesting**: Agents with `Agent` tool access can spawn sub-agents (max
  5 levels deep).

The pin below shows what agents exist on disk. Read an agent's full `.md`
file to understand its system prompt and detailed behavior.
