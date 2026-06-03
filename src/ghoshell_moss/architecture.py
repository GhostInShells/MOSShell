"""
MOSS Architecture Map — 核心抽象地图
=====================================

本模块是 MOSS 项目架构的唯一真理源。它手动策展所有关键模块的 import，
通过 ``moss codex architecture`` 命令反射输出，让 AI 在进入会话时一次性
获得完整的心智模型。

维护方式：手动 import 所有核心模块。每个模块的 ``__doc__`` 即其描述。
添加新模块 = 添加一行 import。无需额外维护。

使用方式：
    moss codex architecture          # AI 纯文本输出
    moss codex architecture          # 人类 Rich 表格输出

See: FEATURE.md codex-architecture
"""

# ============================================================================
# Core Concepts — MOSS 是什么
# ghoshell_moss.core.concepts
# ============================================================================

import ghoshell_moss.core.concepts.channel as channel
import ghoshell_moss.core.concepts.command as command
import ghoshell_moss.core.concepts.shell as shell
import ghoshell_moss.core.concepts.interpreter as interpreter
import ghoshell_moss.core.concepts.topic as topic
import ghoshell_moss.core.concepts.errors as errors
import ghoshell_moss.core.concepts.tools as tools

# ============================================================================
# Blueprints — 怎么用 MOSS 构建
# ghoshell_moss.core.blueprint
# ============================================================================

import ghoshell_moss.core.blueprint.channel_builder as channel_builder
import ghoshell_moss.core.blueprint.matrix as matrix
import ghoshell_moss.core.blueprint.mindflow as mindflow
import ghoshell_moss.core.blueprint.host as host
import ghoshell_moss.core.blueprint.ghost as ghost
import ghoshell_moss.core.blueprint.environment as environment
import ghoshell_moss.core.blueprint.manifests as manifests
import ghoshell_moss.core.blueprint.app as app
import ghoshell_moss.core.blueprint.session as session
import ghoshell_moss.core.blueprint.states_channel as states_channel
import ghoshell_moss.core.blueprint.conversation as conversation
import ghoshell_moss.core.blueprint.fractal as fractal

# ============================================================================
# Contracts — IoC 最小基础依赖
# ghoshell_moss.contracts
# ============================================================================

import ghoshell_moss.contracts as contracts

# ============================================================================
# Messages — 统一消息类型
# ghoshell_moss.message
# ============================================================================

import ghoshell_moss.message as message

# ============================================================================
# Channels — 预制 Channel 目录
# ghoshell_moss.channels
# ============================================================================

import ghoshell_moss.channels as channels
