"""
MOSS Architecture Map — 核心抽象地图
=====================================

本模块是 MOSS 项目架构的唯一真理源。它手动策展所有关键模块的 import，
通过 ``moss codex architecture`` 命令反射输出，让 AI 在进入会话时一次性
获得完整的心智模型。

维护方式：手动 import 所有核心模块。每个模块的 ``__doc__`` 即其描述。
添加新模块 = 添加一行 import。无需额外维护。

使用方式：
    moss codex architecture          # 模型纯文本输出
    moss codex architecture          # 人类 Rich 表格输出

See: FEATURE.md codex-architecture

.. attention for model developers::

   本文件由模型实例手动维护。当你阅读 MOSS 源码、发现值得索引的
   模块或包路径时，评估是否添加一条 import。不是每个文件都值得加 —
   关键是那些"下次模型实例可能需要定位"的目录和核心抽象。
   加一条 import = 花 10 秒，为未来的模型实例省 2 分钟。
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
import ghoshell_moss.core.blueprint.session as session
import ghoshell_moss.core.blueprint.states_channel as states_channel
import ghoshell_moss.core.blueprint.memento as memento



# ============================================================================
# Contracts — 系统级依赖/抽象商场
# ============================================================================

import ghoshell_moss.contracts as contracts


# ============================================================================
# Implementations — 核心实现路径
# ============================================================================

import ghoshell_moss.core.py_channel as channel_impl
import ghoshell_moss.core.memento as memento_impl
import ghoshell_moss.core.file_editor as file_editor_impl
import ghoshell_moss.host as host_impl
import ghoshell_moss.host.tui as tui_design
import ghoshell_moss.host.tui_entries as tui_entries
import ghoshell_moss.matrix.session as session_impl
import ghoshell_moss.bridges as bridges
import ghoshell_moss.ghosts as ghosts
import ghoshell_moss.core.topic as topic_service
import ghoshell_moss.core.speech as speech_impl
import ghoshell_moss.cli as cli

# ============================================================================
# Protocol — 系统协议
# ============================================================================

import ghoshell_moss.message as message
import ghoshell_moss.topics as system_topics

# ============================================================================
# Openbox — 预制能力清单
# ============================================================================

import ghoshell_moss.channels as openbox_channels
import ghoshell_moss.core.concepts.tools as tools
