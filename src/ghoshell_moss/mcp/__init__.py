"""MCP integration — first-class MOSS citizens for MCP protocol interop.

GhostBridge is the bidirectional bridge between external MCP agents and MOSS ghost
(request-reply pattern).  Replaces the experimental ``ghoshell_moss_contrib``
mailbox prototype.
"""

from ghoshell_moss.mcp.ghost_bridge import GhostBridge, serve_ghost_bridge

__all__ = ['GhostBridge', 'serve_ghost_bridge']
