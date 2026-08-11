"""Deprecated: use ``ghoshell_moss.mcp.GhostBridge`` instead.

This module is a thin re-export of the first-class ``ghoshell_moss.mcp``
package.  It exists for backward compatibility during the migration window
and will be removed.
"""

from ghoshell_moss.mcp.ghost_bridge import GhostBridge, serve_ghost_bridge, _Envelope

# Backward-compat aliases
MailboxBridge = GhostBridge
McpBridge = GhostBridge
serve_mailbox = serve_ghost_bridge
serve_mcp_bridge = serve_ghost_bridge
