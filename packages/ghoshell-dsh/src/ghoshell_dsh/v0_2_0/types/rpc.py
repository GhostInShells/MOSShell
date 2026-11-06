"""Re-export of `blueprint.models.rpc` — the shapes live in the blueprint now.

This module exists so the 0.2.0 code keeps importing from where it always did, and so the
version keeps one place to add what is *specific to it*: wire conversion lives in
`ghoshell_dsh.v0_2_0.wire`, not here. If this file ever grows a class of its own, that is a
sign the shape belongs in the blueprint instead.
"""

from ghoshell_dsh.blueprint.models.rpc import *  # noqa: F403
from ghoshell_dsh.blueprint.models.rpc import __all__  # noqa: F401
