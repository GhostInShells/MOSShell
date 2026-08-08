"""Openbox node implementations — ready-to-use MOSS nodes shipped with the distribution.

Each module is a self-contained node that can be launched via
``moss nodes run`` or composed into larger deployments.

Dependencies are lazily loaded: importing a module only checks availability
at the point of use, so the entire package remains importable even when
optional extras (mcp, ROS, etc.) are not installed.
"""
