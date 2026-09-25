---
bringup_nodes: []
# To exercise mode bringup, replace the `bringup_nodes: []` line above with the list below
# (do not keep both — YAML duplicate keys). Nodes:
#   probe_hang  — check never exits; verifies a hung probe does not block startup
#   probe_fail  — check exits 1; verifies per-node failure is logged, not fatal
#   hello_world — healthy node; verifies siblings still spawn (isolation)
# bringup_nodes:
# - .moss/system_test_nodes/probe_hang
# - .moss/system_test_nodes/probe_fail
# - .moss/system_test_nodes/hello_world
description: system test mode — dogfood nodes CLI in isolation
exclude_node_paths: []
manifest_package: HOST
name: system_test
node_paths:
- $MOSS_WORKSPACE/system_test_nodes
- nodes
- $MOSS_WORKSPACE/nodes
---
