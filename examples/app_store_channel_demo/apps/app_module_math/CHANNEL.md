---
name: "math_mod"
description: "A module-mode app (wraps a module as commands via ModuleChannel)."
main: "apps.app_module_math.math_ops"
mode: "module"
---

## Usage

- Command: `mul(a, b) -> int`
- Command: `pow2(x) -> int`

This app is launched in `mode=module`, meaning the app's `main` is treated as a python module name.
