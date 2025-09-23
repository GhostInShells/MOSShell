# MOSShell

`MOSShell` (Model-Operated System Shell) is a Bash-like shell not for humans, but for AI models:
a dedicated runtime that translates model reasoning into structured,
executable commands for real-time tool and robot coordination.

In short, MOSShell does:

* `Present`: Presents function's source code directly as model-readable prompts.
* `Parse`: Requires and parses the model's structured `CTML` (Command Token Marked Language) output stream.
* `Execute`: Schedules and executes commands under a synchronous-blocking (same-channel) or asynchronous-parallel (
  cross-channel) strategy for streaming execution.

This allows the model to not just think, but act in real-time, providing a foundational layer for building Embodied AI.