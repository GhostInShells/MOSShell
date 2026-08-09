# {name}

If this node has nothing worth documenting for a human reader, delete this file.
Otherwise fill in the sections below — what this node does, how to set it up,
how to run and debug it.

## What it does

<!-- One paragraph about what capabilities this node provides. -->

## Setup

<!--
Environment, dependencies, install steps.

If this node needs its own venv or packages, create an INSTALL.md with
the install steps. The presence of INSTALL.md triggers `moss nodes install`
behavior — the model will read the file and run the steps before the node
can be launched.

When no install is needed, delete INSTALL.md — the node is then
considered installed by default.
-->

## Usage

<!-- How to launch and interact with this node. Example:

    moss nodes run .moss/nodes/tools/{name}/

After launch, the node enters the Matrix network. Its channel commands
appear in the model's context when accepted. Test via moss-shell.
-->

## Development

<!--
Edit the files in this directory to change the node's behavior:

  NODE.md       — node manifest: name, description, exec command, singleton flag,
                  and the instruction body the model reads at runtime.
                  exec.command: 'python' resolves to the spawner's sys.executable —
                  the default for nodes sharing the MOSS environment. Only use an
                  absolute interpreter path when the node needs its own venv.
                  Reference: moss codex get-interface ghoshell_moss.core.blueprint.cell:NodeManifest

  main.py       — node entry point. Build channels, register into Matrix.
                  Explore: moss codex blueprint channel_builder
                           moss codex blueprint matrix
                           moss ctml read

  .gitignore    — sensible ignores for cell development. Add more as needed.

  INSTALL.md    — (optional) install guide. Delete if not needed.
-->
