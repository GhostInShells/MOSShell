---
# Copy to <space>/GROUND.md — then rewrite it. This file is an example, not a form.
# What has to happen is that somebody says out loud "this space is governed";
# the shape below is only one way of saying it.
$id: <project-or-ghost>/works
name: works
description: <one line: what this space holds, and for whom>
pins:
  # Directory governance: keep one AGENTS.md in every proposition directory, so a reader
  # arriving through ground meets the local rules first. Delete this pin if you do not
  # use that strategy.
  - verb: file
    label: rules
    arguments:
      path: AGENTS.md
  # Optional: let an agent arriving through ground see the list of works directly.
  # Commented out by default — with many works it is information overload, and one grep
  # is cheaper than a rendered list.
  # - verb: frontmatter
  #   label: works
  #   arguments:
  #     path: "*/WORK.md"
  #     keys: [name, status]
---

# Works

## What this space is

<Write it for yourself, or for this space's owner. Not an introduction for strangers: say
what lives here and who reads it. Two examples — rewrite one of them:>

- **a ghost home's works** — these are my own pieces, and the only reader is me. No need to
  manage anyone's impression; the job is to make it possible for me to find them again.
- **the works/ at a moshell project root** — moshell's dogfooding output, plus the public
  products made by moss the ghost and kept because they are meant to be distributed.

**The mechanism can be invented.** A `GROUND.md` exists here because *somebody has to say
out loud that this space is governed*. The three lines below are a reference, not a
requirement: an equivalent of your own is fine, as long as the governing act actually
happens somewhere a reader will look. The scheme is not the goal; the governance is.

## A reference shape (rewrite it into what this space actually does)

- **Topology** — <what the second level means here, and what the third is. Two lines.>
- **Distribution** — `all | process | template-only`: <what enters the repository, what
  does not, and why>
- **Template** — <which WORK template; or "this space defines its own, see AGENTS.md">

## How to use it

`moss ground` renders this file. A reader uses it to learn where a new work goes, and at
which level this space keeps its rules.
