# AI-assisted implementation guide

This file contains repository-specific instructions for coding agents. Read
and follow [CONTRIBUTING.md](CONTRIBUTING.md); it is authoritative for
architecture, tests, documentation, validation, AI disclosure, authorship, and
project communication.

- Before changing code, inspect the relevant implementation, tests, public
  documentation, and nearby repository conventions. Treat examples and
  existing implementations as guidance, not templates to copy mechanically.
- Keep changes within the requested scope. Do not add unrelated refactors,
  dependency changes, cleanup, or repository-wide formatting.
- Prefer the smallest readable, backward-compatible change. Preserve public
  behavior unless the task explicitly changes it, and do not expand the public
  API without prior discussion. Breaking changes require prior discussion and
  should normally follow a documented deprecation path. Add complexity for
  performance only when benchmarks justify it.
- Reuse established abstractions. Private implementation details may change
  when public behavior and architectural boundaries remain intact.
- Make reasonable private implementation choices and state material
  assumptions, but do not turn an inference or assumption into a public
  contract without explicit direction. Stop when requirements conflict or a
  decision would create a broad, difficult-to-reverse public API or
  architectural commitment.
- Do not weaken, skip, or remove tests, lint rules, warnings, error handling,
  or validation merely to make a change pass.
- If blocked, report the exact cause, the smallest safe alternative, and any
  public API or behavior impact.
- Do not complete the AI tools disclosure or human-only confirmation in
  `.github/PULL_REQUEST_TEMPLATE.md`, or perform public GitHub communication
  on the contributor's behalf.
