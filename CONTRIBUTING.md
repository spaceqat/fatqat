# Contributing to fatqat

Contributions to fatqat are welcome. Development setup and test commands are
documented in the [README](README.md#development). Contributors should discuss
substantial features or architectural changes with the maintainers before
investing in a large implementation.

## AI tools usage policy

fatqat welcomes the use of AI tools to improve efficiency and the quality of
work. Contributors may use these tools provided that the following guidelines
are followed. The policy applies to code, tests, documentation, issues, pull
requests, reviews, discussions, and comments.

AI tools may help a contributor:

- explore the codebase and understand how components interact;
- brainstorm approaches and compare design options;
- draft code, tests, or documentation;
- review an implementation for possible bugs or improvements.

However, the contributor must independently validate architectural decisions,
review all generated output and test applicable changes.

### 1. Accountability

By submitting a change, the human contributor accepts responsibility for its quality,
including any AI-generated material it contains.

The contributor must:

- ensure the contribution meets the project's quality and testing standards;
- understand how the change works and enough of the related existing code to
  trace how the change interacts with the rest of the package;
- explain why the approach was chosen, including relevant alternatives and trade-offs;
- explain how the behavior was verified and respond substantively to questions
  during review.

This does not require memorizing every line or knowing the internals of every
dependency. It requires enough command of the contribution to locate and
explain the relevant behavior, reason about its consequences, and debug or
maintain it after merge. A change the contributor cannot explain and maintain is not ready for
review.

### 2. Transparency

The `AI Tools Usage Disclosure` field in the pull-request template is mandatory.
If no AI tool was used, the contributor must state so clearly.
If AI tools were used, the disclosure must name the tools and briefly
explain how they were used.

### 3. Authorship and legal responsibility

The contributor must have the right to submit every part of the contribution
and must respect third-party licenses and the terms of the AI tool.
The contributor must check AI-generated material for copied or derived
third-party content and must not submit material under incompatible licensing
terms.

An AI system, model, or agent must not sign commits or be listed as an author or
co-author. This includes `Author` and `Co-authored-by`, and `Signed-off-by`
metadata. Any such entries must be removed from the proposed commits before submission.

### 4. GitHub communication and prohibited uses

Project communication is expected to be directly with the contributor, not an
agent. Translation and grammar assistance are welcome, but issues, pull-request
descriptions, and review replies must express the contributor's own understanding and judgment.

AI-assisted descriptions and summaries are permitted, but should be reviewed for
clarity, relevance, and concision. Avoid lengthy or generic content that adds little
useful information or obscures the contributor's own explanation and reasoning.
Lengthy file-by-file summaries, generic background, and repetition
can obscure information needed for review and may be returned for rewriting.

Autonomous or unattended agents must not drive the communication on GitHub.
The contributor must personally direct every public interaction.
Repository-owned maintenance bots and other
automation explicitly approved by the maintainers are exempt.

### 5. Enforcement

Maintainers may ask the contributor to explain a change, rewrite a description,
provide verification, or reduce the scope. Maintainers may decline or close a
contribution if the contributor cannot demonstrate understanding and ownership,
violates this policy, or if the effort required to review the contribution would
outweigh its likely benefit to the project.

## Code changes and validation

Install fatqat in editable mode with the full test and lint environment:

```sh
python -m pip install --upgrade pip
python -m pip install --editable . --group test-full --group lint
```

The `test-full` group includes the core `dev` tests and optional integration
dependencies. Use `dev` alone only for lightweight local work that does not need
those integrations.

Run the narrowest relevant tests while developing a change. Before requesting
review, run the full test and lint checks:

```sh
python -m pytest
python -m black --check --diff src tests conftest.py
python -m pylint src/fatqat tests conftest.py
```

New behavior and bug fixes should have focused tests. Test observable behavior,
public contracts, and meaningful failure cases rather than private helpers,
internal call sequences, or data layouts. Before adding a test, check whether
an existing test already protects the behavior; extend or parameterize it when
that remains clear, and avoid duplicate cases that would fail for the same
regression. Seed stochastic tests and use appropriate numerical tolerances.

## Documentation

Documentation is part of fatqat's public interface. A contribution that changes
public behavior, configuration, defaults, return values, result fields, or
errors must update the corresponding documentation in the same pull request.

### Put information in the right place

- User guides in `docs/mkdocs/en/guide/` provide pedagogical explanations of
  concepts and workflows. They should develop topics progressively and give
  readers the context needed to understand how the pieces fit together.
- API pages in `docs/mkdocs/en/api/`, together with public docstrings, define the
  complete public contract. Docstrings should provide concise reference
  material that remains useful through Python `help()`.
- Tutorials in `docs/mkdocs/tutorial-sources/en/` present substantial executable
  workflows or studies. Their Markdown sources are canonical; pages, figures,
  scripts, and notebooks are generated.

A public class with substantial behavior or configuration should have its own
API page. Begin with why and when a user would choose it, then explain its
behavior, constraints, and configuration, include a minimal example, and finish
with the generated API reference. Explain shared behavior once and link to it
from the relevant pages.

### Contributing tutorials

Tutorial contributors should follow the
[executable tutorial instructions](docs/mkdocs/README.md#add-executable-tutorials).
They cover the canonical Markdown source, required metadata and accessible
figures, learning and reproducibility expectations, executable cells, optional
Jupytext conversion, and generated outputs.

### Writing style and API documentation

Write for someone trying to use the feature, not someone tracing its
implementation. Lead with what the API enables, how to use it, and what behavior
to expect. Use direct, concrete language and active voice, and keep terminology
consistent across code, guides, and API pages. Internal planning, caching, and
execution details belong in user documentation only when they affect public
behavior or define an extension contract.

If AI tools assist with documentation, revise the result to match the project's
established style. Let the subject and the user's needs determine the depth and
structure; do not force every page or section into the same pattern or level of
detail. Cut generic or repeated framing, stock phrasing, mechanical transitions,
and tables or checklists that do not make the material easier to understand or
navigate.

A public docstring is a concise reference that must also read well through
Python `help()`. Begin with a one-sentence summary, then add a short behavioral
description when it is useful. Use `Args`, `Returns`, and `Raises` as applicable,
and include `Examples` only when an example materially clarifies the API. List
arguments in signature order, and prefer simple labels and lists to tables,
Sphinx directives, or underlined section headings.

Options need enough information for a user to choose and combine them correctly.
State each option's accepted values and default, explain its user-visible effect
and when it applies or has no effect, and identify incompatible combinations and
relevant errors. If the accepted values form a small finite set, list every value
explicitly. For a dictionary argument, document each public key separately with
its name, accepted type, default, user-visible effect, and important compatibility
constraints. For example:

```text
simulation_config: Optional execution controls. Accepted keys are:

    - "seed" (int | None, default None): Random seed.
    - "mode" (str, "auto" | "thread", default "auto"): Execution mode.
```

Where relevant, also explain object lifetime and ownership, reuse guarantees,
and support for concurrent calls. Examples should be minimal, executable, and
focused on one behavior. Use the repository's normal import style, and seed
stochastic examples when showing exact output.

### Validate documentation changes

Before submitting documentation changes, follow the
[documentation build and validation instructions](docs/mkdocs/README.md#validate-before-submission).
They cover the pinned environment, warnings-as-errors build, rendered output,
public docstrings, verification against the implementation, and diff checks.
