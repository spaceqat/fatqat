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
