# How to work on an issue

An interesting issue can be a doorway into a codebase. You do not need to
understand all of FatQat to begin. This guide helps you take the next steps:
understand the question, explore the relevant code, choose an approach, and
share a change you can explain. AI tools are welcome throughout the process.

!!! tip "Give yourself the chance to learn"

    Please resist the temptation to hand an issue to an AI agent and forward
    its patch straight to GitHub, or simply relay messages between the agent
    and reviewers. We have access to these tools too. What we value in your
    contribution is the understanding you develop and the judgment you bring
    to the design.

## 1. Find a place to start

We deliberately leave some issues open for contributors and mark them with
the `learning` label. These are opportunities to get to know FatQat by working
on a real problem and helping shape its solution. The goal is to learn how
FatQat works, why its interfaces are designed as they are, and how the code is
structured.

Choose one that interests you. You can start with limited knowledge of the
package and build your understanding as you investigate. Some issues will
take time to explore, even when the eventual change is small.

Begin by reading the issue and its discussion. What is someone trying to
accomplish? Try the example if there is one, and ask questions when something
is unclear. You do not need a solution yet. You just need a starting point
for exploring together.

!!! note "The issue is a starting point for discovery"

    Sometimes we open an issue because we have spotted a problem, before we
    have worked out the right solution. A suggested implementation, including
    one from a maintainer, is something you can investigate and improve.
    Finding a better approach is a welcome contribution.

## 2. Follow the example into the code

Start by asking your AI tool for a small map of the relevant code. A useful
first prompt is:

> "Help me understand how this example runs. Explain what each component is
> responsible for, and point me to related tests and documentation. Let's
> explore before making changes."

Open the suggested files and follow one path from the public API to the
implementation and back to the result. If a function is confusing, ask the
tool to walk through it with an example, then check its explanation against
the code.

As the path becomes clearer, ask **why it was designed this way**. Nearby
code, Git history, and earlier discussions can reveal constraints that are
easy to miss when looking at one function. They can also show where an old
decision no longer fits the current needs.

A useful point to pause is when you can tell someone: "This component owns
the behavior, these callers depend on it, and this is where the issue appears."
You can keep investigating the parts you are still unsure about.

## 3. Where would the change belong?

Now consider how the solution would fit into what you have learned. Ask your
tool to compare approaches with you, and look closely at the trade-offs:

- **What can we reuse?** Look for existing functionality before adding a
  second place that does the same job.
- **Who should own the behavior?** A clear responsibility makes the change
  easier to follow and helps keep unrelated components independent.
- **Will this make future changes harder?** Consider whether a new option,
  special case, or fallback will make the code harder to understand, test,
  or extend over time.

Clear interfaces and focused modules make it easier to understand one part
without reconstructing the whole system. This helps both contributors and
AI tools, which also work with limited context.

Sometimes a seemingly small fix needs increasingly awkward workarounds.
For example, passing the same information through several unrelated
components may suggest that the responsibility belongs elsewhere. Pause and
consider whether a modest restructuring would help. Compare it with a direct
fix: what becomes clearer, what else must change, and could existing users
be affected? Keep the restructuring focused on the problem.

For substantial changes, discuss the direction in the issue before committing
to a full implementation. Share what you have learned and invite others to
fill in missing context. A small exploratory prototype can help answer a
design question and give that discussion concrete evidence.

## 4. Try the approach and check the behavior

When trying an approach, choose a small example that will help you judge
whether it fits. Ask your AI tool to help implement it, then examine and
revise the result. Working code is evidence to consider, and the design
still needs your judgment.

Work in small steps so you can see what each change does. When an agent drafts
code, read the diff and trace your example through the changed path. If a
decision surprises you, ask why it is needed and explore alternatives. If the
implementation becomes awkward, revisit the design before adding more.

Read the tests with the same care. You can ask your tools to help, but check
what each test actually demonstrates. Would it catch the behavior going
wrong? Would it still pass if the implementation changed while preserving
that behavior? Passing tests are useful evidence when they protect the
expectations that matter.

When updating documentation, explain what users can do and what behavior
they should expect. Review AI-generated prose for accuracy and clarity, and
remove implementation details or development history that do not help users.

## 5. Help reviewers follow your reasoning

A pull request (PR) invites others to examine your change with you. Explain
the problem, why you chose this approach, and how you checked it in your own
words. Give reviewers enough context to follow your reasoning even if they
have not read the issue discussion.

Read and edit AI-generated summaries before sharing them. Keep details that help
someone assess the change, and cut repetition and lengthy lists that obscure
the reasoning. If a summary is too long for you to read carefully, it needs
another edit before you ask reviewers to read it.

Review is another chance to learn. A question may uncover a constraint you
missed or help you explain a choice more clearly. You can use AI to investigate
a comment, then check the answer and respond from your own understanding.
