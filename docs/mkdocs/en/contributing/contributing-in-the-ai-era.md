# Contributing in the AI era

## If AI can write the code, where can I make the biggest difference?

**By asking a question worth answering.**

AI has made it easier to turn an idea into code. That makes deciding **what to
build, why it matters, and how it fits** increasingly important. A valuable
question grounded in real use can contribute more than a ready-made
implementation.

Perhaps a result surprised you. You or your agent struggled with an
explanation or needed a complicated workaround. Or you looked at a pull
request and saw a simpler approach. A useful question, a clearer example,
or a thoughtful review can all move the project forward.

!!! tip "You don't need a patch to contribute"

    Share a question, investigate an obstacle, or help review an approach.
    Helping the community decide what deserves attention is valuable work.

## What have you built with FatQat?

Have you reproduced a result from a paper, explored a new physical model,
or used FatQat in a way our examples do not cover? Share it. Your workflow
could help someone else get started and show the community where the
package could grow.

A small example, a tutorial, or a walkthrough of your own work can all be
valuable contributions. Explain the question you explored, how you used
FatQat, and what the results mean. Include enough code and context for
someone else to follow, and credit the papers or other work you build on.

**You don't need a polished tutorial to start the conversation.** If you
have a new use case or an example you would like to share,
[open an issue](https://github.com/spaceqat/fatqat/issues) and tell us about
it. We can discuss how it might fit into the examples or
[tutorial collection](../tutorials/index.md).

## Using FatQat through an agent? Follow the friction. { #working-through-an-agent }

You may ask an agent to reproduce a paper, implement an algorithm, or explore
a physical model. Even when it writes most of the code, your goals and
judgment guide the work and can reveal opportunities to improve FatQat.

Start from those experiences. A broad request to "find improvements in the
codebase" gives an agent little basis for deciding what matters. A concrete
experiment gives it a purpose and constraints.

Try these questions during your next task:

=== "A workaround appeared"

    - "Which parts required custom code, and why?"
    - "Is the capability missing, or did we struggle to find and use it?"

=== "A run is too slow"

    - "Where is the runtime going? Can we measure it?"
    - "Can we improve performance while preserving the physical model and
      required accuracy?"

=== "A result is hard to interpret"

    - "What do we need to know to interpret this result, and does FatQat
      explain it clearly? For example, are the units and basis ordering clear?"
    - "What would help the next user: a clearer explanation, a worked example,
      or more information in the output?"

    If you and your agent had to work out a convention, inspect private
    internals, or write extra code to understand an output, describe that
    step. It may reveal a useful improvement even when the result is correct.

Treat the answers as leads to investigate. Check the documentation and the
evidence.

## Found something? Help others see it.

Help someone who has not seen your experiment understand where you got stuck.
Start with what you wanted to achieve, then walk them through what happened.
These questions can help:

- **What were you trying to do?** Describe the calculation, paper, or workflow
  that led you here. This helps others understand why the problem matters.
- **What surprised you?** Explain what you expected and what you observed.
  Mention anything you tried that helped or left you stuck.
- **How could someone explore it with you?** Share a small runnable example
  and the settings needed to reproduce it. Include an error message or
  measurements when they help show the problem.
- **What would you like to understand or improve?** You might need a way to
  express a calculation, an explanation of a result, or a faster way to run
  an experiment. You can ask that question without knowing the solution.

The useful details depend on the question. For a scientific result, a
reference and the physical assumptions help others assess the comparison.
For a slow simulation, the workload and how you measured runtime help them
investigate the cost. Share your experience in an
[issue](https://github.com/spaceqat/fatqat/issues) or a relevant ongoing
conversation; you do not need to diagnose the cause or implement a fix first.

??? example "What might a useful report sound like?"

    "While reproducing this paper's parameter sweep, we measured most of the
    runtime in repeated setup. Here is a reduced example and timings for two
    sweep sizes. We tried this existing option, but it did not address the
    repeated work. Is there a supported way to reuse the setup?"

    A small example and the actual timings let someone investigate this
    question without first reconstructing the whole study.

## Have an approach? Discuss it before spending the tokens.

For substantial changes, **we strongly encourage discussing the direction
before implementation**. Join an existing relevant issue or discussion, or
open an issue if there isn't one.

Explain the intended behavior and sketch the approach. Discuss the proposed
API, where the behavior belongs, numerical methods and their assumptions, and
trade-offs in accuracy, performance, or compatibility as relevant.

> "I'm reproducing this calculation, and this step requires a workaround.
> I'm considering an interface like this, with two possible numerical
> approaches. Does this fit FatQat's direction? Is there existing
> functionality I should build on?"

This conversation is itself a contribution. It helps contributors and
maintainers uncover constraints and consider alternatives before investing
time or AI tokens in code. An unfinished design with clear questions is
enough to begin. Small, obvious corrections can usually go straight to a
pull request.

## Writing code is easier. Thoughtful review matters more than ever. { #review-matters }

**Reviewing a change can be more valuable than writing it.** As AI makes
implementations easier to produce, the community needs people who can judge
whether they solve the right problem, fit the architecture, and remain
understandable.

**You do not need to be a maintainer to review a pull request.** Try an
example, question a design decision, or explain where the reasoning becomes
difficult to follow. State what you checked and where you remain uncertain.

AI can assist with local correctness checks, but correctness still needs
verification. Review also asks:

- **Does it fit the architecture?** Does the responsibility belong in this
  component, with a clear separation from other components?
- **Does it duplicate existing functionality?** Could an existing path serve
  the same purpose and avoid two implementations drifting apart?
- **Is the complexity justified?** Do defensive checks, fallbacks, and special
  cases address real requirements?
- **Can someone follow the behavior?** Do private hooks, hidden dependencies,
  or an obscure sequence of calls make it difficult to reason about?
- **Will it remain easy to extend?** Could a simpler design make the next
  change more straightforward?

??? example "Turn a concern into a constructive comment"

    "This creates a second place that decides which operations are supported.
    Could we reuse the existing validation so the two paths cannot drift
    apart?"

    "What situation requires this fallback? If that state should be
    impossible, would reporting an error make the behavior clearer?"

    "I'm having trouble tracing when this private hook runs. Could the caller
    pass the information explicitly, or could the behavior stay inside the
    component that owns it?"

Explain the consequence, invite discussion, and remain open to constraints
you may have missed. You can improve a design without supplying its
replacement. Browse the [open pull requests](https://github.com/spaceqat/fatqat/pulls)
to find a conversation you can help with.

## Ready to make a change?

For a guided path through investigating an issue, choosing a design, and
opening a PR, read [How to work on an issue](how-to-work-on-an-issue.md).

Keep it focused, verify the relevant behavior, and explain the reasoning.
Use AI to assist investigation and implementation while taking responsibility
for the claims and changes you submit. Read and understand generated material
before sharing it; lead the project conversation yourself.

For development setup, validation steps, and contribution guidelines, see
[CONTRIBUTING.md](https://github.com/spaceqat/fatqat/blob/main/CONTRIBUTING.md).

**Bring your questions. Share your evidence. Help review an approach.** You
do not need a finished implementation to move FatQat forward. Helping the
community decide what to build and how to build it well is a contribution
in its own right.
