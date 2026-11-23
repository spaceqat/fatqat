# Writing documentation

This page explains how to write FatQat's documentation. The User Guide teaches
readers how and when to use a particular FatQat feature and how to interpret the
results. The API reference helps readers choose a public interface and look up its
exact behavior. A case study shows how to investigate a specific scientific or
engineering question with FatQat.

These guidelines cover user-facing documentation, including public API docstrings.
Include internal implementation details only when they help users call an
interface correctly, understand a constraint, or interpret a result.

## Write a User Guide page

A User Guide page usually focuses on one FatQat capability, such as noisy
simulation, circuit compilation, or visualization. Explain what it helps
readers do and when they would use it, then walk through an example and make
sense of its results. Use an approachable teaching voice and explain why the
key steps are needed. A guide to noisy simulation can run the same circuit with
and without noise and explain why the measurement counts differ.

Start with the reader's goal and give enough context to make the first example
meaningful. Introduce a new concept when it first matters, and use a small
example to connect it to the next step. In the
[compiler guide](../guide/compiler.md), for instance, a three-site chain
shows which pairs can run a gate directly and why routing matters before
readers see the simulator call. Put supporting details where readers first
need them, and link to the API for exact signatures, defaults, and errors.

- Explain what the example produced and why it matters. Print and show the
  results the reader needs to compare, such as the counts from two runs. Use
  the existing example tests to check published results where possible and
  set a seed if you show exact counts from a stochastic run. Present the
  result directly unless the page includes a deliberate exercise.
- Use a diagram when it makes a layout, relationship, or sequence easier to
  understand. Prefer SVG for line drawings and TeX for equations. Keep figure
  generation code out of the reading flow unless writing the plot is part of
  the lesson.
- Write in a direct, approachable voice. Keep constraints that affect the
  example. Cut generic introductions, repeated setup, defensive caveats that
  do not affect the task, and sentences that only preview the next section.
  Use short headings that tell readers what they will learn or do. Keep
  sidebar labels short enough to scan, especially in English.

## Write an API reference page

The API reference systematically documents FatQat's public interfaces. Start by
saying what the object or group of interfaces is and what it does. For example, a
simulator overview can introduce the supported systems and kinds of programs. In
each interface entry, document the signature, accepted inputs and defaults, return
values, and any constraints or errors that affect callers. Use short, precise
sentences and consistent terms to explain how the inputs and conditions affect the
behavior and return value. For `ReadoutConfusion`, explain how to pass the
probability matrix and what its rows and columns mean.

Use a table when readers need to compare several interfaces. The
[Program](../api/program.md) target table, for example, shows which methods accept
integers, references, or groups of targets. Keep method-specific behavior in that
method's docstring. The overview can summarize a rule that affects the choice of
interface and link to the details.

Group interfaces by what readers want to do. Use a **Reference** heading when
it separates an overview from generated entries. If task headings already
organize the entries clearly, as on the [Compiler](../api/compiler.md) page,
another heading adds little. On a subclass page, show what the subclass
defines or changes and link to useful inherited interfaces. Keep extension
hooks available where people need them; hide private fields and implementation
classes from ordinary API listings. Preserve existing URLs and anchors when
moving content.

For a family of operations, put a comparison table before the API entries if
it helps readers choose a gate. Put longer matrix definitions and derivations
after the entries. Keep units, target order, and conventions that affect a
call close to the interface that uses them.

### Write public docstrings

Write each public docstring so it works through Python `help()` as well as in
the rendered site. Start with a short description of the behavior. Use
`Args`, `Returns`, `Raises`, and `Examples` where they help:

- List parameters in signature order. Give accepted values, defaults, effects,
  and compatibility restrictions. For a mapping such as `execution_config`,
  use one bullet per public key. Give the type, default, and effect in each
  bullet so readers can compare the options without reading a long paragraph.
- Describe what a call returns or changes. Include shape, ordering,
  ownership, or copy behavior when they affect use. In a multiline
  `Returns` entry, indent continuation lines so the renderer keeps a single
  result together.
- Explain which errors occur and when. Put each rule with the interface that
  owns it. For example, measurement order belongs to `Measurement`; how
  readout noise changes reported values belongs to `ReadoutConfusion`.
- Use short, executable `Examples` for behavior a signature cannot show, such
  as selecting a group of targets or using a measurement as a condition. Show
  output when it helps the reader check the result.

Use exact API names and plain language so people and tools can read each entry
on its own. Link to another API entry when it defines a shared rule. Link to
the User Guide only when the reader needs the full workflow.

## Write a case study

A case study uses FatQat to investigate a concrete scientific or engineering
question. For example, a case study might ask how readout errors affect a
quantum error correction experiment. Introduce the question, explain why the
model fits it, and take the reader through a runnable experiment. Write as you
would describe a computational experiment. Show what you observed, using
figures, tables, or numerical results where they help. Then return to the
original question and explain what conclusions those observations support.

Group cases by topic. Use titles that tell readers what each case investigates.
Link to the User Guide when readers need background on a workflow used in the
case.

## Keep translations and previews in sync

The English source is sufficient for a documentation pull request. The
translation pipeline generates the Chinese and Korean versions by default.
If you have a translation or a revision to an existing translation, open a
separate pull request in the
[translation repository](https://github.com/BoxiLi/fatqat-doc-translator).

When editing a translation, keep identifiers, code, equations, and link
targets intact. Use the translators' glossaries for shared terms and record
agreed changes there. Choose wording that fits the context in each language.
Use the edition's language for headings and navigation, and keep API names and
standard abbreviations as written. Generated API docstrings stay in English
across editions.

Follow the
[documentation validation instructions](https://github.com/spaceqat/fatqat/blob/main/docs/mkdocs/README.md#validate-before-submission).
Check the examples and their outputs. Where docstrings or API listings
changed, compare the rendered page with `help()` and the public signatures.
Check the navigation, links, formulas, and tables in the affected editions.
