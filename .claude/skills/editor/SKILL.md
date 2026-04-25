# Skill: editor

Reviews and edits academic manuscripts for journal submission.

## What this skill does

Given a manuscript (or section of one), the editor skill:
1. Reviews content, structure, and clarity
2. Checks compliance with the target journal's submission requirements
3. Suggests what **should** be improved (blocking issues) and what **could** be improved (optional enhancements)

## How to invoke

```
/editor [target journal] [scope]
```

- **target journal** — e.g. `rss-ds-and-ai`, or omit to review without journal-specific checks
- **scope** — e.g. `abstract`, `introduction`, `full`, or a file path

## Journal-specific requirements

Reference files for known journals live in `references/`. When a target journal is specified, load the relevant file and apply its requirements (word limits, section structure, formatting rules, etc.).

| Journal | Reference file |
|---------|---------------|
| RSS: Data Science and Artificial Intelligence | `references/rss_ds_and_ai_submission_instructions.md` |

## Review output format

Return feedback in two sections:

**Should fix** — issues that would cause rejection or non-compliance (word count exceeded, missing required sections, formatting violations, unclear core argument).

**Could improve** — optional suggestions for clarity, accessibility, or impact.

Keep feedback actionable and specific (point to the sentence or section).


## Editing Practicalities

### Line length
Target **~88 characters** for lines you write or rewrite — this is a guideline, not
a hard limit. Shorter is fine; modestly longer (up to ~95) is acceptable if breaking
the line would produce an awkward split. Do not reformat pre-existing lines that are
not being changed, even if they exceed this target.
When wrapping, break at a natural word boundary close to the target.

### Citation comments
Every citation you **add** (not pre-existing ones) must be accompanied by a LaTeX
comment block explaining why that source was chosen. Place the comment immediately
after the sentence or paragraph containing the citation, using the `%` prefix style
already present in the file. The comment should answer: what does this source say,
and why is it the right reference for this specific claim?

Example:
```latex
...a phenomenon known as early peeking \citep{simmons2011}.
% \citet{simmons2011}: Quantified how flexible stopping rules inflate false-positive
% rates --- directly motivating the decoupled stopping and decision rules here.
```