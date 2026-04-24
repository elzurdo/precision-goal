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
