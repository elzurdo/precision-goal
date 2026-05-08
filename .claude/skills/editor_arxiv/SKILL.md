# Skill: editor_arxiv

Prepares the `latex/` source for arXiv submission into a clean `latex/arxiv_latex/` directory.
Does **not** modify any original files.

## What this skill does

1. Runs `scripts/prepare_arxiv.sh` to produce `latex/arxiv_latex/` containing:
   - All active `\input` `.tex` files with comments stripped
   - `dptig.tex` with comments stripped, `\today` replaced by today's date, and the 4-pass `\typeout` appended
   - `dptig.bbl` (pre-compiled bibliography — `.bib` is excluded)
   - `images/` subdirectory with only the PNG files actually referenced in the tex sources
2. Verifies the output compiles cleanly
3. Reports any warnings or issues found

## How to invoke

```
/editor_arxiv
```

No arguments needed. Always run from the repo root.

## Agent instructions

### Step 1 — run the preparation script

```bash
bash .claude/skills/editor_arxiv/scripts/prepare_arxiv.sh
```

Review the output for any `WARNING` lines and resolve them before proceeding.

### Step 2 — verify compilation

```bash
cd latex/arxiv_latex && latexmk -pdf dptig.tex
```

The `-pdf` flag is required: without it `latexmk` defaults to DVI mode, which cannot determine the size of PNG images and will crash immediately.

If compilation fails, diagnose the error and fix it in `arxiv_latex/` only — never edit the original `latex/` source files.

### Step 3 — report

Tell the user:
- Which files were copied
- Any warnings encountered
- Whether compilation succeeded
- The path to the generated PDF for review

## Key decisions and their rationale

| Decision | Rationale |
|----------|-----------|
| Images kept in `images/` subdirectory | Avoids changing `\graphicspath` in `dptig.tex`; arXiv compiles from root and supports subdirectories. If compilation fails, see the note in the script about flattening. |
| `.bbl` included, `.bib` excluded | arXiv uses the pre-compiled bibliography (per arxiv_uploading_paper_tips.md) |
| `\today` replaced with hardcoded date | arXiv periodically rebuilds PDFs; `\today` would drift (per arxiv_tex_submissions.md) |
| Comments stripped from all `.tex` files | Source is public on arXiv; comments may contain notes not intended for readers |
| Comment-only lines removed entirely (not blanked) | A line that is entirely a comment (e.g. `  %Left: Bernoulli case.`) must be deleted, not left as a blank line. Blank lines inside a `\caption{}` argument are a paragraph break, which crashes LaTeX with `! Paragraph ended before \NR@gettitle was complete`. |
| 4-pass `\typeout` appended | Ensures `cleveref`/`autonum` references resolve correctly |
| `grep -oP` replaced by `perl -ne` throughout the script | macOS ships BSD grep, which does not support `-P` (Perl-compatible regex). All pattern extraction uses `perl -ne '...' ` instead. |

## Reference documents

- [arXiv TeX submission guidelines](https://info.arxiv.org/help/submit_tex.html) → `.claude/skills/editor/references/arxiv_tex_submissions.md`
- [Practical arXiv upload tips](https://trevorcampbell.me/html/arxiv.html) → `.claude/skills/editor/references/arxiv_uploading_paper_tips.md`
