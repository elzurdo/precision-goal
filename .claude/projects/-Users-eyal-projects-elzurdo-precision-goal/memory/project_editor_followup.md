---
name: Editor review — Should Fix progress and Could Improve items
description: Status of /editor review action items for dptig.tex (RSS DS&AI submission)
type: project
---

## Should Fix — status as of 2026-04-23

1. **TODO placeholder in conclusion** (conclusion.tex:54) — still open; user will address later.
2. **Missing submission elements** (keywords, data availability, ORCID, affiliation, conflicts) — user fills these in manually on the journal site; no LaTeX action needed.
3. **Abstract abbreviations** — user will look into.
4. **ePitG/DPitG naming inconsistency in Table 1** — user manually fixed.
5. **Bullet points → lettered lists** — DONE. All 5 itemize instances in methods.tex converted:
   - List 1 (line 87): enumerate[label=(\alph*)] — Accept/Reject/Inconclusive definitions
   - List 2 (line 123): converted to prose, bold panel labels removed
   - List 3 (line 178): enumerate[label=(\alph*)]
   - Lists 4+5 (lines 213–227): merged into one continuous (a)–(d) list using `resume`, with bridging prose in the middle
   - \usepackage{enumitem} added to dptig.tex preamble
6. **Alt text for figures** — DONE for Figure 1 (cherry_posteriors.png) in methods.tex. Remaining figures still need alt text.
7. **Graphical abstract** — user will address later.

## Could Improve — all pending (user asked to be reminded)

1. Add a "this paper in brief" framing paragraph near top of introduction.
2. Reframe HDI+ROPE as baseline comparator rather than cautionary case.
3. Give §3.2 (general trends) more symmetrical treatment relative to §3.1 (fair coin).
4. Tighten Discussion/Conclusion overlap or combine the two sections.
5. Remove \hline from inside table bodies (results.tex:131, 166) — journal prohibits horizontal rules within table body.
6. UK spelling scan (analyze→analyse, color→colour, acknowledgment→acknowledgement).

**How to apply:** Resume from "alt text for remaining figures" and then Could Improve items.
