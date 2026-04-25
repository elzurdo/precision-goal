# Editorial and Technical Decisions

Version-controlled record of decisions made during paper preparation for RSS: Data Science and Artificial Intelligence submission.

---

## Terminology

- **"Sampling premium"** — preferred over "overhead" (negative connotation) or "cost"; consistent with the paper's framing of DPitG as an improvement, not a burden.
- **No "recouple"** — DPitG is not described as recoupling stopping and decision rules (which reads as contradicting the earlier argument that coupling is harmful). Instead: "precision is a necessary but no longer sufficient stopping condition."
- **Method name** — "Decisive Precision is the Goal" (DPitG), not "ePitG". The prefix "e" was an earlier working label.
- **"Programme evaluations"** — UK spelling throughout (RSS requirement); similarly "favour", "analyse", "colour", "acknowledgement".

---

## Numbers and Framing

- **4%, not 4.7%** — The median sampling premium cited in the abstract and introduction is 4%, which is the median for the *conclusive subset* of experiments. The 4.7% figure (all experiments) was used in earlier drafts and has been replaced for consistency.
- **Percentage framing in prose** — e.g. "4% larger", "41% larger"; ratios (1.04, 1.41) reserved for the table column itself.
- **Zero-premium case** — phrased as "median DPitG stop equals $N_{\rm goal}$ (no premium)", not "0% overhead".
- **Fair coin qualifier** — "$(\Delta_{\rm ROPE}=0.1)$" added wherever the 0%–41% sampling premium range is cited outside `sec:goal_impact`, since those numbers are specific to that setting.
- **Undefined ratio** — "---" in table for the Conclusiveness Ratio at $\omega_{\rm goal}=0.10$ where PitG conclusiveness is zero; prose uses "undefined".

---

## Citations

- **Bayes Factors** — cite `kassraftery1995` (Kass & Raftery, 1995) at first mention in the introduction.
- **HDI+ROPE** — cite `\citealp{kruschke2013, kruschke2015doing}` together; renders as "Kruschke, 2013, 2015". The 2013 journal paper is the earliest reference in the bib; the 2011 POPS paper ("Bayesian Assessment of Null Values Via Parameter Estimation and Model Comparison", *Perspectives on Psychological Science* 6(3), 299–312) is the earliest known reference but is not yet in the bib.
- **p-value thresholds** — cite `fisher1925` at first mention.
- **`\citealp` inside parentheticals** — use `\citealp` (not `\citep`) when a citation sits inside an existing parenthetical expression, to avoid nested parentheses.

---

## Abstract

- **Zero false-positive rate** contextualised against the coupled baseline: "the coupled stopping approach achieves comparable conclusiveness but at a 6.2% false-positive cost under the same conditions." The baseline is referred to as "the coupled stopping approach" (not "HDI+ROPE") because HDI+ROPE has not yet been introduced at that point in the abstract.

---

## Introduction Structure

- **DPitG introduced before null-confirmation context** — the proposal paragraph leads with the DPitG definition, immediately ties it back to PitG's limitation, then expands into why null confirmation matters, then the NHST contrast. (Previous order buried the proposal after two motivational sentences.)
- **Quantitative preview condensed** — Para 7 reduced to headline numbers plus forward reference to sections, avoiding duplication of the abstract.
- **Replication crisis and pre-registration** — (`osc2015`, `gelman2014`, `nosek2018`) elevated from appendix-only to the introduction paragraph that introduces DPitG.

---

## RSS Submission Compliance

- **UK spelling** throughout main text.
- **No abbreviations in abstract** — to be reviewed before final submission.
- **Lists** — lettered (a)(b)(c) using `enumitem`, not bullet points. `\usepackage{enumitem}` added to preamble.
- **No `\hline` inside table bodies** — RSS style prohibits horizontal rules within the table body; `results.tex` ~L131 and ~L166 still need fixing.
- **Alt text** — required for all figures. Figure 1 (cherry_posteriors.png) is done; remaining figures still need alt text below their captions.
- **Graphical abstract** — strongly encouraged by RSS; not yet prepared.
- **TODO placeholder** — `conclusion.tex:57` still has `\textcolor{red}{\textbf{[TODO: replace repo URL]}}` pending repo finalisation.
