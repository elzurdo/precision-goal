# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Precision Is The Goal (PitG)** is a research project implementing and evaluating sequential hypothesis testing methods that use HDI (Highest Density Interval) precision as a stopping criterion. The paper proposes **ePitG** — an enhanced dual stopping criterion (precision + conclusiveness) — compared against HDI+ROPE and standard NHST.

## Environment Setup

```bash
pyenv activate scrappy-3.8.11
pip install -r requirements.txt
```

## Common Commands

```bash
# Run all tests
python -m pytest tests

# Run a single test file
python -m pytest tests/test_utils_experiments.py

# Run a single test
python -m pytest tests/test_utils_experiments.py::test_name

# Build LaTeX paper
cd latex && latexmk precision_goal.tex

# Word count (excluding appendices/references/abstract)
cd latex && texcount -inc -sum precision_goal.tex
```

## Architecture

### Core Algorithm Pipeline

The three stopping methods compared throughout the project are:
1. **HDI+ROPE** — stops when HDI conclusively enters/exits the ROPE
2. **PitG** — stops when precision goal is achieved (may be inconclusive)
3. **ePitG** — stops when BOTH precision is achieved AND a conclusive decision is reached

### Key Modules (`py/`)

**Binomial experiments:**
- `utils_experiments.py` — Core algorithm: `BinomialSimulation`, `BinomialHypothesis`, `BinaryAccounting`, `stop_decision_multiple_experiments_multiple_methods()`
- `utils_stats.py` — `HDIofICDF()`, `successes_failures_to_hdi_ci_limits()`, `CI_FRACTION = 0.95`
- `utils_experiments_shared.py` — Cross-domain result aggregation: `stats_dict_to_df()`, `iteration_counts_to_df()`, `create_decision_correctness_df()`
- `utils_viz.py` — All binomial visualization (1,187 lines)

**Continuous experiments:**
- `utils_experiments_continuous.py` — `ContinuousHypothesis`, `stop_decision_continuous_multiple_methods()` using Student-t HDI
- `utils_viz_continuous.py` — Continuous data visualization
- `utils_data.py` — Data generation utilities

**Notebooks:** Live in `notebooks/`; Python equivalents (jupytext-converted, cells marked `# %%`) live in `py/`.

### Data Flow

```
Simulation params → BinomialSimulation/ContinuousHypothesis
  → stop_decision_*() per iteration
  → stats dicts → stats_dict_to_df() → DataFrames
  → plot_*/viz_*() → figures for paper
```

### Test Suite

Tests in `tests/` are deterministic regression tests using handpicked sequences from the paper. They validate stopping iterations and decision correctness for binomial and NHST approaches.

## Paper Structure (`latex/`)

The paper is split across section files (`precision_goal_abstract.tex`, `_intro.tex`, `_methods.tex`, `_results.tex`, `_discussion.tex`, `_conclusions.tex`, etc.) included from `precision_goal.tex`.
