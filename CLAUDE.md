# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
It explains how to *modify* this code. For guidance on how to *use* the library, see
`pair_match/USAGE.md`, also available at runtime via `pair_match.usage()`.

## What this is

`PairMatch` gives randomization inference for **matched pairs with binary outcomes**:
exact confidence sets for the attributable effects `A_1`/`A_0` and for the
ATT/ATU/ATE, worst-case p-values, and Rosenbaum sensitivity analysis (`Γ•`). It
inverts a worst-case McNemar test, which has a closed-form worst-case allocation, so
intervals cost `O(log S)` binomial tail evaluations. It makes **no monotonicity
assumption** by default (`monotonic=True` opts in) and fits no outcome model. How the
pairs were formed is out of scope; the package takes pairs as given. The method is
from Wilson (2026), "Randomization Inference for Matched Pairs with Binary Outcomes"
(https://arxiv.org/abs/2609.03227); docstrings cite it by section, equation, and table
number.

The library and PyPI distribution are named `PairMatch`; the import package is
`pair_match` (`from pair_match import PairedOutcomeTable`).

## Setup

Tooling matches the sibling libraries `../Cvxium` and `../PyRake`: `uv` (with a `dev`
dependency group), hatchling, black, ruff, mypy, and pytest. After cloning, install
the pre-commit hook, which runs the same four checks as CI
(`.github/workflows/ci.yml`):

```bash
uv sync
cp scripts/pre-commit .git/hooks/pre-commit
```

## Build and Test

```bash
# Build a wheel
uv build --wheel

# Run all tests
uv run python -m pytest

# Run a specific test
uv run python -m pytest test/test_net_effects.py::PairedOutcomeTableTest::test_from_outcomes

# Format code
uv run python -m black pair_match/ test/

# Lint
uv run python -m ruff check pair_match/ test/

# Check typing
uv run python -m mypy

# Everything CI runs
scripts/pre-commit

# Build the docs site and run its doctests
uv sync --group docs
cd docs && uv run make html SPHINXOPTS="-W --keep-going" && uv run make doctest
```

The mypy config deliberately omits `python_version`, so mypy checks against the
running interpreter. numpy>=2.5 requires Python 3.12 and ships stubs that a pinned
3.11 target cannot parse. Ruff's `TC` rules move annotation-only imports into
`if TYPE_CHECKING:` blocks; every module uses `from __future__ import annotations`,
so this is safe.

## Architecture

Four modules, layered bottom-up:

- `match_result.py`: `MatchResult`, a frozen record of `treated_index`/`control_index`
  label pairs, and the hand-off from any external matching step.
  `PairedOutcomeTable.from_match_result` accepts any `Pairing`, a `Protocol` requiring
  only those two attributes (not exported in `__all__`).
- `visualizations.py`: private plotting helpers (`_resolve_gamma_max`,
  `_sweep_sensitivity_bands`, `_plot_sensitivity_curve`). They take **callables**
  (band functions, capacity), not tables, so they have no dependency on the inference
  modules and serve both of the modules below.
- `net_effects.py`: the core. `PairedOutcomeTable` (the 2x2 counts `s00, s01, s10,
  s11`) with `analyze()`, `confidence_interval`, `gamma_star`, `capacity`,
  `plot_sensitivity`, and so on; `PairedOutcomeAnalysis` / `PairedOutcomeAnalysisOptions`
  (result object with `__str__` table output and `to_dict`/`serialize`/`deserialize`);
  the `EffectSize` enum; and the public functions (`attributable_effect_interval`,
  `att_confidence_set`, `worst_case_pvalue`, `sensitivity_value`,
  `design_sensitivity_binary`, `mcnemar_ate_interval`). Interval inversion runs by
  `method`: `"exact"` does a binary search over binomial tails, `"normal"` uses the
  closed-form quadratic roots, and `"auto"` chooses by size (`_resolve_method`). Only
  the confidence sets honor `method`; p-values and `Γ•` are always exact. ATE
  combines `A_1` and `A_0` via the Rigdon–Hudgens Bonferroni split.
- `linear_combination.py`: `LinearCombinationEstimator` (affine combination
  `affine + Σ c_i θ_i` of tables measured on the *same* pairs), `DiffInDiff`
  (post minus pre placebo), and `LinearCombinationAnalysis`/`LinearCombinationTerm`.
  Inference is a **Bonferroni union bound**, not variance propagation: `alpha / k`
  per nonzero term, combined sign-aware. The estimator holds only the combination;
  `target`/`monotonic`/`method` go to the analysis methods. It imports many
  **private** names from `net_effects.py` (table headers such as `_CI_HEADER`,
  validators, `_encode_float`/`_decode_float`, `_gamma_star_search`,
  `_attributable_sensitivity_band`). Renaming any of those breaks this module.

`__init__.py` re-exports the public API through `__all__` and overrides `__dir__` to
return it. `usage()` reads `USAGE.md`, which ships as package data.

## Invariants the tests enforce

- `test/test_packaging.py` requires every name in `__all__` to resolve, requires
  `dir(pair_match) == __all__`, and requires every `from X import` line in `USAGE.md` to
  import from `pair_match`. When you add or rename a public symbol, update `__all__`
  and `USAGE.md` along with it.
- `test/test_net_effects.py` uses the running example from the paper
  (`s00, s01, s10, s11 = 800, 30, 70, 100`; Wilson, 2026, sections 3-6) as ground
  truth. Both inference test files import private helpers directly (`_p_greater`,
  `_combined_tail_max`, `_normal_worst_case_interval`, `_resolve_method`,
  `_PVALUE_EPS`, `_validate_alpha`), so renaming one breaks the tests too.
- Tests use `unittest.TestCase` classes, which pytest runs.

## Documentation

`README.md` is the short overview and also the PyPI project description, so edits to it
reach PyPI only with the next release. `pair_match/USAGE.md` is the full user guide
(`pair_match.usage()`). NumPy-style docstrings are the authoritative reference for
parameters and edge cases, so keep them current when behavior changes.

The Sphinx site in `docs/` (pydata theme, modeled on `../bootstrap-stat`) is served at
https://www.convexanalytics.com/PairMatch/index.html. `docs/api.rst` renders the
docstrings via autodoc; `docs/user_guides/reproducing_the_paper.rst` reproduces the
paper's running example section by section. Every number that guide quotes is a
doctest, and its figures are regenerated by matplotlib's `plot` directive, so a library
change that moves a result fails `make doctest`. The paper's Figures 2-3 use
`scipy.stats.binom` directly because the library has no public single-pattern p-value.
`docs/_build/` is not committed.

`.github/workflows/deploy_docs.yml` builds with `-W` (warnings are errors) and runs the
doctests on pull requests and on pushes to master touching `docs/`, `pair_match/`,
`pyproject.toml`, or `uv.lock`, then rsyncs `docs/_build/html/` to the droplet using the
`DOCS_DEPLOY_KEY`, `DOCS_USER`, and `DOCS_HOST` secrets. The rsync is skipped while
those secrets are unset. The known paper/library difference: `mcnemar_ate_interval`
uses the null variance `S01 + S10`, not the Wald variance the paper quotes; the guide
says so rather than hiding it.

## Publishing to PyPI

Pushing a tag matching `v*` triggers `.github/workflows/publish.yml`, which builds with
`uv build` and publishes to PyPI via trusted publishing (no API token; the job runs in
the `pypi` environment with `id-token: write`). To release:

```bash
# Bump `version` in pyproject.toml, commit, push, and wait for CI to pass; then:
git tag v0.2.0
git push origin v0.2.0
```

Nothing checks that the tag matches `version` in `pyproject.toml`, and PyPI never
accepts a version twice, so bump the version before tagging. The publish workflow does
not wait for CI, so tag only a commit that has already passed. Tags are lightweight,
as in Cvxium.
