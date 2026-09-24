# PairMatch Usage Guide

`PairMatch` draws inference from **matched pairs with binary
outcomes**. Each pair contributes one treated unit and one control, and
each unit's outcome is 0 or 1. From the four resulting counts the
library produces confidence sets for the treatment effect that rest on
nothing more than within-pair randomization — no monotonicity
assumption, no outcome model, no distributional assumption beyond the
coin flip — and reports how much unmeasured confounding the finding
withstands.

It implements the *net effects* method: invert a worst-case McNemar test
for the attributable effects `A_1` and `A_0`, then combine them via the
Rigdon–Hudgens Bonferroni proposition. The worst-case allocation has a
closed form, so the confidence sets cost `O(log S)` binomial tail
evaluations rather than an integer program.

**Scope.** This package does the *estimation* half of a matched study.
How the pairs were formed — propensity-score matching, exact matching,
or a pairing already present in the data — is out of scope and up to
you.

Every class and method carries a NumPy-style docstring documenting its
parameters, return values, and edge cases; those docstrings are the
authoritative reference:

```python
from pair_match import PairedOutcomeTable

help(PairedOutcomeTable)
help(PairedOutcomeTable.analyze)
```

## Installation

```bash
pip install PairMatch
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add PairMatch
```

Requires Python 3.11+ and numpy, scipy, pandas, tabulate, and
matplotlib (the last only for the plotting methods).

## The 2x2 Table

Everything starts from the four counts of outcome patterns among the
pairs. The first subscript is the treated unit's outcome, the second the
control's, so `s10` counts pairs where the treated unit succeeded and
its control did not.

From aligned outcome vectors — `treated_outcomes[i]` is paired with
`control_outcomes[i]`:

```python
from pair_match import PairedOutcomeTable

table = PairedOutcomeTable.from_outcomes(treated_outcomes, control_outcomes)
table.hat_a     # S10 - S01, the point estimate of the attributable effect
table.ate_hat   # hat_a / n_pairs, the ATT point estimate
```

Or straight from the counts, if you already have them:

```python
table = PairedOutcomeTable(s00=62, s01=57, s10=79, s11=87)
```

Or from a pairing expressed as index labels into two frames that carry
the outcome as a column:

```python
from pair_match import MatchResult

pairs = MatchResult(
    treated_index=["t0", "t1", "t2"],
    control_index=["c7", "c3", "c9"],
)
table = PairedOutcomeTable.from_match_result(
    pairs, "converted", df_treated=df_treated, df_control=df_control
)
```

Any object carrying `treated_index` and `control_index` works there —
the method reads only those two attributes.

## Analyzing the Table

`analyze()` bundles the whole net-effects analysis into a
`PairedOutcomeAnalysis`, which prints as a compact summary: the ATT and
the attributable effect (`iSuccess`), each with a confidence interval,
plus the worst-case p-value and the Rosenbaum sensitivity value `Γ•`.

```python
table = PairedOutcomeTable(s00=62, s01=57, s10=79, s11=87)
print(table.analyze(alpha=0.10))
```

```
|    ATT |   Conf Int**    |   iSuccesses |  Conf Int**  |   p-Value* |   Γ• |
|--------+-----------------+--------------+--------------+------------+------|
| +7.72% | -1.75%, +17.19% |          +22 |   -5, +49    |      0.193 |    1 |
*  An asterisk in the p-Value column indicates statistical significance at
   level 0.10, provided Γ≤1.
   p-Value is two-sided against the null hypothesis of 0 iSuccesses.
** Confidence intervals have coverage of at least 90%, provided Γ≤1.
ATT is the average treatment effect on the matched treated units (matched controls
   serve as their counterfactuals), not a population ATE.
```

`alpha` (default `0.10`) sets both the significance threshold and the
`1 - alpha` confidence coverage; `gamma` (default `1.0`) is the
sensitivity parameter you *entertain* to widen the intervals — distinct
from `Γ•`, which the data determine. Pass a
`PairedOutcomeAnalysisOptions` to pick which effect-size columns show —
including a `cost_per_isuccess` column when the table carries a
`spend` — and to override headers or number formats.

### Which effect

By default `analyze()` targets `A_1` and labels the scaled column
**ATT**. Pass `target="ATU"` to analyze the effect on the *untreated*
instead: the `iSuccesses` column and its confidence set then describe
`A_0`, and the scaled column is labeled **ATU** — the `A_0 / n_pairs`
average treatment effect on the untreated. The scaled column always
tracks `target`, so it and the `iSuccesses` column describe the same
effect.

Pass `target="ATE"` for the Rigdon–Hudgens **average treatment effect**,
which combines both attributable effects through
`A_1 + A_0 = 2 S · ATE`. The scaled column (labeled **ATE**) is the
expanded ATE confidence interval — the same set
`table.expanded_confidence_interval(...)` returns — and the `iSuccesses`
column is that set on the count scale, `ATE · n_pairs = (A_1 + A_0) / 2`.
Because the ATE has no single McNemar pivot, its p-value and `Γ•` are
*not* a single tail: the test is an intersection-union over the ways
`A_1 + A_0` can split into the tested sum budget, with a Bonferroni price
for combining the two effects, so it agrees with `table.gamma_star(...)`.
`null_value` is then the ATE `iSuccesses` count. The ATE column is
Bonferroni-conservative relative to the single-effect `ATT`/`ATU`
columns — the price of not assuming which side carries the effect.

### Exact or approximate

Pass `method` to choose how the confidence sets are inverted: `"exact"`
(binary search over exact binomial tails), `"normal"` (the closed-form
large-sample approximation), or `"auto"` (the default — picks from the
sample size, matching `expanded_confidence_interval` / `gamma_star`).
Only the confidence sets honor `method`; the p-value and `Γ•` are always
exact, since they need no inversion and so cost nothing to compute
exactly. Use `method="exact"` when you want the guaranteed-conservative
sets regardless of scale, or `"normal"` to skip the binary search on
large tables.

### Monotonicity

The procedure makes **no monotonicity assumption** by default — its
whole point is valid inference without one. When treatment plausibly
never *hurts* any unit, pass `monotonic=True` to `analyze()` (or to
`att_confidence_set` / `attributable_effect_interval` /
`sensitivity_value` / `worst_case_pvalue`): the prevention possibilities
vanish, every confidence set narrows, the p-value sharpens, and `Γ•`
rises. The footer then reads `… (assuming monotonicity)`.

### Convenience entry points

`att_confidence_set(table, confidence=0.90)` returns the same
`PairedOutcomeAnalysis` (at `alpha = 1 - confidence`); read the raw tuple
off `result.effect_interval` and serialize the whole analysis with
`result.serialize()`. Because `att_confidence_set` always targets the
treated side, its scaled interval is the treated-side
attributable-effect set (`result.attributable_interval`, the
`iSuccesses` column) divided by `n_pairs` — not the Rigdon–Hudgens
`(A_1 + A_0) / (2 S)` ATE combination. (In `analyze()`, by contrast, the
scaled interval tracks `target`.) It is deliberately wider than the
textbook McNemar interval (`mcnemar_ate_interval`) because it makes no
monotonicity or multinomial-sampling assumption — it is valid for the
ATT under pure randomization alone.

To bound a single attributable effect rather than the ATT:

```python
from pair_match import attributable_effect_interval

# 90% two-sided prediction set for the net effect among treated units.
attributable_effect_interval(table, target="ATT", confidence=0.90)
```

## Sensitivity Analysis

An observational finding is only as good as its robustness to hidden
bias. `sensitivity_value` returns the Rosenbaum `Γ•` — the largest
departure from random assignment at which the two-sided finding survives
at level `alpha`:

```python
from pair_match import sensitivity_value

gamma_star = sensitivity_value(table, alpha=0.05)  # H0: A_1 <= 0
```

`Γ` is an odds ratio: at `Γ = 2`, two units with identical measured
covariates could differ by up to 2:1 in their odds of being treated. A
`Γ•` near 1 means the result would be overturned by even slight hidden
bias; a large `Γ•` means it is robust. You can also pass any `gamma >= 1`
to `att_confidence_set` or `attributable_effect_interval` to widen the
interval for a fixed level of assumed bias.

`table.plot_sensitivity()` sweeps `gamma` and draws the whole picture —
the point estimate, the confounding-only *sensitivity interval*, the
wider *sensitivity/confidence interval*, and `Γ•` annotated where the
finding stops being significant — returning `(DataFrame, Axes)`:

```python
# The ATT (A_1) by default, matching analyze(); pass target="ATU" for the
# ATU or target="ATE" for the Rigdon-Hudgens average effect.
data, ax = table.plot_sensitivity(target="ATT", alpha=0.10)
```

The left axis is the scaled effect (**ATT** / **ATU** / **ATE**) and a
secondary right axis rescales it to the matching `iSuccesses` count
(`effect · n_pairs`), so the rate and the count read off the same curves.
`Γ•` inverts the *plotted* confidence band, so the dotted line and the
band cross `null_value` together for whichever `target` you choose;
`method` and `monotonic` are forwarded exactly as in `analyze()`.

### Design sensitivity

`design_sensitivity_binary(baseline, ate)` returns the ceiling `Γ̃` that
`Γ•` tends to as the study grows — the robustness the *design* can
deliver at that effect size, no matter the sample size. Use it before
running a study to judge whether the adjustment strategy is thorough
enough to be worth the cost:

```python
from pair_match import design_sensitivity_binary

design_sensitivity_binary(baseline=0.13, ate=0.04)  # -> ~1.31
```

Pass the general-case baseline `p_{+1}` (the control success rate), or
`p_{01}` when assuming monotonicity. A small `Γ̃` means no amount of data
will make the finding robust; the answer is a better design, not a
bigger sample.

## Combining Net Effects

Sometimes the quantity of interest is not one table's net effect but an
affine combination of several: a difference between two outcomes
measured on the same pairs, a weighted blend, or a net effect shifted by
a known constant. `LinearCombinationEstimator` estimates

```
theta = affine + sum_i c_i * theta_i
```

where each `theta_i` is one `PairedOutcomeTable`'s scaled net effect for
whichever `target` the analysis asks for. All the tables must describe
the *same* matched pairs (same `n_pairs`) under different outcomes:

```python
from pair_match import LinearCombinationEstimator, PairedOutcomeTable

# Two 0/1 outcomes measured on the same matched pairs.
retained = PairedOutcomeTable.from_outcomes(t_retained, c_retained)
engaged = PairedOutcomeTable.from_outcomes(t_engaged, c_engaged)

est = LinearCombinationEstimator([(0.5, retained), (0.5, engaged)])
est.point_estimate()                        # the combined scaled effect
est.confidence_interval(alpha=0.10)         # randomized (Gamma = 1)
est.expanded_confidence_interval(alpha=0.10, gamma=2.0)
```

The estimator holds only the *combination* — which tables, which
coefficients, what offset. `target`, `monotonic` and `method` describe an
analysis of it, so they are arguments to the methods that perform one
(`confidence_interval`, `expanded_confidence_interval`, `pvalue`,
`gamma_star`, `sensitivity_analysis`, `plot_sensitivity`, `analyze`),
exactly as `PairedOutcomeTable` takes them. One estimator therefore
reports the same combination as an ATT and as an ATE without being
rebuilt:

```python
est.confidence_interval(alpha=0.10)                 # the ATT
est.confidence_interval(alpha=0.10, target="ATE")   # the ATE, same estimator
```

Nothing ties one call's `target` to another's, which is the price of
taking them per call: a `Γ•` computed for the ATE and an interval
computed for the ATT are not a matched pair, and neither reports the
mismatch. When the numbers are going to be read together, reach for
`analyze`, which runs one target across all of them and records which.

**The inference is a Bonferroni union bound, not a variance sum.** The
components share their matched pairs, so their effects are dependent —
propagating variances would assume independence and be anti-conservative
here. Instead the level is split evenly across the terms with a nonzero
coefficient, each component interval is taken at its share, and the
endpoints are combined sign-aware. That is valid under *arbitrary*
dependence, at the price of ignoring the positive correlation between
components: the interval is conservative, sometimes materially so. Note
that the split compounds with the ones inside each component — a
two-sided component halves its share again per tail, and `target="ATE"`
halves it once more across `A_1` and `A_0` — so an ATE combination is the
most conservative form.

Two entries on the *same* outcome each pay a share, so sum their
coefficients into a single term rather than listing a table twice. A
coefficient of exactly `0.0` costs nothing: zero-coefficient terms are
dropped from the budget entirely.

### Difference-in-Differences

`DiffInDiff` is the combination you reach for most often, with the
coefficients fixed to `(-1, +1)`:

```python
from pair_match import DiffInDiff

# Y = S - P: the post-period net effect minus the pre-period (placebo)
# net effect, on the same pairs.
did = DiffInDiff(pre_table=pre, post_table=post)
did.point_estimate()
did.confidence_interval(alpha=0.10)
```

Under a parallel-trends assumption the pre-period table measures
whatever bias the design carries, so subtracting it removes that bias
from the post-period estimate. A pre-period net effect that is itself
significantly nonzero is a warning: the pairs were already diverging
before treatment. The two tables are keyword-only, because they have the
same type and swapping them silently sign-flips the answer. Beyond the
tables it takes only `affine`; the analysis arguments go to the
inference methods, as they do on the general estimator.

### Analyzing a Combination

`analyze()` bundles the whole thing into a `LinearCombinationAnalysis`,
the combination's counterpart to `PairedOutcomeAnalysis`: it prints as a
table with one row per term plus a `Combined` row, and carries the
Bonferroni p-value and the sensitivity value `Γ•` for the tested null.

The table quotes each effect twice: as a scaled rate (the `ATT` / `ATU` /
`ATE` column) and as an `iSuccesses` count, the same rate times the
matched-pair count the tables share. The `Combined` row adds the p-value
(starred when it clears `alpha`) and `Γ•`, so the reading "significant,
and robust to hidden bias out to this odds ratio" sits on one line.

```python
est = LinearCombinationEstimator(
    [(1.0, retained), (-1.0, engaged)], labels=["retained", "engaged"]
)
result = est.analyze(alpha=0.10, gamma=1.0)
print(result)

result.point_estimate      # the combined effect
result.effect_interval     # the combined Bonferroni interval, as a tuple
result.significant         # whether that interval excludes the null
result.p_value             # the Bonferroni p-value for the tested null
result.n_pairs             # the shared matched-pair count (0 only if no terms)
result.terms               # the per-term breakdown
```

`labels` names the rows (they default to `term 1`, `term 2`, …;
`DiffInDiff` supplies `pre` and `post`). Each term's interval is shown at
the Bonferroni *share* it actually enters the combination with, not at
the full `alpha` — so the per-term intervals are wider than the ones you
would get analyzing each table on its own, and that is the price the
union bound charges made visible.

The rows are shown on the *side* they enter with, too. A one-sided
combination takes one bound from each component — the combined lower
bound wants a positive coefficient's lower bound and a negative
coefficient's upper one — so under `alternative="greater"` a `+1` term
prints as `+1.20%, ∞` and a `-1` term as `-∞, +2.60%`:

```
| Term     |   Coef |    ATT |  Conf Int**  |   iSuccesses |  Conf Int**  |
|----------+--------+--------+--------------+--------------+--------------|
| pre      |     -1 | +1.00% |  -∞, +2.60%  |          +10 |   -∞, +26    |
| post     |     +1 | +4.00% |  +1.20%, ∞   |          +40 |    +12, ∞    |
| Combined |        | +3.00% |  -1.40%, ∞   |          +30 |    -14, ∞    |
```

The table then adds up: `+12 - 26 = -14`, the combined bound. A
two-sided combination consumes both ends of every component, so its rows
stay two-sided, as does any term with a zero coefficient (it enters
neither bound).

`est.pvalue(null_value=0.0)` returns that p-value directly. Lacking a
closed form for the combined test, it inverts the interval by bisection —
the smallest `alpha` at which the band excludes the null — so it is
reported rounded *up*, the conservative direction. `significant` reads
the interval itself and is the authority when the two sit within a hair
of each other. For a one-sided `alternative` it can exceed `0.5`; that is
the case where the data point away from the null, and the size of the
excess is how far away.

`est.gamma_star(alpha=0.10)` returns `Γ•` for the combination directly:
the largest hidden-bias odds ratio at which the interval still excludes
`null_value`. It is `1.0` when even the randomized interval contains the
null, and `math.inf` when no amount of bias overturns the finding.

`result.serialize()` returns a JSON string (and `to_dict()` the
underlying dict) that `LinearCombinationAnalysis.deserialize()` reads
back. Both directions are strict: a payload carrying a field `analyze`
could not have produced — an out-of-range `alpha`, a `gamma` below 1, a
non-finite null — is rejected rather than reconstructed into an object
that quietly reports nonsense.

### Plotting How a Combination Degrades

`plot_sensitivity` is the combination's version of
`PairedOutcomeTable.plot_sensitivity`: sweep `Gamma` upward and watch two
bands widen around the (bias-independent) combined point estimate — the
confounding-only *sensitivity interval*, and the *expanded confidence
interval* that also carries sampling uncertainty.

```python
data, ax = est.plot_sensitivity(alpha=0.10, num_points=50)
```

It returns the swept frame alongside the axes, so the numbers behind the
picture are available without recomputing them. `Γ•` is marked with a
vertical line when it falls inside the swept range. Both bands and `Γ•`
use the `target` and `monotonic` given here, so the whole figure is one
coherent analysis; `method` reaches the wider band alone, the
confounding-only band having no test to invert.

The default `gamma_max` is `min(6, 0.95 * est.capacity(alpha))`. Six is
the conventional benchmark (roughly the hidden bias the smoking /
lung-cancer association withstands); the capacity term is what keeps the
sweep short of the point where the bands blow up. `capacity` is the
smallest capacity over the terms with a nonzero coefficient — the most
binding one, since the combination is uninformative as soon as any
contributing component is. A combination whose capacity is too small to
sweep raises rather than drawing a plot with nothing in it.

## Serialization

`PairedOutcomeTable`, `PairedOutcomeAnalysis` and
`LinearCombinationAnalysis` all follow the same
`to_dict()` / `serialize()` / `deserialize()` trio, so a result can be
persisted to JSON and read back without recomputation:

```python
payload = table.analyze(alpha=0.10).serialize()
restored = PairedOutcomeAnalysis.deserialize(s=payload)
```

## References

- Wilson, Bob. 2026. "Randomization Inference for Matched Pairs with
  Binary Outcomes." arXiv:2609.03227.
  <https://arxiv.org/abs/2609.03227>
- Rosenbaum, Paul R. 2002. "Attributing Effects to Treatment in Matched
  Observational Studies." *Journal of the American Statistical
  Association* 97 (457): 183–192.
- Rosenbaum, Paul R. 2020. *Design of Observational Studies.* 2nd ed.
  Springer Series in Statistics.
- Rigdon, Joseph, and Michael G. Hudgens. 2015. "Randomization Inference
  for Treatment Effects on a Binary Outcome." *Statistics in Medicine*
  34 (6): 924–935.
- Zhao, Qingyuan. 2019. "On Sensitivity Value of Pair-Matched
  Observational Studies." *Journal of the American Statistical
  Association.*
