# PairMatch

Randomization inference for **matched pairs with binary outcomes**.

Given pairs in which one treated unit is matched to one control and each
unit's outcome is 0 or 1, this library produces exact confidence sets for
the treatment effect and reports how much unmeasured confounding the
finding withstands. It assumes nothing beyond the within-pair coin flip —
in particular it does **not** assume the treatment effect is monotonic,
and it fits no outcome model.

The method inverts a worst-case McNemar test for the attributable effects
`A_1` and `A_0` and combines them via the Rigdon–Hudgens Bonferroni
proposition. The worst-case allocation of effects has a closed form, so
the confidence sets cost `O(log S)` binomial tail evaluations — no integer
program and no numerical search.

## Install

```bash
uv pip install .
```

The distribution is named `PairMatch`; the import package is `pair_match`.

Python 3.11+. Depends on numpy, scipy, pandas, tabulate, and matplotlib.

## Quick start

```python
from pair_match import PairedOutcomeTable

# treated_y[i] and control_y[i] are the 0/1 outcomes of the i-th pair.
table = PairedOutcomeTable.from_outcomes(treated_y, control_y)

print(table)             # the 2x2 table, margins, and success rates
print(table.analyze())   # effect, confidence set, p-value, sensitivity value
```

```
|    ATT |   Conf Int**    |   iSuccesses |  Conf Int**  |   p-Value* |   Γ• |
|--------+-----------------+--------------+--------------+------------+------|
| +7.72% | -1.75%, +17.19% |          +22 |   -5, +49    |      0.193 |    1 |
```

The last column is Rosenbaum's sensitivity value: the largest hidden-bias
odds ratio at which the finding still holds. A `Γ•` near 1 means even
slight unmeasured confounding would overturn the result.

## What is in scope

This package covers the **estimation** half of a matched study. How the
pairs were formed — propensity-score matching, exact matching on a few
keys, or a pairing already present in the data — is up to you; supply the
pairs and it takes over from there.

| Capability | Entry point |
| --- | --- |
| The 2x2 table and its summary | `PairedOutcomeTable`, `.analyze()` |
| Attributable effect `A_1` / `A_0` | `attributable_effect_interval` |
| ATT / ATU / ATE confidence sets | `.analyze(target=...)`, `att_confidence_set` |
| Worst-case p-value | `worst_case_pvalue` |
| Sensitivity value `Γ•` | `sensitivity_value`, `.plot_sensitivity()` |
| Design sensitivity `Γ̃` | `design_sensitivity_binary` |
| Textbook comparison | `mcnemar_ate_interval` |
| Weighted sums of net effects | `LinearCombinationEstimator` |
| Difference-in-differences | `DiffInDiff` |

## Documentation

`pair_match/USAGE.md` is the full guide — also available at runtime:

```python
import pair_match
print(pair_match.usage())
```

Every public class and method carries a NumPy-style docstring, which is
the authoritative reference for parameters, defaults, and edge cases.

## Tests

```bash
uv sync
uv run python -m pytest
```

## References

- Rosenbaum, Paul R. 2002. "Attributing Effects to Treatment in Matched
  Observational Studies." *Journal of the American Statistical
  Association* 97 (457): 183–192.
- Rigdon, Joseph, and Michael G. Hudgens. 2015. "Randomization Inference
  for Treatment Effects on a Binary Outcome." *Statistics in Medicine*
  34 (6): 924–935.
- Rosenbaum, Paul R. 2020. *Design of Observational Studies.* 2nd ed.
  Springer Series in Statistics.
