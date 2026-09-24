"""Randomization inference for matched pairs with binary outcomes.

Given matched pairs in which each unit's outcome is 0 or 1, this package answers three
questions without assuming monotonicity of the treatment effect and without any
distributional assumption beyond the within-pair coin flip:

1. **How large is the effect?** :class:`PairedOutcomeTable` gives exact
   randomization confidence sets for the attributable effect and for the
   average treatment effect (ATT, ATU, or ATE).
2. **How fragile is it?** :meth:`PairedOutcomeTable.sensitivity_analysis`
   reports how much unmeasured confounding -- Rosenbaum's ``Gamma`` -- the
   finding withstands before it can no longer be distinguished from bias.
3. **How do effects combine?** :class:`LinearCombinationEstimator` and
   :class:`DiffInDiff` carry that inference through a weighted sum of net
   effects measured on the same pairs, such as a post-period effect minus
   a pre-period placebo.

A one-minute start, from two aligned 0/1 outcome vectors::

    from pair_match import PairedOutcomeTable

    table = PairedOutcomeTable.from_outcomes(treated_y, control_y)
    print(table)            # the 2x2 table, margins, and success rates
    print(table.analyze())  # point estimate, confidence set, sensitivity

See ``USAGE.md`` (also available at runtime via ``pair_match.usage()``) for the full
guide.

"""

from __future__ import annotations

import importlib.resources

from pair_match.linear_combination import (
    DiffInDiff,
    LinearCombinationAnalysis,
    LinearCombinationEstimator,
    LinearCombinationTerm,
)
from pair_match.match_result import MatchResult
from pair_match.net_effects import (
    EffectSize,
    PairedOutcomeAnalysis,
    PairedOutcomeAnalysisOptions,
    PairedOutcomeTable,
    att_confidence_set,
    attributable_effect_interval,
    design_sensitivity_binary,
    mcnemar_ate_interval,
    sensitivity_value,
    worst_case_pvalue,
)

__all__ = [
    "DiffInDiff",
    "EffectSize",
    "LinearCombinationAnalysis",
    "LinearCombinationEstimator",
    "LinearCombinationTerm",
    "MatchResult",
    "PairedOutcomeAnalysis",
    "PairedOutcomeAnalysisOptions",
    "PairedOutcomeTable",
    "att_confidence_set",
    "attributable_effect_interval",
    "design_sensitivity_binary",
    "mcnemar_ate_interval",
    "sensitivity_value",
    "usage",
    "worst_case_pvalue",
]


def usage() -> str:
    """Return the contents of USAGE.md."""
    return importlib.resources.read_text("pair_match", "USAGE.md")


def __dir__() -> list[str]:
    return __all__
