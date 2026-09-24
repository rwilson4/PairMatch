#!/usr/bin/env python3

# pyre-strict
"""Test linear combinations of matched-pair net effects and diff-in-diff."""

import math
import unittest
from collections.abc import Sequence
from dataclasses import replace
from typing import cast

from pair_match.linear_combination import (
    _PVALUE_EPS,
    DiffInDiff,
    LinearCombinationAnalysis,
    LinearCombinationEstimator,
    LinearCombinationTerm,
)
from pair_match.net_effects import (
    PairedOutcomeTable,
    attributable_effect_interval,
)

# Post-period (real outcome) and pre-period (placebo) tables on the same 1000
# matched pairs.
POST = PairedOutcomeTable(s00=800, s01=30, s10=70, s11=100)
PRE = PairedOutcomeTable(s00=900, s01=40, s10=50, s11=10)


def _combine_greater(
    estimator: LinearCombinationEstimator, alpha: float
) -> tuple[float, float]:
    """The exact one-sided combination at ``alpha``, including alpha >= 0.5.

    ``confidence_interval`` refuses a one-sided level at or above 0.5, but ``pvalue``
    bisects the exclusion predicate over the whole unit interval, so those levels are
    reached in practice and are worth pinning.

    """
    return estimator._combine(
        alpha=alpha,
        gamma=1.0,
        alternative="greater",
        target="ATT",
        monotonic=False,
        method="exact",
    )


def _scaled_interval(
    table: PairedOutcomeTable,
    *,
    confidence: float,
    gamma: float = 1.0,
    alternative: str = "two-sided",
    target: str = "ATT",
    monotonic: bool = False,
    method: str = "auto",
) -> tuple[float, float]:
    """The component interval the estimator should be composing, for reference."""
    lo, hi = attributable_effect_interval(
        table,
        target=target,
        confidence=confidence,
        gamma=gamma,
        monotonic=monotonic,
        alternative=alternative,
        method=method,
    )
    n = table.n_pairs
    return (lo / n, hi / n)


class LinearCombinationEstimatorTest(unittest.TestCase):
    """Tests for ``LinearCombinationEstimator``."""

    def test_point_estimate(self) -> None:
        est = LinearCombinationEstimator([(2.0, POST), (0.5, PRE)], affine=0.1)
        expected = 0.1 + 2.0 * POST.ate_hat + 0.5 * PRE.ate_hat
        self.assertAlmostEqual(est.point_estimate(), expected)

    def test_point_estimate_target_invariant(self) -> None:
        # `ate_hat` estimates the ATT, ATU, and ATE alike, which is why the
        # point estimate takes no target: there is nothing for one to select.
        terms: list[tuple[float, PairedOutcomeTable]] = [(1.0, POST), (-1.0, PRE)]
        est = LinearCombinationEstimator(terms)
        self.assertAlmostEqual(est.point_estimate(), POST.ate_hat - PRE.ate_hat)

    def test_confidence_interval_bonferroni_sign_aware(self) -> None:
        # Two nonzero terms -> each interval at alpha / 2.
        est = LinearCombinationEstimator([(2.0, POST), (-1.0, PRE)])
        post_lo, post_hi = _scaled_interval(POST, confidence=0.95)
        pre_lo, pre_hi = _scaled_interval(PRE, confidence=0.95)
        expected_lb = 2.0 * post_lo + (-1.0) * pre_hi
        expected_ub = 2.0 * post_hi + (-1.0) * pre_lo
        lb, ub = est.confidence_interval(alpha=0.10)
        self.assertAlmostEqual(lb, expected_lb)
        self.assertAlmostEqual(ub, expected_ub)

    def test_affine_shifts_both_bounds(self) -> None:
        base = LinearCombinationEstimator([(1.0, POST)])
        shifted = LinearCombinationEstimator([(1.0, POST)], affine=0.25)
        lb0, ub0 = base.confidence_interval()
        lb1, ub1 = shifted.confidence_interval()
        self.assertAlmostEqual(lb1, lb0 + 0.25)
        self.assertAlmostEqual(ub1, ub0 + 0.25)

    def test_zero_coefficients_dropped_from_budget(self) -> None:
        # A zero-coefficient term must not consume Bonferroni budget: the
        # interval should match the single nonzero term at full alpha.
        est = LinearCombinationEstimator([(1.0, POST), (0.0, PRE)])
        lo, hi = _scaled_interval(POST, confidence=0.90)
        lb, ub = est.confidence_interval(alpha=0.10)
        self.assertAlmostEqual(lb, lo)
        self.assertAlmostEqual(ub, hi)

    def test_empty_terms_is_constant(self) -> None:
        est = LinearCombinationEstimator([], affine=0.5)
        self.assertAlmostEqual(est.point_estimate(), 0.5)
        self.assertEqual(est.confidence_interval(), (0.5, 0.5))
        self.assertEqual(
            est.confidence_interval(alternative="greater"), (0.5, math.inf)
        )
        self.assertEqual(est.confidence_interval(alternative="less"), (-math.inf, 0.5))

    def test_expanded_at_gamma_one_matches_randomized(self) -> None:
        est = LinearCombinationEstimator([(1.0, POST), (-1.0, PRE)])
        self.assertEqual(
            est.expanded_confidence_interval(gamma=1.0),
            est.confidence_interval(),
        )

    def test_expanded_widens_with_gamma(self) -> None:
        est = LinearCombinationEstimator([(1.0, POST), (-1.0, PRE)])
        lb1, ub1 = est.confidence_interval()
        lb, ub = est.expanded_confidence_interval(gamma=3.0)
        self.assertLessEqual(lb, lb1)
        self.assertGreaterEqual(ub, ub1)

    def test_one_sided_greater(self) -> None:
        est = LinearCombinationEstimator([(1.0, POST), (-1.0, PRE)])
        post_lo, _ = _scaled_interval(POST, confidence=0.95, alternative="greater")
        _, pre_hi = _scaled_interval(PRE, confidence=0.95, alternative="less")
        lb, ub = est.confidence_interval(alpha=0.10, alternative="greater")
        self.assertAlmostEqual(lb, post_lo - pre_hi)
        self.assertEqual(ub, math.inf)

    def test_one_sided_less(self) -> None:
        est = LinearCombinationEstimator([(1.0, POST), (-1.0, PRE)])
        _, post_hi = _scaled_interval(POST, confidence=0.95, alternative="less")
        pre_lo, _ = _scaled_interval(PRE, confidence=0.95, alternative="greater")
        lb, ub = est.confidence_interval(alpha=0.10, alternative="less")
        self.assertEqual(lb, -math.inf)
        self.assertAlmostEqual(ub, post_hi - pre_lo)

    def test_mismatched_n_pairs_raises(self) -> None:
        smaller = PairedOutcomeTable(s00=10, s01=1, s10=2, s11=1)
        with self.assertRaises(ValueError):
            LinearCombinationEstimator([(1.0, POST), (-1.0, smaller)])

    def test_invalid_target_raises(self) -> None:
        est = LinearCombinationEstimator([(1.0, POST)])
        # Every public entry point validates, since each is handed its own copy
        # rather than reading one the constructor checked.
        with self.assertRaises(ValueError):
            est.confidence_interval(target="ATX")
        with self.assertRaises(ValueError):
            est.expanded_confidence_interval(target="ATX")
        with self.assertRaises(ValueError):
            est.pvalue(target="ATX")
        with self.assertRaises(ValueError):
            est.gamma_star(target="ATX")
        with self.assertRaises(ValueError):
            est.sensitivity_analysis(target="ATX")
        with self.assertRaises(ValueError):
            est.analyze(target="ATX")

    def test_sensitivity_analysis_target_and_monotonic_are_keyword_only(self) -> None:
        # `gamma` stays positional to mirror the single-table method, but this
        # one takes a `target` the single table does not, so its second
        # positional is `target` where the sibling's is `monotonic`. Refuse the
        # call rather than let `sensitivity_analysis(3.0, True)` mean opposite
        # things in the two classes.
        est = LinearCombinationEstimator([(1.0, POST)])
        with self.assertRaises(TypeError):
            # pyre-ignore[19]: deliberately calling the keyword-only
            # parameters positionally.
            est.sensitivity_analysis(3.0, True)  # type: ignore[misc, arg-type]

    def test_invalid_method_raises(self) -> None:
        est = LinearCombinationEstimator([(1.0, POST)])
        with self.assertRaises(ValueError):
            est.confidence_interval(method="bootstrap")
        with self.assertRaises(ValueError):
            est.expanded_confidence_interval(method="bootstrap")
        with self.assertRaises(ValueError):
            est.pvalue(method="bootstrap")
        with self.assertRaises(ValueError):
            est.gamma_star(method="bootstrap")
        with self.assertRaises(ValueError):
            est.analyze(method="bootstrap")

    def test_invalid_alternative_raises(self) -> None:
        est = LinearCombinationEstimator([(1.0, POST)])
        with self.assertRaises(ValueError):
            est.confidence_interval(alternative="sideways")

    def test_gamma_below_one_raises(self) -> None:
        est = LinearCombinationEstimator([(1.0, POST)])
        with self.assertRaises(ValueError):
            est.expanded_confidence_interval(gamma=0.5)

    def test_non_finite_gamma_raises(self) -> None:
        # `gamma < 1.0` is False for both of these, so an ordering test alone
        # lets them through to the worst-case tilt `gamma / (gamma + 1)`, which
        # is NaN for either and makes every bound NaN without erroring.
        est = LinearCombinationEstimator([(1.0, POST)])
        for gamma in (math.nan, math.inf):
            with self.subTest(gamma=gamma), self.assertRaises(ValueError):
                est.expanded_confidence_interval(gamma=gamma)

    def test_alpha_outside_unit_interval_raises(self) -> None:
        # alpha = 0 would ask each component for a level-0 (degenerate)
        # interval, and alpha >= 1 for non-positive coverage. NaN is in the list
        # because it compares False against every bound.
        est = LinearCombinationEstimator([(1.0, POST)])
        for alpha in (0.0, 1.0, -0.1, 1.5, math.nan):
            with self.subTest(alpha=alpha):
                with self.assertRaises(ValueError):
                    est.confidence_interval(alpha=alpha)
                with self.assertRaises(ValueError):
                    est.expanded_confidence_interval(alpha=alpha, gamma=2.0)

    def test_one_sided_alpha_at_least_a_half_raises(self) -> None:
        # A one-sided p-value peaks at 0.5 at the point estimate, so a one-sided
        # bound at alpha >= 0.5 comes back on the far side of the estimate with
        # no coverage reading, so it is rejected outright.
        est = LinearCombinationEstimator([(1.0, POST)])
        for alternative in ("greater", "less"):
            with self.subTest(alternative=alternative):
                with self.assertRaises(ValueError):
                    est.confidence_interval(alpha=0.5, alternative=alternative)
                with self.assertRaises(ValueError):
                    est.confidence_interval(alpha=0.6, alternative=alternative)
        # Still allowed two-sided, where each tail gets alpha / 2.
        est.confidence_interval(alpha=0.6)

    def test_alpha_share_that_underflows_the_coverage_round_trip_raises(
        self,
    ) -> None:
        # The component API is specified in coverage, so the share travels as
        # `1 - alpha`. Below ~5.5e-17 that lands on exactly 1.0 and the
        # component would be inverted at level 0 -- an unbounded interval, i.e.
        # silently never significant. Reachable with an alpha that is itself
        # representable once the union bound splits it across the terms.
        est = LinearCombinationEstimator([(1.0, POST), (1.0, PRE)])
        with self.assertRaises(ValueError):
            est.confidence_interval(alpha=1e-17)

    def test_empty_table_raises(self) -> None:
        # A table with no pairs has no per-pair rate to report; catch it in the
        # constructor rather than dividing by zero downstream.
        with self.assertRaises(ValueError):
            LinearCombinationEstimator(
                [(1.0, PairedOutcomeTable(s00=0, s01=0, s10=0, s11=0))]
            )

    def test_terms_are_frozen(self) -> None:
        # The validated `n_pairs` invariant must survive a caller mutating the
        # list it passed in -- and must not be reachable through the estimator's
        # own attribute either.
        terms: list[tuple[float, PairedOutcomeTable]] = [(1.0, POST)]
        est = LinearCombinationEstimator(terms)
        terms.append((1.0, PairedOutcomeTable(s00=10, s01=1, s10=2, s11=1)))
        self.assertEqual(est.terms, ((1.0, POST),))
        self.assertIsInstance(est.terms, tuple)

    def test_one_shot_iterable_terms_are_still_validated(self) -> None:
        # `terms` is annotated `Sequence`, but the runtime accepts a generator,
        # and the constructor reads it more than once. If it validated the
        # argument instead of the frozen tuple, the finiteness pass would
        # consume the generator and the `n_pairs` pass would see nothing -- so a
        # bad combination would come back as a silent "pure constant" rather
        # than an error. Both halves of that are checked: the defect is caught,
        # and a *valid* generator still materializes in full.
        def gen(
            terms: list[tuple[float, PairedOutcomeTable]],
        ) -> Sequence[tuple[float, PairedOutcomeTable]]:
            # `cast` is a no-op at runtime; it only says the annotation is being
            # deliberately violated the way a caller might violate it.
            return cast(
                "Sequence[tuple[float, PairedOutcomeTable]]", (t for t in terms)
            )

        with self.assertRaises(ValueError):
            LinearCombinationEstimator(gen([(math.nan, POST)]))
        smaller = PairedOutcomeTable(s00=80, s01=3, s10=7, s11=10)
        with self.assertRaises(ValueError):
            LinearCombinationEstimator(gen([(1.0, POST), (-1.0, smaller)]))
        est = LinearCombinationEstimator(gen([(1.0, POST), (0.5, PRE)]), affine=0.1)
        self.assertEqual(est.terms, ((1.0, POST), (0.5, PRE)))
        self.assertAlmostEqual(
            est.point_estimate(), 0.1 + POST.ate_hat + 0.5 * PRE.ate_hat
        )

    def test_labels_are_frozen(self) -> None:
        # Same reasoning as the terms tuple: `analyze` zips labels against terms,
        # so a shortened list must not silently drop a term from the breakdown
        # while the Combined row still counts it.
        labels = ["real", "placebo"]
        est = LinearCombinationEstimator([(1.0, POST), (-1.0, PRE)], labels=labels)
        labels.pop()
        self.assertEqual(est.labels, ("real", "placebo"))
        self.assertIsInstance(est.labels, tuple)
        self.assertEqual([t.label for t in est.analyze().terms], ["real", "placebo"])

    def test_all_zero_coefficients_is_constant(self) -> None:
        # The other route into the k == 0 branch: terms are present but none
        # contributes, so the combination is the offset alone.
        est = LinearCombinationEstimator([(0.0, POST), (0.0, PRE)], affine=0.2)
        self.assertAlmostEqual(est.point_estimate(), 0.2)
        self.assertEqual(est.confidence_interval(), (0.2, 0.2))
        self.assertEqual(
            est.confidence_interval(alternative="greater"), (0.2, math.inf)
        )

    def test_non_finite_coefficient_or_affine_raises(self) -> None:
        # A NaN coefficient passes the `c != 0.0` filter and would silently
        # return NaN bounds; an infinite one poisons the sum. Fail loudly.
        for bad in (math.nan, math.inf, -math.inf):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    LinearCombinationEstimator([(bad, POST)])
                with self.assertRaises(ValueError):
                    LinearCombinationEstimator([(1.0, POST)], affine=bad)

    def test_monotonic_forwarded_and_narrows(self) -> None:
        # Assuming treatment never hurts shrinks every component set, so the
        # combination inherits the narrower interval.
        est = LinearCombinationEstimator([(1.0, POST)])
        self.assertEqual(
            est.confidence_interval(alpha=0.10, monotonic=True),
            _scaled_interval(POST, confidence=0.90, monotonic=True),
        )
        lb0, ub0 = est.confidence_interval()
        lb1, ub1 = est.confidence_interval(monotonic=True)
        self.assertGreaterEqual(lb1, lb0)
        self.assertLessEqual(ub1, ub0)

    def test_method_exact_forwarded_to_components(self) -> None:
        est = LinearCombinationEstimator([(1.0, POST)])
        self.assertEqual(
            est.confidence_interval(alpha=0.10, method="exact"),
            _scaled_interval(POST, confidence=0.90, method="exact"),
        )

    def test_target_atu_and_ate_intervals_use_their_own_sets(self) -> None:
        # The point estimate is target-invariant, but the interval is not: each
        # target inverts a different attributable-effect set.
        for target in ("ATU", "ATE"):
            with self.subTest(target=target):
                est = LinearCombinationEstimator([(1.0, POST), (-1.0, PRE)])
                post_lo, post_hi = _scaled_interval(
                    POST, confidence=0.95, target=target
                )
                pre_lo, pre_hi = _scaled_interval(PRE, confidence=0.95, target=target)
                self.assertEqual(
                    est.confidence_interval(alpha=0.10, target=target),
                    (post_lo - pre_hi, post_hi - pre_lo),
                )

    def test_one_estimator_serves_several_targets(self) -> None:
        # The point of taking the target per call: the same estimator answers
        # for the ATT and the ATU without being rebuilt, and the two answers
        # differ (so neither call is leaking the other's target).
        est = LinearCombinationEstimator([(1.0, POST)])
        att = est.confidence_interval(alpha=0.10)
        atu = est.confidence_interval(alpha=0.10, target="ATU")
        self.assertEqual(att, _scaled_interval(POST, confidence=0.90))
        self.assertEqual(atu, _scaled_interval(POST, confidence=0.90, target="ATU"))
        self.assertNotEqual(att, atu)
        # And asking again for the first still gives the first.
        self.assertEqual(est.confidence_interval(alpha=0.10), att)


class DiffInDiffTest(unittest.TestCase):
    """Tests for ``DiffInDiff``."""

    def test_coefficients_and_point_estimate(self) -> None:
        did = DiffInDiff(pre_table=PRE, post_table=POST)
        self.assertEqual(did.terms, ((-1.0, PRE), (1.0, POST)))
        # Y = S - P = post - pre.
        self.assertAlmostEqual(did.point_estimate(), POST.ate_hat - PRE.ate_hat)

    def test_confidence_interval_is_s_minus_p(self) -> None:
        did = DiffInDiff(pre_table=PRE, post_table=POST)
        post_lo, post_hi = _scaled_interval(POST, confidence=0.95)
        pre_lo, pre_hi = _scaled_interval(PRE, confidence=0.95)
        lb, ub = did.confidence_interval(alpha=0.10)
        self.assertAlmostEqual(lb, post_lo - pre_hi)
        self.assertAlmostEqual(ub, post_hi - pre_lo)

    def test_affine_is_forwarded(self) -> None:
        # Only the coefficients are fixed by this subclass, so a known constant
        # offset stays reachable rather than sending the caller back to the
        # general constructor to re-write the `(-1, +1)` terms by hand.
        shift = 0.25
        plain = DiffInDiff(pre_table=PRE, post_table=POST)
        shifted = DiffInDiff(pre_table=PRE, post_table=POST, affine=shift)
        self.assertEqual(shifted.affine, shift)
        self.assertAlmostEqual(shifted.point_estimate(), plain.point_estimate() + shift)
        lb, ub = plain.confidence_interval(alpha=0.10)
        s_lb, s_ub = shifted.confidence_interval(alpha=0.10)
        self.assertAlmostEqual(s_lb, lb + shift)
        self.assertAlmostEqual(s_ub, ub + shift)

    def test_affine_defaults_to_zero(self) -> None:
        self.assertEqual(DiffInDiff(pre_table=PRE, post_table=POST).affine, 0.0)

    def test_exposes_pre_and_post(self) -> None:
        did = DiffInDiff(pre_table=PRE, post_table=POST)
        self.assertIs(did.pre_table, PRE)
        self.assertIs(did.post_table, POST)

    def test_pre_and_post_are_read_only_views_of_terms(self) -> None:
        # `terms` is frozen so the validated invariants survive construction.
        # Stored copies of the same tables would sidestep that: a rebound
        # `post_table` would disagree with the tables every computation uses.
        did = DiffInDiff(pre_table=PRE, post_table=POST)
        with self.assertRaises(AttributeError):
            # pyre-ignore[41]: deliberately assigning to a read-only property.
            did.post_table = PRE  # type: ignore[misc]

    def test_tables_are_keyword_only(self) -> None:
        # The two tables are interchangeable to the type checker and to every
        # runtime check, so a swapped positional call would estimate `P - S`
        # and merely flip the sign of the answer. Pinned as a contract: the
        # `*` is load-bearing, not incidental formatting.
        with self.assertRaises(TypeError):
            # pyre-ignore[20]: deliberately calling the keyword-only
            # parameters positionally.
            DiffInDiff(PRE, POST)  # type: ignore[misc]

    def test_mismatched_n_pairs_raises(self) -> None:
        smaller = PairedOutcomeTable(s00=10, s01=1, s10=2, s11=1)
        with self.assertRaises(ValueError):
            DiffInDiff(pre_table=smaller, post_table=POST)


class AnalyzeTest(unittest.TestCase):
    """Tests for ``LinearCombinationEstimator.analyze`` and its result class."""

    def test_analyze_effect_matches_point_estimate_and_interval(self) -> None:
        est = LinearCombinationEstimator([(1.0, POST), (-1.0, PRE)])
        result = est.analyze(alpha=0.10)
        self.assertAlmostEqual(result.effect, est.point_estimate())
        self.assertAlmostEqual(result.point_estimate, est.point_estimate())
        self.assertEqual(result.effect_interval, est.confidence_interval(alpha=0.10))

    def test_analyze_term_breakdown(self) -> None:
        # Two nonzero terms -> each component interval taken at the alpha / 2
        # Bonferroni share, i.e. 95% confidence.
        est = LinearCombinationEstimator(
            [(2.0, POST), (-1.0, PRE)], labels=["real", "placebo"]
        )
        result = est.analyze(alpha=0.10)
        self.assertEqual([t.label for t in result.terms], ["real", "placebo"])
        self.assertEqual([t.coefficient for t in result.terms], [2.0, -1.0])
        self.assertAlmostEqual(result.terms[0].effect, POST.ate_hat)
        self.assertAlmostEqual(result.terms[1].effect, PRE.ate_hat)
        self.assertEqual(
            result.terms[0].effect_interval, _scaled_interval(POST, confidence=0.95)
        )
        self.assertEqual(
            result.terms[1].effect_interval, _scaled_interval(PRE, confidence=0.95)
        )

    def test_analyze_term_intervals_take_the_side_the_combination_consumed(
        self,
    ) -> None:
        # A one-sided combination builds its bound from one bound per component,
        # flipped by the sign of the coefficient. The term rows should show that
        # bound, not a two-sided interval the Combined row never touched.
        est = LinearCombinationEstimator(
            [(2.0, POST), (-1.0, PRE)], labels=["real", "placebo"]
        )
        result = est.analyze(alpha=0.10, alternative="greater")
        self.assertEqual(
            result.terms[0].effect_interval,
            _scaled_interval(POST, confidence=0.95, alternative="greater"),
        )
        self.assertEqual(
            result.terms[1].effect_interval,
            _scaled_interval(PRE, confidence=0.95, alternative="less"),
        )
        # The flip follows the coefficient, so `less` mirrors the whole picture.
        flipped = est.analyze(alpha=0.10, alternative="less")
        self.assertEqual(
            flipped.terms[0].effect_interval,
            _scaled_interval(POST, confidence=0.95, alternative="less"),
        )
        self.assertEqual(
            flipped.terms[1].effect_interval,
            _scaled_interval(PRE, confidence=0.95, alternative="greater"),
        )

    def test_analyze_term_intervals_reproduce_the_combined_bound(self) -> None:
        # The point of showing the consumed side: the term rows now add up. A
        # reader can take the finite endpoint from each row, weight it by the
        # coefficient, and land on the Combined row's bound.
        est = LinearCombinationEstimator([(2.0, POST), (-1.0, PRE)])
        result = est.analyze(alpha=0.10, alternative="greater")
        rebuilt = sum(
            # `greater` for a positive coefficient puts the informative bound at
            # the low end, `less` for a negative one puts it at the high end.
            t.coefficient
            * (t.effect_interval[0] if t.coefficient > 0 else t.effect_interval[1])
            for t in result.terms
        )
        # `assertAlmostEqual`, not `assertEqual`: `_combine` accumulates in
        # counts and divides by `n_pairs` once at the end, while this sums rates
        # that were each divided already. That is the round-off difference
        # `_combine`'s docstring is about, and it is the reason the library does
        # it in counts -- but a reader checking the table by hand works in the
        # displayed rates, which is what this pins.
        self.assertAlmostEqual(rebuilt, result.effect_interval[0])
        self.assertEqual(result.effect_interval[1], math.inf)

    def test_analyze_zero_coefficient_term_stays_two_sided(self) -> None:
        # A zero coefficient contributes to neither combined bound, so there is
        # no side it "consumed" and nothing for a one-sided row to mean.
        est = LinearCombinationEstimator([(1.0, POST), (0.0, PRE)])
        result = est.analyze(alpha=0.10, alternative="greater")
        # k counts only the nonzero terms, so the share is the whole alpha.
        self.assertEqual(
            result.terms[0].effect_interval,
            _scaled_interval(POST, confidence=0.90, alternative="greater"),
        )
        self.assertEqual(
            result.terms[1].effect_interval,
            _scaled_interval(PRE, confidence=0.90, alternative="two-sided"),
        )

    def test_analyze_carries_options_and_gamma_star(self) -> None:
        est = LinearCombinationEstimator([(1.0, POST), (-1.0, PRE)])
        result = est.analyze(alpha=0.10, gamma=2.0, null_value=0.0)
        self.assertEqual(result.alpha, 0.10)
        self.assertEqual(result.gamma, 2.0)
        self.assertAlmostEqual(result.confidence, 0.90)
        self.assertEqual(result.target, "ATT")
        self.assertFalse(result.monotonic)
        self.assertAlmostEqual(
            result.gamma_star, est.gamma_star(alpha=0.10, null_value=0.0)
        )

    def test_analyze_effect_interval_uses_gamma(self) -> None:
        est = LinearCombinationEstimator([(1.0, POST), (-1.0, PRE)])
        result = est.analyze(gamma=3.0)
        self.assertEqual(
            result.effect_interval, est.expanded_confidence_interval(gamma=3.0)
        )

    def test_significant_property(self) -> None:
        # POST alone clearly excludes zero (a real, positive net effect).
        pos = LinearCombinationEstimator([(1.0, POST)]).analyze()
        self.assertTrue(pos.significant)
        # A combination pinned at the null by construction does not.
        flat = LinearCombinationEstimator([], affine=0.0).analyze()
        self.assertFalse(flat.significant)

    def test_gamma_star_one_when_randomized_contains_null(self) -> None:
        # PRE - PRE is identically zero; its interval always straddles the null.
        est = LinearCombinationEstimator([(1.0, PRE), (-1.0, PRE)])
        self.assertEqual(est.gamma_star(), 1.0)

    def test_gamma_star_above_one_for_clear_effect(self) -> None:
        est = LinearCombinationEstimator([(1.0, POST)])
        self.assertGreater(est.gamma_star(), 1.0)

    def test_serialize_round_trip(self) -> None:
        est = LinearCombinationEstimator(
            [(2.0, POST), (-1.0, PRE)], affine=0.05, labels=["real", "placebo"]
        )
        result = est.analyze(alpha=0.10, gamma=2.0)
        restored = LinearCombinationAnalysis.deserialize(result.serialize())
        self.assertEqual(restored, result)

    def test_serialize_handles_infinite_bounds(self) -> None:
        est = LinearCombinationEstimator([(1.0, POST), (-1.0, PRE)])
        result = est.analyze(alternative="greater")
        self.assertEqual(result.effect_interval[1], math.inf)
        restored = LinearCombinationAnalysis.deserialize(result.serialize())
        self.assertEqual(restored, result)

    def test_deserialize_rejects_both_and_neither(self) -> None:
        result = LinearCombinationEstimator([(1.0, POST)]).analyze()
        with self.assertRaises(ValueError):
            LinearCombinationAnalysis.deserialize(result.serialize(), result.to_dict())
        with self.assertRaises(ValueError):
            LinearCombinationAnalysis.deserialize()

    def test_deserialize_rejects_invalid_enum_fields(self) -> None:
        result = LinearCombinationEstimator([(1.0, POST)]).analyze()
        for field in ("alternative", "target", "method"):
            with self.subTest(field=field):
                payload = result.to_dict()
                payload[field] = "nonsense"
                with self.assertRaises(ValueError):
                    LinearCombinationAnalysis.deserialize(d=payload)

    def test_deserialize_rejects_invalid_numeric_fields(self) -> None:
        # `analyze` cannot emit any of these, so a payload carrying one is
        # corrupt -- and every one of them reads as a plausible number
        # downstream rather than raising where it is used.
        result = LinearCombinationEstimator([(1.0, POST)]).analyze()
        bad: dict[str, float] = {
            "alpha": 0.0,
            "gamma": 0.5,
            "null_value": math.nan,
            "affine": math.inf,
            "effect": math.nan,
            "gamma_star": 0.5,
        }
        for field, value in bad.items():
            with self.subTest(field=field):
                payload = result.to_dict()
                payload[field] = value
                with self.assertRaises(ValueError):
                    LinearCombinationAnalysis.deserialize(d=payload)

    def test_deserialize_rejects_nan_interval_endpoint(self) -> None:
        # An infinite endpoint is legitimate (a one-sided interval carries
        # one), but a NaN compares False against everything, so the interval
        # would neither contain nor exclude the null.
        result = LinearCombinationEstimator([(1.0, POST)]).analyze()
        for interval in ([math.nan, 1.0], [0.0, math.nan]):
            with self.subTest(interval=interval):
                payload = result.to_dict()
                payload["effect_interval"] = interval
                with self.assertRaises(ValueError):
                    LinearCombinationAnalysis.deserialize(d=payload)

    def test_deserialize_rejects_invalid_p_value_and_n_pairs(self) -> None:
        # A p-value outside [0, 1] prints as-is beside an asterisk that
        # disagrees with it; a negative `n_pairs` sign-flips every iSuccesses
        # cell, and a non-integral one is corrupt rather than roundable -- it
        # would silently floor and rescale the counts.
        result = LinearCombinationEstimator([(1.0, POST)]).analyze()
        bad: dict[str, object] = {
            "p_value": 1.5,
            "n_pairs": -1,
        }
        for field, value in bad.items():
            with self.subTest(field=field):
                payload = result.to_dict()
                payload[field] = value
                with self.assertRaises(ValueError):
                    LinearCombinationAnalysis.deserialize(d=payload)
        for n_pairs in (999.5, float(POST.n_pairs), "1000", True, False):
            # `True` and `False` are in the list because `bool` is a subclass of
            # `int`: JSON `true` would otherwise reconstruct `n_pairs == 1` and
            # quote every iSuccesses cell as a rate over a single pair.
            with self.subTest(n_pairs=n_pairs):
                payload = result.to_dict()
                payload["n_pairs"] = n_pairs
                with self.assertRaises(ValueError):
                    LinearCombinationAnalysis.deserialize(d=payload)

    def test_one_sided_pvalue_may_exceed_a_half(self) -> None:
        # The bisection brackets past 0.5, which `_validate_alpha` refuses for a
        # one-sided *interval*. Not a contradiction: those levels are the search
        # variable of an inversion, not a coverage claim, and a one-sided test
        # of a null the data point away from has p > 0.5. Truncating the bracket
        # would collapse the entire uninformative half onto exactly 0.5.
        est = LinearCombinationEstimator([(1.0, POST)])
        point = est.point_estimate()
        # POST's effect is positive, so "less" tests the wrong-side alternative.
        p = est.pvalue(null_value=0.0, alternative="less")
        self.assertGreater(point, 0.0)
        self.assertGreater(p, 0.5)
        self.assertLessEqual(p, 1.0)

    def test_deserialize_accepts_infinite_gamma_star(self) -> None:
        # `math.inf` is the documented `gamma_star` for a finding no hidden
        # bias can overturn, so it is the one numeric field exempt from a
        # finiteness check.
        result = LinearCombinationEstimator([(1.0, POST)]).analyze()
        payload = result.to_dict()
        payload["gamma_star"] = math.inf
        self.assertEqual(
            LinearCombinationAnalysis.deserialize(d=payload).gamma_star, math.inf
        )

    def test_term_to_from_dict_round_trip(self) -> None:
        term = LinearCombinationTerm(
            label="real", coefficient=-1.0, effect=0.04, effect_interval=(0.0, math.inf)
        )
        self.assertEqual(LinearCombinationTerm.from_dict(term.to_dict()), term)

    def test_str_smoke(self) -> None:
        est = LinearCombinationEstimator(
            [(1.0, POST), (-1.0, PRE)], labels=["real", "placebo"]
        )
        text = str(est.analyze())
        self.assertIn("real", text)
        self.assertIn("placebo", text)
        self.assertIn("Combined", text)
        self.assertIn("coverage of at least", text)

    def test_analyze_gamma_below_one_raises(self) -> None:
        est = LinearCombinationEstimator([(1.0, POST)])
        with self.assertRaises(ValueError):
            est.analyze(gamma=0.5)

    def test_analyze_non_finite_gamma_raises(self) -> None:
        est = LinearCombinationEstimator([(1.0, POST)])
        for gamma in (math.nan, math.inf):
            with self.subTest(gamma=gamma), self.assertRaises(ValueError):
                est.analyze(gamma=gamma)

    def test_analyze_alpha_outside_unit_interval_raises(self) -> None:
        est = LinearCombinationEstimator([(1.0, POST)])
        for alpha in (0.0, 1.0, 1.5, math.nan):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                est.analyze(alpha=alpha)

    def test_gamma_star_alpha_outside_unit_interval_raises(self) -> None:
        # `gamma_star` inverts `_combine` over gamma at a fixed alpha, and a NaN
        # alpha makes every exclusion test False, so the search would report
        # Gamma-dot = 1 ("no robustness") for any finding at all.
        est = LinearCombinationEstimator([(1.0, POST)])
        for alpha in (0.0, 1.0, 1.5, math.nan):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                est.gamma_star(alpha=alpha)

    def test_non_finite_null_value_raises(self) -> None:
        # A non-finite null is never excluded by any interval, so both searches
        # would quietly report the least significant answer they can.
        est = LinearCombinationEstimator([(1.0, POST)])
        for null_value in (math.nan, math.inf, -math.inf):
            with self.subTest(null_value=null_value):
                with self.assertRaises(ValueError):
                    est.gamma_star(null_value=null_value)
                with self.assertRaises(ValueError):
                    est.analyze(null_value=null_value)

    def test_diff_in_diff_analyze_labels(self) -> None:
        result = DiffInDiff(pre_table=PRE, post_table=POST).analyze()
        self.assertEqual([t.label for t in result.terms], ["pre", "post"])

    def test_small_constant_offset_is_not_stated_as_zero(self) -> None:
        # The footer mentions the offset only *because* it is nonzero, so a
        # fixed two-decimal percentage that rounds it to "+0.00%" makes the
        # sentence contradict its own precondition.
        result = LinearCombinationEstimator([(1.0, POST)], affine=1e-5).analyze()
        offset_note = [
            line for line in str(result).splitlines() if "constant offset" in line
        ]
        self.assertEqual(len(offset_note), 1)
        self.assertNotIn("+0.00%", offset_note[0])
        self.assertIn("+0.00100%", offset_note[0])

    def test_ordinary_constant_offset_keeps_the_fixed_format(self) -> None:
        # The escalation to significant digits must fire only for a value the
        # fixed format would flatten; an ordinary offset reads as before.
        result = LinearCombinationEstimator([(1.0, POST)], affine=0.05).analyze()
        self.assertIn("constant offset of +5.00%.", str(result))

    def test_analyze_populates_p_value_and_n_pairs(self) -> None:
        est = LinearCombinationEstimator([(1.0, POST), (-1.0, PRE)])
        result = est.analyze(alpha=0.10, gamma=2.0)
        self.assertEqual(result.n_pairs, POST.n_pairs)
        self.assertAlmostEqual(result.p_value, est.pvalue(gamma=2.0), places=6)

    def test_zero_coefficient_term_still_carries_the_pair_count(self) -> None:
        # Two different notions of "constant" meet here and must not be
        # conflated. `pvalue` / `gamma_star` answer exactly when no coefficient
        # is nonzero; `n_pairs` is 0 only when there is no term at all. A term
        # declares its table's pair set whatever its coefficient, so the offset
        # is still a per-pair rate on a real design and the iSuccesses columns
        # still have a faithful rendering.
        est = LinearCombinationEstimator([(0.0, POST)], affine=0.05)
        self.assertEqual(est.n_pairs, POST.n_pairs)
        self.assertEqual(est.pvalue(), 0.0)  # exact: nothing moves with alpha
        result = est.analyze()
        self.assertEqual(result.n_pairs, POST.n_pairs)
        self.assertIn(f"{round(0.05 * POST.n_pairs):+,}", str(result))
        # ...whereas a bare offset has no design to count over.
        bare = LinearCombinationEstimator([], affine=0.05)
        self.assertEqual(bare.n_pairs, 0)

    def test_p_value_agrees_with_significant(self) -> None:
        # The inverted-interval p-value is below alpha exactly when the interval
        # excludes the null, so it must track `significant`.
        clear = LinearCombinationEstimator([(1.0, POST)]).analyze(alpha=0.10)
        self.assertTrue(clear.significant)
        self.assertLess(clear.p_value, 0.10)
        null = LinearCombinationEstimator([(1.0, PRE), (-1.0, PRE)]).analyze(alpha=0.10)
        self.assertFalse(null.significant)
        self.assertGreaterEqual(null.p_value, 0.10)

    def test_p_value_one_when_null_always_contained(self) -> None:
        # PRE - PRE is identically zero; no level rejects the null.
        est = LinearCombinationEstimator([(1.0, PRE), (-1.0, PRE)])
        self.assertEqual(est.pvalue(), 1.0)

    def test_isuccesses_column_is_effect_times_n_pairs(self) -> None:
        result = DiffInDiff(pre_table=PRE, post_table=POST).analyze()
        n = result.n_pairs
        # Combined iSuccesses is the differenced discordant count.
        expected = round(POST.hat_a - PRE.hat_a)
        self.assertEqual(round(result.effect * n), expected)
        self.assertIn(f"{expected:+,}", str(result))

    def test_str_shows_isuccesses_pvalue_and_gamma_columns(self) -> None:
        text = str(DiffInDiff(pre_table=PRE, post_table=POST).analyze())
        self.assertIn("iSuccesses", text)
        self.assertIn("p-Value", text)
        self.assertIn("Γ•", text)
        # The reused single-table footers.
        self.assertIn("p-Value is two-sided", text)
        self.assertIn("coverage of at least 90%", text)

    def test_pvalue_gamma_below_one_raises(self) -> None:
        est = LinearCombinationEstimator([(1.0, POST)])
        with self.assertRaises(ValueError):
            est.pvalue(gamma=0.5)

    def test_pvalue_invalid_alternative_raises(self) -> None:
        est = LinearCombinationEstimator([(1.0, POST)])
        with self.assertRaises(ValueError):
            est.pvalue(alternative="sideways")

    def test_pvalue_non_finite_null_value_raises(self) -> None:
        # No interval excludes a non-finite null, so the bisection would return
        # 1.0 -- "no evidence" -- for a finding of any strength.
        est = LinearCombinationEstimator([(1.0, POST)])
        for null_value in (math.nan, math.inf, -math.inf):
            with self.subTest(null_value=null_value), self.assertRaises(ValueError):
                est.pvalue(null_value=null_value)

    def test_pvalue_one_sided_beats_two_sided(self) -> None:
        # POST's net effect is positive, so testing only that direction spends
        # the whole budget on the informative side.
        est = LinearCombinationEstimator([(1.0, POST)])
        self.assertLess(est.pvalue(alternative="greater"), est.pvalue())

    def test_pvalue_floors_instead_of_diverging(self) -> None:
        # Overwhelming evidence must bottom out at the bracket floor: pushing
        # `alpha` below it rounds the coverage to exactly 1.0, which the normal
        # inversion cannot invert.
        allsuccess = PairedOutcomeTable(s00=0, s01=0, s10=5000, s11=0)
        est = LinearCombinationEstimator([(1.0, allsuccess)])
        self.assertEqual(est.pvalue(method="normal"), 1e-12)

    def test_pvalue_constant_combination_is_exact(self) -> None:
        # No sampling uncertainty, so `_combine` ignores `alpha` and there is no
        # threshold to bisect for. The answer must be exact rather than a
        # bracket endpoint: `1e-12` would read as overwhelming evidence when in
        # fact the offset simply clears the null deterministically.
        clears = LinearCombinationEstimator([], affine=0.05)
        self.assertEqual(clears.pvalue(), 0.0)
        # `gamma_star` answers the same combination exactly; the two agree.
        self.assertEqual(clears.gamma_star(), math.inf)
        misses = LinearCombinationEstimator([], affine=0.0)
        self.assertEqual(misses.pvalue(), 1.0)

    def test_constant_combination_pvalue_cell_is_not_the_bracket_floor(self) -> None:
        text = str(LinearCombinationEstimator([], affine=0.05).analyze())
        self.assertNotIn("1e-12", text)

    def test_combined_bound_is_the_exact_rate_of_the_composed_counts(self) -> None:
        # Component endpoints are whole iSuccesses over a shared `n_pairs`, so a
        # combination of them that lands on a whole count has an exact double --
        # the one a caller writing that rate gets. Composing rates instead of
        # counts misses it by an ulp: at this level the diff-in-diff lower bound
        # is 34 - 14 = 20 successes, but `34 / 1000 - 14 / 1000` is
        # 0.020000000000000004. Equality, not `assertAlmostEqual`: an ulp is the
        # whole point.
        did = DiffInDiff(pre_table=PRE, post_table=POST)
        alpha = 0.7143
        share = alpha / 2
        post_lo, _ = attributable_effect_interval(
            POST, confidence=1.0 - share, alternative="greater", method="exact"
        )
        _, pre_hi = attributable_effect_interval(
            PRE, confidence=1.0 - share, alternative="less", method="exact"
        )
        self.assertEqual((post_lo, pre_hi), (34, 14))
        # `_combine` rather than `confidence_interval`: a one-sided interval is
        # only offered below alpha = 0.5, but `pvalue` bisects the predicate over
        # the whole unit interval, so this is a level it really does visit.
        lb, _ = _combine_greater(did, alpha)
        self.assertEqual(lb, (post_lo - pre_hi) / POST.n_pairs)

    def test_pvalue_does_not_reject_a_null_the_bound_only_meets(self) -> None:
        # `_excludes` compares strictly, so a bound sitting exactly on the null
        # is not significant. An ulp of round-off in the bound flips that, and
        # anti-conservatively: the level at which the bound first *meets* the
        # null would be reported as the level at which it clears it.
        did = DiffInDiff(pre_table=PRE, post_table=POST)
        alpha = 0.7143
        null_value = 20 / POST.n_pairs
        lb, _ = _combine_greater(did, alpha)
        self.assertEqual(lb, null_value)
        p = did.pvalue(null_value=null_value, alternative="greater", method="exact")
        self.assertGreaterEqual(p, alpha)

    def test_all_zero_coefficient_terms_take_the_constant_path(self) -> None:
        # `terms` is non-empty but contributes nothing, which is the same
        # deterministic case -- the emptiness test must be on the coefficients.
        est = LinearCombinationEstimator([(0.0, POST)], affine=0.05)
        self.assertEqual(est.pvalue(), 0.0)

    def test_footer_integrality_check_is_absolute_not_relative(self) -> None:
        # A relative tolerance would grow with the count: at these magnitudes a
        # genuinely fractional null must not be restated as a round integer.
        big = PairedOutcomeTable(s00=0, s01=1_000_000_000, s10=1_000_000_000, s11=0)
        est = LinearCombinationEstimator([(1.0, big)])
        # null_value * n_pairs = 0.5 * 2e9 + 1e9 -> a half-count off an integer.
        text = str(est.analyze(null_value=0.5 + 0.25 / big.n_pairs))
        self.assertNotIn("iSuccesses =", text)
        self.assertIn("null hypothesis that ATT =", text)

    def test_footer_states_a_whole_count_despite_product_round_off(self) -> None:
        # `null_value` is the nearest double to the rate the caller meant, so a
        # null that is a whole number of successes by construction need not
        # multiply out to one: 0.28 of 2,678,547,400 pairs is exactly
        # 749,993,272, but the product computes to ...272.0000001, one ulp high.
        # A fixed 1e-9 window is a hundred times too narrow to see that, and the
        # footer would fall back to scaled units with a count in hand.
        big = PairedOutcomeTable(s00=0, s01=1_339_273_700, s10=1_339_273_700, s11=0)
        est = LinearCombinationEstimator([(1.0, big)])
        self.assertNotEqual(0.28 * big.n_pairs, round(0.28 * big.n_pairs))
        text = str(est.analyze(null_value=0.28))
        self.assertIn("null hypothesis that iSuccesses = 749,993,272", text)

    def test_footer_integrality_tolerance_is_capped(self) -> None:
        # Four ulps is half a success once the count passes 2**49 (~5.6e14), at
        # which point "is this a whole number?" would answer yes for anything.
        # The cap holds the window below that: a quarter of a success off an
        # integer must still be phrased in scaled units.
        #
        # Built by hand rather than estimated from a table of this size: the
        # footer is pure rendering, and an interval inversion over 2e15 pairs
        # costs minutes to reach it.
        n_pairs = 2_000_000_000_000_000
        null_value = 0.5 + 0.25 / n_pairs
        # Neither the sum nor the product is exact -- the sum lands on
        # `0.5 + 2**-53` and the product onto the 0.125 grid at 1e15 -- so pin
        # the offset actually constructed. A change in how the literals round
        # should fail here, not silently move the test to a different distance.
        count = null_value * n_pairs
        self.assertEqual(count - round(count), 0.25)
        base = LinearCombinationEstimator([(1.0, POST)]).analyze()
        text = str(replace(base, n_pairs=n_pairs, null_value=null_value))
        self.assertNotIn("iSuccesses =", text)
        self.assertIn("null hypothesis that ATT =", text)

    def test_one_sided_str_footer_wording(self) -> None:
        text = str(
            LinearCombinationEstimator([(1.0, POST)]).analyze(alternative="greater")
        )
        self.assertIn("p-Value is one-sided", text)
        self.assertIn("iSuccesses ≤ 0", text)

    def test_constant_combination_leaves_count_cells_blank(self) -> None:
        # No pairs to count over, and `inf * 0` is NaN rather than an infinite
        # count, so the iSuccesses cells must stay empty.
        text = str(
            LinearCombinationEstimator([], affine=0.05).analyze(alternative="greater")
        )
        self.assertNotIn("nan", text.lower())
        # The column stays in the header -- it is the *cells* that go blank --
        # so asserting on the header alone would pass even if they were filled.
        self.assertIn("iSuccesses", text)
        combined = next(
            line for line in text.splitlines() if line.startswith("| Combined")
        )
        cells = [c.strip() for c in combined.strip("|").split("|")]
        # Term, Coef, <target>, CI, iSuccesses, iSuccesses CI, p, Γ•
        self.assertEqual(cells[4], "")
        self.assertEqual(cells[5], "")
        # The scaled columns are still populated; only the counts are dropped.
        self.assertEqual(cells[2], "+5.00%")

    def test_constant_combination_footer_states_the_null_in_scaled_units(self) -> None:
        # With no pairs the count columns are blank, so the footer must not
        # phrase the null in counts -- least of all as a flat `iSuccesses = 0`,
        # which would misreport a nonzero null.
        text = str(
            LinearCombinationEstimator([], affine=0.05).analyze(
                null_value=0.02, target="ATU"
            )
        )
        self.assertIn("null hypothesis that ATU = +2.00%", text)
        self.assertNotIn("iSuccesses = 0", text)

    def test_footer_states_the_null_in_counts_when_there_are_pairs(self) -> None:
        # A whole count, grouped and never in scientific notation, matching the
        # wording `PairedOutcomeTable.analyze` uses for the same note.
        text = str(LinearCombinationEstimator([(1.0, POST)]).analyze(null_value=0.02))
        self.assertIn("null hypothesis that iSuccesses = 20", text)
        big = PairedOutcomeTable(s00=0, s01=1_000_000, s10=2_000_000, s11=0)
        text = str(LinearCombinationEstimator([(1.0, big)]).analyze(null_value=0.5))
        self.assertIn("null hypothesis that iSuccesses = 1,500,000", text)

    def test_footer_falls_back_to_scaled_units_for_a_fractional_null(self) -> None:
        # 0.001 * 1000 pairs = 1 iSuccess, but 0.0015 is a null of one and a
        # half, which no integer count column can state faithfully. Say it in
        # scaled units rather than rounding it into a different hypothesis.
        text = str(LinearCombinationEstimator([(1.0, POST)]).analyze(null_value=0.0015))
        self.assertIn("null hypothesis that ATT = +0.15%", text)
        self.assertNotIn("iSuccesses = ", text)

    def test_footer_renders_a_non_finite_null_without_raising(self) -> None:
        # `analyze` rejects one, but the result is a plain frozen dataclass that
        # can also be built by hand or reconstructed by `deserialize`. The count
        # path calls `round()`, which raises on `inf` and `nan`, so rendering a
        # hand-built analysis must fall back to scaled units rather than blow up
        # inside `__str__`.
        base = LinearCombinationEstimator([(1.0, POST)]).analyze()
        for null_value in (math.inf, -math.inf, math.nan):
            with self.subTest(null_value=null_value):
                text = str(replace(base, null_value=null_value))
                self.assertIn("null hypothesis that ATT ", text)
                self.assertNotIn("iSuccesses = ", text)

    def test_pvalue_floor_renders_as_a_bound_not_a_value(self) -> None:
        # `pvalue` returns the bracket floor when even its widest interval
        # excludes the null: the search saturated, it did not resolve. Printed
        # bare, that is indistinguishable from a threshold the bisection found.
        base = LinearCombinationEstimator([(1.0, POST)]).analyze()
        self.assertIn("<1e-12", str(replace(base, p_value=_PVALUE_EPS)))

    def test_resolved_and_exact_pvalues_render_without_the_bound(self) -> None:
        # The bound must be reserved for the floor. A p-value the bisection
        # resolved, and the exact 0.0 an all-constant combination genuinely
        # has, are both quoted as themselves.
        base = LinearCombinationEstimator([(1.0, POST)]).analyze()
        for p_value in (0.25, 0.0):
            with self.subTest(p_value=p_value):
                self.assertNotIn("<", str(replace(base, p_value=p_value)))

    def test_pvalue_floor_is_reachable(self) -> None:
        # Guards the rendering above against becoming dead code: an
        # overwhelming table saturates the bracket rather than resolving.
        overwhelming = PairedOutcomeTable(s00=100, s01=1, s10=300, s11=100)
        self.assertEqual(
            LinearCombinationEstimator([(1.0, overwhelming)]).pvalue(), _PVALUE_EPS
        )

    def test_small_null_is_not_stated_as_zero_in_the_footer(self) -> None:
        # A null too small to be a whole number of successes falls back to
        # scaled units, where a fixed two-decimal percentage would restate it
        # as +0.00% -- the footer would then name a null that is not the one
        # the p-value was computed against.
        result = LinearCombinationEstimator([(1.0, POST)]).analyze(null_value=1e-5)
        text = str(result)
        self.assertIn("ATT = +0.00100%", text)
        self.assertNotIn("ATT = +0.00%", text)

    def test_deserialize_rejects_n_pairs_inconsistent_with_terms(self) -> None:
        # `analyze` cannot emit either shape: the constructor rejects a
        # component table with zero pairs, so a positive count and a non-empty
        # `terms` stand or fall together. Neither fails loudly on its own --
        # they render as a table with blank iSuccesses cells or with cells
        # scaled against a count no term supports.
        with_terms = LinearCombinationEstimator([(1.0, POST)]).analyze()
        without_terms = LinearCombinationEstimator([], affine=0.5).analyze()
        for name, result, n_pairs in (
            ("zero count with terms", with_terms, 0),
            ("positive count without terms", without_terms, 1000),
        ):
            with self.subTest(name):
                payload = result.to_dict()
                payload["n_pairs"] = n_pairs
                with self.assertRaises(ValueError):
                    LinearCombinationAnalysis.deserialize(d=payload)
