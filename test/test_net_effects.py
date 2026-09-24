#!/usr/bin/env python3

# pyre-strict
"""Test matched-pair net-effects inference."""

import math
import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from pair_match.match_result import (
    MatchResult,
)
from pair_match.net_effects import (
    EffectSize,
    PairedOutcomeAnalysis,
    PairedOutcomeAnalysisOptions,
    PairedOutcomeTable,
    _combined_tail_max,
    _normal_worst_case_interval,
    _p_greater,
    _p_less,
    _resolve_method,
    att_confidence_set,
    attributable_effect_interval,
    design_sensitivity_binary,
    mcnemar_ate_interval,
    sensitivity_value,
    worst_case_pvalue,
)

# The running example from the paper (Wilson, 2026, sections 3-6): "Randomization
# Inference for Matched Pairs with Binary Outcomes," arXiv:2609.03227,
# https://arxiv.org/abs/2609.03227.
S00, S01, S10, S11 = 800, 30, 70, 100


def _paper_table() -> PairedOutcomeTable:
    return PairedOutcomeTable(s00=S00, s01=S01, s10=S10, s11=S11)


class PairedOutcomeTableTest(unittest.TestCase):
    """Tests for ``PairedOutcomeTable``."""

    def test_summary_quantities(self) -> None:
        table = _paper_table()
        self.assertEqual(table.n_pairs, 1000)
        self.assertEqual(table.hat_a, 40)
        self.assertAlmostEqual(table.ate_hat, 0.040)

    def test_from_outcomes(self) -> None:
        treated = [1, 1, 0, 0, 1]
        control = [0, 1, 1, 0, 0]
        table = PairedOutcomeTable.from_outcomes(treated, control)
        self.assertEqual((table.s00, table.s01, table.s10, table.s11), (1, 1, 2, 1))

    def test_from_outcomes_rejects_non_binary(self) -> None:
        with self.assertRaises(ValueError):
            PairedOutcomeTable.from_outcomes([0, 2], [0, 1])

    def test_from_outcomes_rejects_shape_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            PairedOutcomeTable.from_outcomes([0, 1, 1], [0, 1])

    def test_from_match_result(self) -> None:
        result = MatchResult(
            treated_index=["t0", "t1"],
            control_index=["c0", "c1"],
            distances=np.zeros(2),
        )
        df_treated = pd.DataFrame({"y": [1, 0]}, index=pd.Index(["t0", "t1"]))
        df_control = pd.DataFrame({"y": [0, 1]}, index=pd.Index(["c0", "c1"]))
        table = PairedOutcomeTable.from_match_result(
            result, "y", df_treated=df_treated, df_control=df_control
        )
        # t0/c0 -> (1,0) = S10; t1/c1 -> (0,1) = S01.
        self.assertEqual((table.s10, table.s01), (1, 1))

    def test_serialize_round_trip(self) -> None:
        table = PairedOutcomeTable(s00=62, s01=57, s10=79, s11=87, spend=1000.0)
        restored = PairedOutcomeTable.deserialize(s=table.serialize())
        self.assertEqual(restored, table)
        self.assertEqual(restored.spend, 1000.0)

    def test_serialize_round_trip_without_spend(self) -> None:
        table = PairedOutcomeTable(s00=1, s01=2, s10=3, s11=4)
        restored = PairedOutcomeTable.deserialize(s=table.serialize())
        self.assertEqual(restored, table)
        self.assertIsNone(restored.spend)

    def test_deserialize_tolerates_missing_spend(self) -> None:
        # Older payloads predate the `spend` field.
        restored = PairedOutcomeTable.deserialize(
            d={"s00": 1, "s01": 2, "s10": 3, "s11": 4}
        )
        self.assertIsNone(restored.spend)

    def test_deserialize_requires_exactly_one_arg(self) -> None:
        with self.assertRaises(ValueError):
            PairedOutcomeTable.deserialize()
        with self.assertRaises(ValueError):
            PairedOutcomeTable.deserialize(s="{}", d={})

    def test_success_rates(self) -> None:
        table = PairedOutcomeTable(s00=62, s01=57, s10=79, s11=87)
        self.assertAlmostEqual(table.treated_success_rate, 166 / 285)
        self.assertAlmostEqual(table.control_success_rate, 144 / 285)
        # The success-rate gap is exactly the ATE point estimate.
        self.assertAlmostEqual(
            table.treated_success_rate - table.control_success_rate, table.ate_hat
        )

    def test_str_contingency_table(self) -> None:
        table = PairedOutcomeTable(s00=62, s01=57, s10=79, s11=87)
        expected = "\n".join(
            [
                "             Treated",
                "            0    1    *",
                "        0  62   79  141",
                "Control 1  57   87  144  50.5% control",
                "        * 119  166  285",
                "              58.2% treated",
            ]
        )
        self.assertEqual(str(table), expected)


class PairedOutcomeTableAnalysisTest(unittest.TestCase):
    """Tests for ``PairedOutcomeTable.analyze`` and its display."""

    def _table(self) -> PairedOutcomeTable:
        return PairedOutcomeTable(s00=62, s01=57, s10=79, s11=87)

    def test_analyze_matches_underlying_estimators(self) -> None:
        table = self._table()
        result = table.analyze(alpha=0.10, gamma=1.0)
        self.assertAlmostEqual(result.effect, table.ate_hat)
        self.assertEqual(result.attributable, table.hat_a)
        # alpha=0.10 -> the confidence sets have 1 - alpha = 0.90 coverage.
        self.assertEqual(
            result.effect_interval,
            att_confidence_set(table, confidence=0.90).effect_interval,
        )
        self.assertEqual(
            result.attributable_interval,
            attributable_effect_interval(table, target="ATT", confidence=0.90),
        )
        self.assertEqual(
            result.gamma_star, sensitivity_value(table, alpha=0.10, target="ATT")
        )
        self.assertEqual(
            result.p_value,
            worst_case_pvalue(table, target="ATT", null_value=0, gamma=1.0),
        )

    def test_att_interval_is_attributable_over_n(self) -> None:
        # The scaled interval is exactly the target attributable-effect
        # (iSuccesses) confidence set divided by n_pairs -- for the default
        # target A_1 this is the ATT, not the Rigdon-Hudgens (A_1 + A_0) / (2 S)
        # ATE combination.
        table = self._table()
        n = table.n_pairs
        result = table.analyze(alpha=0.10)
        self.assertEqual(
            result.effect_interval,
            (
                result.attributable_interval[0] / n,
                result.attributable_interval[1] / n,
            ),
        )
        # One-sided "greater": the identity holds and one-sidedness is
        # preserved (+inf / n = +inf). null_value shifts only the p-value, not
        # the interval, so any value works here.
        one_sided = table.analyze(null_value=-20, alternative="greater")
        self.assertEqual(one_sided.effect_interval[1], math.inf)
        self.assertEqual(
            one_sided.effect_interval[0], one_sided.attributable_interval[0] / n
        )
        # The consistency this change buys: when the attributable (iSuccesses)
        # one-sided lower bound clears 0, so does the ATT lower bound. Under the
        # old Rigdon-Hudgens ATT the Bonferroni widening could straddle 0 while
        # the iSuccesses column did not.
        self.assertGreater(one_sided.attributable_interval[0], 0)
        self.assertGreater(one_sided.effect_interval[0], 0.0)

    def test_att_interval_tracks_target(self) -> None:
        # The scaled effect follows target: A_1 / n_pairs (the ATT) for
        # target="ATT", A_0 / n_pairs (the ATU) for target="ATU". Each equals its
        # own iSuccesses (attributable) confidence set divided by n_pairs, so
        # the scaled column and the iSuccesses column always describe the same
        # effect.
        table = self._table()
        n = table.n_pairs
        a1 = table.analyze(target="ATT")
        a0 = table.analyze(target="ATU")
        self.assertEqual(
            a1.effect_interval,
            (a1.attributable_interval[0] / n, a1.attributable_interval[1] / n),
        )
        self.assertEqual(
            a0.effect_interval,
            (a0.attributable_interval[0] / n, a0.attributable_interval[1] / n),
        )
        # A_0 and A_1 have different confidence sets, so the scaled column now
        # differs by target too (it no longer hardcodes A_1).
        self.assertNotEqual(a1.effect_interval, a0.effect_interval)
        # The A_1 scaled interval still matches confidence_interval (A_1-only).
        self.assertEqual(a1.effect_interval, table.confidence_interval())

    def test_scaled_interval_brackets_point_estimate(self) -> None:
        # The displayed point estimate is the target-invariant McNemar pivot
        # ate_hat = hat_a / n_pairs; switching target only changes the ceilings
        # (how far the confidence set can extend), never its center, which is
        # always hat_a. So the scaled interval brackets the point estimate for
        # every target -- the ATU point can never sit outside its own CI.
        table = self._table()
        for target in ("ATT", "ATU"):
            with self.subTest(target=target):
                result = table.analyze(target=target)
                self.assertEqual(result.effect, table.ate_hat)
                lo, hi = result.effect_interval
                self.assertLessEqual(lo, result.effect)
                self.assertLessEqual(result.effect, hi)

    def test_scaled_header_tracks_target(self) -> None:
        # The scaled column is labeled ATT for A_1 and ATU for A_0.
        table = self._table()
        a1_text = str(table.analyze(target="ATT"))
        a0_text = str(table.analyze(target="ATU"))
        self.assertIn("ATT", a1_text)
        self.assertNotIn("ATU", a1_text)
        self.assertIn("ATU", a0_text)
        self.assertNotIn("ATT", a0_text)

    def test_target_round_trips(self) -> None:
        # `target` is carried on the result and survives serialization; a legacy
        # payload without it defaults to ATT.
        table = self._table()
        result = table.analyze(target="ATU")
        self.assertEqual(result.target, "ATU")
        revived = PairedOutcomeAnalysis.deserialize(result.serialize())
        self.assertEqual(revived.target, "ATU")
        d = result.to_dict()
        del d["target"]
        self.assertEqual(PairedOutcomeAnalysis.deserialize(d=d).target, "ATT")

    def test_deserialize_rejects_unknown_target(self) -> None:
        # A payload persisted before the A1/A0 -> ATT/ATU rename carries an
        # unrecognized target; there is no back-compat mapping, so deserialize
        # fails loudly rather than reviving it and silently rendering the stale
        # target under the neutral "Effect" label.
        d = self._table().analyze(target="ATU").to_dict()
        d["target"] = "A0"
        with self.assertRaises(ValueError):
            PairedOutcomeAnalysis.deserialize(d=d)

    def test_analyze_default_method_is_auto(self) -> None:
        # analyze() picks the inversion method from the sample size by default,
        # matching expanded_confidence_interval / gamma_star.
        self.assertEqual(self._table().analyze().method, "auto")

    def test_analyze_forwards_method_to_confidence_sets(self) -> None:
        # `method` is threaded only to the confidence-set inversion: the
        # attributable/scaled sets match a direct call at that method, while the
        # p-value and Γ• stay exact regardless.
        table = _scaled(_paper_table(), 4)
        n = table.n_pairs
        exact_p = worst_case_pvalue(table, target="ATT", null_value=0, gamma=1.0)
        for method in ("exact", "normal", "auto"):
            result = table.analyze(alpha=0.10, target="ATT", method=method)
            direct = attributable_effect_interval(
                table, target="ATT", confidence=0.90, method=method
            )
            self.assertEqual(result.attributable_interval, direct, msg=method)
            self.assertEqual(
                result.effect_interval, (direct[0] / n, direct[1] / n), msg=method
            )
            self.assertEqual(result.method, method)
            # p-value is always exact -- it needs no inversion, so `method` never
            # touches it.
            self.assertEqual(result.p_value, exact_p, msg=method)

    def test_analyze_method_round_trips(self) -> None:
        result = self._table().analyze(method="normal")
        revived = PairedOutcomeAnalysis.deserialize(result.serialize())
        self.assertEqual(revived.method, "normal")

    def test_analyze_rejects_invalid_method(self) -> None:
        with self.assertRaises(ValueError):
            self._table().analyze(method="bogus")

    def test_analyze_str_shows_expected_columns(self) -> None:
        text = str(self._table().analyze())
        for token in ("ATT", "Conf Int**", "iSuccesses", "p-Value*", "Γ•"):
            self.assertIn(token, text)
        # Point estimates: ATT +7.72% (22/285) and attributable +22.
        self.assertIn("+7.72%", text)
        self.assertIn("+22", text)
        self.assertIn(", ", text)  # CI separator
        self.assertIn("|", text)  # orgtbl

    def test_analyze_str_exact(self) -> None:
        self.maxDiff = None
        text = str(self._table().analyze())
        expected = "\n".join(
            [
                "|    ATT |   Conf Int**    |   iSuccesses |  Conf Int**  |   p-Value* |   Γ• |",
                "|--------+-----------------+--------------+--------------+------------+------|",
                "| +7.72% | -1.75%, +17.19% |          +22 |   -5, +49    |      0.193 |    1 |",
                "*  An asterisk in the p-Value column indicates statistical significance at",
                "   level 0.10, provided Γ≤1.",
                "   p-Value is two-sided against the null hypothesis that iSuccesses = 0.",
                "** Confidence intervals have coverage of at least 90%, provided Γ≤1.",
            ]
        )
        self.assertEqual(text, expected)

    def test_options_select_columns(self) -> None:
        options = PairedOutcomeAnalysisOptions(
            effect_size_columns=("effect",),
            include_p_value=False,
            include_sensitivity=False,
        )
        text = str(self._table().analyze(options=options))
        self.assertIn("ATT", text)
        self.assertNotIn("iSuccess", text)
        self.assertNotIn("pval", text)
        self.assertNotIn("Γ•", text)

    def test_suppressed_p_value_note_does_not_read_the_alternative(self) -> None:
        # The level / sidedness / relation wording belongs to the p-value note
        # alone. An unrecognized `alternative` reaches a hand-built or
        # deserialized analysis, and suppressing the note must suppress the
        # demand with it -- rendering is not where that should surface.
        options = PairedOutcomeAnalysisOptions(include_p_value=False)
        analysis = replace(
            self._table().analyze(options=options), alternative="sideways"
        )
        self.assertNotIn("one-sided", str(analysis))
        # With the note shown, the bad field is named rather than raising a
        # bare KeyError from inside __str__.
        shown = replace(self._table().analyze(), alternative="sideways")
        with self.assertRaises(ValueError):
            str(shown)

    def test_options_header_and_format_overrides(self) -> None:
        options = PairedOutcomeAnalysisOptions(
            header_overrides={"effect": "Avg TE"},
            format_overrides={EffectSize.EFFECT: "{:+.1%}"},
        )
        text = str(self._table().analyze(options=options))
        self.assertIn("Avg TE", text)
        self.assertIn("+7.7%", text)
        self.assertNotIn("+7.72%", text)

    def test_pval_header_override(self) -> None:
        options = PairedOutcomeAnalysisOptions(header_overrides={"pval": "MyPval"})
        text = str(self._table().analyze(options=options))
        self.assertIn("MyPval", text)
        self.assertNotIn("p-Value*", text)

    def test_effect_size_get(self) -> None:
        self.assertIs(EffectSize.get("effect"), EffectSize.EFFECT)
        self.assertIs(EffectSize.get("EFFECT"), EffectSize.EFFECT)
        self.assertIs(EffectSize.get(EffectSize.ISUCCESSES), EffectSize.ISUCCESSES)
        with self.assertRaises(ValueError):
            EffectSize.get("bogus")

    def test_effect_size_get_accepts_legacy_att_alias(self) -> None:
        # The scaled column used to be named "att"; that string still resolves to
        # EFFECT (case-insensitively) so pre-rename options/payloads keep working.
        self.assertIs(EffectSize.get("att"), EffectSize.EFFECT)
        self.assertIs(EffectSize.get("ATT"), EffectSize.EFFECT)

    def test_options_accept_legacy_att_column_alias(self) -> None:
        # effect_size_columns given the legacy "att" string still selects the
        # scaled column (normalized to EFFECT in Options.__post_init__).
        options = PairedOutcomeAnalysisOptions(
            effect_size_columns=("att",),
            include_p_value=False,
            include_sensitivity=False,
        )
        text = str(self._table().analyze(options=options))
        self.assertIn("ATT", text)
        self.assertNotIn("iSuccess", text)

    def test_footer_reports_coverage_and_gamma(self) -> None:
        # alpha=0.10 -> 90% coverage; the entertained gamma is shown too.
        text = str(self._table().analyze(alpha=0.10, gamma=1.0))
        self.assertIn("90%", text)
        self.assertIn("Γ≤1", text)
        stressed = str(self._table().analyze(alpha=0.10, gamma=2.0))
        self.assertIn("Γ≤2", stressed)

    def test_pvalue_significance_flag(self) -> None:
        # Two-sided p ~ 0.19: flagged at alpha=0.20, not at alpha=0.10. Check the
        # data row (line 2), not the whole text -- the footer always has a "*".
        flagged = str(self._table().analyze(alpha=0.20)).splitlines()[2]
        self.assertIn("*", flagged)
        not_flagged = str(self._table().analyze(alpha=0.10)).splitlines()[2]
        self.assertNotIn("*", not_flagged)

    def test_from_outcomes_carries_spend(self) -> None:
        table = PairedOutcomeTable.from_outcomes([1, 0], [0, 1], spend=500.0)
        self.assertEqual(table.spend, 500.0)

    def test_cost_per_isuccess_column(self) -> None:
        # Paper table: hat_a = 40; with spend 1000 -> $25.00 per iSuccess, and a
        # strictly-positive A1 interval so the cost CI is well defined.
        table = PairedOutcomeTable(s00=800, s01=30, s10=70, s11=100, spend=1000.0)
        options = PairedOutcomeAnalysisOptions(
            effect_size_columns=("isuccesses", "cost_per_isuccess"),
            include_p_value=False,
            include_sensitivity=False,
        )
        text = str(table.analyze(options=options))
        self.assertIn("Cost/iSuccess", text)
        self.assertIn("$25.00", text)

    def test_cost_per_isuccess_requires_spend(self) -> None:
        options = PairedOutcomeAnalysisOptions(
            effect_size_columns=("cost_per_isuccess",)
        )
        with self.assertRaises(ValueError):
            self._table().analyze(options=options)

    def test_cost_per_isuccess_unbounded_when_straddling_zero(self) -> None:
        # The default table's iSuccess CI (-5, +49) straddles 0, so cost/iSuccess
        # is bounded below (spend / 49) but unbounded above (spend / a -> inf as
        # the lower success bound is non-positive).
        table = PairedOutcomeTable(s00=62, s01=57, s10=79, s11=87, spend=1000.0)
        options = PairedOutcomeAnalysisOptions(
            effect_size_columns=("cost_per_isuccess",),
            include_p_value=False,
            include_sensitivity=False,
        )
        text = str(table.analyze(options=options))
        self.assertIn("$45.45", text)  # point: 1000 / 22
        self.assertIn("$20.41", text)  # lower bound: 1000 / 49
        self.assertIn("∞", text)  # upper bound is unbounded

    def test_worst_case_pvalue_no_effect_is_one(self) -> None:
        # Equal discordancies (S10 == S01) => no estimated effect => nothing to
        # reject, so the worst-case p-value is exactly 1.0.
        flat = PairedOutcomeTable(s00=100, s01=50, s10=50, s11=100)
        self.assertEqual(worst_case_pvalue(flat, target="ATT", null_value=0), 1.0)

    def test_monotonic_narrows_everything(self) -> None:
        table = _paper_table()  # strong effect: clear narrowing under monotonicity
        base = table.analyze(alpha=0.10)
        mono = table.analyze(alpha=0.10, monotonic=True)
        self.assertFalse(base.monotonic)
        self.assertTrue(mono.monotonic)
        # Attributable and ATE confidence sets are strictly narrower.
        self.assertGreaterEqual(
            mono.attributable_interval[0], base.attributable_interval[0]
        )
        self.assertLessEqual(
            mono.attributable_interval[1], base.attributable_interval[1]
        )
        base_width = base.attributable_interval[1] - base.attributable_interval[0]
        mono_width = mono.attributable_interval[1] - mono.attributable_interval[0]
        self.assertLess(mono_width, base_width)
        self.assertGreaterEqual(mono.effect_interval[0], base.effect_interval[0])
        self.assertLessEqual(mono.effect_interval[1], base.effect_interval[1])
        # Sharper p-value, larger sensitivity value.
        self.assertLessEqual(mono.p_value, base.p_value)
        self.assertGreaterEqual(mono.gamma_star, base.gamma_star)

    def test_monotonic_free_functions_agree(self) -> None:
        table = _paper_table()
        base_ci = attributable_effect_interval(table, target="ATT", confidence=0.90)
        mono_ci = attributable_effect_interval(
            table, target="ATT", confidence=0.90, monotonic=True
        )
        self.assertGreaterEqual(mono_ci[0], base_ci[0])
        self.assertLessEqual(mono_ci[1], base_ci[1])
        self.assertLessEqual(
            worst_case_pvalue(table, monotonic=True), worst_case_pvalue(table)
        )
        self.assertGreaterEqual(
            sensitivity_value(table, alpha=0.05, monotonic=True),
            sensitivity_value(table, alpha=0.05),
        )

    def test_monotonic_footer_and_round_trip(self) -> None:
        result = _paper_table().analyze(monotonic=True)
        self.assertIn("assuming monotonicity", str(result))
        restored = PairedOutcomeAnalysis.deserialize(s=result.serialize())
        self.assertTrue(restored.monotonic)


class PValueTest(unittest.TestCase):
    """Tests for the binomial tail helpers."""

    def test_p_greater_basic(self) -> None:
        self.assertAlmostEqual(_p_greater(1, 0, 0.5), 0.5)
        self.assertAlmostEqual(_p_greater(2, 0, 0.5), 0.25)
        self.assertAlmostEqual(_p_greater(0, 5, 0.5), 1.0)

    def test_p_less_basic(self) -> None:
        self.assertAlmostEqual(_p_less(0, 2, 0.5), 0.25)
        self.assertAlmostEqual(_p_less(2, 0, 0.5), 1.0)

    def test_worst_case_corner_matches_paper(self) -> None:
        # Paper: p_>(170, 130) ~ 0.0121, the worst-case corner for A_1 <= 0.
        self.assertTrue(0.011 < _p_greater(170, 130, 0.5) < 0.013)


class AttributableEffectIntervalTest(unittest.TestCase):
    """Tests for ``attributable_effect_interval``."""

    def test_a1_90_percent_set_matches_paper(self) -> None:
        # Paper: 90% two-sided prediction set for A_1 is [12, 66].
        interval = attributable_effect_interval(
            _paper_table(), target="ATT", confidence=0.90
        )
        self.assertEqual(interval, (12, 66))

    def test_invalid_target_raises(self) -> None:
        with self.assertRaises(ValueError):
            attributable_effect_interval(_paper_table(), target="bogus")

    def test_sensitivity_widens_interval(self) -> None:
        base = attributable_effect_interval(_paper_table(), gamma=1.0)
        stressed = attributable_effect_interval(_paper_table(), gamma=1.5)
        self.assertLessEqual(stressed[0], base[0])
        self.assertGreaterEqual(stressed[1], base[1])


class ATTConfidenceSetTest(unittest.TestCase):
    """Tests for ``att_confidence_set``."""

    def test_matches_paper(self) -> None:
        # The ATT is now the treated-side attributable effect A_1 alone, scaled
        # by 1 / n_pairs -- not the Rigdon-Hudgens (A_1 + A_0) / (2 S) ATE
        # combination. Paper: 90% two-sided A_1 set is [12, 66], so with 1000
        # pairs the ATT set is [0.012, 0.066].
        table = _paper_table()
        result = att_confidence_set(table, confidence=0.90)
        attributable = attributable_effect_interval(
            table, target="ATT", confidence=0.90
        )
        n = table.n_pairs
        self.assertEqual(
            result.effect_interval, (attributable[0] / n, attributable[1] / n)
        )
        self.assertAlmostEqual(result.effect_interval[0], 0.012)
        self.assertAlmostEqual(result.effect_interval[1], 0.066)

    def test_point_estimate(self) -> None:
        result = att_confidence_set(_paper_table())
        self.assertAlmostEqual(result.point_estimate, 0.040)

    def test_interval_brackets_point_estimate(self) -> None:
        result = att_confidence_set(_paper_table())
        lo, hi = result.effect_interval
        self.assertLess(lo, result.point_estimate)
        self.assertGreater(hi, result.point_estimate)

    def test_serialize_round_trip(self) -> None:
        result = att_confidence_set(_paper_table(), confidence=0.95, gamma=1.2)
        restored = PairedOutcomeAnalysis.deserialize(s=result.serialize())
        self.assertAlmostEqual(restored.effect_interval[0], result.effect_interval[0])
        self.assertAlmostEqual(restored.effect_interval[1], result.effect_interval[1])
        self.assertEqual(restored.attributable_interval, result.attributable_interval)
        self.assertAlmostEqual(restored.gamma, 1.2)
        self.assertEqual(restored.table, result.table)

    def test_serialize_round_trips_spend(self) -> None:
        table = PairedOutcomeTable(s00=800, s01=30, s10=70, s11=100, spend=5000.0)
        result = att_confidence_set(table, confidence=0.90)
        restored = PairedOutcomeAnalysis.deserialize(s=result.serialize())
        self.assertEqual(restored.table.spend, 5000.0)

    def test_deserialize_tolerates_missing_monotonic(self) -> None:
        # Payloads serialized before the `monotonic` field must still load,
        # defaulting to False rather than raising KeyError.
        result = att_confidence_set(_paper_table(), confidence=0.90)
        payload = result.to_dict()
        del payload["monotonic"]
        restored = PairedOutcomeAnalysis.deserialize(d=payload)
        self.assertFalse(restored.monotonic)

    def test_deserialize_tolerates_legacy_att_keys(self) -> None:
        # Payloads serialized before the att -> effect rename carry "att" /
        # "att_interval"; deserialize must still read them into effect /
        # effect_interval.
        result = att_confidence_set(_paper_table(), confidence=0.90)
        payload = result.to_dict()
        payload["att"] = payload.pop("effect")
        payload["att_interval"] = payload.pop("effect_interval")
        restored = PairedOutcomeAnalysis.deserialize(d=payload)
        self.assertAlmostEqual(restored.effect, result.effect)
        self.assertEqual(restored.effect_interval, result.effect_interval)

    def test_deserialize_defaults_method_to_exact(self) -> None:
        # Payloads serialized before the `method` field predate the knob; their
        # confidence sets were computed exactly, so they load as "exact".
        result = att_confidence_set(_paper_table(), confidence=0.90)
        payload = result.to_dict()
        del payload["method"]
        restored = PairedOutcomeAnalysis.deserialize(d=payload)
        self.assertEqual(restored.method, "exact")


class SensitivityValueTest(unittest.TestCase):
    """Tests for ``sensitivity_value``."""

    def test_gamma_star_exceeds_one_for_strong_effect(self) -> None:
        # H_0: A_1 = 0 is rejected at gamma=1 (two-sided p ~ 0.024 < 0.05), so
        # the finding tolerates some hidden bias: Γ• > 1.
        gamma_star = sensitivity_value(_paper_table(), alpha=0.05)
        self.assertGreater(gamma_star, 1.0)

    def test_gamma_star_is_the_crossing_point(self) -> None:
        table = _paper_table()
        gamma_star = sensitivity_value(table, alpha=0.05)
        # Just below Γ• the null is rejected; just above it is not.
        c0, c1 = table.s00 + table.s10, table.s01 + table.s11
        delta = table.hat_a
        b = min(c1, c0 - delta)
        a = b + delta

        def worst_p(gamma: float) -> float:
            # Two-sided, matching sensitivity_value's inverted test.
            return min(1.0, 2.0 * _p_greater(a, b, gamma / (gamma + 1.0)))

        self.assertLessEqual(worst_p(gamma_star - 1e-3), 0.05)
        self.assertGreater(worst_p(gamma_star + 1e-3), 0.05)

    def test_null_effect_returns_one(self) -> None:
        # No estimated effect -> cannot reject even in the randomized case.
        table = PairedOutcomeTable(s00=100, s01=50, s10=50, s11=100)
        self.assertEqual(sensitivity_value(table, alpha=0.05), 1.0)


class DesignSensitivityBinaryTest(unittest.TestCase):
    """Tests for ``design_sensitivity_binary``."""

    def test_paper_running_example(self) -> None:
        # Wilson (2026), section 7: control success rate p_{+1} = 0.13, effect +0.04,
        # for a (post-hoc) general design sensitivity of 1.31.
        self.assertAlmostEqual(
            design_sensitivity_binary(baseline=0.13, ate=0.04), 1.31, places=2
        )

    def test_monotonic_baseline_gives_larger_value(self) -> None:
        # Same formula 1 + ate/baseline; passing the smaller monotonic
        # denominator p_{01} <= p_{+1} yields a larger design sensitivity.
        general = design_sensitivity_binary(baseline=0.13, ate=0.04)
        monotonic = design_sensitivity_binary(baseline=0.03, ate=0.04)
        self.assertGreater(monotonic, general)

    def test_equals_success_rate_ratio(self) -> None:
        # General case: Γ̃ = p_{1+} / p_{+1}.
        baseline, ate = 0.13, 0.04
        self.assertAlmostEqual(
            design_sensitivity_binary(baseline=baseline, ate=ate),
            (baseline + ate) / baseline,
        )

    def test_rejects_nonpositive_ate(self) -> None:
        with self.assertRaises(ValueError):
            design_sensitivity_binary(baseline=0.13, ate=0.0)
        with self.assertRaises(ValueError):
            design_sensitivity_binary(baseline=0.13, ate=-0.04)

    def test_rejects_baseline_out_of_range(self) -> None:
        with self.assertRaises(ValueError):
            design_sensitivity_binary(baseline=0.0, ate=0.04)
        with self.assertRaises(ValueError):
            design_sensitivity_binary(baseline=1.0, ate=0.04)

    def test_rejects_baseline_plus_ate_at_least_one(self) -> None:
        with self.assertRaises(ValueError):
            design_sensitivity_binary(baseline=0.8, ate=0.3)


class GammaStarMethodTest(unittest.TestCase):
    """``PairedOutcomeTable.gamma_star`` summarizes the expanded ATE interval.

    Unlike the free ``sensitivity_value`` (which inverts the ``A_1``-only worst-case
    McNemar p-value), the method reports the largest ``Gamma`` at which the expanded
    (Rigdon-Hudgens) ATE confidence interval -- the band ``plot_sensitivity`` draws --
    still excludes the null, so the dotted line and the band cross the null together.

    """

    def test_gamma_star_matches_expanded_interval_crossing(self) -> None:
        # Just below Γ• the expanded ATE interval excludes 0; just above it
        # straddles 0. For this positive effect that means its lower bound is
        # above 0 below Γ• and at/below 0 above it.
        table = _paper_table()
        gamma_star = table.gamma_star(0.0, 0.10)
        self.assertGreater(gamma_star, 1.0)
        below = table.expanded_confidence_interval(0.10, gamma_star - 1e-3)
        above = table.expanded_confidence_interval(0.10, gamma_star + 1e-3)
        self.assertGreater(below[0], 0.0)
        self.assertLessEqual(above[0], 0.0)

    def test_gamma_star_running_example_value(self) -> None:
        # The corrected definition (expanded ATE interval, alpha/4 per side over
        # A_1 + A_0) lands near 1.035 for the paper's running example -- well
        # below the ~1.07 the A_1-only p-value inversion used to return.
        table = _paper_table()
        self.assertAlmostEqual(table.gamma_star(0.0, 0.10), 1.035, delta=5e-3)

    def test_gamma_star_null_inside_band_returns_one(self) -> None:
        # No estimated effect -> the randomized band already contains 0.
        table = PairedOutcomeTable(s00=100, s01=50, s10=50, s11=100)
        self.assertEqual(table.gamma_star(0.0, 0.10), 1.0)


class CombinedTailMaxTest(unittest.TestCase):
    """The ATE split search maximizes ``min(f, g)`` for ``f`` up, ``g`` down."""

    def test_peak_at_crossover(self) -> None:
        # f non-decreasing, g non-increasing; min(f, g) peaks where they cross.
        f = lambda a: a / 10.0  # noqa: E731
        g = lambda a: 1.0 - a / 10.0  # noqa: E731
        # Both equal 0.5 at a = 5, the crossover, so the peak min is 0.5.
        self.assertAlmostEqual(_combined_tail_max(f, g, 0, 10), 0.5)

    def test_g_below_f_everywhere(self) -> None:
        # When g is always the smaller, the peak is just max(g) at the low end.
        f = lambda a: 0.9  # noqa: E731
        g = lambda a: 0.1  # noqa: E731
        self.assertAlmostEqual(_combined_tail_max(f, g, 0, 10), 0.1)

    def test_empty_range_is_zero(self) -> None:
        self.assertEqual(_combined_tail_max(lambda a: 1.0, lambda a: 1.0, 5, 4), 0.0)


class ATETargetTest(unittest.TestCase):
    """``target='ATE'`` surfaces Rigdon-Hudgens average-effect inference."""

    def test_interval_is_expanded_ate_interval_scaled(self) -> None:
        # attributable_effect_interval(ATE) is the expanded ATE set on the count
        # scale: dividing by n_pairs recovers expanded_confidence_interval.
        table = _paper_table()
        n = table.n_pairs
        count = attributable_effect_interval(table, target="ATE", confidence=0.90)
        rate = table.expanded_confidence_interval(alpha=0.10, gamma=1.0)
        self.assertAlmostEqual(count[0] / n, rate[0])
        self.assertAlmostEqual(count[1] / n, rate[1])

    def test_analyze_ate_wires_expanded_interval(self) -> None:
        table = _paper_table()
        n = table.n_pairs
        result = table.analyze(alpha=0.10, target="ATE")
        # The scaled column is the expanded ATE interval, up to the round-trip
        # through the count scale: analyze() scales the rate-scale set up by
        # n_pairs and back down, and ``x * n / n`` is not guaranteed bit-exact
        # in IEEE-754, so compare approximately rather than for tuple identity.
        expanded = table.expanded_confidence_interval(0.10, 1.0)
        self.assertAlmostEqual(result.effect_interval[0], expanded[0])
        self.assertAlmostEqual(result.effect_interval[1], expanded[1])
        # ... which is the iSuccesses (count-scale) set divided by n_pairs ...
        self.assertEqual(
            result.effect_interval,
            (
                result.attributable_interval[0] / n,
                result.attributable_interval[1] / n,
            ),
        )
        # ... and the point estimate is still the (shared) ate_hat.
        self.assertAlmostEqual(result.effect, table.ate_hat)

    def test_ate_scaled_header(self) -> None:
        text = str(_paper_table().analyze(target="ATE"))
        self.assertIn("ATE", text)
        self.assertNotIn("ATT", text)
        self.assertNotIn("ATU", text)

    def test_ate_gamma_star_matches_method(self) -> None:
        # sensitivity_value(ATE) inverts the same combined test the gamma_star
        # method inverts, so the two agree (~1.035 for the running example).
        table = _paper_table()
        sens = sensitivity_value(table, alpha=0.10, target="ATE")
        self.assertAlmostEqual(sens, table.gamma_star(0.0, 0.10), delta=5e-3)
        self.assertAlmostEqual(sens, 1.035, delta=5e-3)
        # analyze routes gamma_star through the same call.
        self.assertAlmostEqual(table.analyze(alpha=0.10, target="ATE").gamma_star, sens)

    def test_ate_pvalue_rejects_positive_effect(self) -> None:
        # Γ• > 1 means the two-sided ATE null is rejected at gamma=1, so the
        # worst-case p-value is below alpha there.
        table = _paper_table()
        p = worst_case_pvalue(table, target="ATE", alternative="two-sided")
        self.assertLess(p, 0.10)
        # analyze reports exactly this p-value.
        self.assertAlmostEqual(table.analyze(alpha=0.10, target="ATE").p_value, p)

    def test_ate_pvalue_one_sided_ordering(self) -> None:
        table = _paper_table()  # clear positive effect
        greater = worst_case_pvalue(table, target="ATE", alternative="greater")
        two_sided = worst_case_pvalue(table, target="ATE", alternative="two-sided")
        less = worst_case_pvalue(table, target="ATE", alternative="less")
        self.assertLessEqual(greater, two_sided)
        self.assertLessEqual(two_sided, less)
        # A positive effect gives no left-tail evidence.
        self.assertEqual(less, 1.0)

    def test_ate_two_sided_is_double_the_smaller_one_sided(self) -> None:
        # Two-sided = 2 x min(one-sided tails), capped at 1 -- the one-sided
        # tails already carry the Bonferroni factor over the two effects.
        table = _paper_table()
        greater = worst_case_pvalue(table, target="ATE", alternative="greater")
        less = worst_case_pvalue(table, target="ATE", alternative="less")
        two_sided = worst_case_pvalue(table, target="ATE", alternative="two-sided")
        self.assertAlmostEqual(two_sided, min(1.0, 2.0 * min(greater, less)))

    def test_ate_sensitivity_widens_interval(self) -> None:
        base = attributable_effect_interval(_paper_table(), target="ATE", gamma=1.0)
        stressed = attributable_effect_interval(_paper_table(), target="ATE", gamma=1.5)
        self.assertLessEqual(stressed[0], base[0])
        self.assertGreaterEqual(stressed[1], base[1])

    def test_ate_one_sided_interval_unbounded_above(self) -> None:
        lb, ub = attributable_effect_interval(
            _paper_table(), target="ATE", alternative="greater"
        )
        self.assertEqual(ub, math.inf)
        self.assertTrue(math.isfinite(lb))

    def test_ate_target_round_trips(self) -> None:
        result = _paper_table().analyze(target="ATE")
        self.assertEqual(result.target, "ATE")
        revived = PairedOutcomeAnalysis.deserialize(result.serialize())
        self.assertEqual(revived.target, "ATE")

    def test_ate_no_effect_pvalue_is_one(self) -> None:
        # Equal discordancies => no estimated effect => nothing to reject at any
        # split, so the worst-case ATE p-value and Γ• bottom out.
        flat = PairedOutcomeTable(s00=100, s01=50, s10=50, s11=100)
        self.assertEqual(
            worst_case_pvalue(flat, target="ATE", alternative="two-sided"), 1.0
        )
        self.assertEqual(sensitivity_value(flat, target="ATE"), 1.0)

    def test_ate_out_of_range_null_raises(self) -> None:
        # An ATE null beyond the reachable band raises, mirroring the single-
        # effect path. Without the guard the far side's certain-rejection clamp
        # would collapse the two-sided combine to a spurious 0.0 in BOTH
        # directions -- silently reporting a decisive rejection for a mis-scaled
        # (e.g. rate- rather than count-scale) null.
        table = _paper_table()
        for null_value in (100_000, -100_000):
            with self.subTest(null_value=null_value), self.assertRaises(ValueError):
                worst_case_pvalue(
                    table,
                    target="ATE",
                    null_value=null_value,
                    alternative="two-sided",
                )

    def test_ate_in_band_null_still_computes(self) -> None:
        # The guard rejects only unreachable nulls; a null on the reachable band
        # (here the no-effect null=0 for a clear positive effect) still returns
        # the genuine worst-case tail rather than raising. Pin the real
        # contract: the explicit null=0 matches the default-null computation and
        # rejects (< 0.10), not just "some float in [0, 1]".
        table = _paper_table()
        p = worst_case_pvalue(
            table, target="ATE", null_value=0, alternative="two-sided"
        )
        self.assertEqual(
            p, worst_case_pvalue(table, target="ATE", alternative="two-sided")
        )
        self.assertLess(p, 0.10)

    def test_ate_null_band_edges_and_rounding(self) -> None:
        # Pin the guard's inward count-scale rounding. For _paper_table()
        # (s00,s01,s10,s11 = 800,30,70,100, hat_a = 40) the non-monotonic
        # ceilings are ATT (c0,c1) = (s00+s10, s01+s11) = (870, 130) and
        # ATU (c0,c1) = (s10+s11, s00+s01) = (170, 830), so the reachable
        # average-effect count band is
        #   lo = ceil((2*40 - 870 - 170) / 2) = (80 - 870 - 170 + 1)//2 = -480
        #   hi = floor((2*40 + 130 + 830) / 2) = (80 + 130 + 830)//2 =  520.
        table = _paper_table()
        lo, hi = -480, 520
        for edge in (lo, hi):
            with self.subTest(edge=edge):
                p = worst_case_pvalue(table, target="ATE", null_value=edge)
                self.assertTrue(0.0 <= p <= 1.0)
        for outside in (lo - 1, hi + 1):
            with self.subTest(outside=outside), self.assertRaises(ValueError):
                worst_case_pvalue(table, target="ATE", null_value=outside)

    def test_ate_worst_case_pvalue_rejects_unknown_target(self) -> None:
        with self.assertRaises(ValueError):
            worst_case_pvalue(_paper_table(), target="bogus")


class MonotonicSensitivityTest(unittest.TestCase):
    """Monotonicity narrows the ATE sensitivity/confidence sets, raising Γ•.

    Assuming treatment never hurts shrinks the worst-case discordancy box ceilings (see
    ``_ceilings``), so at a fixed Gamma every set nests within its non-monotonic
    counterpart and the sensitivity value ``Γ•`` can only rise.

    """

    def _table(self) -> PairedOutcomeTable:
        # The paper's running example: a clear positive effect, so the box
        # genuinely shrinks under monotonicity and the narrowing is visible.
        return PairedOutcomeTable(s00=800, s01=30, s10=70, s11=100)

    def test_sensitivity_analysis_monotonic_is_strictly_narrower(self) -> None:
        table = self._table()
        base = table.sensitivity_analysis(2.0, monotonic=False)
        mono = table.sensitivity_analysis(2.0, monotonic=True)
        # Nested: mono lower is no lower, mono upper is no higher ...
        self.assertGreaterEqual(mono[0], base[0])
        self.assertLessEqual(mono[1], base[1])
        # ... and strictly narrower on both ends for this positive table.
        self.assertGreater(mono[0], base[0])
        self.assertLess(mono[1], base[1])

    def test_expanded_confidence_interval_monotonic_is_nested(self) -> None:
        table = self._table()
        base = table.expanded_confidence_interval(
            alpha=0.10, gamma=2.0, monotonic=False
        )
        mono = table.expanded_confidence_interval(alpha=0.10, gamma=2.0, monotonic=True)
        self.assertGreaterEqual(mono[0], base[0])
        self.assertLessEqual(mono[1], base[1])

    def test_gamma_star_monotonic_is_at_least_as_large(self) -> None:
        table = self._table()
        self.assertGreaterEqual(
            table.gamma_star(monotonic=True), table.gamma_star(monotonic=False)
        )

    def test_sensitivity_analysis_agrees_at_gamma_one(self) -> None:
        # The confounding-only band collapses to the point estimate at Gamma = 1
        # regardless of monotonicity (both spreads vanish), so the two agree.
        table = self._table()
        base = table.sensitivity_analysis(1.0, monotonic=False)
        mono = table.sensitivity_analysis(1.0, monotonic=True)
        self.assertAlmostEqual(mono[0], base[0])
        self.assertAlmostEqual(mono[1], base[1])

    def test_expanded_confidence_interval_nested_at_gamma_one(self) -> None:
        # Unlike the confounding-only band, the sampling-based confidence set
        # depends on the box ceilings even at Gamma = 1, so monotonicity narrows
        # it here too: mono is nested within (not equal to) non-monotonic.
        table = self._table()
        base = table.expanded_confidence_interval(
            alpha=0.10, gamma=1.0, monotonic=False
        )
        mono = table.expanded_confidence_interval(alpha=0.10, gamma=1.0, monotonic=True)
        self.assertGreaterEqual(mono[0], base[0])
        self.assertLessEqual(mono[1], base[1])

    def test_large_gamma_lower_endpoint_saturates_upward(self) -> None:
        # Under monotonicity the sensitivity-band lower endpoint saturates at the
        # a-priori minimum 0, so it lies at or above the non-monotonic lower,
        # which keeps opening downward.
        table = self._table()
        base = table.sensitivity_analysis(50.0, monotonic=False)
        mono = table.sensitivity_analysis(50.0, monotonic=True)
        self.assertGreaterEqual(mono[0], base[0])
        self.assertGreater(mono[0], base[0])


class McNemarComparisonTest(unittest.TestCase):
    """Tests for the McNemar baseline interval."""

    def test_center_and_width(self) -> None:
        lo, hi = mcnemar_ate_interval(_paper_table(), confidence=0.95)
        self.assertAlmostEqual((lo + hi) / 2.0, 0.040)
        # Wald: 1.96 * sqrt(30 + 70 - 40^2 / 1000) / 1000 = 1.96 * sqrt(98.4) / 1000.
        self.assertAlmostEqual(
            (hi - lo) / 2.0, 1.959963984540054 * math.sqrt(98.4) / 1000.0
        )

    def test_matches_paper_half_width(self) -> None:
        # Wilson (2026), section 6: "1.645 sqrt(98.4) / 1000 ~ 0.0163".
        lo, hi = mcnemar_ate_interval(_paper_table(), confidence=0.90)
        self.assertAlmostEqual((hi - lo) / 2.0, 0.0163, places=4)

    def test_wald_variance_shrinks_with_effect(self) -> None:
        # The (S10 - S01)^2 / S term separates the Wald variance from the null
        # variance S01 + S10; with every discordant pair favoring treatment it
        # removes 300^2 / 1000 = 90 of the 300.
        table = PairedOutcomeTable(s00=700, s01=0, s10=300, s11=0)
        lo, hi = mcnemar_ate_interval(table, confidence=0.95)
        self.assertAlmostEqual(
            (hi - lo) / 2.0, 1.959963984540054 * math.sqrt(210.0) / 1000.0
        )

    def test_all_pairs_discordant_one_way_has_zero_width(self) -> None:
        # Every pair is S10, so the plug-in variance p10 (1 - p10) is zero.
        lo, hi = mcnemar_ate_interval(
            PairedOutcomeTable(s00=0, s01=0, s10=50, s11=0), confidence=0.90
        )
        self.assertEqual((lo, hi), (1.0, 1.0))

    def test_narrower_than_randomization_interval(self) -> None:
        mc_lo, mc_hi = mcnemar_ate_interval(_paper_table(), confidence=0.90)
        rand = att_confidence_set(_paper_table(), confidence=0.90).effect_interval
        self.assertLess(mc_hi - mc_lo, rand[1] - rand[0])


class AlternativePValueTest(unittest.TestCase):
    """One-sided vs two-sided worst-case p-values."""

    def test_greater_is_right_tail_less_is_left_tail(self) -> None:
        table = _paper_table()  # hat_a = 40 > 0: right tail carries the evidence
        right = worst_case_pvalue(table, alternative="greater")
        left = worst_case_pvalue(table, alternative="less")
        two_sided = worst_case_pvalue(table, alternative="two-sided")
        # Right tail is the paper's worst-case corner p_>(170, 130) at gamma=1.
        self.assertAlmostEqual(right, _p_greater(170, 130, 0.5))
        # A positive effect gives no left-tail evidence.
        self.assertEqual(left, 1.0)
        # Two-sided doubles the smaller (right) tail, capped at 1.
        self.assertAlmostEqual(two_sided, min(1.0, 2.0 * right))

    def test_positive_effect_pvalue_ordering(self) -> None:
        table = _paper_table()
        greater = worst_case_pvalue(table, alternative="greater")
        two_sided = worst_case_pvalue(table, alternative="two-sided")
        less = worst_case_pvalue(table, alternative="less")
        # For a clearly-positive effect: greater is most significant.
        self.assertLess(greater, two_sided)
        self.assertLess(two_sided, less)

    def test_nonzero_null_value(self) -> None:
        table = _paper_table()  # hat_a = 40
        null_value = -10
        # delta = 40 - (-10) = 50 -> right-tail corner (180, 130) at gamma=1.
        p_greater = worst_case_pvalue(
            table, null_value=null_value, alternative="greater"
        )
        self.assertAlmostEqual(p_greater, _p_greater(180, 130, 0.5))
        # analyze reports exactly this p-value for the same null and alternative.
        result = table.analyze(null_value=null_value, alternative="greater")
        self.assertAlmostEqual(result.p_value, p_greater)

    def test_invalid_alternative_raises(self) -> None:
        table = _paper_table()
        with self.assertRaises(ValueError):
            worst_case_pvalue(table, alternative="bogus")
        with self.assertRaises(ValueError):
            table.analyze(alternative="bogus")
        with self.assertRaises(ValueError):
            table.confidence_interval(alternative="bogus")
        with self.assertRaises(ValueError):
            attributable_effect_interval(table, alternative="bogus")
        with self.assertRaises(ValueError):
            sensitivity_value(table, alternative="bogus")

    def test_sensitivity_value_accepts_alternative(self) -> None:
        table = _paper_table()
        gs_two = sensitivity_value(table, alpha=0.05, alternative="two-sided")
        gs_greater = sensitivity_value(table, alpha=0.05, alternative="greater")
        # The one-sided p is smaller, so it tolerates more hidden bias.
        self.assertGreaterEqual(gs_greater, gs_two)
        # Γ• inverts the one-sided p: just below it rejects, just above not.
        self.assertLessEqual(
            worst_case_pvalue(table, gamma=gs_greater - 1e-3, alternative="greater"),
            0.05,
        )
        self.assertGreater(
            worst_case_pvalue(table, gamma=gs_greater + 1e-3, alternative="greater"),
            0.05,
        )


class OneSidedConfidenceIntervalTest(unittest.TestCase):
    """One-sided ATT confidence intervals via ``confidence_interval``."""

    def test_greater_interval_unbounded_above(self) -> None:
        lb, ub = _paper_table().confidence_interval(alpha=0.10, alternative="greater")
        self.assertEqual(ub, math.inf)
        self.assertTrue(math.isfinite(lb))

    def test_less_interval_unbounded_below(self) -> None:
        lb, ub = _paper_table().confidence_interval(alpha=0.10, alternative="less")
        self.assertEqual(lb, -math.inf)
        self.assertTrue(math.isfinite(ub))

    def test_two_sided_finite_and_one_sided_is_tighter(self) -> None:
        table = _paper_table()
        lb, ub = table.confidence_interval(alpha=0.10, alternative="two-sided")
        self.assertTrue(math.isfinite(lb))
        self.assertTrue(math.isfinite(ub))
        g_lb, _ = table.confidence_interval(alpha=0.10, alternative="greater")
        _, l_ub = table.confidence_interval(alpha=0.10, alternative="less")
        # One-sided bounds spend the full alpha on one side, so they are tighter.
        self.assertGreaterEqual(g_lb, lb)
        self.assertLessEqual(l_ub, ub)

    def test_confidence_interval_matches_analyze(self) -> None:
        table = _paper_table()
        self.assertEqual(
            table.confidence_interval(alpha=0.10),
            table.analyze(alpha=0.10).effect_interval,
        )


class AnalyzeOneSidedTest(unittest.TestCase):
    """One-sided intervals threaded through ``analyze`` and its display."""

    def test_analyze_greater_intervals_are_one_sided(self) -> None:
        table = _paper_table()
        result = table.analyze(alpha=0.10, alternative="greater")
        self.assertEqual(result.effect_interval[1], math.inf)
        self.assertTrue(math.isfinite(result.effect_interval[0]))
        self.assertEqual(result.attributable_interval[1], math.inf)
        self.assertTrue(math.isfinite(result.attributable_interval[0]))
        # The ATT interval is the attributable set / n_pairs, so one-sidedness
        # is preserved: +inf / n = +inf.
        self.assertEqual(
            result.effect_interval,
            (result.attributable_interval[0] / table.n_pairs, math.inf),
        )

    def test_analyze_less_intervals_are_one_sided(self) -> None:
        result = _paper_table().analyze(alpha=0.10, alternative="less")
        self.assertEqual(result.effect_interval[0], -math.inf)
        self.assertTrue(math.isfinite(result.effect_interval[1]))
        self.assertEqual(result.attributable_interval[0], -math.inf)
        self.assertTrue(math.isfinite(result.attributable_interval[1]))

    def test_analyze_str_renders_infinity(self) -> None:
        text = str(_paper_table().analyze(alpha=0.10, alternative="greater"))
        self.assertIn("∞", text)  # unbounded interval endpoint
        self.assertIn("ATT", text)

    def test_footer_states_null_hypothesis(self) -> None:
        table = _paper_table()
        # two-sided at the default null: "= 0".
        self.assertIn(
            "p-Value is two-sided against the null hypothesis that iSuccesses = 0",
            str(table.analyze()),
        )
        # one-sided "greater" at a non-zero null: "<= null" (rendered with U+2264).
        self.assertIn(
            "p-Value is one-sided against the null hypothesis that iSuccesses ≤ -100",
            str(table.analyze(null_value=-100, alternative="greater")),
        )
        # one-sided "less": ">= null" (U+2265).
        self.assertIn(
            "p-Value is one-sided against the null hypothesis that iSuccesses ≥ 0",
            str(table.analyze(alternative="less")),
        )

    def test_footer_null_hypothesis_when_isuccesses_hidden(self) -> None:
        # When the iSuccesses column is not shown, the footer must not cite it
        # (nor an override set on that hidden column) -- it uses a generic label.
        options = PairedOutcomeAnalysisOptions(
            effect_size_columns=("effect",),
            header_overrides={"isuccesses": "Hidden"},
        )
        text = str(_paper_table().analyze(options=options))
        self.assertIn(
            "null hypothesis that the attributable effect = 0",
            text,
        )
        self.assertNotIn("Hidden", text)


def _scaled(table: PairedOutcomeTable, k: int) -> PairedOutcomeTable:
    """The same table with every cell count multiplied by ``k``."""
    return PairedOutcomeTable(
        s00=table.s00 * k, s01=table.s01 * k, s10=table.s10 * k, s11=table.s11 * k
    )


class LargeSampleApproximationTest(unittest.TestCase):
    """The closed-form Gaussian prediction set (``method='normal'``)."""

    # Diverse tables that drive the two-face root onto each consistency face:
    # balanced, prevention-heavy (c0 >> c1), success-heavy (c1 >> c0), and a
    # large near-symmetric table.
    _TABLES = (
        PairedOutcomeTable(s00=800, s01=30, s10=70, s11=100),
        PairedOutcomeTable(s00=5000, s01=20, s10=400, s11=40),
        PairedOutcomeTable(s00=40, s01=400, s10=20, s11=5000),
        PairedOutcomeTable(s00=2000, s01=1500, s10=1800, s11=1700),
    )

    def test_matches_paper_large_sample_example(self) -> None:
        # Paper S5 running example. The asymptotic (no continuity correction)
        # closed form gives [13, 65]; with the 1/2 continuity correction it
        # recovers the exact bisection endpoints [12, 66] exactly.
        interval = attributable_effect_interval(
            _paper_table(), target="ATT", confidence=0.90, method="normal"
        )
        self.assertEqual(interval, (12, 66))
        exact = attributable_effect_interval(
            _paper_table(), target="ATT", confidence=0.90, method="exact"
        )
        self.assertEqual(interval, exact)

    def test_brackets_exact_within_one_at_gamma_one(self) -> None:
        table = _paper_table()
        for target in ("ATT", "ATU"):
            exact = attributable_effect_interval(
                table, target=target, confidence=0.90, method="exact"
            )
            approx = attributable_effect_interval(
                table, target=target, confidence=0.90, method="normal"
            )
            self.assertLessEqual(abs(approx[0] - exact[0]), 1, msg=f"{target} lo")
            self.assertLessEqual(abs(approx[1] - exact[1]), 1, msg=f"{target} hi")

    def test_two_face_selection_agrees_with_exact(self) -> None:
        # Across imbalanced tables and mild sensitivity the closed form (which
        # selects the binding face by the crossover, not a blind min-ceiling)
        # tracks the exact bisection to within the normal approximation's
        # boundary slack.
        for table in self._TABLES:
            for gamma in (1.0, 1.5):
                for target in ("ATT", "ATU"):
                    exact = attributable_effect_interval(
                        table,
                        target=target,
                        confidence=0.90,
                        gamma=gamma,
                        method="exact",
                    )
                    approx = attributable_effect_interval(
                        table,
                        target=target,
                        confidence=0.90,
                        gamma=gamma,
                        method="normal",
                    )
                    msg = f"{table} target={target} gamma={gamma}"
                    self.assertLessEqual(abs(approx[0] - exact[0]), 2, msg=msg)
                    self.assertLessEqual(abs(approx[1] - exact[1]), 2, msg=msg)

    def test_continuity_corrected_matches_exact_across_scales(self) -> None:
        # With the continuity correction the closed form tracks the exact
        # bisection to within a single count at every scale (the corrected
        # normal approximation removes the leading discretization error).
        base = _paper_table()
        for k in (1, 25, 400):
            table = _scaled(base, k)
            exact = attributable_effect_interval(
                table, target="ATT", confidence=0.90, method="exact"
            )
            approx = attributable_effect_interval(
                table, target="ATT", confidence=0.90, method="normal"
            )
            self.assertLessEqual(abs(approx[0] - exact[0]), 1, msg=f"k={k} lo")
            self.assertLessEqual(abs(approx[1] - exact[1]), 1, msg=f"k={k} hi")

    def test_normal_respects_one_sided_alternative(self) -> None:
        table = _paper_table()
        lb, ub = attributable_effect_interval(
            table, confidence=0.90, alternative="greater", method="normal"
        )
        self.assertEqual(ub, math.inf)
        self.assertTrue(math.isfinite(lb))
        lb2, ub2 = attributable_effect_interval(
            table, confidence=0.90, alternative="less", method="normal"
        )
        self.assertEqual(lb2, -math.inf)
        self.assertTrue(math.isfinite(ub2))

    def test_normal_empty_ceiling_collapses_to_point(self) -> None:
        # No prevention pool on the lower side: like the exact path, the closed
        # form collapses that bound to the point estimate rather than searching
        # an empty range.
        table = PairedOutcomeTable(s00=0, s01=0, s10=50, s11=50)
        c0, c1 = 0, table.s01 + table.s11
        lo, hi = _normal_worst_case_interval(table.hat_a, c0, c1, alpha=0.10, gamma=1.0)
        self.assertEqual(lo, table.hat_a)

    def test_auto_resolves_by_sample_size(self) -> None:
        # Large binding face -> normal; tiny -> exact.
        self.assertEqual(_resolve_method("auto", 870, 130, 1.0), "normal")
        self.assertEqual(_resolve_method("auto", 3, 4, 1.0), "exact")
        # High Gamma pushes pi -> 1, so pi(1-pi) -> 0 and auto reverts to exact
        # even for a large face -- the regime where the normal frays.
        self.assertEqual(_resolve_method("auto", 1000, 1000, 1.0), "normal")
        self.assertEqual(_resolve_method("auto", 1000, 1000, 1000.0), "exact")

    def test_explicit_method_passes_through(self) -> None:
        self.assertEqual(_resolve_method("exact", 1000, 1000, 1.0), "exact")
        self.assertEqual(_resolve_method("normal", 3, 4, 1.0), "normal")

    def test_invalid_method_raises(self) -> None:
        with self.assertRaises(ValueError):
            attributable_effect_interval(_paper_table(), method="bogus")
        with self.assertRaises(ValueError):
            _paper_table().confidence_interval(method="bogus")
        with self.assertRaises(ValueError):
            _paper_table().expanded_confidence_interval(method="bogus")

    def test_confidence_interval_method_matches_direct(self) -> None:
        table = _scaled(_paper_table(), 4)
        ci = table.confidence_interval(alpha=0.10, method="normal")
        direct = attributable_effect_interval(
            table, target="ATT", confidence=0.90, method="normal"
        )
        n = table.n_pairs
        self.assertEqual(ci, (direct[0] / n, direct[1] / n))

    def test_expanded_interval_normal_near_exact_at_scale(self) -> None:
        table = _scaled(_paper_table(), 25)
        exact = table.expanded_confidence_interval(
            alpha=0.10, gamma=1.5, method="exact"
        )
        approx = table.expanded_confidence_interval(
            alpha=0.10, gamma=1.5, method="normal"
        )
        self.assertAlmostEqual(exact[0], approx[0], places=3)
        self.assertAlmostEqual(exact[1], approx[1], places=3)
