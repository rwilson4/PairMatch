# pyre-strict
"""Randomization inference for the ATT in matched-pair binary outcomes."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt
from scipy.stats import binom, norm
from tabulate import tabulate

from pair_match.visualizations import (
    _plot_sensitivity_curve,
    _resolve_gamma_max,
    _sweep_sensitivity_bands,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.typing import LegendLocType

    from pair_match.match_result import (
        Pairing,
    )


def format_with_min_nonzero_digits(
    number: float | None, min_digits: int, percentage: bool = True
) -> str:
    """Format with min nonzero digits.

    Parameters
    ----------
     number : float
        The number.
     min_digits : int
        Minimum digits to include.
     percentage : boolean, optional
        Whether to print as a percentage. Defaults to True.

    Returns
    -------
     formatted_number : str
        A prettily-formatted number.

    """
    if number is None:
        return "None"

    if not math.isfinite(number):
        # NaN/inf have no sensible fixed-point form; show the repr. Zero is finite
        # and falls through to the small-number branch below (log10(0), which is
        # undefined, is never evaluated on this path).
        return str(number)

    if abs(number) <= 1e-6 and not percentage:
        fstr = "{" + f"number:.0{min_digits}g" + "}"
        return fstr.format(number=number)

    if abs(number) > 1:
        num_digits = 0
    elif abs(number) > 1e-6:
        num_zeros = -1 - math.floor(math.log10(abs(number)))
        if percentage:
            num_zeros -= 2
        num_digits = min_digits + num_zeros
    else:
        num_digits = min_digits

    fstr = "{" + f"number:.0{num_digits}f" + "}"
    if percentage:
        fstr += "%"
        number *= 100.0

    return fstr.format(number=number)


class EffectSize(Enum):
    r"""An effect-size column available in a ``PairedOutcomeTable`` analysis.

    ``EFFECT`` is the scaled treatment effect (a proportion) for the analysis
    ``target``: the average effect on the treated (ATT) for ``A_1``, on the untreated
    (ATU) for ``A_0``, or the average treatment effect (ATE). It is the attributable
    effect divided by ``n_pairs``, so for ``A_1`` / ``A_0`` it is an effect on the
    matched units (for whom the matched counterparts stand in as counterfactuals), not a
    population ATE. ``ISUCCESSES`` is the attributable effect -- the net count of
    successes among treated units caused by treatment (``S_10 - S_01``, an integer);
    ``COST_PER_ISUCCESS`` is the treatment cost per incremental success (needs
    ``PairedOutcomeTable.spend``).

    """

    EFFECT = "effect"
    ISUCCESSES = "isuccesses"
    COST_PER_ISUCCESS = "cost_per_isuccess"

    @classmethod
    def get(cls, value: EffectSize | str) -> EffectSize:
        """Coerce a member or a case-insensitive string to an ``EffectSize``.

        The legacy name ``"att"`` -- the pre-target-generic label for the scaled column
        -- is accepted as an alias for :attr:`EFFECT`, so options and payloads written
        before the rename still resolve.

        """
        if isinstance(value, cls):
            return value
        text = str(value).lower()
        if text == "att":
            return cls.EFFECT
        try:
            return cls(text)
        except ValueError:
            raise ValueError(
                f"unknown effect size {value!r}; choose from {[e.value for e in cls]}."
            ) from None


def _p_greater(a: int, b: int, pi: float) -> float:
    """Right tail P(Binomial(a + b, pi) >= a)."""
    n = a + b
    if n <= 0:
        return 1.0
    return float(binom.sf(a - 1, n, pi))


def _p_less(a: int, b: int, pi: float) -> float:
    """Left tail P(Binomial(a + b, pi) <= a)."""
    n = a + b
    if n <= 0:
        return 1.0
    return float(binom.cdf(a, n, pi))


_ALTERNATIVES = ("two-sided", "less", "greater")
_TARGETS = ("ATT", "ATU", "ATE")


def _validate_alternative(alternative: str) -> None:
    """Raise ``ValueError`` unless ``alternative`` is a recognized option."""
    if alternative not in _ALTERNATIVES:
        raise ValueError(
            f"`alternative` must be one of {_ALTERNATIVES}, got {alternative!r}."
        )


def _validate_target(target: str) -> None:
    """Raise ``ValueError`` unless ``target`` is a recognized effect."""
    if target not in _TARGETS:
        raise ValueError(f"`target` must be one of {_TARGETS}, got {target!r}.")


@dataclass(frozen=True)
class PairedOutcomeTable:
    r"""The 2x2 table of outcome patterns among matched pairs.

    Each pair contributes to exactly one cell according to the observed binary outcomes
    of its treated and control units. The first subscript is the treated unit's outcome,
    the second is the control unit's.

    Attributes
    ----------
     s00 : int
        Pairs where both units failed (treated 0, control 0).
     s01 : int
        Pairs where the treated unit failed and the control succeeded.
     s10 : int
        Pairs where the treated unit succeeded and the control failed.
     s11 : int
        Pairs where both units succeeded (treated 1, control 1).
     spend : float, optional
        Total cost of treatment, used by the ``COST_PER_ISUCCESS`` effect size. ``None``
        (the default) when no cost is attached.

    """

    s00: int
    s01: int
    s10: int
    s11: int
    spend: float | None = None

    @property
    def n_pairs(self) -> int:
        """Total number of matched pairs."""
        return self.s00 + self.s01 + self.s10 + self.s11

    @property
    def hat_a(self) -> int:
        r"""Hodges-Lehmann point estimate of the attributable effect.

        Equals ``S_10 - S_01``, the McNemar pivot; the same value estimates both ``A_1``
        and ``A_0``.

        """
        return self.s10 - self.s01

    @property
    def ate_hat(self) -> float:
        """Point estimate of the average treatment effect."""
        return self.hat_a / self.n_pairs

    @property
    def treated_success_rate(self) -> float:
        """Fraction of treated units with outcome 1 (``S_10 + S_11``)."""
        n = self.n_pairs
        return (self.s10 + self.s11) / n if n > 0 else 0.0

    @property
    def control_success_rate(self) -> float:
        """Fraction of control units with outcome 1 (``S_01 + S_11``)."""
        n = self.n_pairs
        return (self.s01 + self.s11) / n if n > 0 else 0.0

    @staticmethod
    def from_outcomes(
        treated_outcomes: npt.ArrayLike,
        control_outcomes: npt.ArrayLike,
        *,
        spend: float | None = None,
    ) -> PairedOutcomeTable:
        r"""Build the table from aligned binary outcome vectors.

        Parameters
        ----------
         treated_outcomes : array-like of {0, 1}
            Outcome of the treated unit in each pair.
         control_outcomes : array-like of {0, 1}
            Outcome of the control unit in each pair, aligned element-wise with
            ``treated_outcomes``.
         spend : float, optional
            Total cost of treatment (see :attr:`spend`).

        """
        t = np.asarray(treated_outcomes)
        c = np.asarray(control_outcomes)
        if t.shape != c.shape:
            raise ValueError(
                "`treated_outcomes` and `control_outcomes` must have the "
                f"same shape (got {t.shape} and {c.shape})."
            )
        if not np.all(np.isin(t, (0, 1))) or not np.all(np.isin(c, (0, 1))):
            raise ValueError("Outcomes must be binary (0 or 1).")

        t = t.astype(bool)
        c = c.astype(bool)
        return PairedOutcomeTable(
            s00=int(np.sum(~t & ~c)),
            s01=int(np.sum(~t & c)),
            s10=int(np.sum(t & ~c)),
            s11=int(np.sum(t & c)),
            spend=spend,
        )

    @staticmethod
    def from_match_result(
        result: Pairing,
        outcome: str,
        *,
        df_treated: pd.DataFrame,
        df_control: pd.DataFrame,
        spend: float | None = None,
    ) -> PairedOutcomeTable:
        r"""Build the table from a matching and an outcome column.

        Reads the binary outcome for each matched unit by index label from the frames
        the matching was built from -- typically the same ``df_treated`` /
        ``df_control`` passed to the matcher, which already carry the outcome as a
        column.

        Parameters
        ----------
         result : Pairing
            A :class:`~pair_match.match_result.MatchResult`, or any object carrying
            aligned ``treated_index`` and ``control_index`` sequences.
         outcome : str
            Name of the binary outcome column, present in both ``df_treated`` and
            ``df_control``.
         df_treated, df_control : DataFrame
            The treated and control frames the matching was built from. The outcome is
            gathered from ``df_treated`` at ``result.treated_index`` and from
            ``df_control`` at ``result.control_index``.
         spend : float, optional
            Total cost of treatment (see :attr:`spend`).

        """
        treated = df_treated.loc[list(result.treated_index), outcome].to_numpy()
        control = df_control.loc[list(result.control_index), outcome].to_numpy()
        return PairedOutcomeTable.from_outcomes(treated, control, spend=spend)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "s00": self.s00,
            "s01": self.s01,
            "s10": self.s10,
            "s11": self.s11,
            "spend": self.spend,
        }

    def serialize(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict())

    @staticmethod
    def deserialize(
        s: str | None = None, d: dict[str, object] | None = None
    ) -> PairedOutcomeTable:
        """Reconstruct a ``PairedOutcomeTable`` from a JSON string or dict.

        Tolerates a missing ``spend`` key (older payloads) by defaulting it to ``None``.

        """
        if s is not None and d is not None:
            raise ValueError("Provide `s` or `d`, not both.")
        if s is not None:
            d = json.loads(s)
        if d is None:
            raise ValueError("Provide either `s` or `d`.")
        spend = d.get("spend")
        return PairedOutcomeTable(
            s00=int(cast("int", d["s00"])),
            s01=int(cast("int", d["s01"])),
            s10=int(cast("int", d["s10"])),
            s11=int(cast("int", d["s11"])),
            spend=None if spend is None else float(cast("float", spend)),
        )

    def __str__(self) -> str:
        r"""A 2x2 contingency view: cell counts, margins, and success rates.

        Rows are the control unit's outcome, columns the treated unit's, with ``*``
        margins. Each group's success rate -- the fraction with outcome 1 -- is labelled
        alongside its margin; their difference is the ATT point estimate.

        """
        r0 = self.s00 + self.s10  # control outcome 0
        r1 = self.s01 + self.s11  # control outcome 1
        c0 = self.s00 + self.s01  # treated outcome 0
        c1 = self.s10 + self.s11  # treated outcome 1
        n = self.n_pairs
        data_rows = [
            ("0", (self.s00, self.s10, r0)),
            ("1", (self.s01, self.s11, r1)),
            ("*", (c0, c1, n)),
        ]
        counts = (self.s00, self.s10, r0, self.s01, self.s11, r1, c0, c1, n)
        w = max(len(f"{v:,}") for v in counts)
        gap = "  "
        group_w = len("Control")
        prefix_w = group_w + 3  # "<group> <rowlabel> "
        block_w = 3 * w + 2 * len(gap)

        lines: list[str] = [
            " " * prefix_w + "Treated".center(block_w),
            " " * prefix_w + gap.join(lbl.rjust(w) for lbl in ("0", "1", "*")),
        ]
        for rlabel, values in data_rows:
            group = "Control" if rlabel == "1" else ""
            prefix = f"{group:<{group_w}} {rlabel} "
            line = prefix + gap.join(f"{v:,}".rjust(w) for v in values)
            if rlabel == "1":
                line = f"{line}{gap}{self.control_success_rate:.1%} control"
            lines.append(line)
        # Treated success rate, its percentage under the treated "1" column.
        t1_start = prefix_w + w + len(gap)
        pct = f"{self.treated_success_rate:.1%}"
        indent = " " * max(0, t1_start + (w - len(pct)) // 2)
        lines.append(f"{indent}{pct} treated")
        return "\n".join(line.rstrip() for line in lines)

    def analyze(
        self,
        *,
        alpha: float = 0.10,
        gamma: float = 1.0,
        target: str = "ATT",
        null_value: int = 0,
        monotonic: bool = False,
        alternative: str = "two-sided",
        method: str = "auto",
        options: PairedOutcomeAnalysisOptions | None = None,
    ) -> PairedOutcomeAnalysis:
        r"""Summarize the matched-pair effect: estimates, CIs, p-value, ``Γ•``.

        Bundles the net-effects analysis of this table into a single displayable result:
        the scaled effect -- the ATT for the default ``target='ATT'``, the ATU for
        ``target='ATU'`` -- and the attributable effect (``ISUCCESSES``) with ``(1 -
        alpha)``-coverage confidence sets at sensitivity ``gamma``, the worst-case
        p-value for ``H_0: <target> = null_value`` at ``gamma``, and the Rosenbaum
        sensitivity value ``Γ•`` at level ``alpha``.

        Parameters
        ----------
         alpha : float
            Significance level: the confidence sets have coverage ``1 - alpha`` and the
            p-value is flagged significant below it (default 0.10, the RL MDS
            convention).
         gamma : float
            Sensitivity parameter *entertained* for the confidence sets and the p-value
            (``1`` = randomized; larger widens the sets). Distinct from ``Γ•``, which is
            a property of the data, not a value we choose.
         target : {'ATT', 'ATU', 'ATE'}
            The effect the p-value and ``Γ•`` concern, and the one reported as
            ``ISUCCESSES`` and (scaled) in the leading column (default ``'ATT'``).
            ``A_1`` is the effect on the treated (scaled column ``ATT``), ``A_0`` the
            effect on the untreated (``ATU``), and ``ATE`` the average effect (``ATE``),
            whose ``ISUCCESSES`` is the average attributable effect ``(A_1 + A_0) / 2``
            and whose test combines both effects under the Rigdon-Hudgens constraint
            (see :func:`_ate_worst_case_pvalue`).
         null_value : int
            Null value tested by the p-value and ``Γ•`` (default 0), on the
            attributable-effect (count) scale (for ``ATE``, the average-effect count
            ``ATE * n_pairs``).
         monotonic : bool
            Assume treatment never hurts any unit (no prevention). The net- effects
            procedure's main contribution is *not* to require this; the default
            (``False``) makes no such assumption. Opting in is valid when monotonicity
            is substantively defensible and narrows every set, sharpens the p-value, and
            raises ``Γ•``.
         alternative : {'two-sided', 'less', 'greater'}
            The kind of test/interval:

              - "two-sided": ``H_0: <target> = null_value``; both the p-value and
                the confidence sets are two-sided (each bound at ``alpha / 2``).
              - "greater": ``H_0: <target> <= null_value``; the p-value is the
                right-tail worst-case tail, and every confidence set becomes
                one-sided ``[lb, +inf)`` (lower bound finite, upper ``+inf``).
              - "less": ``H_0: <target> >= null_value``; the p-value is the
                left-tail worst-case tail, and every confidence set becomes
                one-sided ``(-inf, ub]`` (upper bound finite, lower ``-inf``).

            Defaults to "two-sided".
         method : {'exact', 'normal', 'auto'}
            How the confidence sets are inverted, forwarded to
            :func:`attributable_effect_interval`: ``'exact'`` by binary search over
            exact binomial tails, ``'normal'`` by the closed-form large- sample
            approximation, or ``'auto'`` (default) to choose from the sample size. Only
            the confidence sets honor this -- the p-value and ``Γ•`` are always exact
            (they need no inversion, so exactness is free). Defaults to ``'auto'``,
            matching :meth:`expanded_confidence_interval` and :meth:`gamma_star`.
         options : PairedOutcomeAnalysisOptions, optional
            Display options; defaults to :class:`PairedOutcomeAnalysisOptions`.

        """
        _validate_alternative(alternative)
        _validate_method(method)
        if options is None:
            options = PairedOutcomeAnalysisOptions()
        if (
            EffectSize.COST_PER_ISUCCESS in options.effect_size_columns
            and self.spend is None
        ):
            raise ValueError(
                "the 'cost_per_isuccess' effect size needs a cost; set "
                "`PairedOutcomeTable.spend` (the total cost of treatment)."
            )
        confidence = 1.0 - alpha
        attributable_interval = attributable_effect_interval(
            self,
            target=target,
            confidence=confidence,
            gamma=gamma,
            monotonic=monotonic,
            alternative=alternative,
            method=method,
        )
        # The scaled effect is the `target` effect divided by n_pairs: A_1 /
        # n_pairs is the ATT (effect on the treated), A_0 / n_pairs the ATU
        # (effect on the untreated), and for target="ATE" the attributable
        # interval is already the average (A_1 + A_0) / 2 = ATE * n_pairs, so
        # dividing by n_pairs recovers the ATE set. It tracks `target` so the
        # scaled column and the iSuccesses column always describe the same
        # effect. The point estimate is `ate_hat = hat_a / n_pairs`, which
        # equals A_1 / n_pairs, A_0 / n_pairs and the ATE (the McNemar pivot
        # estimates all three). (+/- math.inf / n stays +/- math.inf, so
        # one-sided sets stay one-sided.)
        n_pairs = self.n_pairs
        effect_interval = (
            attributable_interval[0] / n_pairs,
            attributable_interval[1] / n_pairs,
        )
        return PairedOutcomeAnalysis(
            table=self,
            alpha=alpha,
            gamma=gamma,
            monotonic=monotonic,
            null_value=null_value,
            alternative=alternative,
            target=target,
            method=method,
            effect=self.ate_hat,
            effect_interval=effect_interval,
            attributable=self.hat_a,
            attributable_interval=attributable_interval,
            p_value=worst_case_pvalue(
                self,
                target=target,
                null_value=null_value,
                gamma=gamma,
                monotonic=monotonic,
                alternative=alternative,
            ),
            gamma_star=sensitivity_value(
                self,
                alpha=alpha,
                target=target,
                null_value=null_value,
                monotonic=monotonic,
                alternative=alternative,
            ),
            options=options,
        )

    def confidence_interval(
        self,
        *,
        alpha: float = 0.10,
        alternative: str = "two-sided",
        gamma: float = 1.0,
        monotonic: bool = False,
        method: str = "exact",
    ) -> tuple[float, float]:
        r"""Confidence interval for the ATT.

        The ATT is the treated-side attributable effect ``A_1`` scaled by ``1 /
        n_pairs``: this returns :func:`attributable_effect_interval` for ``A_1`` at
        coverage ``1 - alpha`` under a hidden bias of odds ratio ``gamma``, divided by
        the pair count. It is the matched-pair binary analog of
        ``TreatmentEffectEstimator.confidence_interval`` and agrees with
        :meth:`analyze`'s ``effect_interval`` when ``analyze`` targets the treated side
        (its default); other targets scale a different attributable set, so their scaled
        interval differs.

        Parameters
        ----------
         alpha : float
            Significance level; the interval has coverage ``1 - alpha`` (default 0.10, a
            90% interval).
         alternative : {'two-sided', 'less', 'greater'}
            The kind of interval:

              - "two-sided": both bounds finite, each side at ``alpha / 2``.
              - "greater": ``[lb, +inf)`` -- ``lb`` finite (right-tail test
                inverted at full ``alpha``), ``ub = +inf``.
              - "less": ``(-inf, ub]`` -- ``ub`` finite (left-tail test inverted
                at full ``alpha``), ``lb = -inf``.

            Defaults to "two-sided".
         gamma : float
            Rosenbaum sensitivity parameter (``>= 1``; default 1.0). ``1`` corresponds
            to a randomized experiment; larger widens the interval.
         monotonic : bool
            Assume treatment never hurts (no prevention); narrows the interval. Default
            ``False``.
         method : {'exact', 'normal', 'auto'}
            How the worst-case test is inverted (see
            :func:`attributable_effect_interval`). ``'exact'`` (default) uses binary
            search over exact binomial tails; ``'normal'`` the closed-form large-sample
            approximation; ``'auto'`` chooses from the sample size.

        Returns
        -------
         (float, float)
            Lower and upper bounds on the ATT. For one-sided intervals only one bound is
            finite (the other is ``+/- math.inf``), per ``alternative``.

        """
        _validate_alternative(alternative)
        lo, hi = attributable_effect_interval(
            self,
            target="ATT",
            confidence=1.0 - alpha,
            gamma=gamma,
            monotonic=monotonic,
            alternative=alternative,
            method=method,
        )
        n_pairs = self.n_pairs
        return (lo / n_pairs, hi / n_pairs)

    def point_estimate(self) -> float:
        """ATE point estimate; the location the sensitivity bands widen around."""
        return self.ate_hat

    def capacity(self, alpha: float = 0.05) -> float:
        r"""Design-sensitivity ceiling over the discordant pairs.

        Beyond this ``Gamma`` no discordancy pattern is significant at level ``alpha``.
        Only the ``m = S_01 + S_10`` discordant pairs carry sign information, so the
        ceiling ``((1/alpha)^(1/m) - 1)^{-1}`` uses ``m`` rather than the full pair
        count -- the McNemar analog of :meth:`PairedEstimator.capacity`. Returns ``1.0``
        when there are no discordant pairs (the study is uninformative).

        """
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must lie in (0, 1), got {alpha}.")
        m = self.s01 + self.s10
        if m == 0:
            return 1.0
        return 1.0 / ((1.0 / alpha) ** (1.0 / m) - 1.0)

    def sensitivity_analysis(
        self, gamma: float = 6.0, monotonic: bool = False
    ) -> tuple[float, float]:
        r"""Confounding-only ATE interval: the estimate range implied by bias ``Gamma``.

        The range the ATE estimate could take under a hidden bias of odds ratio
        ``gamma`` from confounding *alone* (no sampling uncertainty) -- the binary
        analog of :meth:`PairedEstimator.sensitivity_analysis`. Collapses to ``ate_hat``
        at ``gamma == 1`` and opens with ``gamma``, saturating at the ATE's a-priori
        range ``[-(S00 + 2 S01 + S11), S00 + 2 S10 + S11] / (2 S)`` (and so never
        leaving ``[-1, 1]``) rather than diverging. See :func:`_ate_sensitivity_band`.

        With ``monotonic`` (treatment never hurts, so the box ceilings shrink; see
        :func:`_ceilings`) the band narrows, and its lower endpoints saturate at the
        a-priori minimum ``0`` once ``gamma >= S10 / S01``.

        """
        if gamma < 1.0:
            raise ValueError(f"gamma must be >= 1, got {gamma}.")
        return _ate_sensitivity_band(self, gamma=gamma, monotonic=monotonic)

    def expanded_confidence_interval(
        self,
        alpha: float = 0.10,
        gamma: float = 6.0,
        method: str = "exact",
        monotonic: bool = False,
    ) -> tuple[float, float]:
        r"""Sensitivity/confidence ATE set: sampling *and* confounding uncertainty.

        The Rigdon-Hudgens confidence set for the ATE at coverage ``1 - alpha`` under a
        hidden bias of odds ratio ``gamma`` -- the binary analog of
        :meth:`PairedEstimator.expanded_confidence_interval`. At ``gamma == 1`` it is
        the randomized ``1 - alpha`` set; larger ``gamma`` widens it.

        ``method`` selects how the worst-case tests are inverted: ``'exact'`` (default)
        by binary search over exact binomial tails, ``'normal'`` by the closed-form
        large-sample approximation (O(1) per endpoint), or ``'auto'`` chosen from the
        sample size (see :func:`attributable_effect_interval`).

        With ``monotonic`` (treatment never hurts) the box ceilings shrink (see
        :func:`_ceilings`), narrowing the set.

        """
        if gamma < 1.0:
            raise ValueError(f"gamma must be >= 1, got {gamma}.")
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must lie in (0, 1), got {alpha}.")
        _validate_method(method)
        _, _, ate_interval = _ate_confidence_core(
            self,
            confidence=1.0 - alpha,
            gamma=gamma,
            method=method,
            monotonic=monotonic,
        )
        return ate_interval

    def gamma_star(
        self,
        null_value: float = 0.0,
        alpha: float = 0.05,
        monotonic: bool = False,
        method: str = "exact",
    ) -> float:
        r"""Rosenbaum sensitivity value ``Γ•`` for the ATE finding.

        The largest ``Gamma`` at which the expanded (Rigdon-Hudgens)
        sensitivity/confidence interval for the ATE at level ``alpha`` still excludes
        ``null_value`` -- the point on a :meth:`plot_sensitivity` sweep where the wider
        band first touches the null and the finding stops being significant. It
        summarizes exactly the set :meth:`expanded_confidence_interval` returns -- both
        attributable effects ``A_1`` and ``A_0`` combined through ``A_1 + A_0 = 2 S *
        ATE`` under the Bonferroni budget -- so the dotted ``Γ•`` line and the band
        cross the null together. This is a strictly smaller (more conservative) value
        than inverting the ``A_1``-only worst-case McNemar p-value alone
        (:func:`sensitivity_value`), which tests a narrower hypothesis than the interval
        it accompanies.

        ``null_value`` is on the ATE scale -- the reference line the band is tested
        against (default ``0``, the no-effect null). A value near ``1`` means the
        finding is fragile; a large value means it is robust to substantial hidden bias.

        With ``monotonic`` (treatment never hurts) the box ceilings shrink (see
        :func:`_ceilings`), which narrows the band and raises ``Γ•``.

        Parameters
        ----------
         null_value : float
            ATE null the expanded band is tested against (default 0.0).
         alpha : float
            Significance level; the band has coverage ``1 - alpha`` (default 0.05).
         monotonic : bool
            Assume treatment never hurts (no prevention); raises ``Γ•``. Default
            ``False``.
         method : {'exact', 'normal', 'auto'}
            How the worst-case tests behind the band are inverted, forwarded to
            :meth:`expanded_confidence_interval` (default ``'exact'``).

        Returns
        -------
         float
            ``Γ• >= 1``. Returns ``1.0`` when the band already contains ``null_value``
            in the randomized case, and ``math.inf`` when it excludes ``null_value`` for
            arbitrarily large ``Gamma``.

        """
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must lie in (0, 1), got {alpha}.")
        _validate_method(method)

        def excludes(gamma: float) -> bool:
            lower, upper = self.expanded_confidence_interval(
                alpha, gamma, method=method, monotonic=monotonic
            )
            return lower > null_value or upper < null_value

        return _gamma_star_search(excludes)

    def plot_sensitivity(
        self,
        *,
        target: str = "ATT",
        null_value: float = 0.0,
        alpha: float = 0.10,
        gamma_max: float | None = None,
        num_points: int = 50,
        method: str = "auto",
        monotonic: bool = False,
        legend_loc: LegendLocType = "lower left",
        title: str | None = None,
        ax: Axes | None = None,
    ) -> tuple[pd.DataFrame, Axes]:
        r"""Sweep the sensitivity parameter and plot how the finding degrades.

        The headline inference-stage diagnostic for the binary (McNemar / net-effects)
        path: as the hidden-bias odds ratio ``Gamma`` grows from ``1`` (a randomized
        experiment) upward, two intervals widen around the (bias-independent) point
        estimate for the chosen ``target`` --

        - the *sensitivity interval*, the range of the effect from confounding
          alone (:meth:`sensitivity_analysis` for the ATE, the ``A_1`` / ``A_0``
          analog otherwise); and
        - the *sensitivity/confidence interval*, which adds sampling uncertainty
          (:meth:`expanded_confidence_interval` for the ATE,
          :func:`attributable_effect_interval` otherwise).

        The left axis is the scaled effect (the ATT for ``A_1``, the ATU for ``A_0``,
        the ATE for ``ATE``); a secondary right axis rescales it to the matching count
        of induced successes (``iSuccesses = effect * n_pairs``), so both the rate and
        the count can be read off the same curves.

        The study's sensitivity value ``Γ•`` -- where the wider interval first touches
        ``null_value`` and the finding stops being significant -- inverts the plotted
        confidence band for ``target`` and is marked when it falls within the swept
        range.

        Parameters
        ----------
         target : {'ATT', 'ATU', 'ATE'}
            Which effect to sweep: ``A_1`` (net effect on the treated, plotted as the
            ATT), ``A_0`` (net effect on the control, the ATU), or ``ATE`` (the
            Rigdon-Hudgens average effect). Defaults to ``'ATT'`` for consistency with
            :meth:`analyze`.
         null_value : float
            The null the wider band is tested against, on the left-axis (scaled-effect)
            scale; ``Γ•`` is computed against it and marked with a vertical line when it
            falls within the swept range.
         alpha : float
            Significance level; the wider band has coverage ``1 - alpha``.
         gamma_max : float, optional
            Largest ``Gamma`` to sweep to. Defaults to ``min(6, 0.95 * capacity)`` (6 is
            the smoking / lung-cancer benchmark; the cap keeps the intervals finite
            below the study's capacity).
         num_points : int
            Number of ``Gamma`` values swept (``>= 2``).
         method : str
            Inference method forwarded to :meth:`expanded_confidence_interval`
            (``'exact'`` / ``'normal'`` / ``'auto'``).
         monotonic : bool
            When ``True`` (treatment never hurts), both bands and ``Γ•`` use the
            shrunken monotonic box ceilings, narrowing the plotted intervals.
         legend_loc : str
            Matplotlib legend location (e.g. ``'lower left'``, ``'lower right'``); use
            it to keep the legend clear of the ``Γ•`` annotation (default ``'lower
            left'``).
         title : str, optional
            Plot title; no title is drawn when omitted.
         ax : Axes, optional
            Axes to draw on; a new figure and axes are created when omitted.

        Returns
        -------
         tuple of (DataFrame, Axes)
            The swept data (columns ``gamma``, ``point``, ``sens_lower``,
            ``sens_upper``, ``ci_lower``, ``ci_upper``) and the axes drawn on.

        """
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must lie in (0, 1), got {alpha}.")
        _validate_target(target)
        if num_points < 2:
            raise ValueError(f"num_points must be at least 2, got {num_points}.")
        gamma_max = _resolve_gamma_max(
            gamma_max, capacity=lambda: self.capacity(alpha), subject="study"
        )
        point = self.point_estimate()
        n_pairs = self.n_pairs

        # Both bands dispatch on `target`: the ATE reuses the Rigdon-Hudgens
        # combined machinery; A_1 / A_0 use the single-effect band and the
        # attributable-effect set, each on the scaled `effect / n_pairs` axis.
        def sens_band(gamma: float) -> tuple[float, float]:
            if target == "ATE":
                return self.sensitivity_analysis(gamma, monotonic=monotonic)
            return _attributable_sensitivity_band(
                self, target=target, gamma=gamma, monotonic=monotonic
            )

        def ci_band(gamma: float) -> tuple[float, float]:
            if target == "ATE":
                return self.expanded_confidence_interval(
                    alpha, gamma, method=method, monotonic=monotonic
                )
            if n_pairs == 0:
                # Mirror `_attributable_sensitivity_band`'s empty-table guard:
                # the scaled effect is undefined with no pairs, so avoid the
                # division by zero rather than emit inf/nan.
                return (0.0, 0.0)
            lo, hi = attributable_effect_interval(
                self,
                target=target,
                confidence=1.0 - alpha,
                gamma=gamma,
                monotonic=monotonic,
                method=method,
            )
            return (lo / n_pairs, hi / n_pairs)

        data = _sweep_sensitivity_bands(
            point=point,
            gamma_max=gamma_max,
            num_points=num_points,
            sens_band=sens_band,
            ci_band=ci_band,
        )

        # Γ• always inverts the plotted confidence band, so the dotted line and
        # the drawn band cross the null together for every target.
        def excludes(gamma: float) -> bool:
            lower, upper = ci_band(gamma)
            return lower > null_value or upper < null_value

        gamma_star = _gamma_star_search(excludes)
        ax = _plot_sensitivity_curve(
            data,
            point=point,
            gamma_star=gamma_star,
            alpha=alpha,
            gamma_max=gamma_max,
            ylabel=_SCALED_EFFECT_HEADERS[target],
            secondary_scale=float(n_pairs),
            secondary_ylabel="iSuccesses",
            legend_loc=legend_loc,
            title=title,
            ax=ax,
        )
        return data, ax


def _ceilings(
    table: PairedOutcomeTable, target: str, monotonic: bool = False
) -> tuple[int, int]:
    r"""Box ceilings ``(c0, c1)`` on the discordancy counts for a target.

    The ``A_1`` and ``A_0`` procedures are identical except for these ceilings: ``A_0``
    swaps the treated marginal for the control marginal.

    Under ``monotonicity`` (treatment never hurts, so ``m00 = 0`` and ``m01 = S01``),
    the prevention pool vanishes and the ``c0`` ceiling on the worst-case discordancy
    collapses to the observed discordant count ``S10`` for both targets -- ``A_1`` drops
    the ``S00`` pool, ``A_0`` the ``S11`` pool. The ``c1`` ceiling is unchanged (its
    induced floor ``S01`` never binds the supremum). This tightens every set/test.

    """
    if target == "ATT":
        c0 = table.s10 if monotonic else table.s00 + table.s10
        return c0, table.s01 + table.s11
    if target == "ATU":
        c0 = table.s10 if monotonic else table.s10 + table.s11
        return c0, table.s00 + table.s01
    raise ValueError(f"`target` must be 'ATT' or 'ATU', got {target!r}.")


def _search_lower(
    pvalue: Callable[[int], float], a_lo: int, a_hi: int, alpha: float
) -> int:
    """Smallest integer in [a_lo, a_hi] whose (increasing) p-value > alpha."""
    if a_lo > a_hi:
        return a_hi + 1
    if pvalue(a_lo) > alpha:
        return a_lo
    lo, hi = a_lo, a_hi
    while lo < hi:
        mid = (lo + hi) // 2
        if pvalue(mid) > alpha:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _gamma_star_search(excludes: Callable[[float], bool]) -> float:
    r"""Largest ``Gamma`` for which ``excludes(Gamma)`` still holds (a ``Γ•``).

    ``excludes`` reports whether the sensitivity/confidence band still excludes the null
    at a given ``Gamma``; it is assumed monotone (once the widening band admits the null
    it never re-excludes it). Doubles ``hi`` to bracket the crossing, then bisects.
    Returns ``1.0`` when the null is already admitted in the randomized case, and
    ``math.inf`` when it is excluded past the search cap.

    """
    if not excludes(1.0):
        return 1.0
    lo, hi = 1.0, 2.0
    while excludes(hi) and hi < 1e6:
        hi *= 2.0
    # The band excludes the null even at the search cap: report an unbounded
    # sensitivity value rather than a spurious number near the cap.
    if hi >= 1e6 and excludes(hi):
        return math.inf
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if excludes(mid):
            lo = mid
        else:
            hi = mid
    return lo


def _search_upper(
    pvalue: Callable[[int], float], a_lo: int, a_hi: int, alpha: float
) -> int:
    """Largest integer in [a_lo, a_hi] whose (decreasing) p-value > alpha."""
    if a_lo > a_hi:
        return a_lo - 1
    if pvalue(a_hi) > alpha:
        return a_hi
    lo, hi = a_lo, a_hi
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if pvalue(mid) > alpha:
            lo = mid
        else:
            hi = mid - 1
    return lo


_METHODS = ("exact", "normal", "auto")

# The Gaussian prediction-set formula is trustworthy only when the worst-case
# McNemar tail rests on many trials. ``"auto"`` switches to the exact binomial
# inversion once the binding consistency face carries too little mass -- the
# usual ``n * pi * (1 - pi) >= 10`` normal-approximation rule of thumb. Because
# ``pi -> 1`` as ``Gamma`` grows, this guard self-reverts to exact at the
# high-``Gamma`` tail of a sensitivity sweep, where the normal frays.
_NORMAL_MIN_CELL = 10.0


def _validate_method(method: str) -> None:
    """Raise ``ValueError`` unless ``method`` is a recognized inference mode."""
    if method not in _METHODS:
        raise ValueError(f"`method` must be one of {_METHODS}, got {method!r}.")


def _resolve_method(method: str, c0: int, c1: int, gamma: float) -> str:
    r"""Resolve ``"auto"`` to ``"exact"`` or ``"normal"`` from the sample size.

    ``"exact"`` and ``"normal"`` pass through. ``"auto"`` chooses ``"normal"`` when the
    binding consistency face carries enough effective mass, ``2 * min(c0, c1) * pi * (1
    - pi) >= _NORMAL_MIN_CELL`` with ``pi = gamma / (gamma + 1)`` (the endpoint sits on
    ``n ~ 2 * min(c0, c1)`` trials), and ``"exact"`` otherwise.

    """
    _validate_method(method)
    if method != "auto":
        return method
    pi = gamma / (gamma + 1.0)
    c_bind = min(c0, c1)
    if 2.0 * c_bind * pi * (1.0 - pi) >= _NORMAL_MIN_CELL:
        return "normal"
    return "exact"


def _positive_quadratic_root(a: float, b: float, c: float) -> float | None:
    r"""Larger real root of ``a x^2 + b x + c = 0`` when it is nonnegative.

    Returns ``None`` when the roots are complex or the larger root is negative. ``a >
    0`` on both consistency faces (the leading coefficient is a squared probability), so
    the larger root is ``(-b + sqrt(disc)) / (2 a)`` -- the branch corresponding to ``Z
    = +z`` rather than the spurious ``Z = -z`` root introduced by squaring.

    """
    if a <= 0.0:
        return None
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    root = (-b + math.sqrt(disc)) / (2.0 * a)
    if root < 0.0:
        return None
    return root


def _two_face_root(z: float, c_fixed: int, c_sliding: int, pi: float) -> float:
    r"""Self-consistent gap ``delta*`` solving the standardized test ``Z = z``.

    The worst-case McNemar statistic standardizes against ``Binom(n, pi)`` with ``n``
    and the statistic set by which face of the consistency box binds. With ``pi = Gamma
    / (Gamma + 1)`` this is the Gamma-sensitivity generalization of the large-sample
    intervals (net_effects.tex, "Large-sample sensitivity intervals": the
    standardization eqn:Z_sensitivity and the two kernels eqn:kernel_M / eqn:kernel_R,
    tabulated in tbl:sensitivity_large_sample); at ``pi = 1/2`` it reduces to the
    equal-selection per-face roots eqn:root_top / eqn:root_right:

      - "fixed" face -- the smaller count is pinned at ``c_fixed`` and the
        statistic is ``c_fixed + delta``, so ``n = 2 c_fixed + delta`` (valid
        while ``delta <= c_sliding - c_fixed``); and
      - "sliding" face -- the statistic is pinned at ``c_sliding`` and
        ``n = 2 c_sliding - delta`` (valid above the crossover).

    ``Z`` is continuous and strictly increasing in ``delta`` with only a slope change at
    the crossover ``delta = c_sliding - c_fixed``, so exactly one root is
    self-consistent. Guarding on this crossover -- rather than always using ``min(c0,
    c1)`` -- is essential: for imbalanced counts the root can sit on the larger-ceiling
    face.

    A 1/2 continuity correction is applied (the statistic enters as ``a - 1/2``, since
    ``P(X >= a) = P(X > a - 1)`` for integer-valued ``X``), which shifts the numerator
    constant by ``-1/2`` but leaves the variance ``n`` untouched. Without it (the
    paper's asymptotic form) ``pi = 1/2`` would reduce to ``delta* = +/- z^2 / 2 +
    sqrt(2 C z^2 + z^4 / 4)``; the correction widens the resulting set by ~``(Gamma + 1)
    / 2`` counts, tracking the exact binomial test more closely at finite samples.

    """
    q = pi * (1.0 - pi)
    z2q = z * z * q
    # Fixed face: statistic a* = c_fixed + delta, n = 2 c_fixed + delta. The
    # ``- 0.5`` on the numerator constant is the continuity correction; the
    # variance term ``2 c_fixed z2q`` (from n) is left uncorrected.
    one_minus_pi = 1.0 - pi
    p_fixed = c_fixed * (1.0 - 2.0 * pi) - 0.5
    delta_fixed = _positive_quadratic_root(
        one_minus_pi * one_minus_pi,
        2.0 * p_fixed * one_minus_pi - z2q,
        p_fixed * p_fixed - 2.0 * c_fixed * z2q,
    )
    # Sliding face: statistic pinned at c_sliding, n = 2 c_sliding - delta.
    p_slide = c_sliding * (1.0 - 2.0 * pi) - 0.5
    delta_slide = _positive_quadratic_root(
        pi * pi,
        2.0 * p_slide * pi + z2q,
        p_slide * p_slide - 2.0 * c_sliding * z2q,
    )
    crossover = c_sliding - c_fixed
    if delta_fixed is not None and delta_fixed <= crossover:
        return delta_fixed
    if delta_slide is not None:
        return delta_slide
    return delta_fixed if delta_fixed is not None else 0.0


def _z_lower(delta: int, c0: int, c1: int, pi: float) -> float:
    r"""Normal-standardized worst-case statistic for the right-tailed lower test.

    Carries the 1/2 continuity correction (``a - 1/2``) matching :func:`_two_face_root`,
    so the local refinement targets the same endpoint the closed-form root seeds.

    """
    b = min(c1, c0 - delta)
    a = b + delta
    n = a + b
    if n <= 0:
        return math.inf
    return (a - 0.5 - n * pi) / math.sqrt(n * pi * (1.0 - pi))


def _z_upper(delta: int, c0: int, c1: int, pi: float) -> float:
    r"""Normal-standardized statistic for the upper (mirror) test.

    The left-tailed test at ``1 / (Gamma + 1)`` equals a right-tailed test on the
    complementary count at ``pi = Gamma / (Gamma + 1)`` (``P(Binom(n, 1 - pi) <= a) =
    P(Binom(n, pi) >= n - a)``), so the mirror reuses ``pi`` with the box ceilings
    swapped. Carries the same 1/2 continuity correction, here on the complementary count
    ``b``.

    """
    a = min(c0, c1 - delta)
    b = a + delta
    n = a + b
    if n <= 0:
        return math.inf
    return (b - 0.5 - n * pi) / math.sqrt(n * pi * (1.0 - pi))


def _largest_unrejected(
    z_of: Callable[[int], float], root: float, ceiling: int, z: float
) -> int:
    r"""Largest integer gap ``<= ceiling`` the normal test does not reject.

    The test rejects at gap ``delta`` when ``delta >= 2`` (outside the width-two
    no-information band) and ``z_of(delta) >= z``. Rejection is monotone in ``delta``,
    so the unrejected gaps form ``[1, d]``; ``d`` is found by a constant-size local
    search seeded from the closed-form ``root``. The local step settles the rare case
    where flooring ``root`` straddles the face crossover (paper, S5).

    """

    def rejects(delta: int) -> bool:
        if delta < 2:
            return False
        if delta > ceiling:
            return True
        return z_of(delta) >= z

    d = max(1, min(ceiling, math.floor(root)))
    while d > 1 and rejects(d):
        d -= 1
    while d < ceiling and not rejects(d + 1):
        d += 1
    return d


def _assemble_interval(
    lower_bound: Callable[[float], int],
    upper_bound: Callable[[float], int],
    alpha: float,
    alternative: str,
) -> tuple[float, float]:
    r"""Combine one-sided bound-finders per the ``alternative`` convention.

    Shared by the exact and normal inversions: "two-sided" spends ``alpha / 2`` on each
    finite bound; "greater" spends the full ``alpha`` on the lower bound (upper
    ``+inf``); "less" the full ``alpha`` on the upper bound (lower ``-inf``).

    """
    if alternative == "two-sided":
        alpha_side = alpha / 2.0
        return lower_bound(alpha_side), upper_bound(alpha_side)
    if alternative == "greater":
        return lower_bound(alpha), math.inf
    if alternative == "less":
        return -math.inf, upper_bound(alpha)
    raise ValueError(
        f"`alternative` must be one of {_ALTERNATIVES}, got {alternative!r}."
    )


def _normal_worst_case_interval(
    hat_a: int,
    c0: int,
    c1: int,
    alpha: float,
    gamma: float,
    alternative: str = "two-sided",
) -> tuple[float, float]:
    r"""Large-sample (Gaussian) prediction set; the O(1) analog of the bisection.

    Inverts the normal approximation to the worst-case McNemar test in closed form
    (net_effects.tex, "Large-sample sensitivity intervals" and
    tbl:sensitivity_large_sample, plus a 1/2 continuity correction) instead of by binary
    search; at ``gamma == 1`` this is the equal-selection case (S5, tbl:large_sample).
    Endpoints are ``L = hat_a - floor(delta*_lower)`` and ``U = hat_a +
    floor(delta*_upper)``, each ``delta*`` the self-consistent two-face root of
    :func:`_two_face_root`, refined to the exact integer boundary by
    :func:`_largest_unrejected`. The ``alternative`` convention matches
    :func:`_worst_case_interval`.

    """
    pi = gamma / (gamma + 1.0)

    def lower_bound(level: float) -> int:
        if c0 == 0:
            return hat_a
        z = float(norm.ppf(1.0 - level))
        root = _two_face_root(z, c1, c0, pi)
        d = _largest_unrejected(lambda delta: _z_lower(delta, c0, c1, pi), root, c0, z)
        return hat_a - d

    def upper_bound(level: float) -> int:
        if c1 == 0:
            return hat_a
        z = float(norm.ppf(1.0 - level))
        root = _two_face_root(z, c0, c1, pi)
        d = _largest_unrejected(lambda delta: _z_upper(delta, c0, c1, pi), root, c1, z)
        return hat_a + d

    return _assemble_interval(lower_bound, upper_bound, alpha, alternative)


def _exact_worst_case_interval(
    hat_a: int,
    c0: int,
    c1: int,
    alpha: float,
    gamma: float,
    alternative: str = "two-sided",
) -> tuple[float, float]:
    r"""Exact worst-case prediction set, inverted by binary search.

    Inverts the worst-case (over compatible sharp nulls) McNemar test on each side by
    binary search over exact binomial tails. Under a sensitivity parameter ``gamma`` the
    success probability moves from 1/2 to ``gamma / (gamma + 1)`` for the right-tailed
    test and to ``1 / (gamma + 1)`` for the left-tailed test.

    """
    pi_right = gamma / (gamma + 1.0)
    pi_left = 1.0 / (gamma + 1.0)

    def p_lower(a0: int) -> float:
        delta = hat_a - a0
        if delta < 2:
            return 1.0  # no-information band: cannot reject
        b = min(c1, c0 - delta)
        a = b + delta
        return _p_greater(a, b, pi_right)

    def p_upper(a0: int) -> float:
        delta = hat_a - a0
        if delta > -2:
            return 1.0  # no-information band: cannot reject
        a = min(c0, c1 + delta)
        b = a - delta
        return _p_less(a, b, pi_left)

    # With an empty marginal on a side there is no discordancy budget there, so
    # no value past ``hat_a`` can be rejected: the bound collapses to ``hat_a``
    # rather than the inverted empty search range ``(hat_a, hat_a - 1)``.
    def lower_bound(level: float) -> int:
        if c0 == 0:
            return hat_a
        return _search_lower(p_lower, hat_a - c0, hat_a - 1, level)

    def upper_bound(level: float) -> int:
        if c1 == 0:
            return hat_a
        return _search_upper(p_upper, hat_a + 1, hat_a + c1, level)

    return _assemble_interval(lower_bound, upper_bound, alpha, alternative)


def _worst_case_interval(
    hat_a: int,
    c0: int,
    c1: int,
    alpha: float,
    gamma: float,
    alternative: str = "two-sided",
    method: str = "exact",
) -> tuple[float, float]:
    r"""Worst-case prediction set for an attributable effect.

    Inverts the worst-case (over compatible sharp nulls) McNemar test on each side. The
    worst-case p-value sits at a single boundary corner of the consistency rectangle
    (eqn:worst); under a sensitivity parameter ``gamma`` the success probability moves
    from 1/2 to ``gamma / (gamma + 1)`` for the right-tailed test and to ``1 / (gamma +
    1)`` for the left-tailed test.

    With ``method="exact"`` the test is inverted by binary search over exact binomial
    tails. With ``method="normal"`` the closed-form Gaussian approximation of
    :func:`_normal_worst_case_interval` is used instead (O(1) per endpoint, for large
    samples). ``method="auto"`` picks between them from the sample size via
    :func:`_resolve_method`.

    The ``alternative`` controls which bounds are informative:
      - "two-sided": both bounds finite, each inverted at level ``alpha / 2``;
        the returned endpoints are integers.
      - "greater": ``[lower, +inf)`` -- the (right-tail) lower bound inverted at
        the full ``alpha``, upper endpoint ``math.inf``.
      - "less": ``(-inf, upper]`` -- the (left-tail) upper bound inverted at the
        full ``alpha``, lower endpoint ``-math.inf``.

    """
    resolved = _resolve_method(method, c0, c1, gamma)
    if resolved == "normal":
        return _normal_worst_case_interval(hat_a, c0, c1, alpha, gamma, alternative)
    return _exact_worst_case_interval(hat_a, c0, c1, alpha, gamma, alternative)


def attributable_effect_interval(
    table: PairedOutcomeTable,
    *,
    target: str = "ATT",
    confidence: float = 0.90,
    gamma: float = 1.0,
    monotonic: bool = False,
    alternative: str = "two-sided",
    method: str = "exact",
) -> tuple[float, float]:
    r"""Confidence set for an attributable effect, ``A_1``, ``A_0``, or the ATE.

    Parameters
    ----------
     table : PairedOutcomeTable
        The matched-pair outcome table.
     target : {'ATT', 'ATU', 'ATE'}
        Which effect to bound. ``A_1`` is the net effect among treated units; ``A_0``
        the net effect among control units. ``ATE`` returns the Rigdon-Hudgens
        average-effect set on the *count* scale -- the average attributable effect
        ``(A_1 + A_0) / 2 = ATE * n_pairs``, the :func:`_ate_confidence_core` ATE set
        scaled back up by ``n_pairs`` (so dividing it by ``n_pairs`` recovers the ATE
        set exactly).
     confidence : float
        Coverage of the returned set (default 0.90). Two-sided, each side is tested at
        level ``(1 - confidence) / 2``; one-sided, the single informative bound is
        tested at the full ``1 - confidence``.
     gamma : float
        Rosenbaum sensitivity parameter (``>= 1``). ``1`` corresponds to a randomized
        experiment.
     monotonic : bool
        Assume treatment never hurts (no prevention). Narrows the set. The default
        (``False``) makes no monotonicity assumption -- the main point of the
        net-effects procedure.
     alternative : {'two-sided', 'less', 'greater'}
        The kind of confidence set. "two-sided" returns two finite integer endpoints;
        "greater" returns ``[lb, +inf)`` and "less" returns ``(-inf, ub]``, with the
        finite bound an integer. Defaults to "two-sided".
     method : {'exact', 'normal', 'auto'}
        How the worst-case test is inverted. ``'exact'`` (default) uses binary search
        over exact binomial tails. ``'normal'`` uses the closed-form large-sample
        Gaussian approximation (O(1) per endpoint), preferable at scale. ``'auto'``
        chooses from the sample size.

    Returns
    -------
     (float, float)
        Inclusive lower and upper endpoints of the confidence set. For the single-effect
        targets (ATT, ATU) two-sided endpoints are integer counts of induced successes;
        for the ATE target they are the averaged set ``(A_1 + A_0) / 2`` and so may be
        half-integers (the ``ISUCCESSES`` column format rounds them for display).
        One-sided sets carry ``+/- math.inf`` on the uninformative side.

    """
    _validate_alternative(alternative)
    _validate_target(target)
    if target == "ATE":
        # The ATE "iSuccesses" is the average attributable effect
        # (A_1 + A_0) / 2 = ATE * n_pairs. Its confidence set is the
        # Rigdon-Hudgens ATE set (A_1 + A_0) / (2 S) scaled back up by n_pairs;
        # analyze() divides it by n_pairs to recover the ATE set (up to IEEE-754
        # round-trip error from the scale-up/scale-down). A one-sided +/-
        # math.inf endpoint scales through unchanged.
        _, _, ate_interval = _ate_confidence_core(
            table,
            confidence=confidence,
            gamma=gamma,
            monotonic=monotonic,
            alternative=alternative,
            method=method,
        )
        n_pairs = table.n_pairs
        return (ate_interval[0] * n_pairs, ate_interval[1] * n_pairs)
    alpha = 1.0 - confidence
    c0, c1 = _ceilings(table, target, monotonic)
    return _worst_case_interval(table.hat_a, c0, c1, alpha, gamma, alternative, method)


def _ate_confidence_core(
    table: PairedOutcomeTable,
    *,
    confidence: float,
    gamma: float,
    monotonic: bool = False,
    alternative: str = "two-sided",
    method: str = "exact",
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    r"""The Rigdon-Hudgens core: the ``A_1``, ``A_0``, and ATE confidence sets.

    Builds prediction sets for ``A_1`` and ``A_0`` and combines them through the linear
    constraint ``A_1 + A_0 = 2 S * ATE`` into a ``confidence``-level set for the ATE.
    Returns ``(a1, a0, ate)``. With ``monotonic`` the box ceilings shrink (see
    :func:`_ceilings`), narrowing all three sets.

    The Bonferroni budget is split across the effects and (for two-sided) the two sides:
    passing ``alpha_total / 2`` to :func:`_worst_case_interval` yields ``alpha_total /
    4`` per two-sided side (2 effects x 2 sides) and ``alpha_total / 2`` per informative
    one-sided bound (2 effects x 1 side). Under a one-sided ``alternative`` the
    ``A_1``/``A_0`` sets carry ``+/- math.inf`` on the uninformative side, and the ATE
    combination formula propagates that infinity to the corresponding ATE bound.

    """
    alpha_total = 1.0 - confidence
    alpha = alpha_total / 2.0

    c0_1, c1_1 = _ceilings(table, "ATT", monotonic)
    c0_0, c1_0 = _ceilings(table, "ATU", monotonic)
    l1, u1 = _worst_case_interval(
        table.hat_a, c0_1, c1_1, alpha, gamma, alternative, method
    )
    l0, u0 = _worst_case_interval(
        table.hat_a, c0_0, c1_0, alpha, gamma, alternative, method
    )

    two_s = 2 * table.n_pairs
    ate_interval = ((l1 + l0) / two_s, (u1 + u0) / two_s)
    return (l1, u1), (l0, u0), ate_interval


def _ate_sensitivity_band(
    table: PairedOutcomeTable, *, gamma: float, monotonic: bool = False
) -> tuple[float, float]:
    r"""Confounding-only (sampling-free) ATE interval at sensitivity ``gamma``.

    The range of the ATE point estimate as the hidden-bias tilt sweeps its worst-case
    patterns, with no sampling error. Equating the worst-case McNemar statistic to its
    tilted null mean gives, for each attributable effect, a spread ``delta`` solving
    ``delta = (gamma - 1) * min(c_near, c_far - delta)`` for the box ceilings ``(c_near,
    c_far)`` from :func:`_ceilings`. The fixed point is the smaller of the two roots,

       delta = min((gamma - 1) c_near, (1 - 1 / gamma) c_far):

    below ``gamma = c_far / c_near`` the first (linear) branch binds and the endpoint
    opens linearly; above it the min switches and ``delta`` saturates at ``c_far`` as
    ``gamma -> inf`` (each attributable effect saturating at its own a-priori bound,
    ``A_1`` at ``[-f_T, s_T]`` and ``A_0`` at its mirror). The ``A_1`` and ``A_0``
    spreads combine through ``A_1 + A_0 = 2 S * ATE``, so the ATE band saturates at
    ``[-(S00 + 2 S01 + S11), S00 + 2 S10 + S11] / (2 S)`` (hence never leaves ``[-1,
    1]``) rather than diverging linearly. At ``gamma == 1`` both spreads vanish and the
    interval collapses to ``ate_hat``.

    Notes
    -----
    The earlier form ``A_hat -/+ c (gamma - 1)`` (net_effects.tex, "Hodges- Lehmann
    point estimate under sensitivity") is the leading, moderate-``gamma`` branch of this
    exact expression.

    """
    # An empty table has no defined ATE (0 / 0); match the success-rate
    # convention (return 0.0) and avoid dividing by ``two_s == 0`` below. This
    # also skips ``table.ate_hat``, which would itself divide by zero.
    if table.n_pairs == 0:
        return (0.0, 0.0)
    c0_1, c1_1 = _ceilings(table, "ATT", monotonic)
    c0_0, c1_0 = _ceilings(table, "ATU", monotonic)
    two_s = 2 * table.n_pairs

    def _spread(c_near: int, c_far: int) -> float:
        # Fixed point of delta = (gamma - 1) * min(c_near, c_far - delta): the
        # linear root (gamma - 1) c_near while the box is slack, switching to the
        # saturating root (1 - 1 / gamma) c_far -> c_far once the box binds.
        return min((gamma - 1.0) * c_near, (1.0 - 1.0 / gamma) * c_far)

    lower = table.ate_hat - (_spread(c1_1, c0_1) + _spread(c1_0, c0_0)) / two_s
    upper = table.ate_hat + (_spread(c0_1, c1_1) + _spread(c0_0, c1_0)) / two_s
    return lower, upper


def _attributable_sensitivity_band(
    table: PairedOutcomeTable,
    *,
    target: str,
    gamma: float,
    monotonic: bool = False,
) -> tuple[float, float]:
    r"""Confounding-only band for a single attributable effect (``A_1``/``A_0``).

    The ``A_1`` (or ``A_0``) analog of :func:`_ate_sensitivity_band`, on the scaled
    ``target / n_pairs`` axis -- the ATT for ``A_1``, the ATU for ``A_0``. Each endpoint
    opens by the same worst-case spread ``delta = min((gamma - 1) c_near, (1 - 1 /
    gamma) c_far)`` used there, but divided by ``n_pairs`` rather than ``2 S`` (a single
    effect, not the two-effect average). Collapses to ``ate_hat`` at ``gamma == 1`` and
    saturates at the effect's a-priori range as ``gamma -> inf``. The point estimate is
    the (target-invariant) McNemar pivot ``ate_hat``.

    """
    if table.n_pairs == 0:
        return (0.0, 0.0)
    c0, c1 = _ceilings(table, target, monotonic)
    n_pairs = table.n_pairs

    def _spread(c_near: int, c_far: int) -> float:
        # Fixed point of delta = (gamma - 1) * min(c_near, c_far - delta); see
        # _ate_sensitivity_band for the derivation.
        return min((gamma - 1.0) * c_near, (1.0 - 1.0 / gamma) * c_far)

    lower = table.ate_hat - _spread(c1, c0) / n_pairs
    upper = table.ate_hat + _spread(c0, c1) / n_pairs
    return lower, upper


def att_confidence_set(
    table: PairedOutcomeTable,
    *,
    confidence: float = 0.90,
    gamma: float = 1.0,
    monotonic: bool = False,
    alternative: str = "two-sided",
) -> PairedOutcomeAnalysis:
    r"""Confidence set for the ATT (the treated-side attributable effect ``A_1``).

    A convenience wrapper over :meth:`PairedOutcomeTable.analyze`: it returns a full
    :class:`PairedOutcomeAnalysis` (with the default ``target='ATT'`` and
    ``null_value=0``) at significance ``alpha = 1 - confidence``. The ATT set is on the
    returned object as ``effect_interval`` -- the ``A_1`` attributable- effect set
    (``attributable_interval``) scaled by ``1 / n_pairs``, not the Rigdon-Hudgens ``(A_1
    + A_0) / (2 S)`` ATE combination.

    Because it runs the full analysis, the returned object also carries the worst-case
    p-value and the sensitivity value ``Γ•``; computing the latter is a search over
    ``gamma``, so callers that need only the interval pay for it.

    Parameters
    ----------
     table : PairedOutcomeTable
        The matched-pair outcome table.
     confidence : float
        Coverage of the returned ATT confidence set (default 0.90).
     gamma : float
        Rosenbaum sensitivity parameter (``>= 1``).
     monotonic : bool
        Assume treatment never hurts (no prevention); narrows the sets. Default
        ``False``.
     alternative : {'two-sided', 'less', 'greater'}
        The kind of confidence set. One-sided sets carry ``+/- math.inf`` on the
        uninformative side (see :meth:`PairedOutcomeTable.analyze`). Defaults to
        "two-sided".

    Returns
    -------
     PairedOutcomeAnalysis

    """
    _validate_alternative(alternative)
    return table.analyze(
        alpha=1.0 - confidence,
        gamma=gamma,
        monotonic=monotonic,
        alternative=alternative,
    )


def _one_sided_effect_tail(
    table: PairedOutcomeTable,
    *,
    target: str,
    null_value: int,
    gamma: float,
    monotonic: bool,
    side: str,
) -> float:
    r"""One-sided worst-case McNemar tail for one attributable effect, clamped.

    The right-tail (``side='greater'``) worst-case p-value for ``H_0: <target> <=
    null_value`` or the left-tail (``side='less'``) tail for ``H_0: <target> >=
    null_value``, for a single effect ``A_1`` or ``A_0``.

    Unlike :func:`worst_case_pvalue`, a ``null_value`` outside the reachable range
    ``[hat_a - c0, hat_a + c1]`` does not raise: the tail is clamped to ``1.0`` where
    the one-sided null is then certainly true (nothing to reject) and ``0.0`` where it
    is certainly false. This keeps the tail monotone in ``null_value`` across the whole
    integer line -- non-decreasing for ``side='greater'`` (a larger null is harder to
    exceed), non-increasing for ``side='less'`` -- which the ATE split search in
    :func:`_ate_worst_case_pvalue` relies on.

    """
    c0, c1 = _ceilings(table, target, monotonic)
    delta = table.hat_a - null_value
    if side == "greater":
        # null below the reachable minimum hat_a - c0: the effect exceeds it
        # for sure, so reject with certainty.
        if delta > c0:
            return 0.0
        # null at or above the estimate (delta < 2 covers the no-information
        # band and every null above the reachable maximum hat_a + c1): nothing
        # to reject on the right.
        if delta < 2:
            return 1.0
        b = min(c1, c0 - delta)
        a = b + delta
        return _p_greater(a, b, gamma / (gamma + 1.0))
    if side == "less":
        # Mirror of the above: null above the reachable maximum -> certain
        # rejection; null at or below the estimate -> nothing to reject.
        if delta < -c1:
            return 0.0
        if delta > -2:
            return 1.0
        a = min(c0, c1 + delta)
        b = a - delta
        return _p_less(a, b, 1.0 / (gamma + 1.0))
    raise ValueError(f"`side` must be 'greater' or 'less', got {side!r}.")


def _combined_tail_max(
    f: Callable[[int], float], g: Callable[[int], float], a_lo: int, a_hi: int
) -> float:
    r"""``max`` over integers in ``[a_lo, a_hi]`` of ``min(f(a), g(a))``.

    Assumes ``f`` is non-decreasing and ``g`` non-increasing, so ``min(f, g)`` is
    unimodal in ``a``: it rises with ``f`` until the two cross, then falls with ``g``.
    The peak sits at the crossover integer ``a*`` (the largest ``a`` with ``f(a) <=
    g(a)``) or its successor; both are found by a binary search on the monotone
    predicate ``f <= g`` and only those two candidates are evaluated.

    """
    if a_lo > a_hi:
        return 0.0
    if f(a_lo) > g(a_lo):
        a_star = a_lo - 1
    elif f(a_hi) <= g(a_hi):
        a_star = a_hi
    else:
        lo, hi = a_lo, a_hi
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if f(mid) <= g(mid):
                lo = mid
            else:
                hi = mid - 1
        a_star = lo
    best = 0.0
    for a in (a_star, a_star + 1):
        if a_lo <= a <= a_hi:
            best = max(best, min(f(a), g(a)))
    return best


def _ate_worst_case_pvalue(
    table: PairedOutcomeTable,
    *,
    null_value: int,
    gamma: float,
    monotonic: bool,
    alternative: str,
) -> float:
    r"""Worst-case p-value for a hypothesis about the ATE (the count-scale null).

    ``null_value`` is the average attributable effect ``(A_1 + A_0) / 2 = ATE *
    n_pairs`` (the ATE "iSuccesses"); the tested sum budget is ``T = 2 * null_value``.
    At ``gamma == 1`` this is a randomization test; larger ``gamma`` widens it, the
    McNemar analog of a Rosenbaum sensitivity tilt.

    Notes
    -----
    The ATE has no single McNemar pivot, so the test is built from the two per-effect
    worst-case tails combined under the Rigdon-Hudgens constraint ``A_1 + A_0 = 2 S *
    ATE``. For the one-sided null ``H_0: ATE <= theta_0`` (``T = 2 S * theta_0``):

      - ``ATE <= theta_0`` holds iff there EXISTS a split ``a_1 + a_0 = T`` with
        ``A_1 <= a_1`` AND ``A_0 <= a_0`` (take ``a_1 = A_1``). So the null is a
        union over splits of a conjunction of two per-effect nulls.
      - Rejecting a union means rejecting EVERY member (an intersection-union
        test, no multiplicity correction across splits): the ATE p-value is the
        supremum over splits of the per-split p-value.
      - Each per-split null is a conjunction ``(A_1 <= a_1) AND (A_0 <= a_0)``,
        rejected when EITHER one-sided tail rejects; the valid combined p-value
        is ``2 * min(p_1, p_0)`` (Bonferroni over the two effects).

    So the one-sided p-value is

       2 * max over integer splits a_1 + a_0 = T of
           min( p_1^>(a_1), p_0^>(a_0) ),

    with ``p_1^>`` / ``p_0^>`` the right-tail tails of :func:`_one_sided_effect_tail`.
    Since ``p_1^>`` is non-decreasing in ``a_1`` and ``p_0^>(T - a_1)`` non-increasing,
    the ``min`` is maximized at the split that balances the two tails -- the
    least-rejectable member of the family -- found in O(log S) by
    :func:`_combined_tail_max`. ``H_0: ATE >= theta_0`` mirrors it with the left tails;
    the two-sided p-value is twice the smaller one-sided p-value (a further factor of
    two, so ``4 * min(...)`` overall, matching the alpha / 4-per-side Bonferroni budget
    of the two-sided expanded ATE interval). Inverting this test over ``gamma`` (see
    :func:`sensitivity_value`) yields a ``Gamma`` that agrees with
    :meth:`PairedOutcomeTable.gamma_star`, which inverts the same interval.

    Raises
    ------
     ValueError
        If ``null_value`` (the average-effect count) is outside the reachable band -- no
        split of the sum budget puts both effects in their boxes, so the tail is not
        meaningful (see the single-effect guard in :func:`worst_case_pvalue`).

    """
    two_a = 2 * null_value  # sum budget T = A_1 + A_0 = 2 S * theta_0
    hat_a = table.hat_a
    c0_1, c1_1 = _ceilings(table, "ATT", monotonic)
    c0_0, c1_0 = _ceilings(table, "ATU", monotonic)
    # Mirror the single-effect reachability guard. The sum budget T = 2 *
    # null_value is reachable only if some split A_1 + A_0 = T places both
    # effects in their boxes, i.e. T in [2 hat_a - c0_att - c0_atu, 2 hat_a +
    # c1_att + c1_atu]. Outside that band no McNemar tail is meaningful: the
    # far-side clamp would otherwise collapse the two-sided combine to a
    # spurious 0.0, so reject the null loudly rather than return it.
    lo_count = (2 * hat_a - c0_1 - c0_0 + 1) // 2  # ceil to the count scale
    hi_count = (2 * hat_a + c1_1 + c1_0) // 2  # floor to the count scale
    if not lo_count <= null_value <= hi_count:
        raise ValueError(
            f"null_value={null_value} is unreachable for target 'ATE'; it must "
            f"lie in [{lo_count}, {hi_count}] (the average-effect iSuccesses "
            f"count (A_1 + A_0) / 2)."
        )
    a_lo, a_hi = hat_a - c0_1, hat_a + c1_1

    def combined(side: str) -> float:
        def tail(target: str, a: int) -> float:
            return _one_sided_effect_tail(
                table,
                target=target,
                null_value=a,
                gamma=gamma,
                monotonic=monotonic,
                side=side,
            )

        # _combined_tail_max wants f non-decreasing, g non-increasing in a_1.
        # For 'greater' that is (A_1 tail, A_0 tail); for 'less' the A_1 tail
        # decreases in a_1 while the A_0 tail increases, so the roles swap.
        if side == "greater":
            f = lambda a1: tail("ATT", a1)  # noqa: E731
            g = lambda a1: tail("ATU", two_a - a1)  # noqa: E731
        else:
            f = lambda a1: tail("ATU", two_a - a1)  # noqa: E731
            g = lambda a1: tail("ATT", a1)  # noqa: E731
        return _combined_tail_max(f, g, a_lo, a_hi)

    if alternative == "greater":
        return min(1.0, 2.0 * combined("greater"))
    if alternative == "less":
        return min(1.0, 2.0 * combined("less"))
    return min(1.0, 4.0 * min(combined("greater"), combined("less")))


def worst_case_pvalue(
    table: PairedOutcomeTable,
    *,
    target: str = "ATT",
    null_value: int = 0,
    gamma: float = 1.0,
    monotonic: bool = False,
    alternative: str = "two-sided",
) -> float:
    r"""Worst-case p-value for a hypothesis about ``<target>``.

    The largest p-value over the sharp nulls consistent with the observed discordancy,
    under Rosenbaum sensitivity parameter ``gamma`` (``1`` = randomized). The worst case
    sits at a boundary corner of the consistency rectangle. At ``gamma == 1`` this
    reduces to the exact McNemar p-value.

    The two worst-case one-sided McNemar tails are combined per ``alternative``:
      - "two-sided" (``H_0: <target> = null_value``): twice the smaller tail,
        capped at ``1``.
      - "greater" (``H_0: <target> <= null_value``): the right tail (evidence
        that ``<target>`` exceeds ``null_value``); no factor of two.
      - "less" (``H_0: <target> >= null_value``): the left tail.

    Each tail is ``1.0`` when the estimate is within one discordant pair of
    ``null_value`` on that side (nothing to reject).

    For ``target='ATE'`` there is no single McNemar pivot, so the test combines the
    ``A_1`` and ``A_0`` tails under ``A_1 + A_0 = 2 S * ATE`` -- an intersection-union
    test over the split of the sum budget, with a Bonferroni price for combining the two
    effects (see :func:`_ate_worst_case_pvalue` for the full logic). ``null_value`` is
    then the ATE "iSuccesses" count ``(A_1 + A_0) / 2 = ATE * n_pairs``.

    Parameters
    ----------
     table : PairedOutcomeTable
        The matched-pair outcome table.
     target : {'ATT', 'ATU', 'ATE'}
        Which effect the null concerns. ``A_1`` / ``A_0`` are the treated- and
        control-side attributable effects; ``ATE`` is the average effect.
     null_value : int
        The null value tested against (default 0), on the attributable-effect count
        scale (for ``ATE``, the average-effect count ``ATE * n_pairs``). A single-effect
        null outside the reachable range ``[hat_a - c0, hat_a + c1]`` raises
        ``ValueError``. The ``ATE`` path raises likewise when the average-effect null
        lies outside its reachable band ``[(2 hat_a - c0_att - c0_atu) / 2, (2 hat_a +
        c1_att + c1_atu) / 2]`` -- a mis-scaled ATE null (e.g. passed on the rate scale
        rather than the count scale) is surfaced as an error rather than a spurious
        ``0.0``. Because ``null_value`` is an integer count, the ATE band is rounded
        *inward* to integers -- the lower edge is ceil'd, the upper edge floor'd -- so
        when the raw halves are non-integer the enforced bound (and the range quoted in
        the ``ValueError``) can sit up to half a unit inside the exact endpoints above.
     gamma : float
        Rosenbaum sensitivity parameter (``>= 1``; default 1.0).
     monotonic : bool
        Assume treatment never hurts (no prevention); yields a smaller p-value. Default
        ``False``.
     alternative : {'two-sided', 'less', 'greater'}
        The kind of test (see above). Defaults to "two-sided".

    """
    _validate_alternative(alternative)
    _validate_target(target)
    if target == "ATE":
        return _ate_worst_case_pvalue(
            table,
            null_value=null_value,
            gamma=gamma,
            monotonic=monotonic,
            alternative=alternative,
        )
    c0, c1 = _ceilings(table, target, monotonic)
    delta = table.hat_a - null_value
    # The tested ``null_value`` must be a reachable value of the attributable
    # effect: outside ``[hat_a - c0, hat_a + c1]`` (the confidence-set search
    # domain) the McNemar counts go negative and the tail is meaningless.
    if not -c1 <= delta <= c0:
        raise ValueError(
            f"null_value={null_value} is unreachable for target {target!r}; it "
            f"must lie in [{table.hat_a - c0}, {table.hat_a + c1}]."
        )
    # Each tail defaults to 1.0 (cannot reject) until its threshold is met; at
    # most one of the two branches fires, since delta cannot be both >= 2 and
    # <= -2.
    right_tail = 1.0
    left_tail = 1.0
    if delta >= 2:
        # Effect above the null: the worst case is the right-tail McNemar test.
        b = min(c1, c0 - delta)
        a = b + delta
        right_tail = _p_greater(a, b, gamma / (gamma + 1.0))
    if delta <= -2:
        # Effect below the null: mirror to the left-tail test.
        a = min(c0, c1 + delta)
        b = a - delta
        left_tail = _p_less(a, b, 1.0 / (gamma + 1.0))
    if alternative == "greater":
        return right_tail
    if alternative == "less":
        return left_tail
    # Two-sided: double the smaller worst-case one-sided tail, capped at 1.
    return min(1.0, 2.0 * min(right_tail, left_tail))


def sensitivity_value(
    table: PairedOutcomeTable,
    *,
    alpha: float = 0.05,
    target: str = "ATT",
    null_value: int = 0,
    monotonic: bool = False,
    alternative: str = "two-sided",
) -> float:
    r"""Rosenbaum sensitivity value ``Γ•`` for a finding.

    Returns the largest ``Gamma`` at which the worst-case test of ``H_0: <target> =
    null_value`` (with the given ``alternative``) can still be rejected at level
    ``alpha``. A value near 1 means the finding is fragile; a large value means it is
    robust to substantial hidden bias. ``Γ•`` inverts the same test as
    :func:`worst_case_pvalue`, so ``alternative`` is threaded through unchanged.

    Parameters
    ----------
     table : PairedOutcomeTable
        The matched-pair outcome table.
     alpha : float
        Significance level (default 0.05).
     target : {'ATT', 'ATU', 'ATE'}
        Which effect the null concerns. For ``ATE`` this inverts the combined
        (Rigdon-Hudgens) worst-case test of :func:`_ate_worst_case_pvalue`, so at the
        no-effect null the two-sided value agrees with
        :meth:`PairedOutcomeTable.gamma_star`. The scales differ for a non-zero null:
        ``null_value`` here is on the count scale (``ATE * n_pairs``) while
        ``gamma_star`` takes its null on the rate (ATE) scale, so convert with
        ``null_value = ate_null * n_pairs`` before comparing the two.
     null_value : int
        The null value being tested against (default 0); for ``ATE`` the average-effect
        count ``ATE * n_pairs``.
     monotonic : bool
        Assume treatment never hurts (no prevention); yields a larger ``Γ•``. Default
        ``False``.
     alternative : {'two-sided', 'less', 'greater'}
        The kind of test whose worst-case rejection is inverted; must match the p-value
        being reported. Defaults to "two-sided".

    Returns
    -------
     float
        ``Γ• >= 1``. Returns ``1.0`` when the null cannot be rejected even in the
        randomized case.

    """
    _validate_alternative(alternative)
    _validate_target(target)

    def worst_p(gamma: float) -> float:
        return worst_case_pvalue(
            table,
            target=target,
            null_value=null_value,
            gamma=gamma,
            monotonic=monotonic,
            alternative=alternative,
        )

    if worst_p(1.0) > alpha:
        return 1.0

    lo, hi = 1.0, 2.0
    while worst_p(hi) <= alpha and hi < 1e6:
        hi *= 2.0
    # The finding survives even at the search cap: report an unbounded
    # sensitivity value rather than a spurious number near the cap.
    if hi >= 1e6 and worst_p(hi) <= alpha:
        return math.inf
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if worst_p(mid) <= alpha:
            lo = mid
        else:
            hi = mid
    return lo


def design_sensitivity_binary(baseline: float, ate: float) -> float:
    r"""Design sensitivity ``Γ̃`` for a matched-pair binary net effect.

    The Rosenbaum sensitivity value ``Γ•`` mixes two ingredients: robustness to hidden
    bias and ordinary stochastic noise. The design sensitivity ``Γ̃`` is the limit of
    ``Γ•`` as the sample size grows without bound at the favorable situation, isolating
    the bias component. If the true (unknown) hidden bias ``Γ`` exceeds ``Γ̃``, no amount
    of data will let us reject the null of zero net effect; robustness is a matter of
    effect size, not sample size. Like a power analysis, it is most useful *before* a
    study is run: computing it post hoc from an observed table carries the same caveats
    as post-hoc power.

    The closed form is ``Γ̃ = 1 + ate / baseline`` in both the general and the monotonic
    regime; only the meaning of ``baseline`` changes:

    - **General (no monotonicity assumed).** Pass the control success rate
      ``p_{+1}``. Then ``Γ̃ = p_{1+} / p_{+1} = 1 + τ / p_{+1}``, the ratio
      of the treated to the control success rate, where ``τ = ate`` is the
      rate difference. This uses only the outcome marginals.
    - **Monotonic (treatment never hurts).** Pass the harmful-discordance
      rate ``p_{01}`` (the pair-type probability of a control success paired
      with a treated failure). Then ``Γ̃ = 1 + τ / p_{01}``. Because
      ``p_{01} <= p_{+1}``, assuming monotonicity yields a larger, more
      favorable design sensitivity, but ``p_{01}`` is a joint quantity that
      cannot be recovered from the marginals alone.

    Parameters
    ----------
     baseline : float
        The denominator of the relative effect size, in ``(0, 1)``. Pass the control
        success rate ``p_{+1}`` for the general design sensitivity, or the
        harmful-discordance rate ``p_{01}`` if assuming monotonicity (see the discussion
        above).
     ate : float
        The net-effect rate difference ``τ = p_{1+} - p_{+1}`` to be detected; must be
        positive.

    Returns
    -------
     float
        ``Γ̃ > 1``, the design sensitivity.

    Notes
    -----
    The clean form ``Γ̃ = 1 + τ / p_{+1}`` for the general case assumes the worst-case
    (least-favorable) bias configuration sits at the top of the 2x2 contingency box --
    equivalently, that failures are more common than successes (``S_{11} <= S_{00}`` at
    the boundary). This is documented, not enforced; outside that regime the general
    closed form is an approximation. See the "Design sensitivity" discussion in the
    net-effects paper.

    The ``baseline + ate < 1.0`` check is the *general*-regime constraint: there
    ``baseline = p_{+1}`` and ``baseline + ate = p_{1+}`` is the treated success
    marginal, which must be a valid rate below 1. In the monotonic regime ``baseline =
    p_{01}`` is a joint pair-type rate and the sum is not a marginal, so the bound is
    not strictly required; it is nonetheless enforced uniformly for both regimes. This
    is conservative in the monotonic case -- it can reject inputs that are formally
    legitimate there -- but the excluded region (``p_{01} + τ >= 1``) requires an
    implausibly large harmful-discordance rate, so the over-restriction is mild and buys
    a single, simple validation path.

    References
    ----------
    .. [1] Rosenbaum, P. R. (2004). Design sensitivity in observational
       studies. Biometrika, 91(1), 153-164.

    """
    if not 0.0 < baseline < 1.0:
        raise ValueError(f"baseline must be in (0, 1); got {baseline}.")
    if not ate > 0.0:
        raise ValueError(f"ate must be positive; got {ate}.")
    if not baseline + ate < 1.0:
        raise ValueError(f"baseline + ate must be below 1; got {baseline + ate}.")
    return 1.0 + ate / baseline


def mcnemar_ate_interval(
    table: PairedOutcomeTable, *, confidence: float = 0.90
) -> tuple[float, float]:
    r"""Textbook McNemar/Wald ATE interval, for comparison only.

    This is the conventional interval based on the discordant pairs. It is narrower than
    ``att_confidence_set`` because it implicitly assumes monotonicity (or multinomial
    sampling); it is provided as a baseline, not as the recommended estimator.

    """
    disc = table.s01 + table.s10
    n_pairs = table.n_pairs
    z = float(norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    half_width = z * math.sqrt(disc) / n_pairs
    center = table.ate_hat
    return (center - half_width, center + half_width)


_ANALYSIS_HEADERS: dict[EffectSize, str] = {
    # Neutral fallback only: the target-specific label (ATT/ATU/ATE) comes from
    # `_SCALED_EFFECT_HEADERS`; this is used solely when `target` is unrecognized
    # (e.g. a hand-built payload), so it must not name any one target.
    EffectSize.EFFECT: "Effect",
    EffectSize.ISUCCESSES: "iSuccesses",
    EffectSize.COST_PER_ISUCCESS: "Cost/iSuccess",
}
# Header for the scaled-effect column, by target. Since `target` now carries the
# estimand names (ATT for the effect on the treated A_1, ATU for the effect on
# the untreated A_0, ATE for the average), this is an identity map -- it no
# longer translates. It is retained as the single target -> header indirection
# point (leaving room for a future non-identity label) and, via
# `header_for_column`'s `.get(...)` fallback, as the implicit membership check
# that routes an unrecognized target to the neutral `_ANALYSIS_HEADERS` label.
_SCALED_EFFECT_HEADERS: dict[str, str] = {
    "ATT": "ATT",
    "ATU": "ATU",
    "ATE": "ATE",
}
_ANALYSIS_FORMATS: dict[EffectSize, str] = {
    EffectSize.EFFECT: "{:+.2%}",
    EffectSize.ISUCCESSES: "{:+,.0f}",
    EffectSize.COST_PER_ISUCCESS: "${:,.2f}",
}
_CI_HEADER = "Conf Int**"
_PVALUE_HEADER = "p-Value*"
_GAMMA_HEADER = "Γ•"
# The count column's header, for tables that have no `header_overrides` to
# honor and so cannot go through `header_for_column`. Named here rather than
# spelled out at each use so the two table families cannot drift apart: they are
# routinely read side by side, and a column labeled differently in each would
# read as two different quantities.
_ISUCCESSES_HEADER = _ANALYSIS_HEADERS[EffectSize.ISUCCESSES]
# How a footnote names the level and the direction of the test. The relation is
# the one the *null* is stated with, so it is the complement of the alternative:
# a "greater" alternative tests against `H_0: effect <= null`.
_FOOTER_RELATIONS = {"two-sided": "=", "greater": "≤", "less": "≥"}


def _footer_phrasing(alpha: float, alternative: str) -> tuple[str, str, str]:
    """The ``(level, sidedness, relation)`` wording shared by the footnotes.

    :meth:`PairedOutcomeAnalysis._footer` and its
    :class:`~...linear_combination.LinearCombinationAnalysis` counterpart print the same
    two notes, so they read the level and direction the same way. Kept here rather than
    duplicated because the pair is only useful while it stays identical -- the
    conventional 0.05/0.10 in particular must render as written in both, not as ``0.05``
    in one and ``0.0500`` in the other.

    """
    # Named rather than looked up bare: an unrecognized `alternative` reaches a
    # hand-built or deserialized analysis, and a `KeyError: 'sideways'` raised
    # from inside a `__str__` says nothing about which field is wrong.
    _validate_alternative(alternative)
    if any(math.isclose(alpha, a) for a in (0.05, 0.10)):
        alpha_f = f"{alpha:.02f}"
    else:
        alpha_f = format_with_min_nonzero_digits(alpha, 3, percentage=False)
    sided = "two-sided" if alternative == "two-sided" else "one-sided"
    return alpha_f, sided, _FOOTER_RELATIONS[alternative]


@dataclass
class PairedOutcomeAnalysisOptions:
    r"""Display options for :meth:`PairedOutcomeTable.analyze`.

    Which effect-size columns appear is controlled by membership in
    ``effect_size_columns`` (each shows its estimate and a confidence interval),
    mirroring ``SpockTestSummaryOptions``. ``include_p_value`` and
    ``include_sensitivity`` toggle the trailing p-value and ``Γ•`` columns.
    ``header_overrides`` / ``format_overrides`` replace a column's header text or its
    Python format string; keys are ``EffectSize`` members (or the equivalent strings)
    for the effect columns. ``header_overrides`` additionally accepts a ``"pval"`` key
    to rename the p-value column header; the p-value column's numeric format is fixed
    and is not overridable.

    Attributes
    ----------
     effect_size_columns : sequence of EffectSize or str
        The effect-size columns to display, in order.
     include_p_value : bool
        Whether to append the worst-case p-value column.
     include_sensitivity : bool
        Whether to append the ``Γ•`` sensitivity-value column.
     ci_separator : str
        Text placed between a confidence interval's bounds.
     header_overrides, format_overrides : dict, optional
        Per-column header-text / format-string overrides.

    """

    effect_size_columns: Sequence[EffectSize | str] = (
        EffectSize.EFFECT,
        EffectSize.ISUCCESSES,
    )
    include_p_value: bool = True
    include_sensitivity: bool = True
    ci_separator: str = ", "
    header_overrides: dict[EffectSize | str, str] | None = None
    format_overrides: dict[EffectSize | str, str] | None = None

    def __post_init__(self) -> None:
        # Normalize the column list to EffectSize members, and override keys that
        # name an effect size to members (leaving "pval" strings untouched), so
        # lookups use a single key type.
        self.effect_size_columns = [
            EffectSize.get(es) for es in self.effect_size_columns
        ]
        self.header_overrides = self._normalize_overrides(self.header_overrides)
        self.format_overrides = self._normalize_overrides(self.format_overrides)

    @staticmethod
    def _normalize_overrides(
        overrides: dict[EffectSize | str, str] | None,
    ) -> dict[EffectSize | str, str]:
        if overrides is None:
            return {}
        normalized: dict[EffectSize | str, str] = {}
        for key, value in overrides.items():
            try:
                normalized[EffectSize.get(key)] = value
            except ValueError:
                normalized[key] = value
        return normalized


def _encode_float(x: float) -> float | str:
    """JSON-safe float: non-finite values become strings, keeping the output valid.

    ``json.dumps`` renders ``inf``/``nan`` as ``Infinity``/``NaN``, which are not valid
    JSON (RFC 8259) and are rejected by strict cross-language parsers. A one-sided
    confidence-set endpoint or a reachable ``gamma_star`` can be infinite, so those are
    emitted as ``"inf"`` / ``"-inf"`` / ``"nan"`` strings; ``_decode_float`` round-trips
    them exactly.

    """
    return x if math.isfinite(x) else str(x)


def _decode_float(x: object) -> float:
    """Parse a JSON number or an ``"inf"`` / ``"-inf"`` / ``"nan"`` string to float."""
    if isinstance(x, int | float):
        return float(x)
    if isinstance(x, str) and x in ("inf", "-inf", "nan"):
        return float(x)
    raise ValueError(
        f"expected a JSON number or one of 'inf'/'-inf'/'nan', got "
        f"{type(x).__name__}: {x!r}."
    )


@dataclass(frozen=True)
class PairedOutcomeAnalysis:
    r"""The full net-effects analysis of a :class:`PairedOutcomeTable`.

    The single result object for the estimation module: it carries the scaled effect
    (the ATT or the ATU, depending on ``target``), the attributable effects, the
    worst-case p-value and the sensitivity value, and knows how to display itself (via
    :class:`PairedOutcomeAnalysisOptions`) and serialize itself. Produced by
    :meth:`PairedOutcomeTable.analyze` and by :func:`att_confidence_set`.

    Attributes
    ----------
     table : PairedOutcomeTable
        The table analyzed.
     alpha : float
        Significance level; confidence sets have coverage ``1 - alpha``.
     gamma : float
        Sensitivity parameter entertained for the confidence sets and p-value.
     null_value : int
        The attributable-effect null the p-value and ``Γ•`` were tested against.
     alternative : {'two-sided', 'less', 'greater'}
        The kind of test the p-value/``Γ•``/confidence sets reflect.
     target : {'ATT', 'ATU', 'ATE'}
        Which effect the analysis concerns. ``A_1`` is the effect on the treated (scaled
        column shown as ``ATT``); ``A_0`` the effect on the untreated (``ATU``); ``ATE``
        the average effect (``ATE``).
     effect : float
        Point estimate of the scaled effect: ``ate_hat = hat_a / n_pairs``, which equals
        ``A_1 / n_pairs`` (ATT), ``A_0 / n_pairs`` (ATU) and the ATE alike.
     effect_interval : (float, float)
        Confidence set for the scaled ``target`` effect: the ``attributable_interval``
        divided by ``n_pairs`` (the ATT when ``target == 'ATT'``, the ATU when ``target
        == 'ATU'``, the ATE set when ``target == 'ATE'``). For one-sided ``alternative``
        one endpoint is ``+/- math.inf``.
     attributable : int
        Attributable-effect point estimate (``S_10 - S_01``); for ``ATE`` this is also
        the average-effect point estimate ``(A_1 + A_0) / 2`` (both effects share the
        ``S_10 - S_01`` pivot).
     attributable_interval : (float, float)
        The confidence set for the ``target`` effect (the one shown as ``ISUCCESS``):
        the attributable-effect set for ``A_1`` / ``A_0``, or the average-effect set
        ``(A_1 + A_0) / 2 = ATE * n_pairs`` for ``ATE``. Two-sided, an ``A_1`` / ``A_0``
        set is tested at ``alpha / 2`` per side with integer endpoints (the ATE set
        carries the extra Bonferroni split); one-sided it carries ``+/- math.inf`` on
        the uninformative side. ``effect_interval`` is exactly this set divided by
        ``n_pairs``.
     p_value : float
        Worst-case p-value for the tested null (two-sided by default).
     gamma_star : float
        Rosenbaum sensitivity value (a property of the data, independent of the
        entertained ``gamma``).
     options : PairedOutcomeAnalysisOptions
        Display options.
     monotonic : bool
        Whether the sets/tests assumed treatment never hurts (no prevention).
     method : {'exact', 'normal', 'auto'}
        How the confidence sets were inverted (the p-value and ``Γ•`` are always exact).
        A legacy payload without it deserializes to ``'exact'``.

    """

    table: PairedOutcomeTable
    alpha: float
    gamma: float
    null_value: int
    alternative: str
    effect: float
    effect_interval: tuple[float, float]
    attributable: int
    attributable_interval: tuple[float, float]
    p_value: float
    gamma_star: float
    options: PairedOutcomeAnalysisOptions
    # Last, with defaults, so the dataclass stays constructible without them
    # (matching ``deserialize``'s back-compatible ``d.get(..., default)``).
    monotonic: bool = False
    target: str = "ATT"
    method: str = "auto"

    @property
    def point_estimate(self) -> float:
        """Point estimate of the scaled effect (ATT for A_1, ATU for A_0, ATE)."""
        return self.table.ate_hat

    @property
    def confidence(self) -> float:
        """Coverage of the confidence sets, ``1 - alpha``."""
        return 1.0 - self.alpha

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation (display options excluded)."""
        return {
            "table": self.table.to_dict(),
            "alpha": _encode_float(self.alpha),
            "gamma": _encode_float(self.gamma),
            "monotonic": self.monotonic,
            "null_value": self.null_value,
            "alternative": self.alternative,
            "target": self.target,
            "method": self.method,
            "effect": _encode_float(self.effect),
            "effect_interval": [_encode_float(v) for v in self.effect_interval],
            "attributable": self.attributable,
            "attributable_interval": [
                _encode_float(v) for v in self.attributable_interval
            ],
            "p_value": _encode_float(self.p_value),
            "gamma_star": _encode_float(self.gamma_star),
        }

    def serialize(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict())

    @staticmethod
    def deserialize(
        s: str | None = None, d: dict[str, object] | None = None
    ) -> PairedOutcomeAnalysis:
        """Reconstruct from a JSON string or dict (with default display options)."""
        if s is not None and d is not None:
            raise ValueError("Provide `s` or `d`, not both.")
        if s is not None:
            d = json.loads(s)
        if d is None:
            raise ValueError("Provide either `s` or `d`.")
        # Back-compat: pre-rename payloads carry "att"/"att_interval" instead of
        # "effect"/"effect_interval". A payload with neither key gets a
        # domain-specific error rather than a bare KeyError naming the legacy
        # field (which would misleadingly point at "att").
        if "effect" in d:
            effect_val = d["effect"]
        elif "att" in d:
            effect_val = d["att"]
        else:
            raise ValueError("payload is missing 'effect' (or legacy 'att').")
        if "effect_interval" in d:
            effect_iv_raw = d["effect_interval"]
        elif "att_interval" in d:
            effect_iv_raw = d["att_interval"]
        else:
            raise ValueError(
                "payload is missing 'effect_interval' (or legacy 'att_interval')."
            )
        effect_iv = cast("list[object]", effect_iv_raw)
        att_i = cast("list[object]", d["attributable_interval"])
        # Reject unknown targets loudly rather than let `header_for_column` fall
        # back to the neutral label. In particular a payload persisted before the
        # A1/A0 -> ATT/ATU rename (target="A0") would otherwise revive with its
        # stale target intact and render under that neutral "Effect" label,
        # silently mislabeling what is really an ATU. There is no back-compat
        # mapping (the old values only ever existed pre-stack), so surface the
        # mismatch instead of hiding it.
        target = cast("str", d.get("target", "ATT"))
        _validate_target(target)
        return PairedOutcomeAnalysis(
            table=PairedOutcomeTable.deserialize(
                d=cast("dict[str, object]", d["table"])
            ),
            alpha=_decode_float(d["alpha"]),
            gamma=_decode_float(d["gamma"]),
            monotonic=bool(d.get("monotonic", False)),
            null_value=int(cast("int", d.get("null_value", 0))),
            alternative=cast("str", d.get("alternative", "two-sided")),
            target=target,
            # Legacy payloads predate the `method` knob; their sets were exact.
            method=cast("str", d.get("method", "exact")),
            effect=_decode_float(effect_val),
            effect_interval=(
                _decode_float(effect_iv[0]),
                _decode_float(effect_iv[1]),
            ),
            attributable=int(cast("int", d["attributable"])),
            attributable_interval=(
                _decode_float(att_i[0]),
                _decode_float(att_i[1]),
            ),
            p_value=_decode_float(d["p_value"]),
            gamma_star=_decode_float(d["gamma_star"]),
            options=PairedOutcomeAnalysisOptions(),
        )

    def _cost_per(self, successes: float) -> float:
        """Treatment cost per incremental success; ``inf`` at zero-or-fewer."""
        spend = self.table.spend
        assert spend is not None  # validated by analyze() when the column is used
        if successes <= 0:
            return math.inf
        return spend / successes

    def _estimate(self, effect: EffectSize) -> float:
        if effect == EffectSize.EFFECT:
            return self.effect
        if effect == EffectSize.ISUCCESSES:
            return float(self.attributable)
        return self._cost_per(self.attributable)

    def _interval(self, effect: EffectSize) -> tuple[float, float]:
        if effect == EffectSize.EFFECT:
            return self.effect_interval
        lo, hi = self.attributable_interval
        if effect == EffectSize.ISUCCESSES:
            return (float(lo), float(hi))
        # COST_PER_ISUCCESS: cost per success is ``spend / a`` for ``a > 0``, else
        # ``inf``, and decreasing in ``a``. So the lower cost bound comes from the
        # upper success bound and vice versa. If the upper success bound is <= 0
        # the whole cost interval is ``inf``; if only the lower success bound is
        # <= 0 the upper cost bound is ``inf`` while the lower stays finite.
        return (self._cost_per(hi), self._cost_per(lo))

    def header_for_column(self, effect: EffectSize) -> str:
        """Display header for an effect-size column (honoring overrides)."""
        overrides = self.options.header_overrides or {}
        if effect in overrides:
            return overrides[effect]
        # The scaled-effect column is labeled by target (ATT for A_1, ATU for
        # A_0); the others have a fixed header.
        if effect == EffectSize.EFFECT:
            return _SCALED_EFFECT_HEADERS.get(self.target, _ANALYSIS_HEADERS[effect])
        return _ANALYSIS_HEADERS[effect]

    def format_for_column(self, effect: EffectSize) -> str:
        """Python format string for an effect-size column (honoring overrides)."""
        overrides = self.options.format_overrides or {}
        return overrides.get(effect, _ANALYSIS_FORMATS[effect])

    @staticmethod
    def _apply_format(fmt: str, value: float) -> str:
        """Format a value, rendering infinities as ``∞`` / ``-∞``."""
        if math.isinf(value):
            return "∞" if value > 0 else "-∞"
        return fmt.format(value)

    def __str__(self) -> str:
        """Render a one-row org-mode table of the analysis, with a footer."""
        options = self.options
        headers: list[str] = []
        row: list[str] = []
        colalign: list[str] = []
        for column in options.effect_size_columns:
            effect = EffectSize.get(column)
            fmt = self.format_for_column(effect)
            lo, hi = self._interval(effect)
            headers.extend([self.header_for_column(effect), _CI_HEADER])
            row.append(self._apply_format(fmt, self._estimate(effect)))
            lo_s = self._apply_format(fmt, lo)
            hi_s = self._apply_format(fmt, hi)
            row.append(f"{lo_s}{options.ci_separator}{hi_s}")
            colalign.extend(["right", "center"])
        if options.include_p_value:
            headers.append(
                (self.options.header_overrides or {}).get("pval", _PVALUE_HEADER)
            )
            pval = format_with_min_nonzero_digits(self.p_value, 3, percentage=False)
            flag = "*" if self.p_value < self.alpha else ""
            row.append(f"{pval}{flag}")
            colalign.append("right")
        if options.include_sensitivity:
            headers.append(_GAMMA_HEADER)
            gamma_star = self.gamma_star
            row.append("inf" if math.isinf(gamma_star) else f"{gamma_star:g}")
            colalign.append("right")
        table = tabulate([row], headers=headers, colalign=colalign, tablefmt="orgtbl")
        return f"{table}\n{self._footer()}"

    def _footer(self) -> str:
        """Explanatory footer: CI coverage / entertained Gamma, significance."""
        coverage = 1.0 - self.alpha
        notes = []
        if self.monotonic:
            monotonicity_note = " (assuming monotonicity)"
        else:
            monotonicity_note = ""
        if self.options.include_p_value:
            # Inside the branch: the wording is the p-value note's alone (the
            # coverage note reads the level as a percentage), and
            # `_footer_phrasing` rejects an unrecognized `alternative` -- which
            # is the test's business, not the coverage note's, so suppressing
            # the note must also suppress the demand.
            alpha_f, sided, relation = _footer_phrasing(self.alpha, self.alternative)
            # Name the tested effect by its column header when that column is
            # shown; otherwise use a generic label, so the note never cites a
            # hidden column (or an override the user set on a hidden column).
            if EffectSize.ISUCCESSES in self.options.effect_size_columns:
                effect_label = self.header_for_column(EffectSize.ISUCCESSES)
            else:
                effect_label = "the attributable effect"
            notes.append(
                "*  An asterisk in the p-Value column indicates statistical significance at\n"
                f"   level {alpha_f}, provided Γ≤{self.gamma:g}{monotonicity_note}.\n"
                f"   p-Value is {sided} against the null hypothesis that "
                f"{effect_label} {relation} {self.null_value}."
            )
        notes.append(
            f"** Confidence intervals have coverage of at least {coverage:.0%}, "
            f"provided Γ≤{self.gamma:g}{monotonicity_note}."
        )
        return "\n".join(notes)
