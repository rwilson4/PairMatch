# pyre-strict
"""Linear combinations of matched-pair net-effect tables, incl. diff-in-diff."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from tabulate import tabulate

from pair_match.net_effects import (
    _CI_HEADER,
    _GAMMA_HEADER,
    _ISUCCESSES_HEADER,
    _PVALUE_HEADER,
    _SCALED_EFFECT_HEADERS,
    PairedOutcomeTable,
    _attributable_sensitivity_band,
    _decode_float,
    _encode_float,
    _footer_phrasing,
    _gamma_star_search,
    _validate_alternative,
    _validate_method,
    _validate_target,
    attributable_effect_interval,
    format_with_min_nonzero_digits,
)
from pair_match.visualizations import (
    _plot_sensitivity_curve,
    _resolve_gamma_max,
    _sweep_sensitivity_bands,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.typing import LegendLocType

# The combined lower bound takes a positive coefficient's lower component bound
# and a negative coefficient's upper one; the combined upper bound mirrors it.
_FLIPPED = {"greater": "less", "less": "greater"}

# Bracket floor for the p-value bisection. What must not underflow is not `alpha`
# itself but the smallest level any *quantile* is finally evaluated at, and the
# level is split three times on the way down: the union bound spends `alpha / k`
# over the k nonzero terms, a two-sided component halves that again per tail, and
# `target='ATE'` halves it once more across `A_1` and `A_0`. The worst case is
# therefore `alpha / (4k)`, not `alpha / (2k)` -- ATE is the binding target. The
# largest double below 1.0 is 1 - 2**-53, so `1 - x` collapses to exactly 1.0 --
# an infinite normal quantile -- once x falls under ~5.5e-17. A floor of 1e-12
# therefore leaves headroom to about k = 4000 terms, far past any combination one
# would write down.
_PVALUE_EPS = 1e-12

# Floor and ceiling for the footer's integrality window; see
# `_count_integrality_tol`.
_COUNT_ABS_TOL = 1e-9
_COUNT_TOL_CAP = 0.125


def _component_side(alternative: str, coefficient: float) -> str:
    """Which interval of a component the combination at ``alternative`` consumes.

    A combined one-sided bound is built from one bound per component, and which one
    depends on the sign of the coefficient: the combined lower bound takes a positive
    coefficient's lower component bound and a negative coefficient's upper one. So a
    component's side is the combined side, flipped when the coefficient is negative.

    Two cases stay two-sided. A two-sided combination consumes both ends of every
    component, and a zero coefficient consumes neither -- it contributes nothing to
    either bound, so there is no side to report and the component's own two-sided
    interval is the honest thing to show.

    Used by both :meth:`LinearCombinationEstimator._combine`, which reads the bound, and
    :meth:`LinearCombinationEstimator.analyze`, which displays the interval it came
    from. Sharing the rule is the point: the per-term rows of a summary should be the
    numbers the Combined row was actually built from.

    """
    if alternative == "two-sided" or coefficient == 0.0:
        return "two-sided"
    return alternative if coefficient > 0.0 else _FLIPPED[alternative]


def _count_integrality_tol(count: float) -> float:
    """Tolerance for asking whether ``count`` is a whole number of successes.

    Absolute rather than relative, deliberately: integrality does not scale, and a
    relative term would loosen the test as the count grows -- past ~1e9 it would exceed
    0.5 and restate a genuinely fractional null as a rounded integer.

    A *fixed* absolute floor is not enough either, because the count is a product
    (``null_value * n_pairs``) and carries round-off of order one ulp of that product. A
    null the caller means as whole -- ``0.28`` against 2,678,547,400 pairs -- can land
    one ulp (~1.2e-7) off the integer, far outside a 1e-9 window, and so be phrased in
    scaled units when a count was available. A few ulps of the product covers that; the
    cap keeps the tolerance below the half-success at which the question stops meaning
    anything. Four ulps reaches half a success at 2**49 (~5.6e14), so the cap binds from
    2**48 (~2.8e14) up.

    The cap is a real ceiling, not a formality, and it costs something at the very top
    of the range. Writing ``null_value`` as ``N / n_pairs`` rounds once and multiplying
    back rounds again, which puts the product up to one ulp from the ``N`` the caller
    meant. That exceeds the 0.125 window on ``[2**50, 2**52)`` (~1.1e15 to 4.5e15): a
    quarter of a success on the lower binade, where the spacing is 0.25, and half a
    success on the upper, where it is 0.5. An intended-integral null in that range is
    phrased in scaled units after all. Above 2**52 every double is an integer and the
    question answers itself. That range needs more matched pairs than any study will
    have; the alternative -- widening the cap -- would restate genuinely fractional
    nulls as counts at magnitudes one might actually reach, which is the worse failure.

    Note this is the opposite failure from the one
    ``test_footer_integrality_tolerance_is_capped`` pins. That test sits a binade lower,
    on ``[2**49, 2**50)``, and drives a *genuinely fractional* null a quarter-success
    off an integer, where falling back to scaled units is the cap working. The paragraph
    above is about the cap misfiring on a whole null, which starts one binade higher.

    """
    return min(max(_COUNT_ABS_TOL, 4.0 * math.ulp(count)), _COUNT_TOL_CAP)


def _validate_alpha(alpha: float, alternative: str = "two-sided") -> None:
    """Raise unless ``alpha`` is a level usable for ``alternative``."""
    # `not 0 < alpha < 1` rather than `alpha <= 0 or alpha >= 1`, so that NaN --
    # for which every comparison is False -- is rejected rather than passed on
    # to divide the Bonferroni share and turn every bound into NaN.
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"`alpha` must lie in (0, 1), got {alpha}.")
    # The guard `estimators._validate_interval_args` applies, for the same
    # reason: a one-sided p-value peaks at 0.5 at the point estimate, so a
    # one-sided bound at `alpha >= 0.5` lands on the far side of the estimate
    # and has no coverage reading. The rule is restated here rather than
    # borrowed from the other family because the two validate `alternative`
    # against their own vocabularies and word their errors in their own house
    # style; what must not drift is the *boundary*, and
    # `test_alpha_ceiling_agrees_with_estimator_family` pins the two together at
    # it. Applied to the caller's `alpha`, not to the
    # per-term share `alpha / k`: with several terms the share can be
    # respectable while the combination is still a "40% one-sided interval",
    # which is not an object anyone means to ask for at either level.
    if alpha >= 0.5 and alternative != "two-sided":
        raise ValueError(
            "one-sided intervals require `alpha` < 0.5 (the one-sided p-value "
            f"peaks at 0.5 at the point estimate), got alpha={alpha} for "
            f"alternative={alternative!r}."
        )


def _validate_gamma(gamma: float) -> None:
    """Raise unless ``gamma`` is a finite Rosenbaum sensitivity parameter >= 1."""
    # Finiteness is checked explicitly: `gamma < 1.0` is False for both NaN and
    # `inf`, and either one reaches the worst-case tilt `gamma / (gamma + 1)` as
    # a NaN and silently produces NaN bounds. `gamma_star` may *return* `inf`,
    # but it never evaluates the band there (the search caps at 1e6), so no
    # internal caller needs an infinite gamma admitted here.
    if not math.isfinite(gamma) or gamma < 1.0:
        raise ValueError(f"`gamma` must be finite and >= 1, got {gamma}.")


def _validate_null_value(null_value: float) -> None:
    """Raise unless ``null_value`` is a finite hypothesized effect."""
    # A null is only ever compared against the interval's endpoints, and every
    # comparison with NaN is False -- so a NaN null is never excluded, making
    # `gamma_star` report no robustness and `pvalue` report 1.0 for a finding
    # that may be overwhelming. An infinite null is contained by every interval
    # and reads the same way. Both fail in the "not significant" direction, so
    # nothing downstream ever raises to tell the caller their null was junk.
    if not math.isfinite(null_value):
        raise ValueError(f"`null_value` must be finite, got {null_value}.")


def _validate_payload_numbers(
    *,
    alpha: float,
    alternative: str,
    gamma: float,
    null_value: float,
    affine: float,
    effect: float,
    effect_interval: tuple[float, float],
    gamma_star: float,
    p_value: float,
) -> None:
    """Raise unless a decoded payload's numbers are ones ``analyze`` could emit.

    The numeric counterpart of the enum checks in
    :meth:`LinearCombinationAnalysis.deserialize`, and there for the same reason:
    ``analyze`` cannot produce a payload that violates any of these, so one that does is
    corrupt -- and every violation reads as a plausible number downstream rather than
    raising where it is used. An ``alpha`` of 1.5 would report ``confidence == -0.5``; a
    NaN ``null_value`` makes ``significant`` ``False`` for every interval, however
    overwhelming.

    It lives out here, rather than inline, because the two halves of the decode read
    differently: the body of ``deserialize`` is the part that shows which keys the
    format carries, while these are order-independent guards that say nothing about the
    format. Keeping them apart leaves the format legible.

    """
    _validate_alpha(alpha, alternative)
    _validate_gamma(gamma)
    _validate_null_value(null_value)
    if not math.isfinite(affine) or not math.isfinite(effect):
        raise ValueError(
            f"`affine` and `effect` must be finite, got {affine} and {effect}."
        )
    lo, hi = effect_interval
    if math.isnan(lo) or math.isnan(hi):
        # The infinities are legitimate -- a one-sided interval carries one --
        # but a NaN endpoint compares False against everything, so the interval
        # would neither contain nor exclude the null.
        raise ValueError(
            f"`effect_interval` endpoints must not be NaN, got {effect_interval}."
        )
    # `gamma_star` is exempt from finiteness: `math.inf` is its documented value
    # for a finding no hidden bias can overturn.
    if not gamma_star >= 1.0:
        raise ValueError(f"`gamma_star` must be >= 1, got {gamma_star}.")
    if not 0.0 <= p_value <= 1.0:
        # Prints as-is beside an asterisk that disagrees with it.
        raise ValueError(f"`p_value` must lie in [0, 1], got {p_value}.")


def _validated_n_pairs(value: object) -> int:
    """Return ``value`` as a matched-pair count, or raise if it is not one.

    Separate from :func:`_validate_payload_numbers` because it is the one field that is
    not a float: it both checks and *narrows*, so the decode path gets an ``int`` it can
    hand to the constructor without a cast that would paper over the very thing being
    checked.

    A count is corrupt when it is non-integral rather than roundable -- `int()` would
    quietly floor it and rescale every iSuccesses cell against the wrong denominator --
    and a negative one sign-flips them.

    """
    # `bool` is a subclass of `int`, so a bare `isinstance(value, int)` accepts
    # JSON `true` and reconstructs `n_pairs == 1`: every count in the table then
    # reads as a rate over a single pair, which is exactly the silent rescaling
    # this check exists to prevent.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"`n_pairs` must be an integer, got {value!r}.")
    if value < 0:
        raise ValueError(f"`n_pairs` must be non-negative, got {value}.")
    return value


def _fmt_effect(value: float) -> str:
    """Format a scaled effect as a signed percentage, rendering infinities."""
    if math.isinf(value):
        return "∞" if value > 0 else "-∞"
    return f"{value:+.2%}"


def _fmt_prose_effect(value: float) -> str:
    """Format a scaled value quoted in prose, never rounding nonzero to zero.

    The fixed two-decimal percentage of :func:`_fmt_effect` is right for a *column*,
    where every cell shares a width and the reader is comparing magnitudes down the
    page. It is wrong for a sentence that names a specific quantity, because a value
    below ``0.005%`` rounds to ``+0.00%`` and the sentence then asserts the opposite of
    the thing it exists to state: a constant offset the footer only mentions *because*
    it is nonzero, or a null the p-value was computed against.

    Only that case escalates to significant digits; everything else keeps the fixed
    form, so the ordinary footer is unchanged.

    """
    # `round(x * 100, 2)` is exactly what `:+.2%` does -- same half-even
    # rounding at the same place -- so this asks the precise question "would
    # the fixed format print this nonzero value as zero?" rather than guessing
    # at a magnitude cutoff.
    if value != 0.0 and math.isfinite(value) and round(value * 100.0, 2) == 0.0:
        sign = "-" if value < 0 else "+"
        return f"{sign}{format_with_min_nonzero_digits(abs(value), 3)}"
    return _fmt_effect(value)


def _fmt_count(value: float) -> str:
    """Format an iSuccesses count as a signed integer, rendering infinities."""
    if math.isinf(value):
        return "∞" if value > 0 else "-∞"
    return f"{value:+,.0f}"


@dataclass(frozen=True)
class LinearCombinationTerm:
    r"""One term's contribution to a :class:`LinearCombinationAnalysis`.

    Attributes
    ----------
     label : str
        Display label for the term.
     coefficient : float
        The term's coefficient in the combination.
     effect : float
        The component's scaled point estimate (``ate_hat``).
     effect_interval : (float, float)
        The component's net-effect interval at the Bonferroni share ``alpha / k`` (``k``
        = number of nonzero terms) -- the level at which it actually enters the combined
        interval, and on the side at which it enters. Under a one-sided combination that
        is one-sided too, flipped for a negative coefficient, so the term rows reproduce
        the combined bound rather than quoting a bound the combination never used.
        Two-sided when the combination is, or when the coefficient is zero.

    """

    label: str
    coefficient: float
    effect: float
    effect_interval: tuple[float, float]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "label": self.label,
            "coefficient": _encode_float(self.coefficient),
            "effect": _encode_float(self.effect),
            "effect_interval": [_encode_float(v) for v in self.effect_interval],
        }

    @staticmethod
    def from_dict(d: dict[str, object]) -> LinearCombinationTerm:
        """Reconstruct a term from its :meth:`to_dict` representation."""
        iv = cast("list[object]", d["effect_interval"])
        return LinearCombinationTerm(
            label=cast("str", d["label"]),
            coefficient=_decode_float(d["coefficient"]),
            effect=_decode_float(d["effect"]),
            effect_interval=(_decode_float(iv[0]), _decode_float(iv[1])),
        )


@dataclass(frozen=True)
class LinearCombinationAnalysis:
    r"""The displayable, serializable result of analyzing a linear combination.

    Produced by :meth:`LinearCombinationEstimator.analyze`. Carries the combined scaled
    effect and its Bonferroni interval at the entertained ``gamma``, the Bonferroni
    p-value for the tested null, the matched-pair count the effects are rates over (so
    the table can also report them as ``iSuccesses`` counts), the per-term breakdown,
    and the sensitivity value ``Γ•``.

    Attributes
    ----------
     affine : float
        The constant offset of the combination.
     terms : tuple of LinearCombinationTerm
        Per-term breakdown (coefficient, point estimate, component interval).
     effect : float
        Combined scaled point estimate, ``affine + sum_i c_i * effect_i``.
     effect_interval : (float, float)
        Combined Bonferroni interval at coverage ``1 - alpha`` and sensitivity
        ``gamma``. One-sided ``alternative`` carries ``+/- math.inf`` on the
        uninformative side.
     p_value : float
        Worst-case Bonferroni p-value for ``H_0: <combination> = null_value`` at
        sensitivity ``gamma`` -- the p-value dual of ``effect_interval``, below
        ``alpha`` when the interval excludes the null. See
        :meth:`LinearCombinationEstimator.pvalue` for the numerical tolerance on that
        correspondence; :attr:`significant` reads the interval itself.
     n_pairs : int
        The shared matched-pair count of the component tables (``0`` only when the
        combination has no terms at all), used to render the ``iSuccesses`` (count)
        columns as ``effect * n_pairs``.
     alpha : float
        Significance level; the interval has coverage ``1 - alpha``.
     gamma : float
        Sensitivity parameter entertained for the interval.
     null_value : float
        The null the interval and ``Γ•`` were tested against.
     alternative : {'two-sided', 'less', 'greater'}
        The kind of interval/test.
     target : {'ATT', 'ATU', 'ATE'}
        The effect each component reports.
     monotonic : bool
        Whether the component sets assumed treatment never hurts.
     method : {'exact', 'normal', 'auto'}
        How the component sets were inverted.
     gamma_star : float
        Sensitivity value ``Γ•`` for ``H_0: <combination> = null_value``.

    """

    affine: float
    terms: tuple[LinearCombinationTerm, ...]
    effect: float
    effect_interval: tuple[float, float]
    p_value: float
    n_pairs: int
    alpha: float
    gamma: float
    null_value: float
    alternative: str
    target: str
    monotonic: bool
    method: str
    gamma_star: float

    @property
    def point_estimate(self) -> float:
        """Combined scaled point estimate."""
        return self.effect

    @property
    def confidence(self) -> float:
        """Coverage of the confidence interval, ``1 - alpha``."""
        return 1.0 - self.alpha

    @property
    def significant(self) -> bool:
        """Whether the interval excludes ``null_value`` (at the entertained gamma)."""
        lb, ub = self.effect_interval
        return lb > self.null_value or ub < self.null_value

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "affine": _encode_float(self.affine),
            "terms": [t.to_dict() for t in self.terms],
            "effect": _encode_float(self.effect),
            "effect_interval": [_encode_float(v) for v in self.effect_interval],
            "p_value": _encode_float(self.p_value),
            "n_pairs": self.n_pairs,
            "alpha": _encode_float(self.alpha),
            "gamma": _encode_float(self.gamma),
            "null_value": _encode_float(self.null_value),
            "alternative": self.alternative,
            "target": self.target,
            "monotonic": self.monotonic,
            "method": self.method,
            "gamma_star": _encode_float(self.gamma_star),
        }

    def serialize(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict())

    @staticmethod
    def deserialize(
        s: str | None = None, d: dict[str, object] | None = None
    ) -> LinearCombinationAnalysis:
        """Reconstruct from a JSON string or dict.

        Every key is read strictly, with no defaults. That is deliberate: this class has
        never landed, so no payload written by an earlier version of it exists anywhere
        to be compatible with, and a missing key means a corrupt or hand-edited payload
        -- better a `KeyError` here than a silently defaulted field in a reported
        estimate.

        """
        if s is not None and d is not None:
            raise ValueError("Provide `s` or `d`, not both.")
        if s is not None:
            d = json.loads(s)
        if d is None:
            raise ValueError("Provide either `s` or `d`.")
        iv = cast("list[object]", d["effect_interval"])
        terms = tuple(
            LinearCombinationTerm.from_dict(cast("dict[str, object]", t))
            for t in cast("list[object]", d["terms"])
        )
        # `cast` is a no-op at runtime, so without these an edited or
        # hand-written payload reconstructs an object carrying an invalid enum
        # and only fails much later, in whichever consumer dispatches on it.
        alternative = cast("str", d["alternative"])
        target = cast("str", d["target"])
        method = cast("str", d["method"])
        _validate_alternative(alternative)
        _validate_target(target)
        _validate_method(method)
        alpha = _decode_float(d["alpha"])
        gamma = _decode_float(d["gamma"])
        null_value = _decode_float(d["null_value"])
        affine = _decode_float(d["affine"])
        effect = _decode_float(d["effect"])
        p_value = _decode_float(d["p_value"])
        gamma_star = _decode_float(d["gamma_star"])
        lo, hi = _decode_float(iv[0]), _decode_float(iv[1])
        n_pairs = _validated_n_pairs(d["n_pairs"])
        # `_validated_n_pairs` can only judge the field in isolation, but the
        # count and the terms are not independent: the constructor rejects a
        # component table with zero pairs, so `analyze` emits a positive count
        # for any non-empty `terms` and zero only for none at all. A payload
        # that breaks the biconditional is one `analyze` could not have
        # written, and it does not fail loudly -- it renders, with every
        # iSuccesses cell blank and the footer quietly switched to scaled
        # units. That is a degraded table passing for a real one, which is the
        # thing this decode path exists to refuse.
        if (n_pairs > 0) != bool(terms):
            raise ValueError(
                f"`n_pairs` must be positive if and only if `terms` is "
                f"non-empty, got n_pairs={n_pairs} with {len(terms)} term(s)."
            )
        _validate_payload_numbers(
            alpha=alpha,
            alternative=alternative,
            gamma=gamma,
            null_value=null_value,
            affine=affine,
            effect=effect,
            effect_interval=(lo, hi),
            gamma_star=gamma_star,
            p_value=p_value,
        )
        return LinearCombinationAnalysis(
            affine=affine,
            terms=terms,
            effect=effect,
            effect_interval=(lo, hi),
            p_value=p_value,
            n_pairs=n_pairs,
            alpha=alpha,
            gamma=gamma,
            null_value=null_value,
            alternative=alternative,
            target=target,
            monotonic=bool(d["monotonic"]),
            method=method,
            gamma_star=gamma_star,
        )

    def _pvalue_cell(self) -> str:
        """The p-value cell: the value, asterisked when significant at ``alpha``.

        The asterisk reads :attr:`significant` -- the interval itself -- rather than
        comparing ``p_value`` to ``alpha``, so the two can never disagree at a level
        where the bisected p-value lands on the exclusion threshold.

        A p-value at the bisection floor is rendered ``<1e-12`` rather than ``1e-12``.
        :meth:`LinearCombinationEstimator.pvalue` returns the floor when even the
        *widest* interval in its bracket excludes the null, which is a saturation of the
        search and not a resolved threshold: the true p-value is somewhere at or below
        it. Printing the bare number quotes a precision the inversion never had, and
        reads identically to a value the bisection did resolve. Note this is not the
        same as an exact ``0.0``, which an all-constant combination genuinely has and
        which still prints as itself.

        """
        if 0.0 < self.p_value <= _PVALUE_EPS:
            floor = format_with_min_nonzero_digits(_PVALUE_EPS, 3, percentage=False)
            pval = f"<{floor}"
        else:
            pval = format_with_min_nonzero_digits(self.p_value, 3, percentage=False)
        flag = "*" if self.significant else ""
        return f"{pval}{flag}"

    def _gamma_star_cell(self) -> str:
        """The ``Γ•`` cell, rendering an unbounded sensitivity value as ``inf``."""
        return "inf" if math.isinf(self.gamma_star) else f"{self.gamma_star:g}"

    def _row(
        self,
        label: str,
        coef: str,
        effect: float,
        ci: tuple[float, float],
        *,
        combined: bool,
    ) -> list[str]:
        """One table row; the inferential cells are populated on Combined only."""
        lo, hi = ci
        n = self.n_pairs
        if n:
            counts = [
                _fmt_count(effect * n),
                f"{_fmt_count(lo * n)}, {_fmt_count(hi * n)}",
            ]
        else:
            # A combination with no terms has no pairs to count over (and
            # `inf * 0` is NaN, not an infinite count), so leave them blank.
            # Note the gate is on the terms, not on the coefficients: a term
            # with a zero coefficient still declares the pair set.
            counts = ["", ""]
        return [
            label,
            coef,
            _fmt_effect(effect),
            f"{_fmt_effect(lo)}, {_fmt_effect(hi)}",
            *counts,
            self._pvalue_cell() if combined else "",
            self._gamma_star_cell() if combined else "",
        ]

    def __str__(self) -> str:
        """Render a per-term-plus-combined org-mode table, with a footer."""
        headers = [
            "Term",
            "Coef",
            self.target,
            _CI_HEADER,
            _ISUCCESSES_HEADER,
            _CI_HEADER,
            _PVALUE_HEADER,
            _GAMMA_HEADER,
        ]
        colalign = [
            "left",
            "right",
            "right",
            "center",
            "right",
            "center",
            "right",
            "right",
        ]
        rows = [
            self._row(
                term.label,
                f"{term.coefficient:+g}",
                term.effect,
                term.effect_interval,
                combined=False,
            )
            for term in self.terms
        ]
        rows.append(
            self._row("Combined", "", self.effect, self.effect_interval, combined=True)
        )
        table = tabulate(rows, headers=headers, colalign=colalign, tablefmt="orgtbl")
        return f"{table}\n{self._footer()}"

    def _footer(self) -> str:
        """Explanatory footer: significance and coverage.

        Reuses the ``*`` (p-value) and ``**`` (coverage) notes of
        :meth:`PairedOutcomeTable.analyze`'s summary verbatim (the ``Γ•`` value has its
        own column, so it needs no sentence).

        """
        mono = " (assuming monotonicity)" if self.monotonic else ""
        # Shared with the single-table footer rather than restated: the notes
        # below are that footer's, verbatim, and the two are only worth having
        # in common while they stay identical.
        alpha_f, sided, relation = _footer_phrasing(self.alpha, self.alternative)
        null_count = self.null_value * self.n_pairs
        if (
            self.n_pairs
            # `analyze` rejects a non-finite null, but this class is a plain
            # frozen dataclass that can also be built by hand or by
            # `deserialize`, and `round(inf)` raises where `_fmt_effect`
            # renders. Rendering must not be the thing that blows up.
            and math.isfinite(null_count)
            # Absolute-only, deliberately -- integrality does not scale -- but
            # sized to the product's own round-off rather than fixed, which is
            # not the same concession. `math.isfinite` above short-circuits
            # ahead of it, so `ulp` never sees an infinity or a NaN.
            and math.isclose(
                null_count,
                round(null_count),
                rel_tol=0.0,
                abs_tol=_count_integrality_tol(null_count),
            )
        ):
            # A grouped integer, like the iSuccesses cells -- but unsigned, as
            # `PairedOutcomeTable.analyze` writes this same note. A null is a
            # threshold; the `+` the cells carry is there to make the direction
            # of an estimate scannable down the column.
            null_desc = f"{_ISUCCESSES_HEADER} {relation} {null_count:,.0f}"
        else:
            # A combination with no terms renders no iSuccesses cells, and a null
            # that is not a whole number of successes has no faithful rendering
            # in a column of integers. Either way, fall back to the scaled units
            # the table does display.
            null_desc = f"{self.target} {relation} {_fmt_prose_effect(self.null_value)}"
        notes = [
            "*  An asterisk in the p-Value column indicates statistical "
            "significance at\n"
            f"   level {alpha_f}, provided Γ≤{self.gamma:g}{mono}.\n"
            f"   p-Value is {sided} against the null hypothesis that "
            f"{null_desc}.",
            f"** Confidence intervals have coverage of at least "
            f"{self.confidence:.0%}, provided Γ≤{self.gamma:g}{mono}.",
        ]
        if self.affine != 0.0:
            notes.append(
                f"Includes a constant offset of {_fmt_prose_effect(self.affine)}."
            )
        return "\n".join(notes)


class LinearCombinationEstimator:
    r"""Affine combination of matched-pair net effects on shared pairs.

    Estimates

    .. math::

        \theta = \text{affine} + \sum_i c_i\,\theta_i,

    where each :math:`\theta_i` is the scaled net effect (a proportion) of one
    :class:`~pair_match.net_effects.PairedOutcomeTable` for the ``target`` the analysis
    asks for -- the ATT for ``target='ATT'``, the ATU for ``target='ATU'``, or the ATE
    for ``target='ATE'``.

    The estimator holds only the *combination* -- which tables, which coefficients, what
    offset. ``target``, ``monotonic`` and ``method`` describe an analysis of it and are
    passed to the method that performs one, exactly as
    :class:`~pair_match.net_effects.PairedOutcomeTable` takes them. One estimator can
    therefore report the same combination as an ATT and an ATE without being rebuilt.

    **Inference is Bonferroni, not variance-propagated.** Every component is computed on
    the *same* matched pairs, so the component effects are dependent (a pair that
    discordantly favors treatment on one outcome tends to on another). We split the
    level -- ``alpha / k`` across the ``k`` terms with nonzero coefficient -- take each
    component's exact net-effects interval at its share, and combine the endpoints
    sign-aware (a union bound, valid under arbitrary dependence). The cost is
    conservatism: the interval ignores the positive correlation between components. The
    ``D_pair`` follow-up (pair-level differencing + Pagano-Tritchler) reclaims that
    correlation with a single signed-score inference and no Bonferroni penalty.

    ``alpha / k`` is the level handed to each component, not the level reaching each
    *quantile*. A two-sided component splits its share again across the two tails, and
    ``target='ATE'`` splits once more across ``A_1`` and ``A_0`` (see
    :func:`~pair_match.net_effects._ate_confidence_core`), so the smallest level inverted
    anywhere is ``alpha / (2 k)`` for ATT/ATU and ``alpha / (4 k)`` for ATE. Coverage is
    still at least ``1 - alpha`` -- the extra splits only make the interval wider -- but
    an ATE combination is materially more conservative than the same combination on ATT.

    Parameters
    ----------
     terms : sequence of (float, PairedOutcomeTable)
        Coefficients paired with the tables they scale. All tables must share the same
        ``n_pairs`` (they describe the same matched pairs under different outcomes); a
        mismatch raises ``ValueError``. Every coefficient must be finite. One entry per
        *outcome*: two entries on the same outcome each pay a Bonferroni share, so sum
        their coefficients into a single term rather than listing the table twice.
     affine : float, optional
        Constant offset added to the combination (finite). Defaults to ``0.0``.
     labels : sequence of str, optional
        Display labels for the terms, one per entry in ``terms``, used by
        :meth:`analyze`'s summary. Defaults to ``"term 1"``, ``"term 2"``, ...

    Notes
    -----
    The constructor's arguments are validated once and nothing re-validates them
    afterwards, so treat the attributes it sets as read-only: reassigning ``est.affine``
    is not prevented, it simply skips that validation and will produce invalid or NaN
    bounds. ``terms`` and ``labels`` are the exceptions -- they are frozen into tuples,
    because a shared list can be mutated *without* any assignment to the estimator (the
    caller need only keep the list they passed in), which is the subtler footgun of the
    two and the only one a tuple can close. Build a new estimator rather than editing
    one in place. (The analysis arguments carry no such caveat: each method validates
    the ones it is handed, every time.)

    Nothing ties one call's ``target`` to another's, which is the price of taking them
    per call: a ``Γ•`` computed for the ATE and an interval computed for the ATT are not
    a matched pair, and neither reports the mismatch. Prefer :meth:`analyze`, which runs
    one target across all of them and records which, when the numbers are going to be
    read together.

    """

    def __init__(
        self,
        terms: Sequence[tuple[float, PairedOutcomeTable]],
        *,
        affine: float = 0.0,
        labels: Sequence[str] | None = None,
    ) -> None:
        # Freeze first, and validate the frozen tuple -- not the argument. Two
        # reasons, and the order matters for both. The `n_pairs` invariant
        # checked below has to hold for the life of the estimator, which it
        # cannot if a caller's list (or `est.terms`) is still mutable behind it.
        # And the checks read `terms` more than once: the annotation says
        # `Sequence`, but the runtime accepts a one-shot iterable, and a
        # generator validated in place would be *consumed* by the finiteness
        # pass, leaving the `n_pairs` pass nothing to look at -- the estimator
        # would come out an empty "pure constant" that silently reports the
        # offset alone instead of raising.
        self.terms: tuple[tuple[float, PairedOutcomeTable], ...] = tuple(terms)
        if not math.isfinite(affine) or not all(
            math.isfinite(c) for c, _ in self.terms
        ):
            # A NaN coefficient survives the `c != 0.0` filter and silently
            # turns every bound into NaN; an infinite one poisons the sum.
            raise ValueError("`affine` and every coefficient must be finite.")
        n_pairs = {table.n_pairs for _, table in self.terms}
        if len(n_pairs) > 1:
            raise ValueError(
                "all component tables must describe the same matched pairs "
                f"(got differing `n_pairs`: {sorted(n_pairs)})."
            )
        if 0 in n_pairs:
            # Every component effect is a per-pair rate, so an empty table has
            # nothing to divide by. (An empty `terms` -- a pure constant -- is
            # still legal; only a table with no pairs is not.)
            raise ValueError("component tables must have at least one pair.")
        if labels is not None and len(labels) != len(self.terms):
            raise ValueError(
                f"`labels` has {len(labels)} entries but there are "
                f"{len(self.terms)} terms."
            )
        self.affine = affine
        # Frozen for the same reason, and one more: `analyze` zips labels
        # against terms, so a shortened list would silently drop a term from the
        # breakdown while the Combined row still counted it.
        self.labels: tuple[str, ...] = (
            tuple(labels)
            if labels is not None
            else tuple(f"term {i + 1}" for i in range(len(self.terms)))
        )

    @property
    def n_pairs(self) -> int:
        """The matched-pair count shared by every component table.

        The constructor requires the tables to agree, so any of them reports the shared
        count. A combination with no terms -- a bare offset with no table behind it --
        has no pairs and reports ``0``.

        Note this is *not* the "all-constant" condition :meth:`pvalue` and
        :meth:`gamma_star` answer exactly, which is the weaker "no term has a nonzero
        coefficient". A term carries its table's pair set whatever its coefficient, so
        ``[(0.0, table)]`` is all-constant for inference -- nothing in it moves with
        ``alpha`` or ``gamma`` -- and still reports ``table.n_pairs`` here. That is
        deliberate: the pair count describes the *design* the combination is stated
        over, and the offset is a per-pair rate on that design, so ``affine * n_pairs``
        is a real count of successes. Zeroing it would silently blank a column that has
        a faithful rendering.

        """
        return self.terms[0][1].n_pairs if self.terms else 0

    def point_estimate(self) -> float:
        r"""The affine combination of the component point estimates.

        Each component contributes its McNemar pivot ``ate_hat`` -- the same value
        estimates the ATT, ATU, and ATE -- so the point estimate is target-invariant.

        """
        return self.affine + sum(c * table.ate_hat for c, table in self.terms)

    def _component_count_interval(
        self,
        table: PairedOutcomeTable,
        *,
        alpha: float,
        gamma: float,
        alternative: str,
        target: str,
        monotonic: bool,
        method: str,
    ) -> tuple[float, float]:
        """A single component's net-effect interval at ``alpha``, in iSuccesses.

        Counts rather than rates, because that is the scale the endpoints are actually
        attained on -- see :meth:`_combine` for why the division is deferred to the end
        of the combination rather than done here.

        """
        # `attributable_effect_interval` is specified in coverage, so the share
        # makes a round trip through `1 - alpha` and back. That is lossless for
        # any level worth quoting, but the largest double below 1.0 is
        # 1 - 2**-53: once the share falls under ~5.5e-17 the subtraction lands
        # on exactly 1.0 and the component is inverted at level *zero*, which
        # returns the whole line and so is silently never significant. The share
        # is the caller's alpha divided across the k nonzero terms, so enough
        # terms can underflow a level the caller thought was fine.
        confidence = 1.0 - alpha
        if confidence >= 1.0:
            raise ValueError(
                f"the per-term level {alpha} is too small to express as a "
                "coverage: `1 - alpha` rounds to 1.0, which would invert the "
                "component at level 0 and return an unbounded interval. Raise "
                "`alpha` (the per-term level is the caller's alpha split across "
                "the nonzero terms)."
            )
        return attributable_effect_interval(
            table,
            target=target,
            confidence=confidence,
            gamma=gamma,
            monotonic=monotonic,
            alternative=alternative,
            method=method,
        )

    def _component_interval(
        self,
        table: PairedOutcomeTable,
        *,
        alpha: float,
        gamma: float,
        alternative: str,
        target: str,
        monotonic: bool,
        method: str,
    ) -> tuple[float, float]:
        """A single component's scaled net-effect interval at level ``alpha``."""
        lo, hi = self._component_count_interval(
            table,
            alpha=alpha,
            gamma=gamma,
            alternative=alternative,
            target=target,
            monotonic=monotonic,
            method=method,
        )
        n = table.n_pairs
        return (lo / n, hi / n)

    def _combine(
        self,
        *,
        alpha: float,
        gamma: float,
        alternative: str,
        target: str,
        monotonic: bool,
        method: str,
    ) -> tuple[float, float]:
        r"""Bonferroni combine the component intervals, sign-aware.

        The level ``alpha`` is split evenly across the ``k`` nonzero-coefficient terms.
        For a two-sided combination each component uses its two-sided interval at
        ``alpha / k``; for a one-sided combination each contributes only the bound the
        combined sign needs, taken one-sided at ``alpha / k`` (so a positive coefficient
        wants its lower bound for the combined lower bound, a negative coefficient wants
        its upper bound, and vice versa).

        The combination is accumulated in **counts** and scaled once at the end, rather
        than scaling each component and summing rates. The two differ by round-off, and
        the difference is not cosmetic: every component endpoint is a whole number of
        iSuccesses over a shared ``n_pairs``, so a combined bound that lands on a whole
        count is exactly representable as ``count / n_pairs`` -- the same double a
        caller writing that rate as a null gets. Summing rates instead can miss it by an
        ulp (``34 / 1000 - 14 / 1000 == 0.020000000000000004``, not ``0.02``), and
        :meth:`_excludes` compares against the null strictly, so an ulp decides
        significance when the bound sits exactly on the null.

        """
        nonzero = [(c, table) for c, table in self.terms if c != 0.0]
        k = len(nonzero)
        if k == 0:
            # A pure constant: no sampling uncertainty enters.
            if alternative == "greater":
                return (self.affine, math.inf)
            if alternative == "less":
                return (-math.inf, self.affine)
            return (self.affine, self.affine)
        share = alpha / k

        lb_count = ub_count = 0.0
        for c, table in nonzero:
            if alternative == "two-sided":
                lo, hi = self._component_count_interval(
                    table,
                    alpha=share,
                    gamma=gamma,
                    alternative="two-sided",
                    target=target,
                    monotonic=monotonic,
                    method=method,
                )
                near, far = (lo, hi) if c > 0 else (hi, lo)
                lb_count += c * near
                ub_count += c * far
                continue
            # One-sided: only the informative combined bound is built, from the
            # one component bound its sign contributes, taken one-sided.
            side = _component_side(alternative, c)
            lo, hi = self._component_count_interval(
                table,
                alpha=share,
                gamma=gamma,
                alternative=side,
                target=target,
                monotonic=monotonic,
                method=method,
            )
            bound = lo if side == "greater" else hi
            if alternative == "greater":
                lb_count += c * bound
            else:
                ub_count += c * bound

        # One shared `n_pairs`: the estimator rejects terms that disagree.
        n = self.n_pairs
        if alternative == "greater":
            return (self.affine + lb_count / n, math.inf)
        if alternative == "less":
            return (-math.inf, self.affine + ub_count / n)
        return (self.affine + lb_count / n, self.affine + ub_count / n)

    def confidence_interval(
        self,
        *,
        alpha: float = 0.10,
        alternative: str = "two-sided",
        target: str = "ATT",
        monotonic: bool = False,
        method: str = "auto",
    ) -> tuple[float, float]:
        r"""Randomized (``gamma = 1``) Bonferroni interval for the combination.

        The union-bound interval at coverage ``1 - alpha`` assuming no hidden bias. Use
        :meth:`expanded_confidence_interval` to entertain a sensitivity parameter.

        Parameters
        ----------
         alpha : float, optional
            Significance level; the interval has coverage ``1 - alpha`` (default 0.10,
            the RL MDS convention).
         alternative : {'two-sided', 'less', 'greater'}, optional
            The kind of interval. ``'two-sided'`` returns two finite bounds;
            ``'greater'`` returns ``[lb, +inf)`` and ``'less'`` ``(-inf, ub]``. Defaults
            to ``'two-sided'``.
         target : {'ATT', 'ATU', 'ATE'}, optional
            The effect each component reports, and so the scale of the result. Defaults
            to ``'ATT'``.
         monotonic : bool, optional
            Assume treatment never hurts any unit; narrows every component set. Defaults
            to ``False`` (the net-effects default -- no such assumption).
         method : {'exact', 'normal', 'auto'}, optional
            How each component's worst-case test is inverted, forwarded to
            :func:`~pair_match.net_effects.attributable_effect_interval`. Defaults to
            ``'auto'``.

        """
        # `alternative` first: `_validate_alpha` reads it to decide whether the
        # one-sided ceiling applies, so an unrecognized value must be named for
        # what it is rather than reported as an alpha problem.
        _validate_alternative(alternative)
        _validate_alpha(alpha, alternative)
        _validate_target(target)
        _validate_method(method)
        return self._combine(
            alpha=alpha,
            gamma=1.0,
            alternative=alternative,
            target=target,
            monotonic=monotonic,
            method=method,
        )

    def expanded_confidence_interval(
        self,
        *,
        alpha: float = 0.10,
        gamma: float = 6.0,
        alternative: str = "two-sided",
        target: str = "ATT",
        monotonic: bool = False,
        method: str = "auto",
    ) -> tuple[float, float]:
        r"""Sensitivity-expanded Bonferroni interval at hidden bias ``gamma``.

        Widens :meth:`confidence_interval` to allow a hidden bias of odds ratio
        ``gamma`` in the pair assignment. At ``gamma == 1`` it equals the randomized
        interval; larger ``gamma`` widens each component set (and so the combination).
        The union bound holds at every ``gamma``.

        Parameters
        ----------
         alpha : float, optional
            Significance level; coverage ``1 - alpha`` (default 0.10).
         gamma : float, optional
            Rosenbaum sensitivity parameter (finite and ``>= 1``; default 6.0).
         alternative : {'two-sided', 'less', 'greater'}, optional
            The kind of interval; see :meth:`confidence_interval`. Defaults to
            ``'two-sided'``.
         target : {'ATT', 'ATU', 'ATE'}, optional
            The effect each component reports; see :meth:`confidence_interval`. Defaults
            to ``'ATT'``.
         monotonic : bool, optional
            Assume treatment never hurts any unit; narrows every component set. Defaults
            to ``False``.
         method : {'exact', 'normal', 'auto'}, optional
            How each component's worst-case test is inverted; see
            :meth:`confidence_interval`. Defaults to ``'auto'``.

        """
        _validate_alternative(alternative)
        _validate_alpha(alpha, alternative)
        _validate_gamma(gamma)
        _validate_target(target)
        _validate_method(method)
        return self._combine(
            alpha=alpha,
            gamma=gamma,
            alternative=alternative,
            target=target,
            monotonic=monotonic,
            method=method,
        )

    def _excludes(
        self,
        *,
        alpha: float,
        gamma: float,
        alternative: str,
        null_value: float,
        target: str,
        monotonic: bool,
        method: str,
    ) -> bool:
        """Whether the level-``alpha``, ``gamma``-expanded interval excludes the null.

        The predicate that both :meth:`gamma_star` and :meth:`pvalue` invert -- one
        bisecting it over ``gamma``, the other over ``alpha``. Shared so the two
        searches cannot come to disagree about what "significant" means.

        """
        lb, ub = self._combine(
            alpha=alpha,
            gamma=gamma,
            alternative=alternative,
            target=target,
            monotonic=monotonic,
            method=method,
        )
        return lb > null_value or ub < null_value

    def gamma_star(
        self,
        *,
        null_value: float = 0.0,
        alpha: float = 0.10,
        alternative: str = "two-sided",
        target: str = "ATT",
        monotonic: bool = False,
        method: str = "auto",
    ) -> float:
        r"""Rosenbaum sensitivity value ``Γ•`` for the combined finding.

        The largest hidden bias ``gamma`` at which the expanded (Bonferroni) interval at
        level ``alpha`` still excludes ``null_value`` -- the point where the widening
        interval first admits the null and the finding stops being significant. Returns
        ``1.0`` when the randomized interval already contains ``null_value``, and
        ``math.inf`` when the interval excludes it for arbitrarily large ``gamma`` (e.g.
        an all-constant combination whose offset alone clears the null).

        Parameters
        ----------
         null_value : float, optional
            The value the interval is tested against (default ``0.0``).
         alpha : float, optional
            Significance level; the interval has coverage ``1 - alpha`` (default 0.10).
         alternative : {'two-sided', 'less', 'greater'}, optional
            The kind of interval inverted; see :meth:`confidence_interval`. Defaults to
            ``'two-sided'``.
         target : {'ATT', 'ATU', 'ATE'}, optional
            The effect the sensitivity value concerns; see :meth:`confidence_interval`.
            Defaults to ``'ATT'``.
         monotonic : bool, optional
            Assume treatment never hurts any unit; raises ``Γ•``. Defaults to ``False``.
         method : {'exact', 'normal', 'auto'}, optional
            How the inverted intervals are computed; see :meth:`confidence_interval`.
            Defaults to ``'auto'``.

        """
        _validate_alternative(alternative)
        _validate_alpha(alpha, alternative)
        _validate_null_value(null_value)
        _validate_target(target)
        _validate_method(method)
        return _gamma_star_search(
            lambda gamma: self._excludes(
                alpha=alpha,
                gamma=gamma,
                alternative=alternative,
                null_value=null_value,
                target=target,
                monotonic=monotonic,
                method=method,
            )
        )

    def pvalue(
        self,
        *,
        null_value: float = 0.0,
        gamma: float = 1.0,
        alternative: str = "two-sided",
        target: str = "ATT",
        monotonic: bool = False,
        method: str = "auto",
    ) -> float:
        r"""Bonferroni p-value for ``H_0: <combination> = null_value``.

        The smallest level ``alpha`` at which the ``gamma``-expanded Bonferroni interval
        excludes ``null_value`` -- the p-value dual of
        :meth:`expanded_confidence_interval`, obtained by inverting it. Valid
        (conservative) under arbitrary dependence via the same union bound. Lacking a
        closed form for the combined test, it is found by bisection: the interval
        narrows as ``alpha`` grows, so exclusion is monotone and the threshold is the
        p-value. The bracket is ``[1e-12, 1 - 1e-12]``; see :data:`_PVALUE_EPS` for why
        that floor is where it is. Returns ``1.0`` when even the narrowest
        (near-zero-coverage) interval contains the null, and the floor when even the
        widest interval excludes it.

        An all-constant combination (no term with a nonzero coefficient) is answered
        exactly instead: it carries no sampling uncertainty, so the interval does not
        move with ``alpha`` and there is no threshold to bisect for. The offset either
        clears the null -- ``0.0`` -- or it does not -- ``1.0``. This mirrors
        :meth:`gamma_star`, which reports an exact ``math.inf`` for the same
        combination.

        The bracket deliberately runs past ``0.5``, which :func:`_validate_alpha`
        refuses for a one-sided ``alternative``, and the two are not in conflict: that
        guard is about what a caller may *ask* for. A one-sided interval at ``alpha >=
        0.5`` lands on the far side of the point estimate and has no coverage reading,
        so nobody should be handed one. Here the levels are not coverage claims but the
        search variable of an inversion, and the answer is a p-value: a one-sided test
        of a null the data point *away* from has a p-value above 0.5, and it is exactly
        the levels above 0.5 that measure how far above. Truncating the bracket at 0.5
        would report every such null as ``p = 0.5``, collapsing the whole uninformative
        half onto one number.

        The value returned is the *upper* end of the final bracket, so it is an
        over-estimate of the true threshold -- the conservative direction, since a
        p-value rounded up never overstates the evidence. The over-estimate is bounded
        by whichever of the two stopping rules binds first: the relative break gives
        ``1e-9 * p``, and the 40-iteration cap gives ``2**-40`` (~9.1e-13) absolute. The
        relative rule is the binding one down to ``p ~ 9e-4``; below that the bracket
        simply runs out of iterations and the absolute bound governs, which is the
        tighter guarantee anyway.

        The consequence is that agreement with
        :attr:`LinearCombinationAnalysis.significant` is exact only outside a band of
        that width around ``alpha``: if the threshold falls inside it, the reported
        p-value can sit a hair above ``alpha`` while the interval genuinely excludes the
        null. ``significant`` reads the interval directly and is authoritative there;
        the p-value is the numerically-inverted summary of the same fact.

        Parameters
        ----------
         null_value : float, optional
            The value tested against (default ``0.0``).
         gamma : float, optional
            Rosenbaum sensitivity parameter entertained for the test (``>= 1``; default
            1.0, the randomized case).
         alternative : {'two-sided', 'less', 'greater'}, optional
            The kind of test inverted; see :meth:`confidence_interval`. Defaults to
            ``'two-sided'``.
         target : {'ATT', 'ATU', 'ATE'}, optional
            The effect tested; see :meth:`confidence_interval`. Defaults to ``'ATT'``.
         monotonic : bool, optional
            Assume treatment never hurts any unit; sharpens the p-value. Defaults to
            ``False``.
         method : {'exact', 'normal', 'auto'}, optional
            How the inverted intervals are computed; see :meth:`confidence_interval`.
            Defaults to ``'auto'``.

        """
        _validate_alternative(alternative)
        _validate_gamma(gamma)
        _validate_null_value(null_value)
        _validate_target(target)
        _validate_method(method)

        def excludes(alpha: float) -> bool:
            return self._excludes(
                alpha=alpha,
                gamma=gamma,
                alternative=alternative,
                null_value=null_value,
                target=target,
                monotonic=monotonic,
                method=method,
            )

        # A pure constant has no sampling uncertainty: `_combine` ignores
        # `alpha` entirely, so `excludes` is constant and there is nothing to
        # invert. Answer exactly rather than letting the bisection report a
        # bracket endpoint -- the offset either clears the null or it does not,
        # and `1e-12` would read as a very small p-value when the truth is that
        # there is no sampling error at all. This is the same exactness
        # `gamma_star` already gives the case, where it returns `math.inf`.
        if not any(c != 0.0 for c, _ in self.terms):
            return 0.0 if excludes(0.5) else 1.0

        hi = 1.0 - _PVALUE_EPS
        if not excludes(hi):
            return 1.0
        lo = _PVALUE_EPS
        if excludes(lo):
            return lo
        # 40 halvings of a unit bracket resolve the threshold to ~9.1e-13, the
        # floor's own scale -- far past the discreteness of the exact interval.
        # Stop early once the bracket is tight in *relative* terms: a p-value is
        # read as a significance, so 1e-9 of relative width is already more than
        # any caller can use, and it saves ~8 `_combine` sweeps at p ~ 0.3. Below
        # p ~ 9e-4 the relative test cannot fire before the cap does, so the cap
        # is what stops the loop there.
        for _ in range(40):
            if hi - lo <= 1e-9 * hi:
                break
            mid = 0.5 * (lo + hi)
            if excludes(mid):
                hi = mid
            else:
                lo = mid
        return hi

    def sensitivity_analysis(
        self, gamma: float = 6.0, *, target: str = "ATT", monotonic: bool = False
    ) -> tuple[float, float]:
        r"""Confounding-only band for the combination at hidden bias ``gamma``.

        The range the combined estimate could take under a hidden bias of odds ratio
        ``gamma`` from confounding *alone* -- no sampling uncertainty -- mirroring
        :meth:`PairedOutcomeTable.sensitivity_analysis`. Each component's
        confounding-only band is combined sign-aware (a positive coefficient contributes
        its lower bound to the combined lower bound, a negative coefficient its upper
        bound, and vice versa). Collapses to :meth:`point_estimate` at ``gamma == 1``
        and opens with ``gamma``, saturating at each component's a-priori range rather
        than diverging.

        Like the Bonferroni interval this is conservative: the components share matched
        pairs, so the true joint worst case is a subset of the independent per-component
        worst cases combined here. No level is split, though -- a confounding-only band
        carries no sampling error, so there is no Bonferroni penalty.

        Parameters
        ----------
         gamma : float, optional
            Rosenbaum sensitivity parameter (``>= 1``; default 6.0).
         target : {'ATT', 'ATU', 'ATE'}, optional
            The effect the band is drawn for. Defaults to ``'ATT'``.
         monotonic : bool, optional
            Assume treatment never hurts any unit; narrows the band. Defaults to
            ``False``.

        Notes
        -----
        There is no ``method`` here, unlike the interval methods: a confounding-only
        band inverts nothing, so there is no test to choose an exact or normal form for.

        ``gamma`` stays positional to mirror
        :meth:`PairedOutcomeTable.sensitivity_analysis`, which users move between. The
        mirror stops there: the sibling's second positional is ``monotonic``, and this
        one has a ``target`` the single table does not, so a positional second argument
        would mean different things in the two classes. Keyword-only from ``target`` on,
        which is also the convention every other analysis method here follows.

        """
        _validate_gamma(gamma)
        _validate_target(target)
        lb = ub = self.affine
        for c, table in self.terms:
            if c == 0.0:
                continue
            if target == "ATE":
                lo, hi = table.sensitivity_analysis(gamma, monotonic=monotonic)
            else:
                lo, hi = _attributable_sensitivity_band(
                    table, target=target, gamma=gamma, monotonic=monotonic
                )
            if c > 0:
                lb += c * lo
                ub += c * hi
            else:
                lb += c * hi
                ub += c * lo
        return (lb, ub)

    def capacity(self, alpha: float = 0.05) -> float:
        r"""Design-sensitivity ceiling of the combination.

        The smallest :meth:`PairedOutcomeTable.capacity` over the components with a
        nonzero coefficient -- the most binding one, since the combination is
        uninformative once any contributing component is. Each component's level is its
        Bonferroni share ``alpha / k``, the share the union bound spends on it. Returns
        ``math.inf`` when no coefficient is nonzero (no component can degrade).

        For ``target='ATE'`` the share is not the last split: the ATE interval halves
        its level once more across ``A_1`` and ``A_0``, so the level a component is
        really inverted at is ``alpha / (2k)``, and a capacity falls as its level does.
        The share is quoted at ``alpha / k`` anyway, because
        :meth:`PairedOutcomeTable.capacity` quotes an ATE capacity the same way and a
        capacity is only worth reading against another quoted on the same convention.
        The cost is that the ceiling reported for an ATE combination sits slightly above
        the level-consistent one, so the ``gamma_max`` default :meth:`plot_sensitivity`
        derives from it can sweep a little past the point where the wider band has
        already saturated -- cosmetic, and in the conservative direction for a *sweep*
        range.

        Parameters
        ----------
         alpha : float, optional
            Level split across the components (default 0.05). Note this is
            :meth:`PairedOutcomeTable.capacity`'s default, not the 0.10 the inference
            methods on this class use: a capacity is a property of the design that is
            quoted against the conventional level, and the two capacities have to be
            comparable to be worth comparing.

        """
        _validate_alpha(alpha)
        nonzero = [table for c, table in self.terms if c != 0.0]
        if not nonzero:
            return math.inf
        share = alpha / len(nonzero)
        return min(table.capacity(share) for table in nonzero)

    def plot_sensitivity(
        self,
        *,
        null_value: float = 0.0,
        alpha: float = 0.10,
        gamma_max: float | None = None,
        num_points: int = 50,
        target: str = "ATT",
        monotonic: bool = False,
        method: str = "auto",
        legend_loc: LegendLocType = "lower left",
        title: str | None = None,
        ax: Axes | None = None,
    ) -> tuple[pd.DataFrame, Axes]:
        r"""Sweep ``Gamma`` and plot how the combined finding degrades.

        The combination's analog of :meth:`PairedOutcomeTable.plot_sensitivity`, and its
        visual companion: as the hidden-bias odds ratio ``Gamma`` grows from ``1`` (a
        randomized experiment) upward, two bands widen around the (bias-independent)
        combined point estimate --

        - the *sensitivity interval*, the range of the combination from
          confounding alone (:meth:`sensitivity_analysis`); and
        - the *sensitivity/confidence interval*, which adds sampling
          uncertainty (:meth:`expanded_confidence_interval`, Bonferroni across
          the terms).

        The left axis is the scaled combined effect; a secondary right axis rescales it
        to the matching count (``iSuccesses = effect * n_pairs``), matching
        :meth:`analyze`'s columns. A combination with no terms has no pairs to count
        over, so it is drawn without that second axis, just as :meth:`analyze` leaves
        its count columns blank. The sensitivity value ``Γ•`` -- where the wider band
        first touches ``null_value`` -- inverts the plotted band, so the dotted line and
        the band cross the null together.

        Both bands and ``Γ•`` use the ``target`` and ``monotonic`` given here, so the
        whole figure is one coherent analysis. ``method`` reaches the wider band alone
        -- it selects the test inverted for the sampling component, and the
        confounding-only band has none to invert, so it is unaffected. The plot is
        always two-sided (a one-sided band has an infinite edge and cannot be drawn),
        independent of the ``alternative`` used elsewhere.

        Parameters
        ----------
         null_value : float, optional
            The null the wider band is tested against, on the scaled-effect axis; ``Γ•``
            is computed against it (default 0.0).
         alpha : float, optional
            Significance level; the wider band has coverage ``1 - alpha`` (default
            0.10).
         gamma_max : float, optional
            Largest ``Gamma`` swept. Defaults to ``min(6, 0.95 * capacity)`` (6 is the
            smoking / lung-cancer benchmark; the cap keeps the bands finite below the
            combination's :meth:`capacity`).
         num_points : int, optional
            Number of ``Gamma`` values swept (``>= 2``; default 50).
         target : {'ATT', 'ATU', 'ATE'}, optional
            The effect plotted, and the left axis's label. Defaults to ``'ATT'``.
         monotonic : bool, optional
            Assume treatment never hurts any unit; narrows both bands and raises ``Γ•``.
            Defaults to ``False``.
         method : {'exact', 'normal', 'auto'}, optional
            How the wider band's tests are inverted; see :meth:`confidence_interval`.
            Defaults to ``'auto'``.
         legend_loc : str, optional
            Matplotlib legend location, forwarded to ``ax.legend``; use it to keep the
            legend clear of the ``Γ•`` annotation.
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
        _validate_alpha(alpha)
        _validate_target(target)
        _validate_method(method)
        # `Γ•` is computed against `null_value`, and a NaN or infinite null is
        # never excluded by any band -- the plot would draw an annotation at
        # `Γ• = 1` for a finding of any strength rather than raise.
        _validate_null_value(null_value)
        # `num_points` is `_sweep_sensitivity_bands`'s precondition and is
        # enforced there for every caller; it is checked again here only so the
        # rejection lands before any band is computed, rather than after a
        # sweep's worth of interval inversions has already been paid for.
        if num_points < 2:
            raise ValueError(f"`num_points` must be at least 2, got {num_points}.")
        gamma_max = _resolve_gamma_max(
            gamma_max, capacity=lambda: self.capacity(alpha), subject="combination"
        )
        point = self.point_estimate()
        data = _sweep_sensitivity_bands(
            point=point,
            gamma_max=gamma_max,
            num_points=num_points,
            sens_band=lambda g: self.sensitivity_analysis(
                g, target=target, monotonic=monotonic
            ),
            # Through the public method, not `_combine`: the docstring promises
            # the plotted band *is* `expanded_confidence_interval`, and going
            # via it keeps that true if the method ever gains behavior of its
            # own (a subclass override, say). Its per-call validation is
            # negligible against one interval inversion per component.
            ci_band=lambda g: self.expanded_confidence_interval(
                alpha=alpha,
                gamma=g,
                alternative="two-sided",
                target=target,
                monotonic=monotonic,
                method=method,
            ),
        )
        gamma_star = self.gamma_star(
            null_value=null_value,
            alpha=alpha,
            alternative="two-sided",
            target=target,
            monotonic=monotonic,
            method=method,
        )
        ax = _plot_sensitivity_curve(
            data,
            point=point,
            gamma_star=gamma_star,
            alpha=alpha,
            gamma_max=gamma_max,
            ylabel=_SCALED_EFFECT_HEADERS[target],
            secondary_scale=float(self.n_pairs),
            secondary_ylabel="iSuccesses",
            legend_loc=legend_loc,
            title=title,
            ax=ax,
        )
        return data, ax

    def analyze(
        self,
        *,
        alpha: float = 0.10,
        gamma: float = 1.0,
        null_value: float = 0.0,
        alternative: str = "two-sided",
        target: str = "ATT",
        monotonic: bool = False,
        method: str = "auto",
    ) -> LinearCombinationAnalysis:
        r"""Bundle the combination into a displayable, serializable result.

        Computes the combined point estimate and Bonferroni interval at sensitivity
        ``gamma``, a per-term breakdown, the worst-case p-value, and the sensitivity
        value ``Γ•`` for ``H_0: <combination> = null_value``. See
        :class:`LinearCombinationAnalysis`.

        Parameters
        ----------
         alpha : float, optional
            Significance level; the interval has coverage ``1 - alpha`` (default 0.10).
            The level is Bonferroni-split across the nonzero terms.
         gamma : float, optional
            Sensitivity parameter entertained for the interval (``>= 1``; default 1.0,
            the randomized case). Distinct from ``Γ•``.
         null_value : float, optional
            The null the interval and ``Γ•`` are tested against (default 0.0).
         alternative : {'two-sided', 'less', 'greater'}, optional
            The kind of interval; see :meth:`confidence_interval`. Defaults to
            ``'two-sided'``.
         target : {'ATT', 'ATU', 'ATE'}, optional
            The effect reported throughout the summary; see :meth:`confidence_interval`.
            Defaults to ``'ATT'``.
         monotonic : bool, optional
            Assume treatment never hurts any unit; narrows every set. Defaults to
            ``False``.
         method : {'exact', 'normal', 'auto'}, optional
            How the sets are inverted; see :meth:`confidence_interval`. Defaults to
            ``'auto'``.

        """
        _validate_alternative(alternative)
        _validate_alpha(alpha, alternative)
        _validate_gamma(gamma)
        _validate_null_value(null_value)
        _validate_target(target)
        _validate_method(method)
        k = sum(1 for c, _ in self.terms if c != 0.0)
        share = alpha / k if k else alpha
        term_summaries = tuple(
            LinearCombinationTerm(
                label=label,
                coefficient=c,
                effect=table.ate_hat,
                effect_interval=self._component_interval(
                    table,
                    alpha=share,
                    gamma=gamma,
                    # The side the combination consumed, not always two-sided:
                    # under a one-sided `alternative` the Combined row is built
                    # from one bound per component, and showing the other one
                    # alongside it invites the reader to add up numbers that
                    # were never added up. With the consumed side displayed, the
                    # per-term bounds reproduce the combined bound directly.
                    alternative=_component_side(alternative, c),
                    target=target,
                    monotonic=monotonic,
                    method=method,
                ),
            )
            # `strict` so a lost label is a loud error rather than a breakdown
            # that silently omits terms the Combined row still counts.
            for label, (c, table) in zip(self.labels, self.terms, strict=True)
        )
        return LinearCombinationAnalysis(
            affine=self.affine,
            terms=term_summaries,
            effect=self.point_estimate(),
            effect_interval=self._combine(
                alpha=alpha,
                gamma=gamma,
                alternative=alternative,
                target=target,
                monotonic=monotonic,
                method=method,
            ),
            p_value=self.pvalue(
                null_value=null_value,
                gamma=gamma,
                alternative=alternative,
                target=target,
                monotonic=monotonic,
                method=method,
            ),
            n_pairs=self.n_pairs,
            alpha=alpha,
            gamma=gamma,
            null_value=null_value,
            alternative=alternative,
            target=target,
            monotonic=monotonic,
            method=method,
            gamma_star=self.gamma_star(
                null_value=null_value,
                alpha=alpha,
                alternative=alternative,
                target=target,
                monotonic=monotonic,
                method=method,
            ),
        )


class DiffInDiff(LinearCombinationEstimator):
    r"""Difference-in-differences of two matched-pair net effects.

    Syntactic sugar over :class:`LinearCombinationEstimator` with the coefficients fixed
    to :math:`(-1, +1)`, estimating

    .. math::

        Y = S - P,

    the post-period net effect ``S`` minus the pre-period (placebo) net effect ``P`` on
    the same matched pairs. Under **parallel trends** the hidden bias on the real
    outcome is approximated by the placebo net effect, so subtracting ``P`` removes it;
    a placebo net effect far from zero is itself evidence of bias. DiD trades the
    ignorability premise for parallel trends and earns its keep when ``P`` cannot simply
    be balanced away in the design (no overlap / selection-on-trend).

    Parameters
    ----------
     pre_table : PairedOutcomeTable
        The placebo (pre-period) outcome table, ``P``.
     post_table : PairedOutcomeTable
        The real (post-period) outcome table, ``S``. Must share ``pre_table``'s
        ``n_pairs``.
     affine : float, optional
        Forwarded to :class:`LinearCombinationEstimator`; shifts the estimand to ``Y =
        affine + S - P``. It is only fixing the *coefficients* that makes this class
        sugar, so a known constant offset stays available rather than forcing a caller
        who needs one back to the general constructor and a hand-written ``(-1, +1)``.
        Defaults to ``0.0``. ``target``, ``monotonic`` and ``method`` are not
        construction arguments here either -- pass them to
        :meth:`~LinearCombinationEstimator.analyze` and the other inference methods.

    Notes
    -----
    Both tables are keyword-only. They have the same type and the same shape, so a
    positional call offers nothing to catch a swap: transposing them estimates ``P - S``
    instead of ``S - P`` and every diagnostic still looks healthy -- the sign of the
    effect simply flips. The chronological ``(pre, post)`` order is also the reverse of
    the ``S - P`` the estimand is written as above, which is precisely the sort of thing
    a reader supplies from memory. Naming them at the call site costs one word and
    removes the failure mode.

    """

    def __init__(
        self,
        *,
        pre_table: PairedOutcomeTable,
        post_table: PairedOutcomeTable,
        affine: float = 0.0,
    ) -> None:
        super().__init__(
            terms=[(-1.0, pre_table), (1.0, post_table)],
            affine=affine,
            labels=["pre", "post"],
        )

    # Read off `terms` rather than stored beside it. `terms` is frozen at
    # construction precisely so the validated invariants cannot be broken
    # afterwards, and a second copy of the same two tables would undo that:
    # rebinding `did.post_table` would leave every computation -- all of which
    # go through `terms` -- reading the original table while the attribute
    # reported the new one.
    @property
    def pre_table(self) -> PairedOutcomeTable:
        """The placebo (pre-period) table ``P``, the ``-1`` term."""
        return self.terms[0][1]

    @property
    def post_table(self) -> PairedOutcomeTable:
        """The real (post-period) table ``S``, the ``+1`` term."""
        return self.terms[1][1]
