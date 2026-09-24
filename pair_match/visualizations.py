# pyre-strict
"""Inference-stage visualizations for matched-pair sensitivity analysis."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.lines import Line2D

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.axes import Axes
    from matplotlib.typing import LegendLocType

# The conventional upper end of a sensitivity sweep: Gamma = 6 is roughly the
# hidden-bias odds ratio the smoking / lung-cancer association withstands, so a
# band still clear of the null there is about as robust as an observational
# finding is usually asked to be.
_GAMMA_BENCHMARK = 6.0

# How far below the capacity the default sweep stops. The bands are unbounded at
# the capacity itself, so the sweep has to stop short of it; 5% is enough margin
# that the last few points are still readable.
_CAPACITY_MARGIN = 0.95


def _resolve_gamma_max(
    gamma_max: float | None, *, capacity: Callable[[], float], subject: str
) -> float:
    """Fill in the sweep's upper ``Gamma`` from ``capacity``, and check it.

    Shared by the three sweeps rather than restated at each: the default is a claim
    about what a sensitivity plot should show (out to the benchmark, but not past the
    point where the bands blow up), and it is only worth making once. ``capacity`` is a
    callback rather than a number so that a caller who named its own ``gamma_max`` does
    not compute one it will not use -- today's capacities are closed-form and cheap (the
    expensive work on this path is the band sweep, ``num_points`` interval inversions
    per term), so this buys tidiness rather than speed; it also keeps a *failing*
    capacity out of the path of a caller that never asked for the default.

    The check is a courtesy, not the guarantee -- ``_sweep_sensitivity_bands`` is the
    backstop that also rejects a non-finite ``gamma_max`` and a degenerate
    ``num_points``, for every caller. What it cannot do is name the likely cause, which
    is almost always that ``subject``'s capacity is too small to sweep at all.

    """
    if gamma_max is None:
        ceiling = capacity()
        # `min` ignores a NaN argument -- every comparison against it is False,
        # so `min(6.0, nan)` is 6.0 -- and a degenerate capacity would then
        # sweep out to the benchmark as though the finding supported it, with
        # nothing in the plot to say otherwise. An *infinite* capacity is
        # legitimate (no Gamma overturns the finding) and `min` handles it.
        if math.isnan(ceiling):
            raise ValueError(
                f"the {subject}'s capacity is undefined (NaN), so no default "
                "`gamma_max` can be chosen; pass one explicitly."
            )
        gamma_max = min(_GAMMA_BENCHMARK, _CAPACITY_MARGIN * ceiling)
    if gamma_max <= 1.0:
        raise ValueError(
            f"`gamma_max` must exceed 1, got {gamma_max}; the {subject}'s "
            "capacity may be too small to sweep (it is essentially "
            "uninformative)."
        )
    return gamma_max


def _sweep_sensitivity_bands(
    *,
    point: float,
    gamma_max: float,
    num_points: int,
    sens_band: Callable[[float], tuple[float, float]],
    ci_band: Callable[[float], tuple[float, float]],
) -> pd.DataFrame:
    """Tabulate both bands over an evenly spaced ``Gamma`` sweep from ``1``.

    Builds the frame :func:`_plot_sensitivity_curve` consumes, and lives beside it for
    the same reason: the single-table and linear-combination sweeps differ only in their
    two band callbacks, so sharing the grid and the column names keeps the two callers
    from drifting apart.

    The grid's preconditions are enforced here rather than left to each caller, for the
    same reason. A ``gamma_max`` of ``inf`` or a ``num_points`` below 2 does not raise
    on its own -- ``np.linspace`` happily returns a grid of infinities or a single point
    -- so the failure would surface as an empty or degenerate plot much further
    downstream. Callers may still check earlier to give a more specific message (an
    infeasible ``gamma_max`` usually means the study's capacity is too small to sweep);
    this is the backstop that a third caller cannot forget.

    """
    if num_points < 2:
        raise ValueError(f"`num_points` must be at least 2, got {num_points}.")
    if not math.isfinite(gamma_max) or gamma_max <= 1.0:
        raise ValueError(f"`gamma_max` must be finite and > 1, got {gamma_max}.")
    records = []
    for gamma in np.linspace(1.0, gamma_max, num_points):
        g = float(gamma)
        records.append((g, point, *sens_band(g), *ci_band(g)))
    return pd.DataFrame(
        records,
        columns=[
            "gamma",
            "point",
            "sens_lower",
            "sens_upper",
            "ci_lower",
            "ci_upper",
        ],
    )


def _plot_sensitivity_curve(
    data: pd.DataFrame,
    *,
    point: float,
    gamma_star: float,
    alpha: float,
    gamma_max: float,
    ylabel: str = "Sensitivity Interval",
    secondary_scale: float | None = None,
    secondary_ylabel: str | None = None,
    legend_loc: LegendLocType = "lower left",
    title: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    r"""Draw the sensitivity plot from a pre-computed Gamma sweep.

    The headline inference-stage diagnostic, factored out so both estimator families can
    reuse it: as the hidden-bias odds ratio ``Gamma`` grows from ``1`` (a randomized
    experiment) upward, two intervals widen around the (bias-independent) point estimate
    --

    - the *sensitivity interval*, the range of the point estimate from
      confounding alone (``sensitivity_analysis``); and
    - the *sensitivity/confidence interval*, which adds sampling uncertainty
      (``expanded_confidence_interval``).

    The study's sensitivity value ``Γ•`` -- where the wider interval first touches the
    null and the finding stops being significant -- is marked when it falls within the
    swept range.

    Parameters
    ----------
     data : DataFrame
        The pre-computed Gamma sweep, with columns ``gamma``, ``point``, ``sens_lower``,
        ``sens_upper``, ``ci_lower`` and ``ci_upper``.
     point : float
        The bias-independent point estimate the bands widen around.
     gamma_star : float
        The study's sensitivity value; marked with a vertical line when it lies in ``(1,
        gamma_max]``.
     alpha : float
        Significance level; the wider band has coverage ``1 - alpha`` and labels the
        confidence-interval legend entry.
     gamma_max : float
        Largest ``Gamma`` swept; sets the x-axis limit and the ``Γ•`` guard.
     ylabel : str
        Label for the (left) effect axis (default ``'Sensitivity Interval'``).
     secondary_scale : float, optional
        When given, add a secondary right y-axis linked to the left one by this
        multiplicative factor -- e.g. ``n_pairs`` to read the scaled effect off as a
        count of induced successes. Omitted (``None``) draws no second axis.
     secondary_ylabel : str, optional
        Label for the secondary right axis; used only when ``secondary_scale`` is given.
     legend_loc : str
        Matplotlib legend location (e.g. ``'lower left'``, ``'lower right'``); forwarded
        to ``ax.legend`` (default ``'lower left'``).
     title : str, optional
        Plot title; no title is drawn when omitted.
     ax : Axes, optional
        Axes to draw on; a new figure and axes are created when omitted.

    Returns
    -------
     Axes
        The axes drawn on.

    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8.0, 5.0))

    # blue sensitivity band, orange expanded band, a black solid point estimate, and the
    # plain (Gamma = 1) confidence interval as black dashed lines. The three fills are
    # disjoint so the orange margins and the blue centre read cleanly rather than
    # blending where they overlap. The Gamma = 1 expanded CI is already the first swept
    # row (gammas start at 1.0); reuse it rather than recomputing (a McNemar inversion
    # on the binary path).
    ci0_lower = float(data["ci_lower"].iloc[0])
    ci0_upper = float(data["ci_upper"].iloc[0])
    ax.fill_between(
        data["gamma"], data["ci_lower"], data["sens_lower"], alpha=0.2, color="orange"
    )
    ax.fill_between(
        data["gamma"], data["sens_lower"], data["sens_upper"], alpha=0.2, color="blue"
    )
    ax.fill_between(
        data["gamma"], data["sens_upper"], data["ci_upper"], alpha=0.2, color="orange"
    )
    ax.axhline(point, color="black", linewidth=1.5)
    ax.axhline(ci0_lower, color="black", linestyle="--", linewidth=1.0)
    ax.axhline(ci0_upper, color="black", linestyle="--", linewidth=1.0)

    if 1.0 < gamma_star <= gamma_max:
        ax.axvline(gamma_star, color="black", linestyle=":", linewidth=1.0)
        ax.annotate(
            f"$\\Gamma^\\bullet = {gamma_star:.2f}$",
            xy=(gamma_star, ax.get_ylim()[0]),
            xytext=(4.0, 4.0),
            textcoords="offset points",
        )

    legend_handles = [
        Line2D([0], [0], color="black", linestyle="-", label="Point Estimate"),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            label=f"{1.0 - alpha:.0%} Confidence Interval",
        ),
        Line2D([0], [0], color="blue", linestyle="-", label="Sensitivity Interval"),
        Line2D(
            [0],
            [0],
            color="orange",
            linestyle="-",
            label="Expanded Confidence Interval",
        ),
    ]
    ax.set_xlim(1.0, gamma_max)
    ax.set_xlabel("Gamma")
    ax.set_ylabel(ylabel)

    # Optional right axis: the same curves read on a rescaled (count) axis. A
    # linked secondary_yaxis keeps its ticks in lock-step with the left axis as
    # the limits change, so the two scales never drift apart.
    if secondary_scale is not None and secondary_scale > 0.0:
        scale = secondary_scale

        def _to_secondary(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
            return np.asarray(values, dtype=np.float64) * scale

        def _to_primary(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
            return np.asarray(values, dtype=np.float64) / scale

        secax = ax.secondary_yaxis("right", functions=(_to_secondary, _to_primary))
        if secondary_ylabel is not None:
            secax.set_ylabel(secondary_ylabel)

    if title is not None:
        ax.set_title(title)
    ax.legend(handles=legend_handles, loc=legend_loc)
    return ax
