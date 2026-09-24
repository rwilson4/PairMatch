"""The matched pairs a net-effects analysis is computed on.

This package does the *estimation* half of a matched study -- it takes pairs as given
and draws inference from their binary outcomes. How the pairs were formed is out of
scope: propensity-score matching, exact matching on a few keys, or a pairing that
already exists in the data all work equally well.

:class:`MatchResult` is the hand-off point. It is a plain record of which treated unit
was paired with which control, by index label, and is consumed by
:meth:`pair_match.PairedOutcomeTable.from_match_result`. Anything else carrying
``treated_index`` and ``control_index`` attributes of equal length works there too --
the method reads only those two.

If you are starting from two aligned 0/1 outcome vectors rather than from index labels,
skip this module entirely and use :meth:`pair_match.PairedOutcomeTable.from_outcomes`.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Sequence


class Pairing(Protocol):
    """Anything :meth:`pair_match.PairedOutcomeTable.from_match_result` accepts.

    Declared as read-only properties so that a frozen dataclass, a plain class
    attribute, or a property all satisfy it.

    """

    @property
    def treated_index(self) -> Sequence[object]:
        """Index labels of the treated units, one per pair."""
        ...

    @property
    def control_index(self) -> Sequence[object]:
        """Index labels of the matched controls, aligned with ``treated_index``."""
        ...


@dataclass(frozen=True)
class MatchResult:
    """One treated unit paired with one control, repeated ``n_pairs`` times.

    Attributes
    ----------
     treated_index : list
        Index labels of the treated units, one per pair.
     control_index : list
        Index labels of the matched controls, aligned element-wise with
        ``treated_index``: ``control_index[i]`` is the control matched to
        ``treated_index[i]``.
     distances : ndarray, optional
        Covariate distance within each pair, in the same order. Carried for reporting
        only -- nothing in the net-effects inference reads it -- so it defaults to an
        empty array when the pairing came from somewhere that does not compute
        distances.

    """

    treated_index: list[object]
    control_index: list[object]
    distances: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )

    def __post_init__(self) -> None:
        if len(self.treated_index) != len(self.control_index):
            raise ValueError(
                "`treated_index` and `control_index` must have the same "
                f"length (got {len(self.treated_index)} and "
                f"{len(self.control_index)})."
            )

    @property
    def n_pairs(self) -> int:
        """Number of matched pairs."""
        return len(self.treated_index)
