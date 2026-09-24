PairMatch
=========

PairMatch implements randomization inference for **matched pairs with
binary outcomes**, following Wilson's "Randomization Inference for
Matched Pairs with Binary Outcomes" [Wil26]_. Given pairs in which one
treated unit is matched to one control and each unit's outcome is 0 or
1, it produces exact confidence sets for the treatment effect and
reports how much unmeasured confounding the finding withstands. It
assumes nothing beyond the within-pair coin flip: it does **not** assume
the treatment effect is monotonic, and it fits no outcome model.

The library provides:

- **Attributable effects**: exact prediction sets for the net number of
  successes treatment caused among the treated (``A_1``) or the
  untreated (``A_0``)
- **Average treatment effects**: ATT, ATU, and ATE confidence sets,
  combining ``A_1`` and ``A_0`` via the Bonferroni proposition of
  Rigdon and Hudgens [RH15]_
- **Worst-case p-values** from a closed-form worst-case allocation of
  effects, with no integer program and no numerical search
- **Sensitivity analysis** under Rosenbaum's :math:`\Gamma`-model: the
  sensitivity value :math:`\Gamma^\bullet`, expanded confidence
  intervals, and the design sensitivity :math:`\tilde{\Gamma}`
- **Combinations of net effects**, including difference-in-differences,
  on the same matched pairs

Install it from PyPI with ``pip install PairMatch`` (or ``uv add
PairMatch``). The distribution is named ``PairMatch``; the import
package is ``pair_match``.

Quickstart
----------

Everything starts from the four counts of outcome patterns among the
pairs. The first subscript is the treated unit's outcome, the second the
control's, so ``s10`` counts pairs in which the treated unit succeeded
and its control did not. Build a
:class:`~pair_match.net_effects.PairedOutcomeTable` from the counts (or
from two aligned 0/1 outcome vectors with
:meth:`~pair_match.net_effects.PairedOutcomeTable.from_outcomes`), then
call :meth:`~pair_match.net_effects.PairedOutcomeTable.analyze`:

.. code-block:: python

    from pair_match import PairedOutcomeTable

    # The running example of [Wil26]: 1,000 pairs.
    table = PairedOutcomeTable(s00=800, s01=30, s10=70, s11=100)
    print(table.analyze(alpha=0.10, target="ATE"))

.. code-block:: text

    |    ATE |   Conf Int**   |   iSuccesses |  Conf Int**  |   p-Value* |      Γ• |
    |--------+----------------+--------------+--------------+------------+---------|
    | +4.00% | +0.55%, +7.45% |          +40 |   +6, +74    |    0.0484* | 1.03459 |

The last column is the sensitivity value: the largest hidden-bias odds
ratio at which the finding still holds. Here it is barely above 1, so
even slight unmeasured confounding could explain the effect. The
:doc:`user_guides/reproducing_the_paper` guide derives every number in
this table, and every other calculation in [Wil26]_, step by step.

The full usage guide ships with the package and is available at runtime
through ``pair_match.usage()``. Full citations are on the
:doc:`references` page.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   user_guides/index
   api
   references

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
