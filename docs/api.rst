API Reference
*************

.. automodule:: pair_match

.. note::

   ``alpha`` is the *total* error rate of an interval, so ``alpha=0.10``
   produces a **90%** two-sided interval (5% in each tail). Functions
   that take ``confidence`` instead use ``confidence = 1 - alpha``.

Paired Outcome Tables
=====================
.. automodule:: pair_match.net_effects
   :members:

Combinations of Net Effects
===========================
.. automodule:: pair_match.linear_combination
   :members:

Matched Pairs
=============
.. automodule:: pair_match.match_result
   :members:
