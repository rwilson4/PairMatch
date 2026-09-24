"""Tests for the packaging surface: exports, docs, and the pairing input.

The rest of the suite tests the inference itself. This file covers the parts a packaging
or install mistake would break without any inference test noticing: the public
``__init__`` re-exports, ``usage()`` finding its data file, and the :class:`MatchResult`
hand-off.

"""

import re
import unittest

import numpy as np
import pandas as pd

import pair_match
from pair_match import MatchResult, PairedOutcomeTable


class PublicApiTest(unittest.TestCase):
    def test_every_exported_name_resolves(self) -> None:
        # A name in __all__ that does not resolve breaks `from pair_match
        # import *` and `dir()` without breaking any other test.
        for name in pair_match.__all__:
            with self.subTest(name=name):
                self.assertTrue(
                    hasattr(pair_match, name),
                    f"{name} is exported but not importable",
                )

    def test_dir_matches_all(self) -> None:
        # `dir()` sorts whatever `__dir__` returns, so compare as sets.
        self.assertEqual(set(dir(pair_match)), set(pair_match.__all__))

    def test_usage_is_packaged(self) -> None:
        # USAGE.md ships as package data; if the packaging config drops it,
        # `usage()` raises at runtime rather than at build time.
        text = pair_match.usage()
        self.assertIn("# PairMatch Usage Guide", text)
        self.assertGreater(len(text), 5000)

    def test_usage_imports_resolve_against_this_package(self) -> None:
        # Every `from X import` in the guide must name this package, so a
        # reader can paste any snippet and have it run.
        imports = re.findall(r"^from ([\w.]+) import", pair_match.usage(), re.M)
        self.assertTrue(imports, "the guide has no import examples")
        for module in set(imports):
            with self.subTest(module=module):
                self.assertTrue(module.startswith("pair_match"), module)


class MatchResultTest(unittest.TestCase):
    def test_from_match_result_reads_outcomes_by_label(self) -> None:
        pairs = MatchResult(treated_index=["t0", "t1"], control_index=["c0", "c1"])
        df_treated = pd.DataFrame({"y": [1, 0]}, index=pd.Index(["t0", "t1"]))
        df_control = pd.DataFrame({"y": [0, 1]}, index=pd.Index(["c0", "c1"]))

        table = PairedOutcomeTable.from_match_result(
            pairs, "y", df_treated=df_treated, df_control=df_control
        )

        # t0/c0 -> (1, 0) = S10; t1/c1 -> (0, 1) = S01.
        self.assertEqual((table.s10, table.s01), (1, 1))

    def test_distances_are_optional(self) -> None:
        # A pairing that came from somewhere without a distance metric is
        # a first-class input: nothing in the inference reads distances.
        pairs = MatchResult(treated_index=["a"], control_index=["b"])
        self.assertEqual(pairs.n_pairs, 1)
        self.assertEqual(len(pairs.distances), 0)

    def test_distances_round_trip_when_supplied(self) -> None:
        pairs = MatchResult(
            treated_index=["a", "b"],
            control_index=["c", "d"],
            distances=np.array([0.5, 1.5]),
        )
        np.testing.assert_allclose(pairs.distances, [0.5, 1.5])

    def test_mismatched_lengths_raise(self) -> None:
        # Silently zipping to the shorter list would drop pairs and bias the
        # table toward whichever side was truncated.
        with self.assertRaises(ValueError):
            MatchResult(treated_index=["a", "b"], control_index=["c"])


class DuckTypedPairingTest(unittest.TestCase):
    def test_any_object_with_the_two_index_attributes_works(self) -> None:
        # Documented contract: `from_match_result` reads only these two
        # attributes, so a caller's own pairing object can be passed in
        # without adopting our MatchResult.
        class TheirPairing:
            treated_index = ("t0",)
            control_index = ("c0",)

        df_treated = pd.DataFrame({"y": [1]}, index=pd.Index(["t0"]))
        df_control = pd.DataFrame({"y": [0]}, index=pd.Index(["c0"]))

        table = PairedOutcomeTable.from_match_result(
            TheirPairing(), "y", df_treated=df_treated, df_control=df_control
        )

        self.assertEqual(table.s10, 1)
