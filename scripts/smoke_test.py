"""scripts/smoke_test.py — run the learned-AI smoke suite end to end."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    targets = [
        "tests.test_state_encoder",
        "tests.test_action_encoder",
        "tests.test_phase_detection",
        "tests.test_model_routing",
        "tests.test_self_play",
        "tests.test_heuristic_vs_learned",
        "tests.test_legal_moves",
        "tests.test_checkpoint_save_load",
    ]
    for name in targets:
        suite.addTests(loader.loadTestsFromName(name))
    runner = unittest.TextTestRunner(verbosity=2)
    res = runner.run(suite)
    return 0 if res.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
