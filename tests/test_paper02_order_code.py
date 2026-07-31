"""Data-independent regression checks for the Paper-02 v2 order code."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

import paper02_consolidate_prepool_results as consolidate  # noqa: E402
import paper02_prepool_order_permutation as order  # noqa: E402


def test_tail_fraction_ledger_has_no_threshold_decisions() -> None:
    rows = {
        "a": {"shuffle_reference_tail_fraction_one_sided": 0.006},
        "b": {"shuffle_reference_tail_fraction_one_sided": 0.03},
        "c": {"shuffle_reference_tail_fraction_one_sided": 0.031},
    }
    assert consolidate.tail_fraction_ledger(rows) == {
        "a": {"tail_fraction": 0.006, "rank_by_tail_fraction": 1},
        "b": {"tail_fraction": 0.03, "rank_by_tail_fraction": 2},
        "c": {"tail_fraction": 0.031, "rank_by_tail_fraction": 3},
    }


def test_each_shuffle_realization_is_matched_across_configurations(monkeypatch) -> None:
    calls: list[tuple[int, int | None]] = []

    def fake_cv_r2(sequences, y, *, batch_size, config_seed, shuffle_seed):
        calls.append((config_seed, shuffle_seed))
        value = 0.2 if shuffle_seed is None else 0.1 + shuffle_seed * 1e-6
        return value, 5.0, 0.4, [5, 5, 5, 5, 5]

    monkeypatch.setattr(order.sp, "cv_r2", fake_cv_r2)
    row = order.run_model(
        "test",
        np.zeros((8, 6, 3), dtype=np.float32),
        np.linspace(0.5, 0.8, 8),
        batch_size=4,
        n_perm=3,
        config_seeds=[0, 1, 2, 3],
    )

    shuffled = [call for call in calls if call[1] is not None]
    for permutation_index in range(3):
        expected_seed = 1000 + permutation_index
        assert [
            config for config, seed in shuffled if seed == expected_seed
        ] == [0, 1, 2, 3]
    assert row["shuffle_seed_formula"].startswith("1000 + k")
    assert row["n_permutations"] == 3
    assert row["config_seeds"] == [0, 1, 2, 3]
