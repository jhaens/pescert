"""Result helpers: score normalization, aggregation, JSON serialization."""

from __future__ import annotations

import json

import numpy as np

from pescert import EvalResult, aggregate, score_from_defect


def test_score_from_defect_monotone():
    assert score_from_defect(0.0, 1.0) == 1.0
    assert np.isclose(score_from_defect(1.0, 1.0), np.exp(-1.0))
    assert score_from_defect(10.0, 1.0) < score_from_defect(1.0, 1.0)
    assert score_from_defect(float("inf"), 1.0) == 0.0


def test_result_json_serializable():
    r = EvalResult(
        name="x",
        section="NEW-1",
        target=0.0,
        raw_defect=1e-3,
        score=0.9,
        n_model_calls=10,
        details={"arr": np.arange(3), "val": np.float64(1.5), "flag": np.bool_(True)},
        gate=True,
    )
    text = r.to_json()
    parsed = json.loads(text)
    assert parsed["details"]["arr"] == [0, 1, 2]
    assert parsed["details"]["val"] == 1.5
    assert parsed["gate"] is True


def test_aggregate_geometric_mean():
    results = [
        EvalResult("a", "KNOWN", 0.0, 0.0, 1.0, 5, {}, gate=True),
        EvalResult("b", "KNOWN", 0.0, 0.0, 0.25, 5, {}, gate=False),
    ]
    agg = aggregate(results)
    assert np.isclose(agg["overall"], np.sqrt(1.0 * 0.25))
    assert agg["sub_scores"] == {"a": 1.0, "b": 0.25}
    assert agg["gates_passed"] == 1
    assert agg["gates_total"] == 2
    assert agg["n_total_calls"] == 10


def test_aggregate_ignores_nan_scores():
    results = [
        EvalResult("a", "KNOWN", 0.0, 0.0, 1.0, 5, {}),
        EvalResult("skipped", "OOB-2", 0.0, float("nan"), float("nan"), 0, {"skipped": True}),
    ]
    agg = aggregate(results)
    assert np.isclose(agg["overall"], 1.0)
