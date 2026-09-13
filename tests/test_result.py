"""Result helpers: score normalization, aggregation, JSON serialization."""

from __future__ import annotations

import json

import numpy as np

from pescert import EvalResult, aggregate, combine_scores, score_from_defect


def test_score_from_defect_monotone():
    assert score_from_defect(0.0, 1.0) == 1.0
    assert np.isclose(score_from_defect(1.0, 1.0), np.exp(-1.0))
    assert score_from_defect(10.0, 1.0) < score_from_defect(1.0, 1.0)
    assert score_from_defect(float("inf"), 1.0) == 0.0


def test_result_json_serializable():
    r = EvalResult(
        name="x",
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


def test_aggregate_defaults_to_arithmetic_mean():
    results = [
        EvalResult("a", 0.0, 0.0, 1.0, 5, {}, gate=True),
        EvalResult("b", 0.0, 0.0, 0.25, 5, {}, gate=False),
    ]
    agg = aggregate(results)
    assert agg["overall_method"] == "arithmetic"
    assert np.isclose(agg["overall"], (1.0 + 0.25) / 2)
    assert agg["sub_scores"] == {"a": 1.0, "b": 0.25}
    assert agg["gates_passed"] == 1
    assert agg["gates_total"] == 2
    assert agg["n_total_calls"] == 10


def test_aggregate_method_selects_mean():
    results = [
        EvalResult("a", 0.0, 0.0, 1.0, 5, {}),
        EvalResult("b", 0.0, 0.0, 0.25, 5, {}),
    ]
    assert np.isclose(aggregate(results, "geometric")["overall"], np.sqrt(1.0 * 0.25))
    assert np.isclose(aggregate(results, "harmonic")["overall"], 2 / (1 / 1.0 + 1 / 0.25))


def test_combine_scores_methods_and_ordering():
    s = [1.0, 0.5, 0.25]
    ar = combine_scores(s, "arithmetic")
    ge = combine_scores(s, "geometric")
    ha = combine_scores(s, "harmonic")
    # harmonic <= geometric <= arithmetic (AM-GM-HM inequality)
    assert ha <= ge <= ar
    assert np.isclose(ar, np.mean(s))
    assert np.isnan(combine_scores([], "arithmetic"))


def test_combine_scores_unknown_method_raises():
    import pytest

    with pytest.raises(ValueError):
        combine_scores([1.0], "median")


def test_aggregate_ignores_nan_scores():
    results = [
        EvalResult("a", 0.0, 0.0, 1.0, 5, {}),
        EvalResult("skipped", 0.0, float("nan"), float("nan"), 0, {"skipped": True}),
    ]
    agg = aggregate(results)
    assert np.isclose(agg["overall"], 1.0)
