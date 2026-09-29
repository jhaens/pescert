"""Suite orchestration, Report serialization, and the CLI entry point."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
from _potentials import clean_lj

from pescert import Suite, available, from_ase_calculator
from pescert.cli import main as cli_main


def test_suite_default_runs_all(lj_engine, cluster13, trimer3, bulk_ar, p2mm_ar):
    subs = {"cluster": cluster13, "trimer": trimer3, "bulk": bulk_ar, "p2mm": p2mm_ar}
    report = Suite.default().run(lj_engine, subs, seed=0, budget_per_eval=2500)
    assert len(report.results) == len(available())
    agg = report.aggregate
    assert agg["overall_method"] == "geometric"
    assert 0.0 <= agg["overall"] <= 1.0
    assert agg["n_total_calls"] > 0


def test_suite_subset_and_single_atoms(lj_engine, cluster13):
    report = Suite.from_names(["equivariance", "betti"]).run(lj_engine, cluster13, seed=0)
    assert [r.name for r in report.results] == ["equivariance", "betti"]


def test_report_json_and_summary(tmp_path, lj_engine, cluster13):
    report = Suite.from_names(["equivariance", "zero_modes"]).run(lj_engine, cluster13, seed=0)
    out = tmp_path / "report.json"
    report.to_json(str(out))
    data = json.loads(out.read_text())
    assert "results" in data and "aggregate" in data and "metadata" in data
    assert {r["name"] for r in data["results"]} == {"equivariance", "zero_modes"}
    table = report.summary()
    assert "Probe" in table and "overall score" in table
    # padded columns: every table line has the header's width
    lines = [ln for ln in table.splitlines() if ln.startswith("|")]
    assert len({len(ln) for ln in lines}) == 1


def test_single_element_list_is_that_element():
    # ["Ar"] used to reach the trimer builder as a list and abort the whole run
    engine = from_ase_calculator(clean_lj(rc=8.0))
    suite = Suite.from_names(["equivariance", "trimer"])
    as_list = suite.run(engine, ["Ar"], seed=0)
    as_symbol = suite.run(engine, "Ar", seed=0)
    assert [r.score for r in as_list.results] == [r.score for r in as_symbol.results]


def test_suite_builds_substrate_from_element():
    # element string -> auto-built default substrates per proxy
    engine = from_ase_calculator(clean_lj(rc=8.0))
    report = Suite.from_names(["equivariance"]).run(engine, "Ar", seed=0)
    assert report.results[0].score > 0.99


def test_suite_multi_element_averages():
    engine = from_ase_calculator(clean_lj(rc=8.0))
    suite = Suite.from_names(["equivariance", "conservativeness"])
    # both a list and a comma-separated string select the multi-element averaging path
    for spec in (["Ar", "Ne", "Xe"], "Ar, Ne, Xe"):
        report = suite.run(engine, spec, seed=0)
        assert report.metadata["elements"] == ["Ar", "Ne", "Xe"]
        assert set(report.metadata["per_element_overall"]) == {"Ar", "Ne", "Xe"}
        # each averaged proxy records its per-element scores
        assert set(report.results[0].details["per_element"]) == {"Ar", "Ne", "Xe"}
        assert report.results[0].score > 0.99


def test_aggregation_method_changes_overall():
    engine = from_ase_calculator(clean_lj(rc=8.0))
    suite = Suite.from_names(["equivariance", "conservativeness"])
    for method in ("arithmetic", "geometric", "harmonic"):
        report = suite.run(engine, "Ar", seed=0, agg_method=method)
        assert report.aggregate["overall_method"] == method
        assert 0.0 <= report.aggregate["overall"] <= 1.0


def test_unknown_eval_raises():
    with pytest.raises(KeyError):
        Suite.from_names(["does_not_exist"])


def test_cli_list_runs():
    rc = cli_main(["list"])
    assert rc == 0


def test_cli_run_writes_json(tmp_path):
    out = tmp_path / "cli.json"
    rc = cli_main(
        [
            "run",
            "--calc",
            "_potentials:clean_lj",
            "--element",
            "Ar",
            "--evals",
            "equivariance,betti",
            "--json",
            str(out),
        ]
    )
    assert rc == 0
    data = json.loads(out.read_text())
    assert len(data["results"]) == 2
    # same overall score as Suite.run's default
    assert data["aggregate"]["overall_method"] == "geometric"


def test_cli_subprocess_entrypoint():
    # the installed console script is importable and runs
    res = subprocess.run(
        [sys.executable, "-m", "pescert.cli", "list"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "zero_modes" in res.stdout
