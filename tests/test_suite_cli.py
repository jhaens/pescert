"""Suite orchestration, Report serialization, and the CLI entry point."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
from _potentials import clean_lj

from pescert import Suite, available, from_ase_calculator
from pescert.cli import main as cli_main


def test_suite_default_runs_all(lj_engine, cluster13, trimer3, bulk_ar):
    subs = {"cluster": cluster13, "trimer": trimer3, "bulk": bulk_ar}
    report = Suite.default().run(lj_engine, subs, seed=0, budget_per_eval=2000)
    assert len(report.results) == len(available())
    agg = report.aggregate
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
    assert "Proxy" in table and "overall score" in table


def test_suite_builds_substrate_from_element():
    # element string -> auto-built default substrates per proxy
    engine = from_ase_calculator(clean_lj(rc=8.0))
    report = Suite.from_names(["equivariance"]).run(engine, "Ar", seed=0)
    assert report.results[0].score > 0.99


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


def test_cli_subprocess_entrypoint():
    # the installed console script is importable and runs
    res = subprocess.run(
        [sys.executable, "-m", "pescert.cli", "list"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "zero_modes" in res.stdout
