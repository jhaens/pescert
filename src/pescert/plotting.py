"""Optional per-proxy diagnostic figures (requires the ``[plots]`` extra).

Never imported by the core path; only :meth:`pescert.suite.Report.plot` pulls it in.
"""

from __future__ import annotations

import os

from .result import EvalResult


def plot_report(report, directory: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(directory, exist_ok=True)
    written: list[str] = []
    for r in report.results:
        fig = _plot_one(plt, r)
        if fig is None:
            continue
        path = os.path.join(directory, f"{r.name}.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written


def _plot_one(plt, r: EvalResult):
    d = r.details
    fig, ax = plt.subplots(figsize=(5, 3.2))
    title = f"[{r.section}] {r.name}  score={r.score:.3f}"

    if r.name == "smoothness" and "s_grid" in d:
        ax.plot(d["s_grid"], d["energies"], label="E(s)")
        ax2 = ax.twinx()
        ax2.plot(d["s_grid"], d["projected_force"], color="C1", label="F·u")
        ax.set_xlabel("bond displacement s (Å)")
        ax.set_ylabel("energy (eV)")
        ax2.set_ylabel("projected force (eV/Å)")
    elif r.name == "conservativeness" and "s_grid" in d:
        s = d["s_grid"]
        ax.plot(s, d["projected_force"], label="model F·u")
        ax.plot(s[2:-2], d["force_from_energy"], "--", label="-dE/ds")
        ax.set_xlabel("bond displacement s (Å)")
        ax.set_ylabel("force (eV/Å)")
        ax.legend()
    elif r.name == "equipartition" and d.get("R_alpha"):
        ra = d["R_alpha"]
        ax.plot(range(len(ra)), ra, "o", ms=3)
        ax.axhline(1.0, color="k", lw=0.8)
        ax.set_xlabel("mode index α")
        ax.set_ylabel(r"$R_\alpha$")
    elif r.name == "trimer" and "three_body_vanishing" in d:
        tb = d["three_body_vanishing"]
        ax.plot(tb["separations"], tb["delta_e3"], "o-")
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_xlabel("vertex separation (Å)")
        ax.set_ylabel(r"$\Delta E_3$ (eV)")
    elif r.name == "zero_modes" and "lowest_eigenvalues" in d:
        ev = d["lowest_eigenvalues"]
        ax.plot(range(len(ev)), ev, "o")
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_xlabel("eigenvalue index")
        ax.set_ylabel("Hessian eigenvalue (eV/Å²)")
    elif r.name == "config_temperature" and d.get("ratio_series"):
        ax.plot(d["ratio_series"], "o-", ms=3)
        ax.axhline(1.0, color="k", lw=0.8)
        ax.set_xlabel("sample")
        ax.set_ylabel(r"$T_{config}/T_{kin}$")
    else:
        ax.text(0.5, 0.5, f"defect={r.raw_defect:.3e}", ha="center", va="center")
        ax.set_axis_off()

    ax.set_title(title, fontsize=9)
    return fig
