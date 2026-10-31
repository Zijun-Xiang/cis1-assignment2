"""
Fill report/template.md from results/notebook_results.json (or a stand-in results file) and render
it to PDF (+ DOCX) with pandoc.

    python report/make_report.py [results.json] [--authors "Name 1, Name 2"] [--tests "29 passed in 41 s"]
"""
import argparse
import datetime as dt
import json
import shutil
import subprocess
from pathlib import Path

import jinja2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def fig_frustum(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.plot([1000, 2200, 2200, 1000, 1000], [-500, -1000, 1000, 500, -500], color="#0284c7", lw=1.5)
    ax.fill([1000, 2200, 2200, 1000], [-500, -1000, 1000, 500], color="#e0f2fe", alpha=.8)
    c, h = 1625, 475
    ax.fill([c - h, c + h, c + h, c - h], [-h, -h, h, h], color="#dcfce7", alpha=.9)
    ax.plot([c - h, c + h, c + h, c - h, c - h], [-h, -h, h, h, -h], color="#16a34a", lw=2)
    ax.plot([0, 2200], [0, 0], ls="--", color="#94a3b8", lw=1)
    ax.plot(0, 0, marker="s", color="#475569", ms=10)
    ax.text(0, -120, "camera", ha="center", fontsize=9)
    ax.text(c, 0, "marker envelope\n950 mm cube, centre at 1625 mm", ha="center", va="center", fontsize=9, color="#166534")
    ax.annotate("", xy=(c - h, 562.5), xytext=(c - h, 475), arrowprops=dict(arrowstyle="<->", color="#b45309"))
    ax.text(c - h - 30, 600, "87.5 mm", fontsize=8, color="#b45309", ha="right")
    ax.annotate("", xy=(2200, 0), xytext=(c + h, 0), arrowprops=dict(arrowstyle="<->", color="#b45309"))
    ax.text(2150, 60, "100 mm", fontsize=8, color="#b45309", ha="center")
    ax.set_xlabel("depth along the camera axis [mm]")
    ax.set_ylabel("lateral [mm]")
    ax.set_aspect("equal")
    ax.set_xlim(-150, 2350)
    ax.set_ylim(-1100, 1100)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_sv(path, sv):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sv = np.asarray(sv, float)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.semilogy(np.arange(1, len(sv) + 1), sv / sv[0], "o-", color="#0e7490", ms=4)
    ax.axhline(1e-6, ls="--", color="#b91c1c", lw=1, label="rank tolerance")
    ax.set_xlabel("singular value index")
    ax.set_ylabel("relative singular value")
    ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default=str(ROOT / "results" / "notebook_results.json"))
    ap.add_argument("--authors", default="[Your names here]")
    ap.add_argument("--tests", default="run `python -m pytest tests -q` and paste the result here")
    ap.add_argument("--n-tests", default="29")
    ap.add_argument("--out", default=str(ROOT / "report"))
    a = ap.parse_args()
    R = json.load(open(a.results))
    standin = "stand-in" in str(R.get("simulator", ""))
    vpath = Path(a.results).parent / "variants.json"
    if vpath.exists():                                   # the 10-seed study from report/variant_study.py
        V = json.load(open(vpath))
        R["variants"] = V["variants"]
        R["variant_seeds"] = len(V["seeds"])
    else:
        R.setdefault("variant_seeds", 1)
    seeds = R["seeds"]
    s0 = seeds[0]
    plan = dict(R["plan_info"])
    plan["slide_stops_count"] = int(round(600.0 / plan["slide_steps_mm"][0])) + 1
    plan["slide_steps_count"] = 3 * plan["slide_stops_count"]
    plan["wrist_stops_count"] = 3 * len(plan["wrist_stops_rad"])
    pt = plan.get("predicted_time", {})
    plan["predicted_total"] = pt.get("total_s", float("nan"))
    plan["predicted_motion"] = pt.get("motion_s", float("nan"))
    plan["predicted_sampling"] = pt.get("sampling_s", float("nan"))
    pvt = dict(R["params_vs_truth"])
    pvt["axis_angles_err_mrad"] = (np.radians(np.asarray(pvt["axis_angles_deg_err"], float)) * 1e3).tolist()
    rank = R["rank"]
    weak = rank["weakest"]["direction"] if isinstance(rank["weakest"], dict) else rank["weakest"]
    names = [n for n, w in weak]
    rank_weak_names = ", ".join(f"`{n}`" for n in names)
    wall = R.get("wall_seconds")
    wall_note = f"{wall:.0f} s on the machine the results were produced on" if wall else "see the notebook's last printed line"

    env = jinja2.Environment(undefined=jinja2.StrictUndefined, trim_blocks=False, lstrip_blocks=False,
                             comment_start_string="<#", comment_end_string="#>")
    env.filters["f0"] = lambda x: f"{float(x):.0f}"
    env.filters["f2"] = lambda x: f"{float(x):.2f}"
    env.filters["f3"] = lambda x: f"{float(x):.3f}"
    env.filters["e2"] = lambda x: f"{float(x):.2e}"
    env.filters["pct"] = lambda x: f"{100 * float(x):.0f}%"
    env.filters["mrad"] = lambda x: f"{1e3 * float(x):.3f}"
    env.filters["mrad2"] = lambda x: f"{1e3 * float(x):.2f}"
    env.filters["met"] = lambda s: "yes" if (s["max_pos"] <= 0.2 and s["max_angle"] <= 1e-3) else "no"
    env.filters["fmtlist"] = lambda xs, d=3: ", ".join(f"{float(x):.{d}f}" for x in xs)
    env.filters["fmtlist_e"] = lambda xs, d=1: ", ".join(f"{float(x):.{d}e}" for x in xs)
    tpl = env.from_string((HERE / "template.md").read_text())
    md = tpl.render(authors=a.authors, date=dt.date.today().isoformat(), standin=standin, s0=s0, seeds=seeds, plan=plan,
                    fit=R["fit_info"], rank=rank, rank_weak_names=rank_weak_names, pvt=pvt, test_grid_size=R["test_grid_size"],
                    variants=R.get("variants", []), variant_seeds=R.get("variant_seeds", 1), n_tests=a.n_tests,
                    tests_status=a.tests, wall_note=wall_note)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(md)
    fig_frustum(out / "fig_frustum.png")
    fig_sv(out / "fig_singular_values.png", rank["singular_values"])
    if shutil.which("pandoc"):
        base = ["pandoc", str(out / "report.md"), "--resource-path", str(out), "-f", "markdown+pipe_tables+implicit_figures"]
        pdf_args = ["-o", str(out / "Assignment2_Report.pdf"), "--pdf-engine=xelatex", "-V", "colorlinks=true", "-V", "fontsize=10pt"]
        tpl_path = HERE / "pandoc_latex.tpl"
        if tpl_path.exists():      # pandoc's default template minus lmodern (not installed everywhere) + system fonts
            pdf_args += ["--template", str(tpl_path), "-V", "mainfont=DejaVu Serif", "-V", "sansfont=DejaVu Sans", "-V", "monofont=DejaVu Sans Mono"]
        try:
            subprocess.run(base + pdf_args, check=True)
        except subprocess.CalledProcessError:
            print("PDF rendering failed (LaTeX); report.md and the DOCX are still written")
        subprocess.run(base + ["-o", str(out / "Assignment2_Report.docx")], check=True)
        print("wrote", out / "Assignment2_Report.pdf", "and .docx")
    else:
        print("pandoc not found: wrote report.md only")


if __name__ == "__main__":
    main()
