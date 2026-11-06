"""Compare archived model scores on PACE; never refit or recompute fields.

Usage: python scripts/compare_sweep_models.py /scratch/path/to/figure2_directory
"""

import argparse
import csv
import hashlib
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


METRICS = ("closure_accuracy_A", "flux_accuracy_A", "mean_flux_ratio")


def number(value):
    if value in (None, "", "None", "null"):
        return float("nan")
    result = float(value)
    if math.isinf(result):
        raise ValueError("Infinite archived metric")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--budgets", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--output-root", type=Path,
                        help="Parent for a new unique result directory; defaults to input archive")
    args = parser.parse_args()
    budgets = sorted(set(args.budgets))
    if not budgets or min(budgets) < 1:
        parser.error("Predictor budgets must be positive")
    archive = args.archive.resolve()
    if not (archive / "COMPLETE").is_file():
        raise ValueError("Input comparison archive must contain COMPLETE")
    analysis_path = archive / "analysis.json"
    csv_path = archive / "metrics.csv"
    analysis = json.loads(analysis_path.read_bytes())
    length = analysis["config"].get("reference_length")
    if length is not None and (not math.isfinite(length) or length <= 0):
        raise ValueError("Invalid archived reference length")
    widths = sorted(float(s["delta"]) for s in analysis["sources"])
    if len(widths) < 2 or len(set(widths)) != len(widths):
        raise ValueError("Expected at least two unique archived widths")
    if any(not math.isfinite(w) or w <= 0 for w in widths):
        raise ValueError("Invalid archived filter width")

    labels = [f"Discovered P{p}" for p in budgets] + [
        "Nonlinear/gradient/Clark model", "NGM4"]
    selected = {}
    with csv_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            label = None
            if row["family"] == "general" and row["predictor_budget"]:
                budget = int(row["predictor_budget"])
                if budget in budgets:
                    label = f"Discovered P{budget}"
            elif row["family"] == "prescribed":
                label = {1: labels[-2], 2: labels[-1]}.get(int(row["model_order"]))
            if label is None or row["target"] not in ("tau", "phi"):
                continue
            width = float(row["delta"])
            if width not in widths:
                raise ValueError(f"CSV width absent from analysis: {width}")
            key = (row["target"], label, width)
            if key in selected:
                raise ValueError(f"Duplicate candidate for {key}; refusing to select a winner")
            selected[key] = row

    missing = [key for key in itertools.product(("tau", "phi"), labels, widths)
               if key not in selected]
    if missing:
        raise ValueError(f"Missing model/width combinations: {missing}")
    for row in selected.values():
        for metric in METRICS:
            number(row[metric])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parent = args.output_root.resolve() if args.output_root else archive
    output = parent / ("model_comparison_" + datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    x = [w / length if length else w for w in widths]
    colors = [plt.get_cmap("tab10")(i % 10) for i in range(len(budgets))] + ["#555555", "#d55e00"]
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    for col, target in enumerate(("tau", "phi")):
        for index, label in enumerate(labels):
            for row_index, metric in enumerate(METRICS):
                y = [number(selected[target, label, w][metric]) for w in widths]
                axes[row_index, col].plot(
                    x, y, label=label, color=colors[index],
                    linestyle="-" if index < len(budgets) else ("--" if index == len(budgets) else "-."),
                    marker="o", markersize=4)
        axes[0, col].set_title("SGS stress tensor" if target == "tau" else "SGS scalar flux (convection extension)")
        symbol = r"\tau" if target == "tau" else r"\phi"
        flux = r"\Pi" if target == "tau" else r"\Pi_{\phi}"
        ylabels = (
            rf"$\mathcal{{A}}({symbol})$",
            rf"$\mathcal{{A}}({flux})$",
            rf"$\langle {flux}^{{\mathrm{{LES}}}}\rangle/\langle {flux}^{{\mathrm{{DNS}}}}\rangle$",
        )
        for row_index, ylabel in enumerate(ylabels):
            ax = axes[row_index, col]
            ax.axhline(1, color="0.65", linewidth=.8, linestyle=":")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=.2)
        axes[2, col].set_xlabel(r"$\delta$" if analysis.get("scale_plan") and length else
                                (r"$\Delta/L$" if length else r"$\Delta$"))
        axes[2, col].set_xticks(x)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=3, frameon=False)
    fig.text(.5, .015, r"A priori benchmarks on convection data. $\delta=\Delta/\ell_i$."
             if analysis.get("scale_plan") and length else
             "A priori benchmarks on convection data.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, .91))
    for extension in ("png", "pdf"):
        fig.savefig(output / f"models_vs_dns.{extension}", dpi=200)
    plt.close(fig)

    with (output / "selected_models.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["comparison_label"] + list(next(iter(selected.values()))))
        writer.writeheader()
        for (_, label, _), row in selected.items():
            writer.writerow(dict(row, comparison_label=label))
    with (output / "pairwise_score_differences.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["target", "delta", "model_a", "model_b", "metric", "score_a", "score_b", "a_minus_b"])
        for target, width in itertools.product(("tau", "phi"), widths):
            for a, b in itertools.combinations(labels, 2):
                for metric in METRICS:
                    va = number(selected[target, a, width][metric])
                    vb = number(selected[target, b, width][metric])
                    writer.writerow([target, width, a, b, metric, va, vb, va-vb])
    notes = (
        "A priori benchmarks: all models are compared with quantities computed from filtered DNS data.\n"
        "Stress panels: accuracy of the SGS stress tensor A(tau), local energy flux A(Pi),\n"
        "and mean energy flux normalized by the DNS value <Pi^LES>/<Pi^DNS>.\n"
        "delta = Delta/ell_i when the measured scale plan is supplied.\n"
        "The scalar-flux column extends this notation to convection; it is not a paper benchmark.\n"
        "The gradient/Clark and NGM4 legend entries denote analogous scalar expansions in that column.\n"
        "Discovered P1-P3 retain their archive identities; none is relabeled as NGMR.\n"
        "Accuracy A: higher is better (ideal 1). Mean flux ratio: closer to 1 is better, not larger.\n"
        "Pairwise CSV contains differences of reference-based scores, NOT pairwise field errors.\n"
        "The saved scores do not supply cross-moments between arbitrary models.\n"
        "P is requested predictor budget; tau uses equal budgets for trace and STF.\n"
        "Models are refitted at each width. No best model or independent validation is claimed.\n"
        "Undefined scores remain NaN/gaps. Input archives are never modified.\n"
        "This replot consumes the earlier audit; it does not repeat the full archive verification.\n"
    )
    (output / "README.txt").write_text(notes, encoding="utf-8")
    inputs = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (csv_path, analysis_path)}
    script = Path(__file__).read_bytes()
    (output / "compare_sweep_models.py").write_bytes(script)
    (output / "provenance.json").write_text(json.dumps(dict(
        input_archive=str(archive), input_sha256=inputs, budgets=budgets,
        script_sha256=hashlib.sha256(script).hexdigest()), indent=2), encoding="utf-8")
    (output / "COMPLETE").write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
    print(f"Comparison saved to: {output}")


if __name__ == "__main__":
    main()
