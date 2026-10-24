"""Aggregate results/*.csv into markdown tables and a comparison figure.

    python experiments/make_report.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results")
RATES = [0.1, 0.3, 0.5, 0.7, 0.9]
METHODS = ["TGAIN (1 model)", "TGAIN (3 models)", "MissForest", "MICE", "KNNImputer",
           "IterativeImputer"]
# Thesis Table 5.2 (German Credit, 382 train / 50 test rows)
PAPER = {
    "TGAIN (1 model)":  [0.2487, 0.2543, 0.2557, 0.3176, 0.3197],
    "TGAIN (3 models)": [0.2416, 0.2605, 0.2685, 0.2941, 0.2948],
    "MissForest":       [0.3250, 0.3187, 0.3140, 0.3527, 0.3980],
    "MICE":             [0.2840, 0.3140, 0.3054, 0.3084, 0.3002],
    "KNNImputer":       [0.2525, 0.2941, 0.2941, 0.3290, 0.3155],
    "IterativeImputer": [0.3335, 0.3368, 0.3581, 0.3572, 0.3412],
}
# categorical palette slots 1-6 (validated), plus distinct markers as secondary encoding
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
REPRO = METHODS + ["MeanMode"]  # MeanMode: trivial reference, not in the thesis


def table(df, col, methods):
    g = df.groupby(["method", "rate"])[col].agg(["mean", "std", "count"])
    lines = ["| Method | " + " | ".join(f"{int(r*100)}%" for r in RATES) + " |",
             "|---|" + "---|" * len(RATES)]
    for m in methods:
        cells = []
        for r in RATES:
            if (m, r) in g.index:
                mu, sd, n = g.loc[(m, r)]
                cells.append(f"{mu:.4f} ± {sd:.4f}" if n > 1 else f"{mu:.4f}")
            else:
                cells.append("–")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(lines), g


def paper_table():
    lines = ["| Method | " + " | ".join(f"{int(r*100)}%" for r in RATES) + " |",
             "|---|" + "---|" * len(RATES)]
    for m in METHODS:
        lines.append(f"| {m} | " + " | ".join(f"{v:.4f}" for v in PAPER[m]) + " |")
    return "\n".join(lines)


def rank_table(df, col):
    """Mean rank of each method per rate (1 = lowest error), averaged over reps."""
    d = df[df.method.isin(METHODS)].copy()
    d["rank"] = d.groupby(["rate", "rep"])[col].rank()
    g = d.groupby(["method", "rate"])["rank"].mean().unstack()
    pr = pd.DataFrame(PAPER, index=RATES).T.rank(axis=0)
    lines = ["| Method | " + " | ".join(f"{int(r*100)}%" for r in RATES) + " |",
             "|---|" + "---|" * len(RATES)]
    for m in METHODS:
        lines.append(f"| {m} | " + " | ".join(
            f"{g.loc[m, r]:.1f} (paper {int(pr.loc[m, r])})" for r in RATES) + " |")
    return "\n".join(lines)


def plot(g, path, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    x = [int(r * 100) for r in RATES]
    for i, m in enumerate(METHODS):
        kw = dict(color=COLORS[i], marker=MARKERS[i], markersize=7, linewidth=2, label=m)
        axes[0].plot(x, PAPER[m], **kw)
        mu = [g.loc[(m, r), "mean"] for r in RATES]
        sd = [g.loc[(m, r), "std"] for r in RATES]
        axes[1].errorbar(x, mu, yerr=sd, capsize=3, **kw)
    if ("MeanMode", RATES[0]) in g.index:
        axes[1].plot(x, [g.loc[("MeanMode", r), "mean"] for r in RATES], color="#8a8a8a",
                     linestyle="--", linewidth=1.5, label="Mean/mode (reference)")
    axes[0].set_title("Thesis, Table 5.2 (single run)")
    axes[1].set_title(title)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xlabel("Missing rate (%)")
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Prediction error (lower is better)")
    axes[1].legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    out = []
    df = pd.read_csv(os.path.join(RES, "german_results.csv"))
    n_reps = df.rep.nunique()
    out.append("## Thesis Table 5.2\n\n" + paper_table())
    t, g = table(df, "error_original", REPRO)
    out.append(f"## Reproduction - original metric (mean ± std over {n_reps} repetitions)\n\n" + t)
    plot(g, os.path.join(RES, "german_error_original.png"),
         f"Reproduction, original metric (mean ± sd, {n_reps} reps)")
    t, g2 = table(df, "error_corrected", REPRO)
    out.append("## Reproduction - corrected metric\n\n" + t)
    plot(g2, os.path.join(RES, "german_error_corrected.png"),
         f"Reproduction, corrected metric (mean ± sd, {n_reps} reps)")
    out.append("## Mean rank per missing rate (original metric)\n\n" +
               rank_table(df, "error_original"))
    t, _ = table(df, "num_err", REPRO)
    out.append("## Mean absolute error on numerical cells (shared [0,1] scale)\n\n" + t)
    t, _ = table(df, "cat_err", REPRO)
    out.append("## Error rate on categorical cells\n\n" + t)
    p = os.path.join(RES, "german_results_discrete.csv")
    if os.path.exists(p):
        dd = pd.read_csv(p)
        t, _ = table(dd, "error_original", ["TGAIN (1 model)", "TGAIN (3 models)"])
        out.append("## Ablation - categorical codes declared discrete in CTGAN "
                   f"(original metric, {dd.rep.nunique()} reps)\n\n" + t)
    p = os.path.join(RES, "german_results_baselines_testonly.csv")
    if os.path.exists(p):
        tb = pd.read_csv(p)
        t, _ = table(tb, "error_original", REPRO[2:])
        out.append("## Ablation - baselines fitted on the 50 test rows only "
                   f"(original metric, {tb.rep.nunique()} reps)\n\n" + t)
    p = os.path.join(RES, "german_results_legacy.csv")
    if os.path.exists(p):
        lg = pd.read_csv(p)
        out.append("## Unmodified original code (Python 3.8, sdv 0.15)\n\n" +
                   lg[["method", "rate", "rep", "error_original", "seconds"]]
                   .to_markdown(index=False, floatfmt=".4f"))
    tim = df.groupby("method")["seconds"].mean().round(1)
    out.append("## Mean runtime per run (s)\n\n" + tim.to_markdown())
    with open(os.path.join(RES, "summary.md"), "w") as f:
        f.write("\n\n".join(out) + "\n")
    print("\n\n".join(out))


if __name__ == "__main__":
    main()
