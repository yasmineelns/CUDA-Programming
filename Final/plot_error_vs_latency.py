# /// script
# dependencies = [
#   "pandas",
#   "matplotlib",
# ]
# ///
"""
error_vs_latency.py – Pareto-style error-vs-latency plot for the convexe benchmark.

Reads results_detailed.csv and generates error_vs_latency.png:
  Two log-log scatter plots (Phase 1 and Phase 2) showing relative error
  vs latency for all methods.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RAW_CSV    = Path("results_detailed.csv")
OUTPUT_PNG = Path("error_vs_latency.png")

P1_DET_LABELS = ["P1_CUDA_Det"]
P1_STO_LABELS = ["P1_CUDA_Sto"]
P2_DET_LABELS = ["P2_CUDA_Det"]
P2_STO_LABELS = ["P2_CUDA_Sto"]

COLOR_MAP = {
    "P1_CUDA_Det": "#2ca02c",
    "P1_CUDA_Sto": "#d62728",
    "P2_CUDA_Det": "#17becf",
    "P2_CUDA_Sto": "#e377c2",
}
MARKERS = {
    "P1_CUDA_Det": "D",
    "P1_CUDA_Sto": "^",
    "P2_CUDA_Det": "D",
    "P2_CUDA_Sto": "^",
}


def load_and_aggregate(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found: {csv_path}. Run ./convexe first."
        )
    df = pd.read_csv(csv_path)
    df["Function"]   = df["Function"].astype(str).str.strip()
    df["Latency_ms"] = pd.to_numeric(df["Latency_ms"], errors="coerce")
    df["Error_abs"]  = pd.to_numeric(df["Error_abs"],  errors="coerce")
    df["Error_rel"]  = pd.to_numeric(df["Error_rel"],  errors="coerce")

    for phase in ("P1", "P2"):
        prefix = f"{phase}_OMP_t"
        mask   = df["Function"].str.startswith(prefix, na=False)
        if mask.any():
            omp_df = df[mask].copy()
            omp_df["Function"] = f"{phase}_OMP"
            df = pd.concat([df[~mask], omp_df], ignore_index=True)

    agg = (
        df.groupby(["N", "Function"])
        .agg(
            mean_latency=("Latency_ms", "mean"),
            std_latency =("Latency_ms", "std"),
            mean_error  =("Error_abs",  "mean"),
            mean_rel    =("Error_rel",  "mean"),
        )
        .reset_index()
    )
    return agg


def plot_error_vs_latency(agg: pd.DataFrame, output_png: Path) -> None:
    """Pareto-style log-log scatter: relative error vs latency, P1 and P2 side by side."""
    phases = [
        ("Phase 1 – no occlusion",  P1_DET_LABELS + P1_STO_LABELS),
        ("Phase 2 – with obstacle", P2_DET_LABELS + P2_STO_LABELS),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (title, labels) in zip((ax1, ax2), phases):
        for label in labels:
            sub = agg[agg["Function"] == label].sort_values("mean_latency")
            if sub.empty:
                continue
            x = sub["mean_latency"].to_numpy()
            y = sub["mean_rel"].to_numpy()
            ax.plot(x, y, f"{MARKERS.get(label, 'o')}-",
                    color=COLOR_MAP.get(label),
                    label=label, linewidth=1.5, markersize=5)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("Relative Error  |F_est − F_ref| / F_ref")
        ax.set_title(title)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.legend(loc="best", frameon=False, fontsize=9)

    fig.suptitle("Convex Polyhedra View-Factor – Error vs Latency (log-log)", y=1.01)
    fig.tight_layout()
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    print(f"Saved {output_png}")
    try:
        plt.show()
    except Exception:
        pass


def main():
    agg = load_and_aggregate(RAW_CSV)
    plot_error_vs_latency(agg, OUTPUT_PNG)


if __name__ == "__main__":
    main()
