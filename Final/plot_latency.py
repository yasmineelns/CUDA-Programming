# /// script
# dependencies = [
#   "pandas",
#   "matplotlib",
# ]
# ///
"""
plot_latency.py – Latency plots for the convexe benchmark.

Reads results_detailed.csv and generates latency.png with 4 sub-plots:

  Row 1: Phase 1 all methods – latency (linear / log)
  Row 2: Phase 2 all methods – latency (linear / log)

The CSV has columns:
  N, Function, Latency_ms, Estimation, Error_abs, Error_rel
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RAW_CSV    = Path("results_detailed.csv")
OUTPUT_PNG = Path("latency.png")

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


def _plot_det_latency(ax, agg, labels):
    """Det series: x = N (samples per face edge, integer 1-16)."""
    for label in labels:
        sub = agg[agg["Function"] == label].sort_values("N")
        if sub.empty:
            continue
        x    = sub["N"].to_numpy()
        y    = sub["mean_latency"].to_numpy()
        yerr = sub["std_latency"].fillna(0.0).to_numpy()
        ax.errorbar(
            x, y, yerr=yerr,
            fmt=f"{MARKERS.get(label, 'o')}-",
            color=COLOR_MAP.get(label),
            label=label,
            capsize=3, linewidth=1.5, markersize=5,
        )


def _plot_sto_latency(ax, agg, labels):
    """Sto series: x = N (total rays, log scale)."""
    for label in labels:
        sub = agg[agg["Function"] == label].sort_values("N")
        if sub.empty:
            continue
        x    = sub["N"].to_numpy()
        y    = sub["mean_latency"].to_numpy()
        yerr = sub["std_latency"].fillna(0.0).to_numpy()
        ax.errorbar(
            x, y, yerr=yerr,
            fmt=f"{MARKERS.get(label, 'o')}-",
            color=COLOR_MAP.get(label),
            label=label,
            capsize=3, linewidth=1.5, markersize=5,
        )
    ax.set_xscale("log")


def plot_latency(agg: pd.DataFrame, output_png: Path) -> None:
    # 2×2: rows = Phase 1 / Phase 2,  cols = Deterministic / Stochastic
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    (ax_p1_det, ax_p1_sto), (ax_p2_det, ax_p2_sto) = axes

    # Row 1 – Phase 1
    _plot_det_latency(ax_p1_det, agg, P1_DET_LABELS)
    ax_p1_det.set_title("Phase 1 (no occlusion) – Deterministic latency")
    ax_p1_det.set_xlabel("N  (samples per face edge)")

    _plot_sto_latency(ax_p1_sto, agg, P1_STO_LABELS)
    ax_p1_sto.set_title("Phase 1 (no occlusion) – Stochastic latency (log x)")
    ax_p1_sto.set_xlabel("N  (total rays)")

    # Row 2 – Phase 2
    _plot_det_latency(ax_p2_det, agg, P2_DET_LABELS)
    ax_p2_det.set_title("Phase 2 (with obstacle) – Deterministic latency")
    ax_p2_det.set_xlabel("N  (samples per face edge)")

    _plot_sto_latency(ax_p2_sto, agg, P2_STO_LABELS)
    ax_p2_sto.set_title("Phase 2 (with obstacle) – Stochastic latency (log x)")
    ax_p2_sto.set_xlabel("N  (total rays)")

    for ax in axes.flat:
        ax.set_ylabel("Latency (ms)")
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.legend(loc="best", frameon=False)

    fig.suptitle("Convex Polyhedra View-Factor – Latency", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_png, dpi=200)
    print(f"Saved {output_png}")
    try:
        plt.show()
    except Exception:
        pass



def main():
    agg = load_and_aggregate(RAW_CSV)
    plot_latency(agg, OUTPUT_PNG)


if __name__ == "__main__":
    main()
