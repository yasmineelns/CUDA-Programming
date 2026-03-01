# /// script
# dependencies = [
#   "pandas",
#   "matplotlib",
# ]
# ///
"""
plot_error.py – Error convergence plots for the convexe benchmark.

Reads results_detailed.csv (produced by ./convexe) and generates
error.png with a 2×2 grid:

  Columns: Deterministic (N = samples per face edge) | Stochastic (N = rays)
  Rows:    Phase 1 (no occlusion)                   | Phase 2 (with obstacle)

Keeping det and sto on separate x-axes avoids the "vertical line" artifact
that occurs when N_det (1-16) and N_sto (1k-10M) share the same axis.

The CSV has columns:
  N, Function, Latency_ms, Estimation, Error_abs, Error_rel
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RAW_CSV    = Path("results_detailed.csv")
OUTPUT_PNG = Path("error.png")

# GPU-only labels after CPU removal
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
    df["Function"] = df["Function"].astype(str).str.strip()
    df["Latency_ms"] = pd.to_numeric(df["Latency_ms"], errors="coerce")
    df["Error_abs"]  = pd.to_numeric(df["Error_abs"],  errors="coerce")
    df["Error_rel"]  = pd.to_numeric(df["Error_rel"],  errors="coerce")

    agg = (
        df.groupby(["N", "Function"])
        .agg(
            mean_latency=("Latency_ms", "mean"),
            std_latency =("Latency_ms", "std"),
            mean_error  =("Error_abs",  "mean"),
            std_error   =("Error_abs",  "std"),
            mean_rel    =("Error_rel",  "mean"),
            mean_est    =("Estimation", "mean"),
        )
        .reset_index()
    )
    return agg


def _plot_det_series(ax, agg, labels):
    """Det series: x = N (samples per face edge, small integers 1-16), y = relative error (log)."""
    for label in labels:
        sub = agg[agg["Function"] == label].sort_values("N")
        if sub.empty:
            continue
        x    = sub["N"].to_numpy()
        y    = sub["mean_rel"].to_numpy()
        ax.plot(
            x, y,
            f"{MARKERS.get(label, 'o')}-",
            color=COLOR_MAP.get(label),
            label=label,
            linewidth=1.5, markersize=5,
        )
    ax.set_yscale("log")


def _plot_sto_series(ax, agg, labels):
    """Sto series: x = N (total rays), y = relative error — both axes log."""
    for label in labels:
        sub = agg[agg["Function"] == label].sort_values("N")
        if sub.empty:
            continue
        x    = sub["N"].to_numpy()
        y    = sub["mean_rel"].to_numpy()
        ax.plot(
            x, y,
            f"{MARKERS.get(label, 'o')}-",
            color=COLOR_MAP.get(label),
            label=label,
            linewidth=1.5, markersize=5,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")


def plot_error(agg: pd.DataFrame, output_png: Path) -> None:
    # 2×2: rows = Phase 1 / Phase 2,  cols = Deterministic / Stochastic
    # Keeping det and sto on *separate* x-axes avoids N_det (1-16) and
    # N_sto (1k-10M) competing on the same scale → no more "vertical line".
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    (ax_p1_det, ax_p1_sto), (ax_p2_det, ax_p2_sto) = axes

    # Row 1 – Phase 1
    _plot_det_series(ax_p1_det, agg, P1_DET_LABELS)
    ax_p1_det.set_title("Phase 1 (no occlusion) – Deterministic")
    ax_p1_det.set_xlabel("N  (samples per face edge)")

    _plot_sto_series(ax_p1_sto, agg, P1_STO_LABELS)
    ax_p1_sto.set_title("Phase 1 (no occlusion) – Stochastic (log-log)")
    ax_p1_sto.set_xlabel("N  (total rays)")

    # Row 2 – Phase 2
    _plot_det_series(ax_p2_det, agg, P2_DET_LABELS)
    ax_p2_det.set_title("Phase 2 (with obstacle) – Deterministic")
    ax_p2_det.set_xlabel("N  (samples per face edge)")

    _plot_sto_series(ax_p2_sto, agg, P2_STO_LABELS)
    ax_p2_sto.set_title("Phase 2 (with obstacle) – Stochastic (log-log)")
    ax_p2_sto.set_xlabel("N  (total rays)")

    for ax in axes.flat:
        ax.set_ylabel("Relative Error  |F_est − F_ref| / F_ref")
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.legend(loc="best", frameon=False)

    fig.suptitle("Convex Polyhedra View-Factor – Error Convergence", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_png, dpi=200)
    print(f"Saved {output_png}")
    try:
        plt.show()
    except Exception:
        pass


def main():
    agg = load_and_aggregate(RAW_CSV)
    plot_error(agg, OUTPUT_PNG)


if __name__ == "__main__":
    main()
