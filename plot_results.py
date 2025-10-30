# /// script
# dependencies = [
#   "pandas",
#   "matplotlib",
# ]
# ///

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RAW_CSV = Path("results_detailed.csv")
OUTPUT_PNG = Path("benchmark_results.png")

LABEL_ORDER = ["CPU", "OpenMP", "OpenBLAS", "GPU", "cuBLAS"]
COLOR_MAP = {
    "CPU": "#1f77b4",  # blue
    "OpenMP": "#ff7f0e",  # orange
    "OpenBLAS": "#2ca02c",  # green
    "GPU": "#d62728",  # red
    "cuBLAS": "#9467bd",  # purple
}
MARKERS = {
    "CPU": "o",
    "OpenMP": "s",
    "OpenBLAS": "^",
    "GPU": "D",
    "cuBLAS": "v",
}


def load_and_aggregate(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw CSV not found: {csv_path}. Run the benchmark executable first to generate it."
        )

    df = pd.read_csv(csv_path)
    # Normalize function labels just in case
    df["Function"] = df["Function"].str.strip()

    # Aggregate per N and Function (keep all labels; we'll filter for top/bottom plots later)
    agg = (
        df.groupby(["N", "Function"])  # type: ignore[arg-type]
        .agg(
            mean_ms=("Measurement_ms", "mean"),
            std_ms=("Measurement_ms", "std"),
            count=("Measurement_ms", "count"),
        )
        .reset_index()
    )
    return agg


def plot_benchmarks(agg: pd.DataFrame, output_png: Path) -> None:
    # Prepare top-row data: only the main labels in LABEL_ORDER
    agg_top = agg[agg["Function"].isin(LABEL_ORDER)].copy()
    # Ensure consistent presence for plotting across Ns for top row
    if not agg_top.empty:
        all_idx = pd.MultiIndex.from_product(
            [sorted(agg_top["N"].unique()), LABEL_ORDER], names=["N", "Function"]
        )
        agg_top = (
            agg_top.set_index(["N", "Function"]).reindex(all_idx).reset_index()
        )

    # Prepare bottom-row data: CPU + OpenMP_t{threads}
    is_openmp_t = agg["Function"].str.startswith("OpenMP_t", na=False)
    openmp_t_labels = (
        agg.loc[is_openmp_t, "Function"].dropna().unique().tolist()
    )
    # Sort OpenMP_t labels by numeric thread count
    def _thread_num(label: str) -> int:
        try:
            return int(label.split("t", 1)[1])
        except Exception:
            return 0

    openmp_t_labels = sorted(openmp_t_labels, key=_thread_num)

    # Layout: 2 rows x 2 cols — top row (overall), bottom row (CPU vs OpenMP threads)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharex=True)
    (top_linear_ax, top_log_ax), (bottom_linear_ax, bottom_log_ax) = axes

    # Top row: overall comparison
    for label in LABEL_ORDER:
        sub = agg_top[agg_top["Function"] == label].sort_values("N")
        sub = sub.dropna(subset=["N", "mean_ms"]).copy()
        if sub.empty:
            continue
        x = sub["N"].to_numpy()
        y = sub["mean_ms"].to_numpy()
        yerr = sub["std_ms"].fillna(0.0).to_numpy()

        top_linear_ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt=f"{MARKERS[label]}-",
            color=COLOR_MAP[label],
            label=label,
            capsize=3,
            linewidth=1.5,
            markersize=5,
        )
        top_log_ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt=f"{MARKERS[label]}-",
            color=COLOR_MAP[label],
            label=label,
            capsize=3,
            linewidth=1.5,
            markersize=5,
        )

    # Bottom row: CPU vs OpenMP with varying threads
    # Plot CPU first
    cpu_sub = agg[agg["Function"] == "CPU"].sort_values("N").dropna(subset=["N", "mean_ms"])  # type: ignore[arg-type]
    if not cpu_sub.empty:
        x = cpu_sub["N"].to_numpy()
        y = cpu_sub["mean_ms"].to_numpy()
        yerr = cpu_sub["std_ms"].fillna(0.0).to_numpy()
        bottom_linear_ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt=f"{MARKERS['CPU']}-",
            color=COLOR_MAP["CPU"],
            label="CPU",
            capsize=3,
            linewidth=1.5,
            markersize=5,
        )
        bottom_log_ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt=f"{MARKERS['CPU']}-",
            color=COLOR_MAP["CPU"],
            label="CPU",
            capsize=3,
            linewidth=1.5,
            markersize=5,
        )

    # Color map for multiple OpenMP thread counts
    if openmp_t_labels:
        cmap = plt.cm.get_cmap("viridis", len(openmp_t_labels))
        for idx, label in enumerate(openmp_t_labels):
            sub = agg[agg["Function"] == label].sort_values("N")
            sub = sub.dropna(subset=["N", "mean_ms"]).copy()
            if sub.empty:
                continue
            x = sub["N"].to_numpy()
            y = sub["mean_ms"].to_numpy()
            yerr = sub["std_ms"].fillna(0.0).to_numpy()
            color = cmap(idx)
            # Use same marker shape as OpenMP
            bottom_linear_ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt=f"{MARKERS.get('OpenMP', 's')}-",
                color=color,
                label=label,
                capsize=3,
                linewidth=1.3,
                markersize=5,
            )
            bottom_log_ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt=f"{MARKERS.get('OpenMP', 's')}-",
                color=color,
                label=label,
                capsize=3,
                linewidth=1.3,
                markersize=5,
            )

    # Titles and axis formatting
    top_linear_ax.set_title("All methods — linear scale")
    top_linear_ax.set_xlabel("Matrix size N")
    top_linear_ax.set_ylabel("Time (ms)")
    top_linear_ax.grid(True, linestyle="--", alpha=0.3)

    top_log_ax.set_title("All methods — log scale")
    top_log_ax.set_xlabel("Matrix size N")
    top_log_ax.set_yscale("log")
    top_log_ax.grid(True, which="both", linestyle="--", alpha=0.3)

    bottom_linear_ax.set_title("CPU vs OpenMP (varied threads) — linear scale")
    bottom_linear_ax.set_xlabel("Matrix size N")
    bottom_linear_ax.set_ylabel("Time (ms)")
    bottom_linear_ax.grid(True, linestyle="--", alpha=0.3)

    bottom_log_ax.set_title("CPU vs OpenMP (varied threads) — log scale")
    bottom_log_ax.set_xlabel("Matrix size N")
    bottom_log_ax.set_yscale("log")
    bottom_log_ax.grid(True, which="both", linestyle="--", alpha=0.3)

    # Legends inside axes to avoid overlapping figure footer
    top_linear_ax.legend(loc="best", frameon=False)
    top_log_ax.legend(loc="best", frameon=False)
    # If many thread counts, split legend columns for readability
    ncols = 1 if len(openmp_t_labels) < 6 else 2
    bottom_linear_ax.legend(loc="best", frameon=False, ncols=ncols)
    bottom_log_ax.legend(loc="best", frameon=False, ncols=ncols)

    fig.suptitle(
        "Matrix Multiplication Benchmark\nTop: All methods • Bottom: CPU vs OpenMP threads",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    fig.savefig(output_png, dpi=200)

    # Also display if running interactively
    try:
        plt.show()
    except Exception:
        pass


def main():
    agg = load_and_aggregate(RAW_CSV)
    plot_benchmarks(agg, OUTPUT_PNG)


if __name__ == "__main__":
    main()
