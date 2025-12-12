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
OUTPUT_PNG = Path("latency.png")

LABEL_ORDER = ["CPU", "OpenMP", "CUDA"]
COLOR_MAP = {
    "CPU": "#1f77b4",
    "OpenMP": "#ff7f0e",
    "CUDA": "#d62728",
}
MARKERS = {
    "CPU": "o",
    "OpenMP": "s",
    "CUDA": "D",
}


def load_and_aggregate(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {csv_path}. Run the prototype benchmark to generate it.")

    df = pd.read_csv(csv_path)
    df["Function"] = df["Function"].astype(str).str.strip()

    df["Latency_ms"] = pd.to_numeric(df["Latency_ms"], errors="coerce")
    df["Error"] = pd.to_numeric(df["Error"], errors="coerce")
    df["Estimation"] = pd.to_numeric(df["Estimation"], errors="coerce")

    agg = (
        df.groupby(["N", "Function"])  # type: ignore[arg-type]
        .agg(
            mean_latency=("Latency_ms", "mean"),
            std_latency=("Latency_ms", "std"),
            mean_error=("Error", "mean"),
            std_error=("Error", "std"),
            mean_est=("Estimation", "mean"),
        )
        .reset_index()
    )
    return agg


def plot_latency(agg: pd.DataFrame, output_png: Path) -> None:
    # Prepare OpenMP aggregated series (mean across OpenMP_t* labels)
    is_openmp_t = agg["Function"].str.startswith("OpenMP_t", na=False)
    openmp_agg = None
    if is_openmp_t.any():
        openmp_agg = (
            agg[is_openmp_t]
            .groupby("N")
            .agg(
                mean_latency=("mean_latency", "mean"),
                std_latency=("std_latency", "mean"),
            )
            .reset_index()
        )
        openmp_agg["Function"] = "OpenMP"

    # Top row dataset for CPU, OpenMP (agg), CUDA
    top_frames = []
    for label in LABEL_ORDER:
        if label == "OpenMP" and openmp_agg is not None:
            df_label = openmp_agg.copy()
        else:
            df_label = agg[agg["Function"] == label].copy()
        if not df_label.empty:
            top_frames.append(df_label[["N", "Function", "mean_latency", "std_latency"]])

    top_df = pd.concat(top_frames, ignore_index=True) if top_frames else pd.DataFrame(columns=["N", "Function", "mean_latency", "std_latency"])

    # Bottom row: CPU and individual OpenMP_t labels
    cpu_sub = agg[agg["Function"] == "CPU"].sort_values("N").dropna(subset=["N", "mean_latency"])  # type: ignore[arg-type]
    openmp_t_labels = (agg[agg["Function"].str.startswith("OpenMP_t", na=False)]["Function"].unique().tolist())

    def _thread_num(label: str) -> int:
        try:
            return int(label.split("t", 1)[1])
        except Exception:
            return 0

    openmp_t_labels = sorted(openmp_t_labels, key=_thread_num)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharex=True)
    (top_linear_ax, top_log_ax), (bottom_linear_ax, bottom_log_ax) = axes

    # Top row: overall comparison (latency)
    for label in LABEL_ORDER:
        sub = top_df[top_df["Function"] == label].sort_values("N")
        if sub.empty:
            continue
        x = sub["N"].to_numpy()
        y = sub["mean_latency"].to_numpy()
        yerr = sub["std_latency"].fillna(0.0).to_numpy()
        for ax in (top_linear_ax, top_log_ax):
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt=f"{MARKERS.get(label, 'o')}-",
                color=COLOR_MAP.get(label, None),
                label=label,
                capsize=3,
                linewidth=1.5,
                markersize=5,
            )

    top_linear_ax.set_title("Methods — latency (linear)")
    top_linear_ax.set_xlabel("N (samples)")
    top_linear_ax.set_ylabel("Time (ms)")
    top_linear_ax.grid(True, linestyle="--", alpha=0.3)

    top_log_ax.set_title("Methods — latency (log)")
    top_log_ax.set_xlabel("N (samples)")
    top_log_ax.set_yscale("log")
    top_log_ax.grid(True, which="both", linestyle="--", alpha=0.3)

    top_linear_ax.legend(loc="best", frameon=False)
    top_log_ax.legend(loc="best", frameon=False)

    # Bottom: CPU and OpenMP_t* (different thread counts)
    if not cpu_sub.empty:
        x = cpu_sub["N"].to_numpy()
        y = cpu_sub["mean_latency"].to_numpy()
        yerr = cpu_sub["std_latency"].fillna(0.0).to_numpy()
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

    if openmp_t_labels:
        cmap = plt.cm.get_cmap("viridis", len(openmp_t_labels))
        for idx, label in enumerate(openmp_t_labels):
            sub = agg[agg["Function"] == label].sort_values("N").dropna(subset=["N", "mean_latency"]).copy()
            if sub.empty:
                continue
            x = sub["N"].to_numpy()
            y = sub["mean_latency"].to_numpy()
            yerr = sub["std_latency"].fillna(0.0).to_numpy()
            color = cmap(idx)
            bottom_linear_ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt="s-",
                color=color,
                label=label,
                capsize=3,
                linewidth=1.2,
                markersize=5,
            )
            bottom_log_ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt="s-",
                color=color,
                label=label,
                capsize=3,
                linewidth=1.2,
                markersize=5,
            )

    bottom_linear_ax.set_title("Latency — CPU vs OpenMP (varied threads)")
    bottom_linear_ax.set_xlabel("N (samples)")
    bottom_linear_ax.set_ylabel("Time (ms)")
    bottom_linear_ax.grid(True, linestyle="--", alpha=0.3)

    bottom_log_ax.set_title("Latency — CPU vs OpenMP (varied threads, log)")
    bottom_log_ax.set_xlabel("N (samples)")
    bottom_log_ax.set_yscale("log")
    bottom_log_ax.grid(True, which="both", linestyle="--", alpha=0.3)

    # Legends
    bottom_linear_ax.legend(loc="best", frameon=False)
    bottom_log_ax.legend(loc="best", frameon=False)

    fig.suptitle("Prototype — Latency: methods and CPU vs OpenMP threads", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_png, dpi=200)

    try:
        plt.show()
    except Exception:
        pass


def main():
    agg = load_and_aggregate(RAW_CSV)
    plot_latency(agg, OUTPUT_PNG)


if __name__ == "__main__":
    main()
