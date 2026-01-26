# /// script
# dependencies = [
#   "pandas",
#   "matplotlib",
# ]
# ///

"""
Plot comparison of deterministic vs non-deterministic view factor estimation methods.
Shows execution time vs error trade-offs for all implementations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# Read the CSV files
prototype_file = "prototype/results_detailed.csv"
deterministic_file = "prototype_deterministic/results_detailed.csv"

# Check if files exist
if not os.path.exists(prototype_file):
    print(
        f"Error: {prototype_file} not found. Please run the prototype benchmark first."
    )
    exit(1)
if not os.path.exists(deterministic_file):
    print(
        f"Error: {deterministic_file} not found. Please run the deterministic benchmark first."
    )
    exit(1)

# Read data
df_proto = pd.read_csv(prototype_file)
df_det = pd.read_csv(deterministic_file)

# Add method type column
df_proto["Method_Type"] = "Monte Carlo"
df_det["Method_Type"] = "Deterministic"


# Simplify function names for better grouping
def simplify_function_name(func_name):
    """Simplify function names by removing thread count details for grouping."""
    if func_name.startswith("OpenMP"):
        return "OpenMP"
    return func_name


df_proto["Function_Simplified"] = df_proto["Function"].apply(simplify_function_name)
df_det["Function_Simplified"] = df_det["Function"].apply(simplify_function_name)


# Filter OpenMP to only keep maximum thread count
def get_max_thread_openmp(df):
    """Filter to keep only OpenMP with maximum thread count."""
    openmp_funcs = df[df["Function"].str.startswith("OpenMP")]["Function"].unique()
    if len(openmp_funcs) > 0:
        # Extract thread counts and find max
        thread_counts = []
        for func in openmp_funcs:
            if "_t" in func:
                try:
                    t_count = int(func.split("_t")[1])
                    thread_counts.append((func, t_count))
                except:
                    pass
        if thread_counts:
            max_func = max(thread_counts, key=lambda x: x[1])[0]
            # Keep only the max thread count OpenMP and all non-OpenMP
            return df[
                (df["Function"] == max_func)
                | (~df["Function"].str.startswith("OpenMP"))
            ]
    return df


df_proto = get_max_thread_openmp(df_proto)
df_det = get_max_thread_openmp(df_det)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Define colors and markers for each method
colors = {
    "CPU": "#1f77b4",  # blue
    "OpenMP": "#ff7f0e",  # orange
    "CUDA": "#2ca02c",  # green
    "CUDA_Naive": "#d62728",  # red
    "CUDA_Optimized": "#9467bd",  # purple
}

markers = {"Monte Carlo": "o", "Deterministic": "s"}

# Plot 1: All data points (log-log scale)
print("Plotting all data points...")
for method_type in ["Monte Carlo", "Deterministic"]:
    if method_type == "Monte Carlo":
        df = df_proto
        methods = ["CPU", "OpenMP", "CUDA"]
    else:
        df = df_det
        methods = ["CPU", "OpenMP", "CUDA_Naive", "CUDA_Optimized"]

    for method in methods:
        if method == "OpenMP":
            # Use only max thread count OpenMP (already filtered)
            mask = df["Function_Simplified"] == method
        else:
            mask = df["Function"] == method

        if mask.any():
            data = df[mask]
            label = f"{method} ({method_type})"
            ax1.scatter(
                data["Latency_ms"],
                data["Error"],
                alpha=0.6,
                s=30,
                color=colors[method],
                marker=markers[method_type],
                label=label,
            )

ax1.set_xlabel("Execution Time (ms)", fontsize=12)
ax1.set_ylabel("Absolute Error", fontsize=12)
ax1.set_title("Execution Time vs Error (All Runs)", fontsize=14, fontweight="bold")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.grid(True, alpha=0.3, which="both", linestyle="--")
ax1.legend(loc="best", fontsize=9)

# Plot 2: Median values per N and method
print("Computing median statistics...")
# Combine dataframes for easier processing
df_proto["Implementation"] = df_proto["Function"].apply(lambda x: f"{x} (MC)")
df_det["Implementation"] = df_det["Function"].apply(lambda x: f"{x} (Det)")

combined = pd.concat([df_proto, df_det], ignore_index=True)

# Group by N and Implementation to get median values
median_stats = (
    combined.groupby(["N", "Implementation"])
    .agg({"Latency_ms": "median", "Error": "median"})
    .reset_index()
)

# Plot median values with different markers
for impl in median_stats["Implementation"].unique():
    data = median_stats[median_stats["Implementation"] == impl]

    # Determine color and marker
    method_type = "Monte Carlo" if "(MC)" in impl else "Deterministic"
    method_name = impl.split(" (")[0]
    method_name_simplified = simplify_function_name(method_name)

    # Get base color
    if method_name_simplified in colors:
        color = colors[method_name_simplified]
    elif method_name in colors:
        color = colors[method_name]
    else:
        color = "#7f7f7f"  # grey for unknown

    marker = markers[method_type]

    ax2.plot(
        data["Latency_ms"],
        data["Error"],
        marker=marker,
        linestyle="-",
        linewidth=1.5,
        markersize=8,
        color=color,
        alpha=0.8,
        label=impl,
    )

ax2.set_xlabel("Execution Time (ms)", fontsize=12)
ax2.set_ylabel("Absolute Error", fontsize=12)
ax2.set_title("Execution Time vs Error (Median per N)", fontsize=14, fontweight="bold")
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.grid(True, alpha=0.3, which="both", linestyle="--")
ax2.legend(loc="best", fontsize=8, ncol=2)

plt.tight_layout()
output_file = "deterministic_vs_stochastic.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"\nPlot saved as: {output_file}")

plt.show()
