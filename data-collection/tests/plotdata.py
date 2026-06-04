import os

import matplotlib.pyplot as plt
import pandas as pd

# Set font styling (Geist Mono preferred)
plt.rcParams["font.family"] = "monospace"
plt.rcParams["font.size"] = 10

# Unified comprehensive benchmark dataset
data = {
    "Episodes": [
        10,
        20,
        30,
        40,
        10,
        20,
        30,
        40,
        10,
        20,
        30,
        40,
        10,
        20,
        30,
        40,
        10,
        100,
        500,
        10,
        100,
        500,
        10,
        100,
    ],
    "Max_Steps": [
        100,
        100,
        100,
        100,
        200,
        200,
        200,
        200,
        300,
        300,
        300,
        300,
        400,
        400,
        400,
        400,
        100,
        100,
        100,
        300,
        300,
        300,
        1000,
        1000,
    ],
    "Serial": [
        32.91,
        48.93,
        69.79,
        91.11,
        44.05,
        43.67,
        63.71,
        84.90,
        31.26,
        57.74,
        150.77,
        204.54,
        69.72,
        131.36,
        183.02,
        244.34,
        32.70,
        221.09,
        1077.19,
        55.95,
        501.37,
        2432.47,
        142.22,
        1350.92,
    ],
    "Parallel": [
        13.72,
        19.49,
        26.64,
        29.05,
        17.70,
        18.00,
        27.81,
        29.90,
        14.12,
        40.41,
        51.93,
        58.27,
        26.79,
        41.17,
        61.60,
        68.83,
        13.86,
        63.59,
        290.54,
        22.52,
        136.95,
        632.55,
        48.93,
        352.51,
    ],
    "Speedup": [
        2.40,
        2.51,
        2.62,
        3.14,
        2.49,
        2.43,
        2.29,
        2.84,
        2.21,
        1.43,
        2.90,
        3.51,
        2.60,
        3.19,
        2.97,
        3.55,
        2.36,
        3.48,
        3.71,
        2.48,
        3.66,
        3.85,
        2.91,
        3.83,
    ],
}

df = pd.DataFrame(data)

# Sort strictly by scale dependencies to ensure proper directional line mapping
df = df.sort_values(by=["Max_Steps", "Episodes"]).drop_duplicates(
    subset=["Max_Steps", "Episodes"], keep="last"
)

unique_steps = sorted(df["Max_Steps"].unique())
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
markers = ["o", "s", "^", "v", "D"]

fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

# Left Subplot: Execution Time Performance (Log-Log scale to encompass high variance scales)
for i, steps in enumerate(unique_steps):
    sub_df = df[df["Max_Steps"] == steps]

    axes[0].plot(
        sub_df["Episodes"],
        sub_df["Serial"],
        label=f"Serial (Steps={steps})",
        linestyle="--",
        marker=markers[i],
        color=colors[i],
        alpha=0.8,
    )
    axes[0].plot(
        sub_df["Episodes"],
        sub_df["Parallel"],
        label=f"Parallel (Steps={steps})",
        linestyle="-",
        marker=markers[i],
        color=colors[i],
        linewidth=2,
    )

axes[0].set_title("Unified Execution Time Profile", pad=15)
axes[0].set_xlabel("Episodes")
axes[0].set_ylabel("Runtime (seconds)")
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xticks([10, 20, 30, 40, 100, 500])
axes[0].get_xaxis().set_major_formatter(plt.ScalarFormatter())
axes[0].grid(True, linestyle=":", alpha=0.6, which="both")
axes[0].legend(frameon=True, fontsize=9, loc="upper left")

# Right Subplot: Speedup Scaling Trajectories
for i, steps in enumerate(unique_steps):
    sub_df = df[df["Max_Steps"] == steps]

    axes[1].plot(
        sub_df["Episodes"],
        sub_df["Speedup"],
        label=f"Steps={steps}",
        linestyle="-",
        marker=markers[i],
        color=colors[i],
        linewidth=2,
    )

axes[1].axhline(
    y=4.0, color="black", linestyle=":", alpha=0.6, label="Theoretical Ideal (4 Cores)"
)
axes[1].set_title("Unified Speedup Optimization Trajectory", pad=15)
axes[1].set_xlabel("Episodes")
axes[1].set_ylabel("Speedup Factor (x)")
axes[1].set_xscale("log")
axes[1].set_xticks([10, 20, 30, 40, 100, 500])
axes[1].get_xaxis().set_major_formatter(plt.ScalarFormatter())
axes[1].set_ylim(1.0, 4.3)
axes[1].grid(True, linestyle=":", alpha=0.6, which="both")
axes[1].legend(frameon=True, fontsize=9, loc="lower right")

plt.tight_layout()

output_path = "parallel_benchmark_results.png"
plt.savefig(output_path, dpi=300)
print(f"Unified plot saved to: {os.path.abspath(output_path)}")
