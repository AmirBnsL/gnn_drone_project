#!/usr/bin/env python3
"""
Benchmark Script for Dataset Generator Parallelization

This script benchmarks the performance of the dataset generator under serial
(1 worker) and parallel (multiple workers) execution. It tests various combinations
of episode counts and maximum step limits to demonstrate how parallelization efficiency
scales with the workload. The results are visualized using Plotly and saved as an HTML file.

Usage:
    python test_parallelization.py --workers 4
"""

import argparse
import os
import shutil
import sys
import time

import numpy as np

# Add the directory containing dataset_generator to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("Warning: 'plotly' is not installed. Plotting will be skipped.")
    plotly_available = False
else:
    plotly_available = True

from dataset_generator import generate_dataset_parallel

# Resolve datasets directory relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATASETS_DIR = os.path.join(REPO_ROOT, "datasets")


def cleanup_benchmark_files(prefix="benchmark_"):
    """Removes temporary files generated during benchmarking to keep the disk clean."""
    if not os.path.exists(DATASETS_DIR):
        return
    cleaned_count = 0
    for file in os.listdir(DATASETS_DIR):
        if file.startswith(prefix):
            file_path = os.path.join(DATASETS_DIR, file)
            try:
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                else:
                    os.remove(file_path)
                cleaned_count += 1
            except Exception as e:
                print(f"Failed to remove {file_path}: {e}")
    if cleaned_count > 0:
        print(f"Cleaned up {cleaned_count} benchmark-related files from {DATASETS_DIR}")


def run_benchmark(episodes_list, max_steps_list, num_workers):
    """Runs a matrix of benchmark configurations and measures execution times."""
    results = []

    print("=" * 70)
    print("STARTING DATASET GENERATOR PARALLELIZATION BENCHMARK")
    print(f"Target Tasks: setpoint_prediction on 'hovering' type")
    print(f"Configurations:")
    print(f"  - Episodes: {episodes_list}")
    print(f"  - Max Steps: {max_steps_list}")
    print(f"  - Parallel Workers: {num_workers}")
    print("=" * 70)

    # Clean up any leftover files first
    cleanup_benchmark_files()

    for max_steps in max_steps_list:
        for episodes in episodes_list:
            print(
                f"\n--- Benchmarking Configuration: {episodes} Episodes, {max_steps} Max Steps ---"
            )

            # --- 1. Serial Run (1 Worker) ---
            print(f"Running Serial Baseline (1 Worker)...")
            start_serial = time.time()
            generate_dataset_parallel(
                num_workers=1,
                num_episodes=episodes,
                max_steps=max_steps,
                dataset_name="benchmark_serial",
                dataset_type="hovering",
                task_type="setpoint_prediction",
                conv_stopping=False,  # Disable to keep workload constant and repeatable
                apf_enabled=True,
                worker_batch_size=max(
                    1, episodes
                ),  # Ensure all go in one chunk for 1 worker
            )
            time_serial = time.time() - start_serial
            print(f"Serial Run completed in {time_serial:.2f} seconds.")

            # Clean up serial files
            cleanup_benchmark_files("benchmark_serial")

            # --- 2. Parallel Run (N Workers) ---
            print(f"Running Parallel Generation ({num_workers} Workers)...")
            start_parallel = time.time()
            generate_dataset_parallel(
                num_workers=num_workers,
                num_episodes=episodes,
                max_steps=max_steps,
                dataset_name="benchmark_parallel",
                dataset_type="hovering",
                task_type="setpoint_prediction",
                conv_stopping=False,  # Disable to keep workload constant and repeatable
                apf_enabled=True,
                worker_batch_size=max(1, episodes // num_workers),
            )
            time_parallel = time.time() - start_parallel
            print(f"Parallel Run completed in {time_parallel:.2f} seconds.")

            # Clean up parallel files
            cleanup_benchmark_files("benchmark_parallel")

            # --- Compute Statistics ---
            speedup = time_serial / time_parallel
            efficiency = (speedup / num_workers) * 100
            print(
                f"--> Results: Serial = {time_serial:.2f}s | Parallel = {time_parallel:.2f}s"
            )
            print(f"--> Speedup Factor: {speedup:.2f}x | Efficiency: {efficiency:.1f}%")

            results.append(
                {
                    "episodes": episodes,
                    "max_steps": max_steps,
                    "time_serial": time_serial,
                    "time_parallel": time_parallel,
                    "speedup": speedup,
                    "efficiency": efficiency,
                }
            )

    return results


def generate_plots(results, num_workers, output_html_path):
    """Generates beautiful dual-axis Plotly plots illustrating parallelization speedups."""
    if not plotly_available:
        print("Skipping plot generation as Plotly is not installed.")
        return

    print("\nGenerating plotly visualizations...")

    # Group results by max_steps to plot separate lines/bars
    max_steps_groups = sorted(list(set(r["max_steps"] for r in results)))

    # Create a subplot with 1 row and 2 columns
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Execution Time: Serial vs. Parallel",
            f"Parallel Speedup Factor ({num_workers} Workers)",
        ),
        horizontal_spacing=0.15,
    )

    # Colors for the different max_steps
    colors_serial = ["#FFA07A", "#CD5C5C", "#8B0000"]
    colors_parallel = ["#ADD8E6", "#4682B4", "#00008B"]
    line_colors = ["#32CD32", "#008000", "#006400"]

    # 1. Left Subplot: Bar Chart for Execution Time Comparison
    for i, max_steps in enumerate(max_steps_groups):
        cfg_results = [r for r in results if r["max_steps"] == max_steps]
        ep_labels = [f"{r['episodes']} Ep" for r in cfg_results]

        serial_times = [r["time_serial"] for r in cfg_results]
        parallel_times = [r["time_parallel"] for r in cfg_results]

        # Serial Bars
        fig.add_trace(
            go.Bar(
                x=ep_labels,
                y=serial_times,
                name=f"Serial (Max Steps: {max_steps})",
                marker_color=colors_serial[i % len(colors_serial)],
                legendgroup=f"group_{max_steps}",
                hovertemplate="<b>%{x}</b><br>Serial Time: %{y:.2f}s<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Parallel Bars
        fig.add_trace(
            go.Bar(
                x=ep_labels,
                y=parallel_times,
                name=f"Parallel (Max Steps: {max_steps})",
                marker_color=colors_parallel[i % len(colors_parallel)],
                legendgroup=f"group_{max_steps}",
                hovertemplate="<b>%{x}</b><br>Parallel Time: %{y:.2f}s<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # 2. Right Subplot: Line Plot for Speedup Factor
    for i, max_steps in enumerate(max_steps_groups):
        cfg_results = [r for r in results if r["max_steps"] == max_steps]
        episodes = [r["episodes"] for r in cfg_results]
        speedups = [r["speedup"] for r in cfg_results]

        fig.add_trace(
            go.Scatter(
                x=episodes,
                y=speedups,
                mode="lines+markers",
                name=f"Speedup (Max Steps: {max_steps})",
                line=dict(color=line_colors[i % len(line_colors)], width=3),
                marker=dict(size=8),
                hovertemplate="<b>%{x} Episodes</b><br>Speedup: %{y:.2f}x<extra></extra>",
            ),
            row=1,
            col=2,
        )

    # Add Ideal Reference Line
    fig.add_trace(
        go.Scatter(
            x=[
                min(r["episodes"] for r in results),
                max(r["episodes"] for r in results),
            ],
            y=[num_workers, num_workers],
            mode="lines",
            name="Ideal Speedup",
            line=dict(color="gray", width=2, dash="dash"),
            hovertemplate="Ideal Speedup: %{y}x<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # Update Layout
    fig.update_layout(
        title=dict(
            text=f"<b>Dataset Generator Parallelization Scaling Benchmark ({num_workers} Workers)</b>",
            x=0.5,
            font=dict(size=18),
        ),
        barmode="group",
        template="plotly_white",
        height=500,
        width=1100,
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5
        ),
    )

    # Update axes labels
    fig.update_yaxes(title_text="Execution Time (seconds)", row=1, col=1)
    fig.update_xaxes(title_text="Workload Configuration", row=1, col=1)

    fig.update_yaxes(title_text="Speedup Factor (Serial / Parallel)", row=1, col=2)
    fig.update_xaxes(title_text="Number of Episodes", row=1, col=2)

    # Save to file
    fig.write_html(output_html_path)
    print(
        f"Successfully generated and saved interactive report plot to: {output_html_path}"
    )

    # If user wants a quick terminal output representing the plot:
    print("\n" + "=" * 40)
    print("      BENCHMARK SUMMARY TABLE       ")
    print("=" * 40)
    print(
        f"{'Episodes':<10} | {'Max Steps':<10} | {'Serial (s)':<10} | {'Parallel (s)':<12} | {'Speedup':<8}"
    )
    print("-" * 60)
    for r in results:
        print(
            f"{r['episodes']:<10} | {r['max_steps']:<10} | {r['time_serial']:<10.2f} | {r['time_parallel']:<12.2f} | {r['speedup']:<8.2f}x"
        )
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark Dataset Generator Parallelization"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker processes to use (default: 4)",
    )
    parser.add_argument(
        "--episodes",
        type=str,
        default="5,10,20",
        help="Comma-separated list of episode numbers to test",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="100,250",
        help="Comma-separated list of max steps to test",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="parallelization_benchmark.html",
        help="Output path for the Plotly HTML plot",
    )
    args = parser.parse_args()

    # Parse arguments
    try:
        episodes_list = [int(e) for e in args.episodes.split(",")]
        max_steps_list = [int(s) for s in args.steps.split(",")]
    except ValueError:
        print("Error: Episodes and steps must be comma-separated lists of integers.")
        sys.exit(1)

    # Run the benchmark
    results = run_benchmark(episodes_list, max_steps_list, args.workers)

    # Plot results
    output_path = os.path.join(SCRIPT_DIR, args.output)
    generate_plots(results, args.workers, output_path)

    print(
        "\nDone! Run the script with more episodes to observe maximum parallel efficiency."
    )
