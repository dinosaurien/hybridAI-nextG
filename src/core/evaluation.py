#!/usr/bin/env python3
"""
Evaluation & Aggregation Plotter (Master's Thesis Edition)

Reads KPI data from multiple kpms.csv runs (e.g., 5 baselines, 5 AI runs) and generates
aggregated evaluation plots:
  1. Ablation Bar Charts: Average Latency, Throughput, Recovery Time, Anomalies.
  2. Pareto Trade-off Graph: 2D Scatter of Energy/MCS vs Latency.

Usage:
  python evaluate_baseline.py \
      --baseline baseline1.csv baseline2.csv ... \
      --managed no_reflexion1.csv no_reflexion2.csv ... \
      --reflexion reflexion1.csv reflexion2.csv ... \
      --energy energy1.csv energy2.csv ...
"""

import argparse
import csv
import os
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

def _import_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt

def load_kpms_csv(path: str) -> Dict:
    """Load kpms.csv and dynamically extract all numerical metrics."""
    cell_rows =[]
    ue_rows = defaultdict(list)
    timestamps = set()

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_str = row.get("timestamp", "")
            if not ts_str:
                continue
            try:
                ts = datetime.fromisoformat(ts_str)
            except ValueError:
                continue

            cell_id = row.get("cell_id", "")
            
            # Cell-level metrics
            if cell_id.startswith("CELL_") or cell_id == "unknown":
                entry = {"timestamp": ts}
                for k, v in row.items():
                    if k not in ["timestamp", "meid", "cell_id", "node_id", "format"] and v:
                        try:
                            entry[k] = float(v)
                        except ValueError:
                            pass
                if len(entry) > 1:
                    cell_rows.append(entry)
                    timestamps.add(ts)

            # UE-level metrics
            elif cell_id.startswith("UE_"):
                ue_id = cell_id
                entry = {"timestamp": ts, "ue_id": ue_id}
                for k, v in row.items():
                    if k not in["timestamp", "meid", "cell_id", "node_id", "format", "ue_id"] and v:
                        try:
                            entry[k] = float(v)
                        except ValueError:
                            pass
                if len(entry) > 2:
                    ue_rows[ue_id].append(entry)
                    timestamps.add(ts)

    return {
        "cell": sorted(cell_rows, key=lambda x: x["timestamp"]),
        "ue": {uid: sorted(rows, key=lambda x: x["timestamp"]) for uid, rows in ue_rows.items()},
        "timestamps": sorted(timestamps),
    }

def analyze_recovery_times(data: Dict, metric: str = "DRB_PdcpSduDelayDl", threshold: float = 40.0) -> Tuple[List[dict], int]:
    """Detect degradations and measure time-to-recover."""
    vals_with_ts = [(r["timestamp"], r[metric]) for r in data["cell"] if metric in r]
    if not vals_with_ts: 
        return[], 0
    
    events =[]
    unrecovered = 0
    in_degradation = False
    start_ts = None
    
    for t, v in vals_with_ts:
        if not in_degradation and v > threshold:
            in_degradation = True
            start_ts = t
        elif in_degradation and v <= threshold:
            in_degradation = False
            recovery_s = (t - start_ts).total_seconds()
            events.append({"start": start_ts, "end": t, "recovery_s": recovery_s})
    
    if in_degradation:
        unrecovered = 1
        
    return events, unrecovered

def compute_run_metrics(data: Dict) -> dict:
    """Compute scalar summary statistics for a single simulation run."""
    metrics = {}
    
    # 1. Average Cell Latency
    lat_vals = [r["DRB_PdcpSduDelayDl"] for r in data["cell"] if "DRB_PdcpSduDelayDl" in r]
    metrics["latency_mean"] = np.mean(lat_vals) if lat_vals else np.nan
    
    # 2. Average Aggregate Throughput
    thp_by_ts = defaultdict(float)
    for uid, rows in data["ue"].items():
        for r in rows:
            if "UE_DRB_UEThpDl_UEID" in r:
                thp_by_ts[r["timestamp"]] += r["UE_DRB_UEThpDl_UEID"]
    thp_vals = list(thp_by_ts.values())
    metrics["throughput_mean"] = np.mean(thp_vals) if thp_vals else np.nan
    
    # 3. Tx Power
    tx_vals = [r["tx_power_dbm"] for r in data["cell"] if "tx_power_dbm" in r]
    metrics["tx_power_mean"] = np.mean(tx_vals) if tx_vals else np.nan
    
    # 4. MCS Average
    mcs_vals =[r["mcs_dl_avg"] for r in data["cell"] if "mcs_dl_avg" in r]
    metrics["mcs_mean"] = np.mean(mcs_vals) if mcs_vals else np.nan
    
    # 5. PRB Usage
    prb_vals = [r["RRU_PrbUsedDl"] for r in data["cell"] if "RRU_PrbUsedDl" in r]
    metrics["prb_mean"] = np.mean(prb_vals) if prb_vals else np.nan

    # 6. Recovery Time & Anomalies (Spikes over 40ms)
    events, unrecovered = analyze_recovery_times(data, metric="DRB_PdcpSduDelayDl", threshold=40.0)
    metrics["anomaly_count"] = len(events) + unrecovered
    
    rec_times = [e["recovery_s"] for e in events]
    if metrics["anomaly_count"] == 0:
        metrics["recovery_time_mean"] = 0.0
    elif len(events) > 0:
        metrics["recovery_time_mean"] = np.mean(rec_times)
    else:
        metrics["recovery_time_mean"] = np.nan # It broke but never recovered
        
    return metrics

def print_summary_table(categories: Dict[str, List[dict]]):
    """Print the aggregated data to console in a clean ASCII table."""
    print("\n" + "=" * 95)
    print(f"{'Metric':<30} | " + " | ".join([f"{lbl[:16]:>13}" for lbl in categories.keys()]))
    print("-" * 95)
    
    metrics =[
        ("latency_mean", "Avg Latency (ms)"),
        ("throughput_mean", "Avg Throughput (kbps)"),
        ("tx_power_mean", "Avg Tx Power (dBm)"),
        ("mcs_mean", "Avg MCS (Index)"),
        ("anomaly_count", "Total Anomalies"),
        ("recovery_time_mean", "Recovery Time (s)")
    ]
    
    for m_key, m_name in metrics:
        row = f"{m_name:<30} | "
        for lbl, runs in categories.items():
            vals = [r[m_key] for r in runs if not np.isnan(r.get(m_key, np.nan))]
            if vals:
                mean_val = np.mean(vals)
                std_val = np.std(vals)
                if std_val > 0.01:
                    row += f"{mean_val:>6.1f}±{std_val:<6.1f} | "
                else:
                    row += f"{mean_val:>6.1f}       | "
            else:
                row += f"{'N/A':>13} | "
        print(row)
    print("=" * 95 + "\n")

def plot_aggregated_bars(categories: Dict[str, List[dict]], out_dir: str):
    """Plot bar charts with error bars for the ablation study."""
    plt = _import_plt()
    
    metrics_to_plot =[
        ("latency_mean", "Average Latency (ms)"),
        ("throughput_mean", "Aggregate Throughput (kbps)"),
        ("recovery_time_mean", "Average Recovery Time (s)"),
        ("anomaly_count", "Total Anomalies Detected (Count)")
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    labels = list(categories.keys())
    x = np.arange(len(labels))
    colors =["tab:red", "tab:blue", "tab:green", "tab:orange"][:len(labels)]
    
    for i, (metric_key, title) in enumerate(metrics_to_plot):
        ax = axes[i]
        means = []
        stds =[]
        
        for lbl in labels:
            vals = [run[metric_key] for run in categories[lbl] if not np.isnan(run.get(metric_key, np.nan))]
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            else:
                means.append(0)
                stds.append(0)
        
        ax.bar(x, means, yerr=stds, capsize=6, color=colors, alpha=0.8, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.grid(True, axis="y", linestyle="--", alpha=0.6)
        
    plt.tight_layout()
    path = os.path.join(out_dir, "ablation_metrics_aggregated.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved Ablation Bar Charts: {path}")

def plot_tradeoff(categories: Dict[str, List[dict]], out_dir: str):
    """Plot 2D Scatter for Intent-Steering Trade-off (Pareto Shift)."""
    plt = _import_plt()
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = {"Baseline": "tab:red", "AI (No Reflexion)": "tab:blue", "AI (Reflexion)": "tab:green", "Energy Intent": "tab:orange"}
    
    # Determine X-Axis dynamically based on what the simulator successfully logged
    has_tx = any(not np.isnan(run.get("tx_power_mean", np.nan)) for runs in categories.values() for run in runs)
    if has_tx:
        x_key, x_label = "tx_power_mean", "Average Tx Power (dBm)"
    else:
        has_mcs = any(not np.isnan(run.get("mcs_mean", np.nan)) for runs in categories.values() for run in runs)
        if has_mcs:
            x_key, x_label = "mcs_mean", "Average MCS (Index)"
        else:
            x_key, x_label = "prb_mean", "Average PRB Usage (%)"
            
    y_key, y_label = "latency_mean", "Average Latency (ms)"
    
    for label, runs in categories.items():
        color = colors.get(label, "tab:gray")
        x_vals = [run[x_key] for run in runs if not np.isnan(run.get(x_key, np.nan))]
        y_vals = [run[y_key] for run in runs if not np.isnan(run.get(y_key, np.nan))]
        
        if not x_vals or not y_vals:
            continue
            
        # Plot individual runs
        ax.scatter(x_vals, y_vals, label=f"{label} (Individual Runs)", color=color, alpha=0.5, s=80, edgecolors="white")
        # Plot centroid
        ax.scatter(np.mean(x_vals), np.mean(y_vals), color=color, marker="X", s=300, edgecolor="black", zorder=5, label=f"{label} (Centroid)")
            
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title("Multi-Objective Trade-off (Pareto Shift)", fontsize=14, fontweight="bold")
    
    # Shrink current axis by 20% to fit legend outside
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    
    ax.grid(True, linestyle="--", alpha=0.6)
    
    ax.annotate("Ideal Region\n(Low Latency, Low Energy)", 
                xy=(0.02, 0.02), xycoords='axes fraction', 
                bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", alpha=0.3),
                fontsize=11)
                
    path = os.path.join(out_dir, "intent_tradeoff_pareto.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved Pareto Trade-off Graph: {path}")

def main():
    parser = argparse.ArgumentParser(description="Aggregated Thesis Evaluator for 5G AI Architecture")
    parser.add_argument("--baseline", nargs='+', help="List of baseline CSV files (e.g., baseline1.csv baseline2.csv)")
    parser.add_argument("--managed", nargs='+', help="List of AI (No Reflexion) CSV files")
    parser.add_argument("--reflexion", nargs='+', help="List of AI (With Reflexion) CSV files")
    parser.add_argument("--energy", nargs='+', help="List of AI (Energy Intent) CSV files")
    parser.add_argument("--out-dir", type=str, default="evaluation_plots", help="Output directory for plots")
    
    args = parser.parse_args()

    if not any([args.baseline, args.managed, args.reflexion, args.energy]):
        print("Please provide at least one category of CSV files (e.g., --baseline run1.csv run2.csv)")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    categories = {}

    print("Loading datasets and computing metrics...")
    if args.baseline:
        categories["Baseline"] =[compute_run_metrics(load_kpms_csv(f)) for f in args.baseline if os.path.exists(f)]
    if args.managed:
        categories["AI (No Reflexion)"] =[compute_run_metrics(load_kpms_csv(f)) for f in args.managed if os.path.exists(f)]
    if args.reflexion:
        categories["AI (Reflexion)"] =[compute_run_metrics(load_kpms_csv(f)) for f in args.reflexion if os.path.exists(f)]
    if args.energy:
        categories["Energy Intent"] =[compute_run_metrics(load_kpms_csv(f)) for f in args.energy if os.path.exists(f)]

    # Print Text Summary
    print_summary_table(categories)

    # Generate Plots
    plot_aggregated_bars(categories, args.out_dir)
    plot_tradeoff(categories, args.out_dir)
    
    print("\nEvaluation complete! Add these plots to your Results section.")

if __name__ == "__main__":
    main()