#!/usr/bin/env python3
"""
Reads KPI data from multiple kpms.csv runs (e.g., 5 baselines, 5 AI runs) and generates
aggregated evaluation plots:
  Ablation Bar Charts: Average Latency, Throughput, Recovery Time, Anomalies.

It also creates a summary table in LaTex format for easy inclusion in the thesis.

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
    
    # Average Cell Latency
    lat_vals = [r["DRB_PdcpSduDelayDl"] for r in data["cell"] if "DRB_PdcpSduDelayDl" in r]
    metrics["latency_mean"] = np.mean(lat_vals) if lat_vals else np.nan
    
    # Average Aggregate Throughput
    thp_by_ts = defaultdict(float)
    for uid, rows in data["ue"].items():
        for r in rows:
            if "UE_DRB_UEThpDl_UEID" in r:
                thp_by_ts[r["timestamp"]] += r["UE_DRB_UEThpDl_UEID"]
    thp_vals = list(thp_by_ts.values())
    metrics["throughput_mean"] = np.mean(thp_vals) if thp_vals else np.nan
    
    # Tx Power
    tx_vals = [r["tx_power_dbm"] for r in data["cell"] if "tx_power_dbm" in r]
    metrics["tx_power_mean"] = np.mean(tx_vals) if tx_vals else np.nan
    
    # MCS Average
    mcs_vals =[r["mcs_dl_avg"] for r in data["cell"] if "mcs_dl_avg" in r]
    metrics["mcs_mean"] = np.mean(mcs_vals) if mcs_vals else np.nan
    
    # PRB Usage
    prb_vals = [r["RRU_PrbUsedDl"] for r in data["cell"] if "RRU_PrbUsedDl" in r]
    metrics["prb_mean"] = np.mean(prb_vals) if prb_vals else np.nan

    # Recovery Time & Anomalies (Spikes over 40ms)
    events, unrecovered = analyze_recovery_times(data, metric="DRB_PdcpSduDelayDl", threshold=40.0)
    metrics["anomaly_count"] = len(events) + unrecovered
    
    rec_times = [e["recovery_s"] for e in events]
    if metrics["anomaly_count"] == 0:
        metrics["recovery_time_mean"] = np.nan # No anomalies
    elif len(events) > 0:
        metrics["recovery_time_mean"] = np.mean(rec_times)
    else:
        metrics["recovery_time_mean"] = np.nan # It broke but never recovered
        
    return metrics

def print_summary_table(categories: Dict[str, List[dict]]):
    """Print the aggregated data to console in a clean ASCII table and output LaTeX code."""
    
    metrics =[
        ("latency_mean", "Avg Latency (ms)"),
        ("throughput_mean", "Avg Throughput (kbps)"),
        ("prb_mean", "Avg PRB Usage (%)"),
        ("anomaly_count", "Total Anomalies"),
        ("recovery_time_mean", "Recovery Time (s)")
    ]
    
    headers =["Metric"] + list(categories.keys())
    
    table_rows =[]
    for m_key, m_name in metrics:
        row = [m_name]
        for lbl, runs in categories.items():
            vals = [r[m_key] for r in runs if not np.isnan(r.get(m_key, np.nan))]
            if vals:
                mean_val = np.mean(vals)
                std_val = np.std(vals)
                if std_val > 0.01:
                    row.append(f"{mean_val:.1f} ± {std_val:.1f}")
                else:
                    row.append(f"{mean_val:.1f}")
            else:
                row.append("N/A")
        table_rows.append(row)
        
    # Console Print
    print("\n" + "=" * 105)
    print(f"{'AGGREGATED SYSTEM PERFORMANCE METRICS (5 RUNS PER PHASE)':^105}")
    print("=" * 105)
    
    col_width = 18
    header_format = f"{{:<24}} | " + " | ".join([f"{{:^{col_width}}}" for _ in categories.keys()])
    row_format = f"{{:<24}} | " + " | ".join([f"{{:^{col_width}}}" for _ in categories.keys()])
    
    print(header_format.format(*headers))
    print("-" * 105)
    for row in table_rows:
        print(row_format.format(*row))
    print("=" * 105 + "\n")
    
    # LaTeX Generation
    print("\\begin{table}[h!]")
    print("\\centering")
    print("\\caption{Aggregated Performance Metrics across 5 Simulation Runs per Phase}")
    print("\\label{tab:performance_metrics}")
    print("\\begin{tabular}{l" + "c" * len(categories) + "}")
    print("\\toprule")
    print("\\textbf{Metric} & " + " & ".join([f"\\textbf{{{c}}}" for c in categories.keys()]) + " \\\\")
    print("\\midrule")
    
    for row in table_rows:
        # Escape the % sign for LaTeX
        safe_row = [str(item).replace("%", "\\%") for item in row]
        print(" & ".join(safe_row) + " \\\\")
        
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print("==================================================\n")

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
    
    print("\nEvaluation complete! Add these plots to your Results section.")

if __name__ == "__main__":
    main()