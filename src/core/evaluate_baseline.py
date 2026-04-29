#!/usr/bin/env python3
"""
Single Run Trace Plotter (High Resolution - 3 Panels)

Usage:
  python plot_trace.py --csv energy5.csv
"""

import argparse
import csv
import os
from datetime import datetime
from collections import defaultdict
import numpy as np

def _import_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt

def load_kpms_csv(path: str) -> dict:
    """Load the KPI stream and sort it by timestamp."""
    cell_rows =[]
    ue_rows = defaultdict(list)
    all_timestamps = set()

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_str = row.get("timestamp", "")
            if not ts_str: continue

            try:
                ts = datetime.fromisoformat(ts_str)
            except ValueError:
                continue

            cell_id = row.get("cell_id", "")
            
            # Cell-level metrics
            if cell_id.startswith("CELL_") or cell_id == "unknown":
                entry = {"timestamp": ts}
                for key in["DRB_PdcpSduDelayDl", "RRU_PrbUsedDl", "TB_TotNbrDlInitial_Qpsk", "TB_TotNbrDlInitial_16Qam", "TB_TotNbrDlInitial_64Qam"]:
                    val = row.get(key, "")
                    if val:
                        try: entry[key] = float(val)
                        except ValueError: pass
                if len(entry) > 1:
                    cell_rows.append(entry)
                    all_timestamps.add(ts)

            # UE-level metrics
            elif cell_id.startswith("UE_"):
                entry = {"timestamp": ts}
                for key in["UE_DRB_UEThpDl_UEID", "UE_DRB_PdcpSduDelayDl_UEID"]:
                    val = row.get(key, "")
                    if val:
                        try: entry[key] = float(val)
                        except ValueError: pass
                if len(entry) > 1:
                    ue_rows[cell_id].append(entry)
                    all_timestamps.add(ts)

    return {
        "cell": sorted(cell_rows, key=lambda x: x["timestamp"]),
        "ue": {uid: sorted(rows, key=lambda x: x["timestamp"]) for uid, rows in ue_rows.items()},
        "timestamps": sorted(all_timestamps),
    }

def plot_single_run(data: dict, out_dir: str, file_name: str = "single_run_trace.png"):
    """Plot the raw timeseries data without averaging in a 3-panel layout."""
    plt = _import_plt()
    os.makedirs(out_dir, exist_ok=True)

    if not data["timestamps"]:
        print("No valid data found to plot.")
        return

    t0 = data["timestamps"][0]
    ue_colors =["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:cyan"]
    ue_ids = sorted(data["ue"].keys())

    # Extract MCS Data for Distribution
    mcs_labels =[]
    for row in data["cell"]:
        qpsk = row.get("TB_TotNbrDlInitial_Qpsk", 0)
        qam16 = row.get("TB_TotNbrDlInitial_16Qam", 0)
        qam64 = row.get("TB_TotNbrDlInitial_64Qam", 0)
        if qpsk + qam16 + qam64 == 0:
            continue
        dominant = max([("QPSK", qpsk), ("16QAM", qam16), ("64QAM", qam64)], key=lambda x: x[1])
        mcs_labels.append(dominant[0])

    # Create Figure (3 panels)
    fig, axes = plt.subplots(3, 1, figsize=(12, 13))

    # ==========================================
    # Panel 1: Per-UE Latency
    # ==========================================
    ax1 = axes[0]
    if ue_ids:
        for i, uid in enumerate(ue_ids):
            l_ts = [r["timestamp"] for r in data["ue"][uid] if "UE_DRB_PdcpSduDelayDl_UEID" in r]
            l_vals = [r["UE_DRB_PdcpSduDelayDl_UEID"] for r in data["ue"][uid] if "UE_DRB_PdcpSduDelayDl_UEID" in r]
            l_mins =[(t - t0).total_seconds() / 60.0 for t in l_ts]
            
            if l_mins:
                short_id = uid.replace("UE_", "")[-4:]
                ax1.plot(l_mins, l_vals, color=ue_colors[i % len(ue_colors)], alpha=0.8, linewidth=1.2, label=f"UE ...{short_id}")
        
        ax1.set_ylabel("Latency (ms)", fontweight="bold")
        ax1.set_title("Per-UE Downlink Latency Spikes", fontsize=12)
        ax1.legend(loc="upper left", ncol=len(ue_ids), fontsize=9)
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "No UE Latency Data", ha='center', va='center')

    # ==========================================
    # Panel 2: Per-UE Throughput
    # ==========================================
    ax2 = axes[1]
    if ue_ids:
        for i, uid in enumerate(ue_ids):
            t_ts = [r["timestamp"] for r in data["ue"][uid] if "UE_DRB_UEThpDl_UEID" in r]
            t_vals = [r["UE_DRB_UEThpDl_UEID"] for r in data["ue"][uid] if "UE_DRB_UEThpDl_UEID" in r]
            t_mins =[(t - t0).total_seconds() / 60.0 for t in t_ts]
            
            if t_mins:
                short_id = uid.replace("UE_", "")[-4:]
                ax2.plot(t_mins, t_vals, color=ue_colors[i % len(ue_colors)], alpha=0.7, linewidth=1.0, label=f"UE ...{short_id}")
        
        ax2.set_ylabel("Throughput (kbps)", fontweight="bold")
        ax2.set_xlabel("Time (Minutes)", fontweight="bold")
        ax2.set_title("Per-UE Downlink Throughput", fontsize=12)
        ax2.legend(loc="upper right", ncol=len(ue_ids), fontsize=9)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No UE Throughput Data", ha='center', va='center')

    # ==========================================
    # Panel 3: MCS Bar Chart ("Boxes")
    # ==========================================
    ax3 = axes[2]
    if mcs_labels:
        from collections import Counter
        counts = Counter(mcs_labels)
        labels_order = ["QPSK", "16QAM", "64QAM"]
        freqs =[counts.get(m, 0) for m in labels_order]
        bar_colors =["tab:red", "tab:orange", "tab:green"]
        
        ax3.bar(labels_order, freqs, color=bar_colors, edgecolor="black", alpha=0.8, width=0.4)
        ax3.set_ylabel("Frequency (Frames)", fontweight="bold")
        ax3.set_xlabel("Modulation Type", fontweight="bold")
        ax3.set_title("Total Distribution of Applied Modulation", fontsize=12)
        ax3.grid(True, alpha=0.3, axis="y")
    else:
        ax3.text(0.5, 0.5, "No MCS Bar Data Available", ha='center', va='center')

    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Leaves room for main title
    path = os.path.join(out_dir, file_name)
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved Qualitative Trace Plot: {path}")

def main():
    parser = argparse.ArgumentParser(description="Plot a single KPM CSV run in high resolution.")
    parser.add_argument("--csv", type=str, required=True, help="Path to the kpms.csv file")
    parser.add_argument("--out-dir", type=str, default="evaluation_plots", help="Output directory")
    parser.add_argument("--file-name", type=str, default="single_trace.png", help="Output file name")
    
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: {args.csv} does not exist.")
        sys.exit(1)

    print(f"Loading {args.csv}...")
    data = load_kpms_csv(args.csv)
    plot_single_run(data, args.out_dir, args.file_name)

if __name__ == "__main__":
    main()