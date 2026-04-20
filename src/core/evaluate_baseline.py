#!/usr/bin/env python3
"""
Baseline Evaluation & Comparison Plotter

Reads KPI data from kpms.csv (produced by xApp adapter) and generates
evaluation plots for:
  1. Baseline (no AI actions) — vanilla ns-3 mmWave performance
  2. AI-managed runs — with the LLM/KB/Reflexion control loop active
  3. Side-by-side comparison

Usage:
  # Collect baseline: run ns-3 sim with AI in baseline mode (no actions)
  python main.py --mode baseline --port 6000

  # Then plot baseline only:
  python evaluate_baseline.py --csv kpms_baseline.csv --label baseline

  # Compare baseline vs AI-managed:
  python evaluate_baseline.py --baseline kpms_baseline.csv --managed kpms_managed.csv

  # Plot from existing kpms.csv (auto-detect sessions by date):
  python evaluate_baseline.py --csv kpms.csv --list-sessions
  python evaluate_baseline.py --csv kpms.csv --session 2026-02-17
"""

import argparse
import csv
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

def _import_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    return plt, mdates

def load_kpms_csv(path: str, session_date: Optional[str] = None) -> Dict[str, List[dict]]:
    """Load kpms.csv and split into cell-level and per-UE rows.

    Returns dict with keys:
      "cell": list of cell-level metric dicts
      "ue":   dict of ue_id -> list of metric dicts
      "timestamps": sorted list of unique timestamps
    """
    cell_rows = []
    ue_rows = defaultdict(list)
    all_timestamps = set()

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_str = row.get("timestamp", "")
            if not ts_str:
                continue

            # Filter by session date if specified
            if session_date and not ts_str.startswith(session_date):
                continue

            try:
                ts = datetime.fromisoformat(ts_str)
            except ValueError:
                continue

            cell_id = row.get("cell_id", "")
            node_id = row.get("node_id", "")

            # Cell-level rows: CELL_* or "unknown" with cell metrics
            if cell_id.startswith("CELL_") or cell_id == "unknown":
                entry = {"timestamp": ts}
                for key in ["DRB_PdcpSduDelayDl", "RRU_PrbUsedDl", "DRB_MeanActiveUeDl",
                            "TB_TotNbrDlInitial_Qpsk", "TB_TotNbrDlInitial_16Qam",
                            "TB_TotNbrDlInitial_64Qam"]:
                    val = row.get(key, "")
                    if val:
                        try:
                            entry[key] = float(val)
                        except ValueError:
                            pass
                if len(entry) > 1:  # has at least one metric
                    cell_rows.append(entry)
                    all_timestamps.add(ts)

            # UE-level rows: UE_* cell_id
            elif cell_id.startswith("UE_"):
                ue_id = cell_id
                entry = {"timestamp": ts, "ue_id": ue_id}
                for key in ["UE_DRB_UEThpDl_UEID", "UE_DRB_PdcpSduDelayDl_UEID",
                            "UE_RRU_PrbUsedDl_UEID", "UE_DRB_BlerDl_UEID",
                            "UE_DRB_EstabSucc_5QI_UEID",
                            "UE_HO_SrcCellQual_RS-SINR_UEID",
                            "UE_TB_TotNbrDlInitial_Qpsk_UEID",
                            "UE_TB_TotNbrDlInitial_16Qam_UEID",
                            "UE_TB_TotNbrDlInitial_64Qam_UEID"]:
                    val = row.get(key, "")
                    if val:
                        try:
                            entry[key] = float(val)
                        except ValueError:
                            pass
                if len(entry) > 2:  # has at least one metric beyond ts+ue_id
                    ue_rows[ue_id].append(entry)
                    all_timestamps.add(ts)

    return {
        "cell": sorted(cell_rows, key=lambda x: x["timestamp"]),
        "ue": {uid: sorted(rows, key=lambda x: x["timestamp"]) for uid, rows in ue_rows.items()},
        "timestamps": sorted(all_timestamps),
    }


def list_sessions(path: str):
    """List available date sessions in the CSV."""
    dates = defaultdict(int)
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = row.get("timestamp", "")[:10]
            if ts:
                dates[ts] += 1

    print(f"\nAvailable sessions in {path}:")
    print(f"{'Date':<15} {'Rows':>10}")
    print("-" * 27)
    for date in sorted(dates):
        print(f"{date:<15} {dates[date]:>10,}")
    print()


# Metric Extraction

def extract_timeseries(data: Dict, metric: str, source: str = "cell",
                       ue_id: Optional[str] = None) -> Tuple[List[datetime], List[float]]:
    """Extract a (timestamps, values) pair for a given metric."""
    if source == "cell":
        rows = data["cell"]
    elif source == "ue" and ue_id:
        rows = data["ue"].get(ue_id, [])
    else:
        return [], []

    ts_list, val_list = [], []
    for row in rows:
        if metric in row:
            ts_list.append(row["timestamp"])
            val_list.append(row[metric])
    return ts_list, val_list


def compute_mcs_from_tb(data: Dict) -> Tuple[List[datetime], List[str]]:
    """Infer dominant MCS from TB transport block counts.

    Returns (timestamps, mcs_labels) where label is one of
    'QPSK', '16QAM', '64QAM', or 'None'.
    """
    ts_list, mcs_list = [], []
    for row in data["cell"]:
        qpsk = row.get("TB_TotNbrDlInitial_Qpsk", 0)
        qam16 = row.get("TB_TotNbrDlInitial_16Qam", 0)
        qam64 = row.get("TB_TotNbrDlInitial_64Qam", 0)
        total = qpsk + qam16 + qam64
        if total == 0:
            continue
        # Dominant modulation
        dominant = max([(qpsk, "QPSK"), (qam16, "16QAM"), (qam64, "64QAM")], key=lambda x: x[0])
        ts_list.append(row["timestamp"])
        mcs_list.append(dominant[1])
    return ts_list, mcs_list


def compute_summary_stats(data: Dict) -> dict:
    """Compute summary statistics for a dataset."""
    stats = {}

    # Cell-level latency
    _, delay_vals = extract_timeseries(data, "DRB_PdcpSduDelayDl", "cell")
    if delay_vals and any(v > 0 for v in delay_vals):
        arr = np.array(delay_vals)
        stats["cell_latency_mean_ms"] = float(np.nanmean(arr))
        stats["cell_latency_median_ms"] = float(np.nanmedian(arr))
        stats["cell_latency_p95_ms"] = float(np.nanpercentile(arr, 95))

    # PRB usage (cell-level)
    _, prb_vals = extract_timeseries(data, "RRU_PrbUsedDl", "cell")
    if prb_vals and any(v > 0 for v in prb_vals):
        arr = np.array(prb_vals)
        stats["cell_prb_mean"] = float(np.nanmean(arr))
        stats["cell_prb_p95"] = float(np.nanpercentile(arr, 95))

    # Per-UE metrics
    for ue_id in sorted(data["ue"].keys()):
        short_id = ue_id.replace("UE_", "")[-4:]

        # Throughput
        _, thp_vals = extract_timeseries(data, "UE_DRB_UEThpDl_UEID", "ue", ue_id)
        if thp_vals:
            arr = np.array(thp_vals)
            stats[f"ue_{short_id}_thp_mean_kbps"] = float(np.nanmean(arr))
            stats[f"ue_{short_id}_thp_median_kbps"] = float(np.nanmedian(arr))
            stats[f"ue_{short_id}_thp_p5_kbps"] = float(np.nanpercentile(arr, 5))

        # Per-UE latency
        _, lat_vals = extract_timeseries(data, "UE_DRB_PdcpSduDelayDl_UEID", "ue", ue_id)
        if lat_vals and any(v > 0 for v in lat_vals):
            arr = np.array(lat_vals)
            stats[f"ue_{short_id}_latency_mean_ms"] = float(np.nanmean(arr))
            stats[f"ue_{short_id}_latency_p95_ms"] = float(np.nanpercentile(arr, 95))

        # Per-UE BLER
        _, bler_vals = extract_timeseries(data, "UE_DRB_BlerDl_UEID", "ue", ue_id)
        if bler_vals and any(v > 0 for v in bler_vals):
            arr = np.array(bler_vals)
            stats[f"ue_{short_id}_bler_mean"] = float(np.nanmean(arr))

    # MCS distribution (only if TB counts have real data)
    _, mcs_labels = compute_mcs_from_tb(data)
    if mcs_labels:
        from collections import Counter
        counts = Counter(mcs_labels)
        total = len(mcs_labels)
        # Only report if there's actual modulation diversity
        if len(counts) > 1 or "QPSK" not in counts:
            for mod in ["QPSK", "16QAM", "64QAM"]:
                stats[f"mcs_{mod}_pct"] = 100.0 * counts.get(mod, 0) / total

    # Aggregate throughput
    all_thp = []
    for ue_id in data["ue"]:
        _, vals = extract_timeseries(data, "UE_DRB_UEThpDl_UEID", "ue", ue_id)
        all_thp.extend(vals)
    if all_thp:
        stats["agg_thp_mean_kbps"] = float(np.nanmean(all_thp))
        stats["agg_thp_median_kbps"] = float(np.nanmedian(all_thp))

    # Time span
    if data["timestamps"]:
        dt = (data["timestamps"][-1] - data["timestamps"][0]).total_seconds()
        stats["duration_s"] = dt
        stats["num_samples"] = len(data["timestamps"])

    return stats


# Plotting

def _relative_minutes(timestamps: List[datetime]) -> List[float]:
    """Convert absolute timestamps to minutes relative to the first."""
    if not timestamps:
        return []
    t0 = timestamps[0]
    return [(t - t0).total_seconds() / 60.0 for t in timestamps]


def _has_nonzero_data(timestamps, values, threshold=0.0) -> bool:
    """Check if a timeseries has any meaningful (non-zero) data."""
    if not values:
        return False
    return any(abs(v) > threshold for v in values)


def _plot_rolling(ax, mins, vals, raw_color, avg_color, raw_lw=0.6, label=None):
    """Plot raw timeseries with rolling average overlay."""
    ax.plot(mins, vals, linewidth=raw_lw, alpha=0.5, color=raw_color, label=label)
    if len(vals) > 20:
        window = min(50, len(vals) // 4)
        avg = np.convolve(vals, np.ones(window)/window, mode="valid")
        lbl = f"rolling avg (w={window})" if label is None else None
        ax.plot(mins[window-1:], avg, linewidth=1.5, color=avg_color, label=lbl)


def plot_single_dataset(data: Dict, label: str, out_dir: str):
    """Generate evaluation plots, adapting panels to whatever data is available."""
    plt, mdates = _import_plt()
    os.makedirs(out_dir, exist_ok=True)

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:cyan"]
    ue_ids = sorted(data["ue"].keys())

    cell_latency_ts, cell_latency_vals = extract_timeseries(data, "DRB_PdcpSduDelayDl", "cell")
    has_cell_latency = _has_nonzero_data(cell_latency_ts, cell_latency_vals)

    cell_prb_ts, cell_prb_vals = extract_timeseries(data, "RRU_PrbUsedDl", "cell")
    has_cell_prb = _has_nonzero_data(cell_prb_ts, cell_prb_vals)

    ts_mcs, mcs_labels = compute_mcs_from_tb(data)
    has_mcs = len(ts_mcs) > 0 and any(m != "QPSK" for m in mcs_labels)  # all-QPSK with zero TB = no real data

    # Check UE-level latency and BLER availability
    has_ue_latency = any(
        _has_nonzero_data(*extract_timeseries(data, "UE_DRB_PdcpSduDelayDl_UEID", "ue", uid))
        for uid in ue_ids
    )
    has_ue_bler = any(
        _has_nonzero_data(*extract_timeseries(data, "UE_DRB_BlerDl_UEID", "ue", uid))
        for uid in ue_ids
    )
    has_ue_prb = any(
        _has_nonzero_data(*extract_timeseries(data, "UE_RRU_PrbUsedDl_UEID", "ue", uid))
        for uid in ue_ids
    )

    # Build panel list dynamically
    panels = []  # list of (title, draw_func)

    # Panel 1: Latency — cell-level if available, else per-UE overlay
    def draw_latency(ax):
        if has_cell_latency:
            mins = _relative_minutes(cell_latency_ts)
            _plot_rolling(ax, mins, cell_latency_vals, "tab:red", "darkred")
            ax.set_ylabel("DL PDCP Delay (ms)")
            ax.set_title("Cell-Level Downlink Latency")
        elif has_ue_latency:
            for i, uid in enumerate(ue_ids):
                ts, vals = extract_timeseries(data, "UE_DRB_PdcpSduDelayDl_UEID", "ue", uid)
                mins = _relative_minutes(ts)
                if mins and _has_nonzero_data(ts, vals):
                    short = uid.replace("UE_", "")[-4:]
                    ax.plot(mins, vals, linewidth=0.5, alpha=0.6,
                            color=colors[i % len(colors)], label=f"UE ...{short}")
            ax.set_ylabel("DL PDCP Delay (ms)")
            ax.set_title("Per-UE Downlink Latency")
            ax.legend(fontsize=7, ncol=3)
        else:
            ax.text(0.5, 0.5, "No latency data available", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="gray")
            ax.set_title("Downlink Latency (no data)")
        ax.grid(True, alpha=0.3)
    panels.append(("latency", draw_latency))

    # Panel 2: Per-UE throughput (always present)
    def draw_throughput(ax):
        for i, uid in enumerate(ue_ids):
            ts, vals = extract_timeseries(data, "UE_DRB_UEThpDl_UEID", "ue", uid)
            mins = _relative_minutes(ts)
            if mins:
                short = uid.replace("UE_", "")[-4:]
                ax.plot(mins, vals, linewidth=0.5, alpha=0.6,
                        color=colors[i % len(colors)], label=f"UE ...{short}")
        ax.set_ylabel("Throughput (kbps)")
        ax.set_title("Per-UE Downlink Throughput")
        ax.legend(fontsize=7, ncol=3)
        ax.grid(True, alpha=0.3)
    panels.append(("throughput", draw_throughput))

    # Panel 3: Per-UE latency (if cell latency was used in panel 1 and UE latency also exists)
    if has_cell_latency and has_ue_latency:
        def draw_ue_latency(ax):
            for i, uid in enumerate(ue_ids):
                ts, vals = extract_timeseries(data, "UE_DRB_PdcpSduDelayDl_UEID", "ue", uid)
                mins = _relative_minutes(ts)
                if mins and _has_nonzero_data(ts, vals):
                    short = uid.replace("UE_", "")[-4:]
                    ax.plot(mins, vals, linewidth=0.5, alpha=0.6,
                            color=colors[i % len(colors)], label=f"UE ...{short}")
            ax.set_ylabel("DL PDCP Delay (ms)")
            ax.set_title("Per-UE Downlink Latency")
            ax.legend(fontsize=7, ncol=3)
            ax.grid(True, alpha=0.3)
        panels.append(("ue_latency", draw_ue_latency))

    # Panel 4: PRB usage — cell-level or per-UE
    if has_cell_prb:
        def draw_prb(ax):
            mins = _relative_minutes(cell_prb_ts)
            ax.fill_between(mins, cell_prb_vals, alpha=0.4, color="tab:green")
            ax.plot(mins, cell_prb_vals, linewidth=0.5, color="tab:green")
            ax.set_ylabel("PRBs Used (DL)")
            ax.set_title("Cell-Level PRB Usage")
            ax.grid(True, alpha=0.3)
        panels.append(("prb", draw_prb))
    elif has_ue_prb:
        def draw_ue_prb(ax):
            for i, uid in enumerate(ue_ids):
                ts, vals = extract_timeseries(data, "UE_RRU_PrbUsedDl_UEID", "ue", uid)
                mins = _relative_minutes(ts)
                if mins and _has_nonzero_data(ts, vals):
                    short = uid.replace("UE_", "")[-4:]
                    ax.plot(mins, vals, linewidth=0.5, alpha=0.6,
                            color=colors[i % len(colors)], label=f"UE ...{short}")
            ax.set_ylabel("PRBs Used (DL)")
            ax.set_title("Per-UE PRB Usage")
            ax.legend(fontsize=7, ncol=3)
            ax.grid(True, alpha=0.3)
        panels.append(("ue_prb", draw_ue_prb))

    # Panel 5: MCS / Modulation (only if TB counts have real data)
    if has_mcs:
        def draw_mcs(ax):
            mins_mcs = _relative_minutes(ts_mcs)
            mcs_map = {"QPSK": 0, "16QAM": 1, "64QAM": 2}
            mcs_nums = [mcs_map.get(m, -1) for m in mcs_labels]
            scatter_colors = {"QPSK": "tab:red", "16QAM": "tab:orange", "64QAM": "tab:green"}
            for mod, num in mcs_map.items():
                idxs = [j for j, n in enumerate(mcs_nums) if n == num]
                if idxs:
                    ax.scatter([mins_mcs[j] for j in idxs], [num]*len(idxs),
                              s=4, alpha=0.5, color=scatter_colors[mod], label=mod)
            ax.set_yticks([0, 1, 2])
            ax.set_yticklabels(["QPSK", "16QAM", "64QAM"])
            ax.legend(fontsize=8)
            ax.set_ylabel("Dominant Modulation")
            ax.set_title("MCS / Modulation Over Time")
            ax.grid(True, alpha=0.3)
        panels.append(("mcs", draw_mcs))

    # Panel 6: BLER (only if non-zero)
    if has_ue_bler:
        def draw_bler(ax):
            for i, uid in enumerate(ue_ids):
                ts, vals = extract_timeseries(data, "UE_DRB_BlerDl_UEID", "ue", uid)
                mins = _relative_minutes(ts)
                if mins and _has_nonzero_data(ts, vals):
                    short = uid.replace("UE_", "")[-4:]
                    ax.plot(mins, vals, linewidth=0.5, alpha=0.6,
                            color=colors[i % len(colors)], label=f"UE ...{short}")
            ax.set_ylabel("BLER (DL)")
            ax.set_title("Per-UE Block Error Rate")
            ax.legend(fontsize=7, ncol=3)
            ax.grid(True, alpha=0.3)
        panels.append(("bler", draw_bler))

    # Render panels
    n_panels = len(panels)
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 4 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]
    fig.suptitle(f"Network KPI Overview — {label}", fontsize=14, fontweight="bold")

    for i, (name, draw_fn) in enumerate(panels):
        draw_fn(axes[i])
    axes[-1].set_xlabel("Time (minutes)")

    plt.tight_layout()
    path = os.path.join(out_dir, f"kpi_overview_{label}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    # Per-UE detail: all observable metrics
    # Each metric: (csv_column, y-label, color, optional_transform)
    ue_metric_cols = [
        ("UE_DRB_UEThpDl_UEID",              "Throughput (kbps)",  "tab:blue"),
        ("UE_DRB_PdcpSduDelayDl_UEID",       "DL Delay (ms)",     "tab:red"),
        ("UE_DRB_BlerDl_UEID",               "BLER (DL)",         "tab:orange"),
        ("UE_RRU_PrbUsedDl_UEID",            "PRBs Used (DL)",    "tab:green"),
        ("UE_HO_SrcCellQual_RS-SINR_UEID",   "SINR (dB)",         "tab:purple"),
    ]
    if ue_ids:
        n_cols = len(ue_metric_cols)
        fig, axes = plt.subplots(len(ue_ids), n_cols,
                                 figsize=(5 * n_cols, 3.5 * len(ue_ids)), squeeze=False)
        fig.suptitle(f"Per-UE Detail — {label}", fontsize=14, fontweight="bold")

        for i, ue_id in enumerate(ue_ids):
            short = ue_id.replace("UE_", "")[-4:]
            for j, (metric, ylabel, color) in enumerate(ue_metric_cols):
                ax = axes[i][j]
                ts, vals = extract_timeseries(data, metric, "ue", ue_id)
                if _has_nonzero_data(ts, vals):
                    mins = _relative_minutes(ts)
                    if mins:
                        _plot_rolling(ax, mins, vals, color, "black", raw_lw=0.5)
                else:
                    ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                            ha="center", va="center", fontsize=10, color="gray")
                ax.set_ylabel(ylabel)
                ax.set_title(f"UE ...{short} — {ylabel}")
                ax.grid(True, alpha=0.3)

        for ax in axes[-1]:
            ax.set_xlabel("Time (minutes)")

        plt.tight_layout()
        path = os.path.join(out_dir, f"ue_detail_{label}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")

    # Per-UE MCS distribution (derived from TB counts)
    if ue_ids:
        fig, axes = plt.subplots(len(ue_ids), 1, figsize=(14, 3 * len(ue_ids)), squeeze=False)
        fig.suptitle(f"Per-UE Modulation (MCS) — {label}", fontsize=14, fontweight="bold")
        mcs_map_colors = {"QPSK": "tab:red", "16QAM": "tab:orange", "64QAM": "tab:green"}

        for i, ue_id in enumerate(ue_ids):
            short = ue_id.replace("UE_", "")[-4:]
            ax = axes[i][0]

            # Extract per-UE TB counts
            ts_q, vals_q = extract_timeseries(data, "UE_TB_TotNbrDlInitial_Qpsk_UEID", "ue", ue_id)
            ts_16, vals_16 = extract_timeseries(data, "UE_TB_TotNbrDlInitial_16Qam_UEID", "ue", ue_id)
            ts_64, vals_64 = extract_timeseries(data, "UE_TB_TotNbrDlInitial_64Qam_UEID", "ue", ue_id)

            if ts_q or ts_16 or ts_64:
                # Build aligned timestamps and derive dominant modulation
                all_ts = sorted(set((ts_q or []) + (ts_16 or []) + (ts_64 or [])))
                q_dict = dict(zip(ts_q or [], vals_q or []))
                s_dict = dict(zip(ts_16 or [], vals_16 or []))
                x_dict = dict(zip(ts_64 or [], vals_64 or []))

                mcs_ts, mcs_labels_ue = [], []
                for t in all_ts:
                    qv = q_dict.get(t, 0); sv = s_dict.get(t, 0); xv = x_dict.get(t, 0)
                    if qv + sv + xv > 0:
                        dominant = max([("QPSK", qv), ("16QAM", sv), ("64QAM", xv)], key=lambda x: x[1])
                        mcs_ts.append(t); mcs_labels_ue.append(dominant[0])

                if mcs_ts:
                    mins_ue = _relative_minutes(mcs_ts)
                    mcs_num_map = {"QPSK": 0, "16QAM": 1, "64QAM": 2}
                    for mod, num in mcs_num_map.items():
                        idxs = [j for j, m in enumerate(mcs_labels_ue) if m == mod]
                        if idxs:
                            ax.scatter([mins_ue[j] for j in idxs], [num]*len(idxs),
                                       s=4, alpha=0.5, color=mcs_map_colors[mod], label=mod)
                    ax.set_yticks([0, 1, 2])
                    ax.set_yticklabels(["QPSK", "16QAM", "64QAM"])
                    ax.legend(fontsize=7)
                else:
                    ax.text(0.5, 0.5, "no MCS data", transform=ax.transAxes,
                            ha="center", va="center", fontsize=10, color="gray")
            else:
                ax.text(0.5, 0.5, "no MCS data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=10, color="gray")
            ax.set_title(f"UE ...{short} — Dominant Modulation")
            ax.grid(True, alpha=0.3)

        axes[-1][0].set_xlabel("Time (minutes)")
        plt.tight_layout()
        path = os.path.join(out_dir, f"ue_mcs_{label}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")


def plot_comparison(baseline: Dict, managed: Dict, out_dir: str):
    """Generate side-by-side comparison plots."""
    plt, _ = _import_plt()
    os.makedirs(out_dir, exist_ok=True)

    # Summary statistics comparison
    b_stats = compute_summary_stats(baseline)
    m_stats = compute_summary_stats(managed)

    print("\n" + "=" * 65)
    print(f"{'Metric':<35} {'Baseline':>12} {'AI-Managed':>12}")
    print("=" * 65)
    all_keys = sorted(set(list(b_stats.keys()) + list(m_stats.keys())))
    for key in all_keys:
        bv = b_stats.get(key)
        mv = m_stats.get(key)
        bs = f"{bv:.2f}" if bv is not None else "N/A"
        ms = f"{mv:.2f}" if mv is not None else "N/A"
        print(f"{key:<35} {bs:>12} {ms:>12}")
    print("=" * 65)

    # Latency comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Baseline vs AI-Managed Comparison", fontsize=14, fontweight="bold")

    # Latency time series
    ax = axes[0][0]
    for dataset, label, color in [(baseline, "Baseline", "tab:red"), (managed, "AI-Managed", "tab:blue")]:
        ts, vals = extract_timeseries(dataset, "DRB_PdcpSduDelayDl", "cell")
        mins = _relative_minutes(ts)
        if mins and len(vals) > 20:
            window = min(50, len(vals) // 4)
            avg = np.convolve(vals, np.ones(window)/window, mode="valid")
            ax.plot(mins[window-1:], avg, linewidth=1.5, color=color, label=label)
    ax.set_ylabel("DL Latency (ms)")
    ax.set_title("Latency (rolling avg)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Latency CDF
    ax = axes[0][1]
    for dataset, label, color in [(baseline, "Baseline", "tab:red"), (managed, "AI-Managed", "tab:blue")]:
        _, vals = extract_timeseries(dataset, "DRB_PdcpSduDelayDl", "cell")
        if vals:
            sorted_v = np.sort(vals)
            cdf = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
            ax.plot(sorted_v, cdf, linewidth=1.5, color=color, label=label)
    ax.set_xlabel("DL Latency (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("Latency CDF")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Aggregate throughput
    ax = axes[1][0]
    for dataset, label, color in [(baseline, "Baseline", "tab:red"), (managed, "AI-Managed", "tab:blue")]:
        # Sum throughput across all UEs at each timestamp
        all_thp = defaultdict(float)
        for ue_id in dataset["ue"]:
            for row in dataset["ue"][ue_id]:
                if "UE_DRB_UEThpDl_UEID" in row:
                    all_thp[row["timestamp"]] += row["UE_DRB_UEThpDl_UEID"]
        if all_thp:
            sorted_ts = sorted(all_thp.keys())
            vals = [all_thp[t] for t in sorted_ts]
            mins = _relative_minutes(sorted_ts)
            if len(vals) > 20:
                window = min(50, len(vals) // 4)
                avg = np.convolve(vals, np.ones(window)/window, mode="valid")
                ax.plot(mins[window-1:], avg, linewidth=1.5, color=color, label=label)
    ax.set_ylabel("Aggregate Throughput (kbps)")
    ax.set_title("Total DL Throughput (rolling avg)")
    ax.set_xlabel("Time (minutes)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # MCS distribution bar chart
    ax = axes[1][1]
    width = 0.35
    x = np.arange(3)
    labels = ["QPSK", "16QAM", "64QAM"]
    for i, (dataset, label, color) in enumerate([(baseline, "Baseline", "tab:red"), (managed, "AI-Managed", "tab:blue")]):
        _, mcs_labels = compute_mcs_from_tb(dataset)
        if mcs_labels:
            from collections import Counter
            counts = Counter(mcs_labels)
            total = len(mcs_labels)
            pcts = [100.0 * counts.get(m, 0) / total for m in labels]
        else:
            pcts = [0, 0, 0]
        ax.bar(x + i * width, pcts, width, label=label, color=color, alpha=0.7)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(labels)
    ax.set_ylabel("% of samples")
    ax.set_title("MCS Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(out_dir, "comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# Recovery Time Analysis

def analyze_recovery_times(data: Dict, metric: str = "DRB_PdcpSduDelayDl",
                           threshold_factor: float = 2.0) -> List[dict]:
    """Detect degradation events and measure time-to-recover.

    A degradation event starts when the metric exceeds threshold_factor * baseline_median
    and ends when it returns below that threshold.
    Tries cell-level first, then falls back to per-UE data.
    """
    ts, vals = extract_timeseries(data, metric, "cell")
    # If cell-level has no meaningful data, try per-UE latency
    if len(vals) < 50 or not any(v > 0 for v in vals):
        ue_metric = "UE_DRB_PdcpSduDelayDl_UEID"
        # Merge all UE latency into a single series (take max across UEs per timestamp)
        all_ue_data = defaultdict(list)
        for uid in data["ue"]:
            for row in data["ue"][uid]:
                if ue_metric in row:
                    all_ue_data[row["timestamp"]].append(row[ue_metric])
        if all_ue_data:
            sorted_ts = sorted(all_ue_data.keys())
            ts = sorted_ts
            vals = [max(all_ue_data[t]) for t in sorted_ts]  # worst-case UE
    if len(vals) < 50:
        return []

    arr = np.array(vals)
    baseline_median = np.nanmedian(arr[:min(100, len(arr) // 4)])
    threshold = baseline_median * threshold_factor

    events = []
    in_degradation = False
    start_ts = None

    for i, (t, v) in enumerate(zip(ts, vals)):
        if not in_degradation and v > threshold:
            in_degradation = True
            start_ts = t
        elif in_degradation and v <= threshold:
            in_degradation = False
            recovery_s = (t - start_ts).total_seconds()
            peak_val = max(vals[j] for j in range(max(0, i - 50), i) if vals[j] is not None)
            events.append({
                "start": start_ts,
                "end": t,
                "recovery_s": recovery_s,
                "peak": peak_val,
                "threshold": threshold,
            })

    return events


def print_recovery_analysis(data: Dict, label: str):
    """Print recovery time analysis."""
    events = analyze_recovery_times(data)
    if not events:
        print(f"\n  {label}: No degradation events detected (or not enough data)")
        return

    recovery_times = [e["recovery_s"] for e in events]
    print(f"\n  {label}: {len(events)} degradation events detected")
    print(f"    Mean recovery time: {np.mean(recovery_times):.1f}s")
    print(f"    Median recovery time: {np.median(recovery_times):.1f}s")
    print(f"    Max recovery time: {np.max(recovery_times):.1f}s")
    print(f"    Min recovery time: {np.min(recovery_times):.1f}s")
    for i, e in enumerate(events[:10]):
        print(f"    Event {i+1}: start={e['start'].strftime('%H:%M:%S')}, "
              f"recovery={e['recovery_s']:.1f}s, peak={e['peak']:.1f}")


# Main

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and plot baseline vs AI-managed network performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available sessions in existing data
  python evaluate_baseline.py --csv kpms.csv --list-sessions

  # Plot a single session
  python evaluate_baseline.py --csv kpms.csv --session 2026-02-17 --label baseline

  # Compare baseline vs AI-managed (two separate CSV files)
  python evaluate_baseline.py --baseline kpms_baseline.csv --managed kpms_managed.csv

  # Compare two sessions from same CSV
  python evaluate_baseline.py --csv kpms.csv --baseline-session 2026-01-26 --managed-session 2026-02-17
        """,
    )

    parser.add_argument("--csv", type=str, help="Single kpms.csv file to analyze")
    parser.add_argument("--baseline", type=str, help="Baseline CSV file path")
    parser.add_argument("--managed", type=str, help="AI-managed CSV file path")
    parser.add_argument("--session", type=str, help="Filter by date (YYYY-MM-DD)")
    parser.add_argument("--baseline-session", type=str, help="Baseline session date")
    parser.add_argument("--managed-session", type=str, help="AI-managed session date")
    parser.add_argument("--label", type=str, default="run", help="Label for single-dataset plots")
    parser.add_argument("--list-sessions", action="store_true", help="List available sessions and exit")
    parser.add_argument("--out-dir", type=str, default="evaluation_plots", help="Output directory for plots")

    args = parser.parse_args()

    # List sessions mode
    if args.list_sessions:
        csv_path = args.csv or "kpms.csv"
        if not os.path.exists(csv_path):
            print(f"File not found: {csv_path}")
            sys.exit(1)
        list_sessions(csv_path)
        sys.exit(0)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Comparison mode: two separate files
    if args.baseline and args.managed:
        print(f"Loading baseline: {args.baseline}")
        b_data = load_kpms_csv(args.baseline)
        print(f"  → {len(b_data['cell'])} cell rows, {sum(len(v) for v in b_data['ue'].values())} UE rows")

        print(f"Loading AI-managed: {args.managed}")
        m_data = load_kpms_csv(args.managed)
        print(f"  → {len(m_data['cell'])} cell rows, {sum(len(v) for v in m_data['ue'].values())} UE rows")

        print("\nGenerating plots...")
        plot_single_dataset(b_data, "baseline", out_dir)
        plot_single_dataset(m_data, "ai_managed", out_dir)
        plot_comparison(b_data, m_data, out_dir)
        print_recovery_analysis(b_data, "Baseline")
        print_recovery_analysis(m_data, "AI-Managed")

    # Comparison mode: two sessions from same CSV
    elif args.baseline_session and args.managed_session:
        csv_path = args.csv or "kpms.csv"
        print(f"Loading baseline session {args.baseline_session} from {csv_path}")
        b_data = load_kpms_csv(csv_path, session_date=args.baseline_session)
        print(f"  → {len(b_data['cell'])} cell rows, {sum(len(v) for v in b_data['ue'].values())} UE rows")

        print(f"Loading AI-managed session {args.managed_session} from {csv_path}")
        m_data = load_kpms_csv(csv_path, session_date=args.managed_session)
        print(f"  → {len(m_data['cell'])} cell rows, {sum(len(v) for v in m_data['ue'].values())} UE rows")

        print("\nGenerating plots...")
        plot_single_dataset(b_data, f"baseline_{args.baseline_session}", out_dir)
        plot_single_dataset(m_data, f"managed_{args.managed_session}", out_dir)
        plot_comparison(b_data, m_data, out_dir)
        print_recovery_analysis(b_data, f"Baseline ({args.baseline_session})")
        print_recovery_analysis(m_data, f"AI-Managed ({args.managed_session})")

    # Single dataset mode
    elif args.csv:
        print(f"Loading {args.csv}" + (f" (session: {args.session})" if args.session else ""))
        data = load_kpms_csv(args.csv, session_date=args.session)
        print(f"  → {len(data['cell'])} cell rows, {sum(len(v) for v in data['ue'].values())} UE rows")
        print(f"  → UE IDs: {sorted(data['ue'].keys())}")

        if not data["cell"] and not data["ue"]:
            print("No data found! Check your --session filter.")
            sys.exit(1)

        print("\nGenerating plots...")
        plot_single_dataset(data, args.label, out_dir)
        print("\nSummary statistics:")
        stats = compute_summary_stats(data)
        for k, v in sorted(stats.items()):
            print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")
        print_recovery_analysis(data, args.label)

    else:
        parser.print_help()
        sys.exit(1)

    print(f"\nAll plots saved to: {out_dir}/")


if __name__ == "__main__":
    main()