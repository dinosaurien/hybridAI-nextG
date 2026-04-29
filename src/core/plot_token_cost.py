#!/usr/bin/env python3
"""
Token Cost & Cognitive Overhead Plotter

Reads tokens.csv files from the evaluation runs and plots:
1. Context Bloat (Prompt tokens vs Adaptation Attempt)
2. Cognitive Delay (Inference Latency Breakdown)

Usage:
  python plot_token_cost.py \
      --managed tokens_no_reflexion1.csv tokens_no_reflexion2.csv ... \
      --reflexion tokens_reflexion1.csv tokens_reflexion2.csv ... \
      --energy tokens_energy1.csv tokens_energy2.csv ...
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

def load_tokens_csv(filepaths: list) -> list:
    """Load and merge multiple token CSVs into a single chronological list."""
    data =[]
    for path in filepaths:
        if not os.path.exists(path):
            continue
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts = datetime.fromisoformat(row['ts'].replace("Z", "+00:00"))
                    data.append({
                        'ts': ts,
                        'call_type': row['call_type'],
                        'prompt_tokens': int(row['prompt_tokens']),
                        'completion_tokens': int(row['completion_tokens']),
                        'total_tokens': int(row['total_tokens']),
                        'latency_ms': float(row['latency_ms'])
                    })
                except (ValueError, KeyError):
                    continue
    return sorted(data, key=lambda x: x['ts'])

def extract_adaptation_sequences(data: list, split_threshold_sec: float = 120.0) -> list:
    """
    Groups LLM calls into continuous 'Adaptation Sequences'.
    If the time between calls is > split_threshold_sec, it's considered a new anomaly/sequence.
    """
    sequences = []
    current_seq =[]
    
    for row in data:
        if not current_seq:
            current_seq.append(row)
        else:
            delta = (row['ts'] - current_seq[-1]['ts']).total_seconds()
            if delta < split_threshold_sec:
                current_seq.append(row)
            else:
                sequences.append(current_seq)
                current_seq = [row]
                
    if current_seq:
        sequences.append(current_seq)
        
    return sequences

def calculate_bloat(sequences: list):
    """Calculates average prompt tokens at Attempt 1, Attempt 2, etc."""
    attempt_tokens = defaultdict(list)
    
    for seq in sequences:
        decisions =[r for r in seq if r['call_type'] == 'decision']
        for i, dec in enumerate(decisions):
            attempt_tokens[i + 1].append(dec['prompt_tokens'])
            
    # Calculate means
    max_attempts = max(attempt_tokens.keys()) if attempt_tokens else 0
    means = [np.mean(attempt_tokens[i]) for i in range(1, min(6, max_attempts + 1))]
    return list(range(1, len(means) + 1)), means

def calculate_average_latencies(data: list):
    """Calculate average latency for decision and reflection calls in seconds."""
    dec_lats = [r['latency_ms'] / 1000.0 for r in data if r['call_type'] == 'decision']
    ref_lats =[r['latency_ms'] / 1000.0 for r in data if r['call_type'] == 'reflection']
    
    avg_dec = np.mean(dec_lats) if dec_lats else 0
    avg_ref = np.mean(ref_lats) if ref_lats else 0
    return avg_dec, avg_ref

def plot_cognitive_overhead(datasets: dict, out_dir: str):
    plt = _import_plt()
    os.makedirs(out_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ---------------------------------------------------------
    # Panel 1: Context Bloat (Line Chart - Averages Only)
    # ---------------------------------------------------------
    ax1 = axes[0]
    colors = {"AI (No Reflexion)": "tab:blue", "AI (Reflexion)": "tab:green", "Energy Intent": "tab:purple"}
    
    for label, data in datasets.items():
        if not data: continue
        seqs = extract_adaptation_sequences(data)
        x, means = calculate_bloat(seqs)
        
        if x:
            ax1.plot(x, means, marker='o', linewidth=2.5, markersize=8, label=label, color=colors.get(label, "black"))
            
    ax1.set_xticks([1, 2, 3, 4, 5])
    ax1.set_xlabel("Adaptation Attempt (Consecutive Failures)", fontweight="bold")
    ax1.set_ylabel("Prompt Context Size (Tokens)", fontweight="bold")
    ax1.set_title("Context Growth over Adaptation Cycles", fontsize=13)
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    # ---------------------------------------------------------
    # Panel 2: Latency Breakdown (Stacked Bar Chart)
    # ---------------------------------------------------------
    ax2 = axes[1]
    labels = list(datasets.keys())
    dec_times = []
    ref_times =[]
    
    for label in labels:
        d_time, r_time = calculate_average_latencies(datasets[label])
        dec_times.append(d_time)
        ref_times.append(r_time)
        
    x_pos = np.arange(len(labels))
    width = 0.5
    
    # Bottom bar: Decision time
    ax2.bar(x_pos, dec_times, width, label='Decision Latency', color='tab:blue', edgecolor='black', alpha=0.8)
    # Top bar: Reflection time (Stacked)
    ax2.bar(x_pos, ref_times, width, bottom=dec_times, label='Reflection Latency', color='tab:orange', edgecolor='black', alpha=0.8)
    
    # Add text labels on top of the bars
    for i in range(len(labels)):
        total_time = dec_times[i] + ref_times[i]
        ax2.text(x_pos[i], total_time + 0.2, f"{total_time:.1f}s", ha='center', fontweight='bold')
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, fontweight="bold")
    ax2.set_ylabel("Average Inference Latency (Seconds)", fontweight="bold")
    ax2.set_title("Average Inference Time per Adaptation", fontsize=13)
    ax2.legend()
    ax2.grid(True, axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    out_path = os.path.join(out_dir, "cognitive_overhead_analysis.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved Token Cost Plot: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot LLM Token Costs and Latency")
    parser.add_argument("--managed", nargs='+', help="List of AI (No Reflexion) token CSVs")
    parser.add_argument("--reflexion", nargs='+', help="List of AI (Reflexion) token CSVs")
    parser.add_argument("--energy", nargs='+', help="List of Energy Intent token CSVs")
    parser.add_argument("--out-dir", type=str, default="evaluation_plots", help="Output directory")
    
    args = parser.parse_args()
    
    datasets = {}
    if args.managed:
        datasets["AI (No Reflexion)"] = load_tokens_csv(args.managed)
    if args.reflexion:
        datasets["AI (Reflexion)"] = load_tokens_csv(args.reflexion)
    if args.energy:
        datasets["Energy Intent"] = load_tokens_csv(args.energy)
        
    if not datasets:
        print("Please provide at least one category of token CSV files.")
        sys.exit(1)
        
    plot_cognitive_overhead(datasets, args.out_dir)

if __name__ == "__main__":
    main()