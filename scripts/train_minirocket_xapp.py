#!/usr/bin/env python3
"""
Inherited from old src/demo/ain/pipeline.

Train MiniRocket model for deviation detection using xApp KPI data.

This script:
1. Reads KPI data from CSV files (gnb_kpis.csv, ue_kpis.csv)
2. Extracts time series for key metrics (delay_p95_ms, etc.)
3. Labels deviations based on thresholds or anomaly detection
4. Trains MiniRocket + classifier
5. Saves model for use in MinirocketAgent
"""

from __future__ import annotations
import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sktime.transformations.panel.rocket import MiniRocket
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import train_test_split
import sys

# Add parent directory to path
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent.parent.parent))


def load_xapp_kpis(gnb_csv: str = None, ue_csv: str = None, kpms_csv: str = None,
                   metric: str = "delay_p95_ms", cell_id: str = None) -> pd.Series:
    """
    Load KPI data from CSV files and extract time series for a specific metric.

    Args:
        gnb_csv: Path to gnb_kpis.csv (legacy, optional)
        ue_csv: Path to ue_kpis.csv (legacy, optional)
        kpms_csv: Path to kpms.csv (new unified format, optional)
        metric: Metric to extract (e.g., "DRB_PdcpSduDelayDl", "UE_DRB_PdcpSduDelayDl_UEID")
        cell_id: Optional cell_id filter (e.g., "CELL_1111") to get consistent data

    Returns:
        Series with metric values over time
    """
    # Try kpms.csv first (new unified format)
    if kpms_csv and Path(kpms_csv).exists():
        df = pd.read_csv(kpms_csv)

        # Filter by cell_id if specified
        if cell_id and "cell_id" in df.columns:
            df = df[df["cell_id"] == cell_id]
            print(f"Filtered to cell_id={cell_id}: {len(df)} rows")

        if metric in df.columns:
            # Use non-null values, sorted by timestamp
            if "timestamp" in df.columns:
                df = df.sort_values("timestamp")
            values = df[metric].dropna()
            if len(values) > 0:
                print(f"Loaded {len(values)} values for {metric} from {kpms_csv}")
                return values

    # Fallback to legacy format: Try gnb_kpis.csv
    if gnb_csv and Path(gnb_csv).exists():
        df_gnb = pd.read_csv(gnb_csv)
        if metric in df_gnb.columns:
            values = df_gnb[metric].dropna()
            if len(values) > 0:
                print(f"Loaded {len(values)} values for {metric} from {gnb_csv}")
                return values

    # Fallback to legacy format: Try ue_kpis.csv
    if ue_csv and Path(ue_csv).exists():
        df_ue = pd.read_csv(ue_csv)
        if metric in df_ue.columns:
            values = df_ue[metric].dropna()
            if len(values) > 0:
                print(f"Loaded {len(values)} values for {metric} from {ue_csv}")
                return values

    raise ValueError(f"Metric {metric} not found in CSV files")


def label_deviations(series: pd.Series, method: str = "threshold",
                     threshold: float = None, percentile: float = 90.0) -> pd.Series:
    """
    Label deviations in time series.

    Args:
        series: Time series values
        method: "threshold" (absolute), "percentile" (relative), or "statistical" (z-score)
        threshold: Absolute threshold for "threshold" method
        percentile: Percentile threshold for "percentile" method

    Returns:
        Series with labels (0=normal, 1=deviation)
    """
    labels = pd.Series(0, index=series.index, dtype=int)

    if method == "threshold":
        if threshold is None:
            # Default: use 75th percentile as threshold
            threshold = series.quantile(0.75)
        labels[series > threshold] = 1
        print(f"Threshold method: threshold={threshold:.2f}, deviations={labels.sum()}/{len(labels)}")

    elif method == "percentile":
        threshold = series.quantile(percentile / 100.0)
        labels[series > threshold] = 1
        print(f"Percentile method: {percentile}th percentile={threshold:.2f}, deviations={labels.sum()}/{len(labels)}")

    elif method == "statistical":
        # Z-score based: deviation if > mean + 2*std
        mean = series.mean()
        std = series.std()
        threshold = mean + 2 * std
        labels[series > threshold] = 1
        print(f"Statistical method: mean={mean:.2f}, std={std:.2f}, threshold={threshold:.2f}, deviations={labels.sum()}/{len(labels)}")

    return labels


def to_windows(x: pd.Series, y: pd.Series, win: int = 128, step: int = 32, deviation_ratio: float = 0.3):
    """
    Convert time series to sliding windows.

    Args:
        x: Time series values
        y: Labels (0=normal, 1=deviation)
        win: Window size
        step: Step size for sliding window
        deviation_ratio: Minimum ratio of deviation points in window to label as deviation (default: 0.3 = 30%)
                         If 0.0, uses "any point" logic (original behavior)

    Returns:
        X: DataFrame with windows (sktime format)
        Y: Array with window labels (1 if window has enough deviation points)
        actual_window_size: The window size actually used
    """
    # Auto-adjust window size if data is too short
    actual_window_size = win
    if len(x) < win:
        # Reduce window size to fit available data (use at least 50% of data)
        old_win = win
        win = max(16, min(win, len(x) // 2))  # At least 16, but not more than half the data
        step = max(4, win // 4)  # Adjust step proportionally
        actual_window_size = win
        print(f"   WARNING: Data length ({len(x)}) < window size ({old_win}). Adjusted to window={win}, step={step}")

    X, Y = [], []
    for s in range(0, len(x) - win + 1, step):
        e = s + win
        window_x = x.iloc[s:e]
        window_y = y.iloc[s:e]
        X.append(pd.Series(window_x.values))

        # Window labeling logic
        if deviation_ratio > 0.0:
            # Label as deviation if at least deviation_ratio% of points are deviations
            deviation_count = window_y.sum()
            deviation_percentage = deviation_count / len(window_y)
            Y.append(1 if deviation_percentage >= deviation_ratio else 0)
        else:
            # Original logic: label as deviation if ANY point is a deviation
            Y.append(int(window_y.max()))

    return pd.DataFrame({"signal": X}), np.array(Y), actual_window_size


def train_single_model(series: pd.Series, metric: str, args, output_path: str):
    """Train a single MiniRocket model on a metric."""
    print(f"\n{'='*60}")
    print(f"Training model for metric: {metric}")
    print(f"{'='*60}")

    # Auto-adjust step size if not specified
    step_size = args.step_size if args.step_size is not None else max(4, args.window_size // 4)

    # Label deviations
    print(f"\n2. Labeling deviations (method: {args.method})")
    labels = label_deviations(series, method=args.method,
                             threshold=args.threshold,
                             percentile=args.percentile)

    if labels.sum() == 0:
        print("   WARNING: No deviations found! Model may not train well.")
        print("   Try adjusting --percentile or --threshold")
        return None

    # Create windows (will auto-adjust window size if needed)
    print(f"\n3. Creating sliding windows (window={args.window_size}, step={step_size})")
    deviation_ratio = getattr(args, 'deviation_ratio', 0.0)  # Default: any point = deviation (original behavior)
    X, Y, actual_window_size = to_windows(series, labels, win=args.window_size, step=step_size, deviation_ratio=deviation_ratio)
    actual_step_size = step_size

    print(f"   Created {len(X)} windows")
    print(f"   Actual window size used: {actual_window_size} (requested: {args.window_size})")
    normal_count = (Y == 0).sum()
    deviation_count = (Y == 1).sum()
    print(f"   Normal windows: {normal_count}, Deviation windows: {deviation_count}")

    if deviation_count == 0:
        print("   ERROR: No deviation windows found! Cannot train model.")
        print("   Suggestions:")
        print("     - Lower --deviation-ratio (e.g., 0.1 or 0.15)")
        print("     - Use lower --percentile (e.g., 85.0 or 90.0)")
        print("     - Collect more data with actual variance")
        return None

    if normal_count == 0:
        print("   WARNING: No normal windows found! Model may not generalize well.")
        print("   Suggestions:")
        print("     - Increase --deviation-ratio (e.g., 0.3 or 0.5)")
        print("     - Use higher --percentile (e.g., 95.0 or 99.0)")
        print("     - Collect more data with actual variance")

    # Warn if dataset is very unbalanced
    total_windows = len(Y)
    if total_windows > 0:
        deviation_ratio_actual = deviation_count / total_windows
        if deviation_ratio_actual < 0.1 or deviation_ratio_actual > 0.9:
            print(f"   WARNING: Very unbalanced dataset ({deviation_ratio_actual*100:.1f}% deviations). Model may not train well.")

    # Train-test split
    print(f"\n4. Splitting data (test_size={args.test_size})")
    # Check if we can use stratified split (need at least 2 samples per class)
    normal_count = (Y == 0).sum()
    deviation_count = (Y == 1).sum()
    can_stratify = normal_count >= 2 and deviation_count >= 2

    if can_stratify:
        Xtr, Xte, ytr, yte = train_test_split(
            X, Y, test_size=args.test_size, stratify=Y, random_state=42
        )
    else:
        print(f"   WARNING: Cannot use stratified split (normal={normal_count}, deviation={deviation_count}). Using random split.")
        Xtr, Xte, ytr, yte = train_test_split(
            X, Y, test_size=args.test_size, random_state=42
        )
    print(f"   Train: {len(Xtr)} windows ({ytr.sum()} deviations)")
    print(f"   Test: {len(Xte)} windows ({yte.sum()} deviations)")

    # Train MiniRocket
    print(f"\n5. Training MiniRocket transformer...")
    mr = MiniRocket()
    mr.fit(Xtr)
    Xtr_f = mr.transform(Xtr)
    Xte_f = mr.transform(Xte)
    print(f"   Transformed features: {Xtr_f.shape[1]} dimensions")

    # Train classifier
    print(f"\n6. Training Ridge Classifier...")
    clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 13))
    clf.fit(Xtr_f, ytr)

    # Evaluate
    train_acc = clf.score(Xtr_f, ytr)
    test_acc = clf.score(Xte_f, yte)
    print(f"   Train accuracy: {train_acc:.3f}")
    print(f"   Test accuracy: {test_acc:.3f}")

    # Save model
    print(f"\n7. Saving model to {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "mr": mr,
        "clf": clf,
        "meta": {
            "metric": metric,
            "window_size": actual_window_size,  # Use actual window size
            "requested_window_size": args.window_size,
            "step_size": actual_step_size,
            "method": args.method,
            "train_acc": float(train_acc),
            "test_acc": float(test_acc),
            "n_windows": len(X),
            "n_deviations": int(Y.sum()),
            "n_data_points": len(series),
        }
    }, output_path)
    print(f"   ✓ Model saved successfully!")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Train MiniRocket model for xApp KPI deviation detection")
    parser.add_argument("--kpms-csv", type=str, default="kpms.csv",
                       help="Path to kpms.csv (unified format, preferred)")
    parser.add_argument("--gnb-csv", type=str, default=None, help="Path to gnb_kpis.csv (legacy format)")
    parser.add_argument("--ue-csv", type=str, default=None, help="Path to ue_kpis.csv (legacy format)")
    parser.add_argument("--metric", type=str, default=None,
                       help="Single metric to train on (e.g., DRB_PdcpSduDelayDl). If not specified, trains on both gNB and UE metrics.")
    parser.add_argument("--train-both", action="store_true",
                       help="Train separate models for gNB and UE level metrics (ignores --metric)")
    parser.add_argument("--gnb-metric", type=str, default="DRB_PdcpSduDelayDl",
                       help="gNB-level metric (used with --train-both)")
    parser.add_argument("--ue-metric", type=str, default="UE_DRB_PdcpSduDelayDl_UEID",
                       help="UE-level metric (used with --train-both)")
    parser.add_argument("--cell-id", type=str, default=None,
                       help="Filter by cell_id (e.g., CELL_1111). If not specified, uses ALL entries.")
    parser.add_argument("--window-size", type=int, default=128,
                       help="Window size for MiniRocket (will auto-adjust if data is too short)")
    parser.add_argument("--step-size", type=int, default=None,
                       help="Step size for sliding windows (defaults to window_size/4 if not specified)")
    parser.add_argument("--method", type=str, default="percentile",
                       choices=["threshold", "percentile", "statistical"],
                       help="Method for labeling deviations")
    parser.add_argument("--threshold", type=float, default=None,
                       help="Threshold value (for threshold method)")
    parser.add_argument("--percentile", type=float, default=85.0,
                       help="Percentile threshold (for percentile method)")
    parser.add_argument("--deviation-ratio", type=float, default=0.0,
                       help="Minimum ratio of deviation points in window to label as deviation (0.0-1.0). Default 0.0 = any point = deviation. Use 0.3-0.5 for more balanced labels.")
    parser.add_argument("--output", type=str, default="models/minirocket_xapp.joblib",
                       help="Output path for trained model (or prefix if training multiple)")
    parser.add_argument("--test-size", type=float, default=0.3, help="Test set size")

    args = parser.parse_args()

    print("=" * 60)
    print("Training MiniRocket Model for xApp KPI Deviation Detection")
    print("=" * 60)

    # Determine training mode
    if args.train_both:
        # Train separate models for gNB and UE
        metrics_to_train = [
            (args.gnb_metric, "gNB"),
            (args.ue_metric, "UE")
        ]
    elif args.metric:
        # Train single model
        metrics_to_train = [(args.metric, "single")]
    else:
        # Default: train both
        metrics_to_train = [
            ("DRB_PdcpSduDelayDl", "gNB"),
            ("UE_DRB_PdcpSduDelayDl_UEID", "UE")
        ]
        print("\nNo --metric specified, training on both gNB and UE metrics by default")
        print("Use --metric <name> to train on a single metric, or --train-both to explicitly train both")

    # Show what we're training on
    cell_filter_info = f" (filtered to {args.cell_id})" if args.cell_id else " (ALL entries - no filter)"
    print(f"\nTraining configuration:")
    print(f"  CSV file: {args.kpms_csv}")
    print(f"  Cell filter: {cell_filter_info}")
    print(f"  Metrics: {[m[0] for m in metrics_to_train]}")

    trained_models = []

    for metric, level in metrics_to_train:
        # Load KPI data
        print(f"\n{'='*60}")
        print(f"1. Loading KPI data for {level}-level metric: {metric}")
        try:
            series = load_xapp_kpis(
                gnb_csv=args.gnb_csv,
                ue_csv=args.ue_csv,
                kpms_csv=args.kpms_csv,
                metric=metric,
                cell_id=args.cell_id
            )
            print(f"   Loaded {len(series)} data points")
            print(f"   Range: [{series.min():.2f}, {series.max():.2f}]")
            print(f"   Mean: {series.mean():.2f}, Std: {series.std():.2f}")
        except Exception as e:
            print(f"   ERROR: {e}")
            # Show available columns from the CSV file that exists
            csv_to_check = args.kpms_csv or args.gnb_csv or "kpms.csv"
            if Path(csv_to_check).exists():
                df = pd.read_csv(csv_to_check)
                print(f"   Available columns in {csv_to_check}:")
                print(f"     {list(df.columns)}")
            continue

        # Determine output path
        if len(metrics_to_train) > 1:
            # Multiple models: add suffix
            output_path = args.output.replace(".joblib", f"_{level.lower()}.joblib")
        else:
            output_path = args.output

        # Train model
        model_path = train_single_model(series, metric, args, output_path)
        if model_path:
            trained_models.append((level, metric, model_path))

    # Summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    if trained_models:
        print(f"\n✓ Successfully trained {len(trained_models)} model(s):")
        for level, metric, path in trained_models:
            print(f"  - {level}-level ({metric}): {path}")
        print(f"\nTo use these models:")
        if len(trained_models) == 1:
            print(f"  --minirocket-model {trained_models[0][2]}")
        else:
            print(f"  Note: You can use either model depending on which metric you want to monitor")
            print(f"  For gNB-level: --minirocket-model {trained_models[0][2]}")
            print(f"  For UE-level: --minirocket-model {trained_models[1][2]}")
    else:
        print("\n✗ No models were successfully trained. Check errors above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
