# observer_rl.py - Enhanced with Contextual Bandit capabilities
from __future__ import annotations
import json
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from core.common.log_config import should_log, LOG_REWARD, LOG_OBSERVER, LOG_INTENT

logger = logging.getLogger(__name__)

# Import contextual bandit components
try:
    from core.control_layer.RL_engines.bandit.context_extractor import ContextExtractor, NetworkContext
    from core.control_layer.RL_engines.bandit.reward_calculator import SLORewardCalculator
    BANDIT_AVAILABLE = True
except ImportError:
    # Fallback if bandit components not yet created
    BANDIT_AVAILABLE = False
    print("[Warning] Contextual bandit components not found. Using basic mode.")


DEFAULT_FEATURES = [
    "DRB_PdcpSduDelayDl", "RRU_PrbUsedDl", "DRB_MeanActiveUeDl",
    "TB_TotNbrDlInitial_Qpsk", "TB_TotNbrDlInitial_16Qam", "TB_TotNbrDlInitial_64Qam",
    "UE_DRB_PdcpSduDelayDl_UEID", "UE_DRB_UEThpDl_UEID",
    "UE_RRU_PrbUsedDl_UEID", "UE_DRB_EstabSucc_5QI_UEID",
]

@dataclass
class Intent:
    type: str                 # e.g., "REDUCE_LATENCY", "INCREASE_THROUGHPUT"
    metric: str               # e.g., "delay_p95_ms"
    target: float             # e.g., 40.0
    direction: str = "lower_better"  # "lower_better" | "higher_better"
    action_cost: float = 0.01        # penalty per action in playbook
    reward_clip: float = 20.0         # clip absolute reward


class RLObserver:
    def __init__(
        self,
        predictor,
        intent: Intent,
        kpi_file: str = "fake_kpi_stream.json",
        window: int = 12,
        features: Optional[List[str]] = None,
        use_internal_encoder: bool = False,
        device: Optional[torch.device] = None,
        enable_contextual_bandit: bool = True,
        min_feature_completeness: float = 0.5,  # Minimum fraction of features that must be present (0.0-1.0)
    ):
        self.predictor = predictor
        self.intent = intent
        self.kpi_file = Path(kpi_file)
        self.window = window
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.buf: Deque[np.ndarray] = deque(maxlen=window)
        self.last_kpi_raw: Optional[Dict] = None
        self.last_state_win: Optional[np.ndarray] = None  # [W, F]
        self.use_internal_encoder = use_internal_encoder
        self.min_feature_completeness = min_feature_completeness
        
        # Track last known values for each metric (to handle fragmented KPIs)
        self.last_metric_values: Dict[str, float] = {}

        # Track last read file state to avoid duplicate KPI reads
        self._last_file_mtime: float = 0.0
        self._last_read_timestamp: Optional[str] = None
        self._last_read_kpi_hash: Optional[str] = None

        # State that was previously lazy-initialized via hasattr in step()
        self.known_kpi_state: Optional[Dict] = None
        self.steps_since_action: int = 0
        self.last_action_time: float = 0.0
        self.active_otm: Optional[Dict] = None

        # NEW: Contextual bandit components
        self.enable_contextual_bandit = enable_contextual_bandit and BANDIT_AVAILABLE
        if self.enable_contextual_bandit:
            self.context_extractor = ContextExtractor(window_size=window)
            self.slo_reward_calculator = SLORewardCalculator(
                reward_clip=intent.reward_clip
            )
            self.current_context: Optional[NetworkContext] = None
            self.context_history: List[NetworkContext] = []
            print("[Observer] Contextual bandit mode enabled")
        else:
            print("[Observer] Basic mode (no contextual bandit)")

        # Select features - USE ACTUAL CSV COLUMN NAMES
        # gNB LEVEL (no UE prefix): DRB_MeanActiveUeDl, DRB_PdcpSduDelayDl, RRU_PrbUsedDl, 
        #                            TB_TotNbrDlInitial_16Qam, TB_TotNbrDlInitial_64Qam, TB_TotNbrDlInitial_Qpsk
        # UE LEVEL (UE_ prefix): UE_DRB_PdcpSduDelayDl_UEID, UE_DRB_UEThpDl_UEID, UE_RRU_PrbUsedDl_UEID, etc.
        # Note: UE metrics are aggregated into cell-level metrics (sum/max) for global context
        self.features = features or list(DEFAULT_FEATURES)

        # Simple fixed scalers (adjust as you like) - USE ACTUAL CSV COLUMN NAMES
        self.scalers = {
            "DRB_PdcpSduDelayDl": 100.0,              # scale to 0-1 range (100ms max)
            "RRU_PrbUsedDl": 100.0,                  # scale to 0-1 range (100 PRBs max)
            "DRB_MeanActiveUeDl": 20.0,              # scale to 0-1 range (20 UEs max)
            "TB_TotNbrDlInitial_Qpsk": 1000.0,       # scale to 0-1 range
            "TB_TotNbrDlInitial_16Qam": 1000.0,     # scale to 0-1 range
            "TB_TotNbrDlInitial_64Qam": 1000.0,     # scale to 0-1 range
            "UE_DRB_PdcpSduDelayDl_UEID": 100.0,    # scale to 0-1 range (100ms max)
            "UE_DRB_UEThpDl_UEID": 100e6,           # scale to 0-1 range (100 Mbps = 100e6 bps)
            "UE_RRU_PrbUsedDl_UEID": 50.0,          # scale to 0-1 range (50 PRBs max)
            "UE_DRB_EstabSucc_5QI_UEID": 100.0,     # scale to 0-1 range
        }

    # ---------- KPI reading & preprocessing ----------
    def _read_latest_kpi(self) -> Optional[Dict]:
        """
        Read the latest KPI from file, but only if it's different from the last read.
        This prevents returning the same KPI multiple times, which would cause s1 = s2.
        """
        if not self.kpi_file.exists():
            return None
        
        # Check file modification time to detect changes
        try:
            current_mtime = self.kpi_file.stat().st_mtime
        except (OSError, FileNotFoundError):
            return None
        
        # If file hasn't changed since last read, return None (skip this step)
        # This prevents reading the same KPI multiple times
        if current_mtime <= self._last_file_mtime:
            return None
        
        # Read the file
        try:
            with open(self.kpi_file, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
        
        stream = data.get("kpi_stream", [])
        if not stream:
            return None
        
        kpi = stream[-1]
        
        # Additional check: compare KPI content hash to detect duplicates
        # This is more reliable than timestamp because fragments share the same timestamp
        # but have different metric values
        import hashlib
        import json as json_module
        
        # Create a hash of the KPI's CellMetrics (the actual data we care about)
        # This detects if metric values have changed, even if timestamp is the same
        cell_metrics = kpi.get("CellMetrics", {})
        # Sort keys for consistent hashing
        metrics_str = json_module.dumps(cell_metrics, sort_keys=True)
        kpi_hash = hashlib.md5(metrics_str.encode()).hexdigest()
        
        # If the hash matches the last read, it's the same KPI content - skip
        if self._last_read_kpi_hash == kpi_hash:
            # Same KPI content as last time - skip to avoid duplicate processing
            if should_log(LOG_OBSERVER):
                logger.debug(f"[OBSERVER] Skipping duplicate KPI (hash={kpi_hash[:8]}...), file mtime changed but content unchanged")
            return None
        
        # Log when we detect a new KPI (different hash)
        if should_log(LOG_OBSERVER) and self._last_read_kpi_hash is not None:
            delay_val = cell_metrics.get('DRB_PdcpSduDelayDl', 'N/A')
            logger.debug(f"[OBSERVER] New KPI detected (hash={kpi_hash[:8]}...), delay={delay_val}, file mtime={current_mtime:.6f}")
        
        # Also check timestamp as a secondary check
        kpi_timestamp = (
            kpi.get("timestamp") or 
            kpi.get("Header", {}).get("window_start") or
            kpi.get("Header", {}).get("window_end") or
            str(kpi.get("Header", {}).get("sequence_number", ""))
        )
        
        # Update tracking variables
        self._last_file_mtime = current_mtime
        self._last_read_timestamp = kpi_timestamp
        self._last_read_kpi_hash = kpi_hash
        
        # Return a deep copy to prevent modifications from affecting stored references
        from copy import deepcopy
        return deepcopy(kpi)

    def _extract_features_row(self, kpi: Dict) -> Tuple[np.ndarray, float]:
        """
        Extract features from KPI, using NaN for missing values.
        
        Returns:
            (feature_vector, completeness): feature array and fraction of features present (0.0-1.0)
        """
        cell = kpi.get("CellMetrics", {})
        
        # build feature vector in configured order - USE ACTUAL METRIC NAMES
        vals: List[float] = []
        present_count = 0
        
        for name in self.features:
            # Direct feature: check if present in cell metrics
            if name in cell and cell[name] is not None:
                try:
                    v = float(cell[name])
                    present_count += 1
                except (ValueError, TypeError):
                    v = np.nan
            else:
                v = np.nan
            
            # scale (only if value is not NaN)
            if not np.isnan(v):
                scale = self.scalers.get(name, 1.0)
                v = v / scale
            
            vals.append(v)
        
        completeness = present_count / len(self.features) if self.features else 0.0
        return np.asarray(vals, dtype=np.float32), completeness

    def _update_window(self, row: np.ndarray) -> np.ndarray:
        """
        Update window buffer and forward-fill NaN values from previous entries.
        This allows the system to work with partial data while waiting for complete features.
        """
        from copy import deepcopy
        self.buf.append(row)
        
        # Forward-fill NaN values: use the last valid value for each feature
        filled_buf = []
        last_valid = None
        for entry in self.buf:
            if last_valid is None:
                # First entry - use as-is (may have NaNs)
                filled_entry = entry.copy()
            else:
                # Fill NaNs with last valid values
                filled_entry = entry.copy()
                nan_mask = np.isnan(filled_entry)
                filled_entry[nan_mask] = last_valid[nan_mask]
            
            # Update last_valid with non-NaN values from this entry
            if last_valid is None:
                last_valid = filled_entry.copy()
            else:
                valid_mask = ~np.isnan(filled_entry)
                last_valid[valid_mask] = filled_entry[valid_mask]
            
            filled_buf.append(filled_entry)
        
        if len(filled_buf) < self.window:
            # Left-pad with the first filled entry until window is full
            first = deepcopy(filled_buf[0]) if filled_buf else row
            padded = [first] * (self.window - len(filled_buf)) + filled_buf
            win = np.stack(padded, axis=0)
        else:
            win = np.stack(filled_buf, axis=0)
        
        # Final cleanup: ensure no NaN values remain in the window
        # For each feature column, if all values are NaN, fill with 0
        # Otherwise, forward-fill then backward-fill
        for f in range(win.shape[1]):
            col = win[:, f]
            if np.isnan(col).any():
                # Check if any values are valid
                valid_mask = ~np.isnan(col)
                if valid_mask.any():
                    # Forward fill from first valid
                    first_valid_idx = np.where(valid_mask)[0][0]
                    first_valid_val = col[first_valid_idx]
                    col[:first_valid_idx] = first_valid_val
                    # Forward fill remaining
                    for i in range(1, len(col)):
                        if np.isnan(col[i]):
                            col[i] = col[i-1]
                else:
                    # All NaN - fill with zero
                    col[:] = 0.0
        
        # Final safety check: replace any remaining NaN/inf with zero
        win = np.nan_to_num(win, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.last_state_win = win
        return win  # [W, F] - guaranteed to have no NaN values

    # ---------- NEW: Context extraction methods ----------
    def _extract_context(self, kpi: Dict) -> Optional[NetworkContext]:
        """Extract rich network context from KPI data."""
        if not self.enable_contextual_bandit:
            return None
        
        try:
            context = self.context_extractor.extract_from_kpi(kpi)
            return context
        except Exception as e:
            print(f"[Observer] Context extraction error: {e}")
            return None

    def _update_context_history(self, context: NetworkContext):
        """Maintain context history for temporal analysis."""
        if not self.enable_contextual_bandit:
            return
        
        self.context_history.append(context)
        if len(self.context_history) > self.window:
            self.context_history.pop(0)

    def _classify_network_situation(self, context: NetworkContext) -> str:
        """Classify current network situation for contextual decisions."""
        if not context:
            return "unknown"
        
        # Multi-criteria situation classification
        situations = []
        
        if context.latency_ms > 70:
            situations.append("high_latency")
        if context.throughput_dl_mbps < 50:
            situations.append("low_throughput")
        if context.bler > 0.05:
            situations.append("high_error_rate")
        if context.network_load > 0.8:
            situations.append("high_load")
        if context.quality_index < 0.5:
            situations.append("poor_quality")
        
        if not situations:
            return "normal"
        elif len(situations) == 1:
            return situations[0]
        else:
            # Multiple issues - return most critical
            if "high_latency" in situations:
                return "high_latency"
            elif "high_error_rate" in situations:
                return "high_error_rate"
            else:
                return situations[0]

    def _calculate_context_bonus(self, playbook: Any, context: NetworkContext) -> float:
        """Calculate bonus reward for contextually appropriate actions."""
        if not playbook or not context or not self.enable_contextual_bandit:
            return 0.0
        
        bonus = 0.0
        situation = self._classify_network_situation(context)
        actions = getattr(playbook, 'actions', [])
        
        for action in actions:
            action_bonus = 0.0
            
            # Get action details
            action_type = getattr(action, 'type', '')
            action_params = getattr(action, 'params', {})
            
            # Situation-specific bonuses (only for action types in the active action space)
            if situation == "high_latency":
                if action_type == "MCS_CAP":
                    mcs_cap = action_params.get('dl_mcs_max', 15)
                    if mcs_cap >= 20:
                        action_bonus = 0.10  # Higher MCS might help latency
                elif action_type == "PRB_WEIGHT":
                    weight = action_params.get('weight', 1.0)
                    if weight > 1.0:
                        action_bonus = 0.08  # More PRBs might help latency
                elif action_type == "TX_POWER":
                    tx = action_params.get('txPowerDbm', 30)
                    if tx >= 45:
                        action_bonus = 0.12  # Higher power improves signal for latency

            elif situation == "low_throughput":
                if action_type == "PRB_WEIGHT":
                    weight = action_params.get('weight', 1.0)
                    if weight > 1.0:
                        action_bonus = 0.20  # Excellent for throughput
                elif action_type == "MCS_CAP":
                    mcs_cap = action_params.get('dl_mcs_max', 15)
                    if mcs_cap >= 18:
                        action_bonus = 0.10  # Higher MCS for throughput
                elif action_type == "TX_POWER":
                    tx = action_params.get('txPowerDbm', 30)
                    if tx >= 45:
                        action_bonus = 0.10  # Higher power for throughput

            elif situation == "high_error_rate":
                if action_type == "MCS_CAP":
                    mcs_cap = action_params.get('dl_mcs_max', 15)
                    if mcs_cap <= 16:
                        action_bonus = 0.18  # Lower MCS for reliability
                elif action_type == "PRB_WEIGHT":
                    weight = action_params.get('weight', 1.0)
                    if weight > 1.0:
                        action_bonus = 0.10  # More resources for reliability

            elif situation == "poor_quality":
                if action_type == "MCS_CAP":
                    mcs_cap = action_params.get('dl_mcs_max', 15)
                    if mcs_cap <= 18:
                        action_bonus = 0.12  # Conservative MCS
                elif action_type == "TX_POWER":
                    tx = action_params.get('txPowerDbm', 30)
                    if tx >= 45:
                        action_bonus = 0.08  # Higher power for quality
            
            bonus += action_bonus
        
        # Cap total bonus
        return min(bonus, 0.5)  # Maximum 0.5 bonus per playbook

    def _extract_metrics_dict(self, kpi: Dict) -> Dict[str, float]:
        """Extract metrics as dictionary for reward calculation, using actual CSV column names."""
        cell = kpi.get("CellMetrics", {})
        def safe_float(val, default=np.nan):
            if val is None:
                return default
            try:
                return float(val)
            except (ValueError, TypeError):
                return default
        
        # Extract metrics using ACTUAL CSV column names (as they appear in CellMetrics)
        # These are the names that the KPI adapter puts into CellMetrics
        metrics = {}
        
        # gNB-level metrics (actual CSV column names)
        if "DRB_PdcpSduDelayDl" in cell:
            metrics["DRB_PdcpSduDelayDl"] = safe_float(cell["DRB_PdcpSduDelayDl"])
        if "RRU_PrbUsedDl" in cell:
            metrics["RRU_PrbUsedDl"] = safe_float(cell["RRU_PrbUsedDl"])
        if "DRB_MeanActiveUeDl" in cell:
            metrics["DRB_MeanActiveUeDl"] = safe_float(cell["DRB_MeanActiveUeDl"])
        if "TB_TotNbrDlInitial_Qpsk" in cell:
            metrics["TB_TotNbrDlInitial_Qpsk"] = safe_float(cell["TB_TotNbrDlInitial_Qpsk"])
        if "TB_TotNbrDlInitial_16Qam" in cell:
            metrics["TB_TotNbrDlInitial_16Qam"] = safe_float(cell["TB_TotNbrDlInitial_16Qam"])
        if "TB_TotNbrDlInitial_64Qam" in cell:
            metrics["TB_TotNbrDlInitial_64Qam"] = safe_float(cell["TB_TotNbrDlInitial_64Qam"])
        
        # UE-level metrics (aggregated into cell-level, actual CSV column names)
        if "UE_DRB_PdcpSduDelayDl_UEID" in cell:
            metrics["UE_DRB_PdcpSduDelayDl_UEID"] = safe_float(cell["UE_DRB_PdcpSduDelayDl_UEID"])
        if "UE_DRB_UEThpDl_UEID" in cell:
            metrics["UE_DRB_UEThpDl_UEID"] = safe_float(cell["UE_DRB_UEThpDl_UEID"])
        if "UE_DRB_BlerDl_UEID" in cell:
            metrics["UE_DRB_BlerDl_UEID"] = safe_float(cell["UE_DRB_BlerDl_UEID"])
        if "UE_RRU_PrbUsedDl_UEID" in cell:
            metrics["UE_RRU_PrbUsedDl_UEID"] = safe_float(cell["UE_RRU_PrbUsedDl_UEID"])
        if "UE_DRB_EstabSucc_5QI_UEID" in cell:
            metrics["UE_DRB_EstabSucc_5QI_UEID"] = safe_float(cell["UE_DRB_EstabSucc_5QI_UEID"])
        
        # Legacy/fallback names (for backward compatibility, but prefer CSV names)
        if "delay_p95_ms" in cell and "DRB_PdcpSduDelayDl" not in metrics:
            metrics["delay_p95_ms"] = safe_float(cell["delay_p95_ms"])
        if "thr_dl_bps" in cell and "UE_DRB_UEThpDl_UEID" not in metrics:
            metrics["thr_dl_bps"] = safe_float(cell["thr_dl_bps"])
        if "thr_ul_bps" in cell:
            metrics["thr_ul_bps"] = safe_float(cell["thr_ul_bps"])
        if "bler_dl" in cell and "UE_DRB_BlerDl_UEID" not in metrics:
            metrics["bler_dl"] = safe_float(cell["bler_dl"])
        if "bler_ul" in cell:
            metrics["bler_ul"] = safe_float(cell["bler_ul"])
        if "cqi_avg" in cell:
            metrics["cqi_avg"] = safe_float(cell["cqi_avg"])
        if "mcs_dl_avg" in cell:
            metrics["mcs_dl_avg"] = safe_float(cell["mcs_dl_avg"])
        if "active_ue_count" in cell and "DRB_MeanActiveUeDl" not in metrics:
            metrics["active_ue_count"] = safe_float(cell["active_ue_count"])
        if "prb_used_dl" in cell and "RRU_PrbUsedDl" not in metrics:
            metrics["prb_used_dl"] = safe_float(cell["prb_used_dl"])
        
        return metrics

    # ---------- Enhanced Reward Calculation ----------
    def _compute_reward(self, prev_kpi: Dict, curr_kpi: Dict, actions_len: int) -> float:
        """Original reward computation (fallback)."""
        metric = self.intent.metric
        target = self.intent.target
        direction = self.intent.direction

        prev_cell = prev_kpi.get("CellMetrics", {})
        curr_cell = curr_kpi.get("CellMetrics", {})
        
        # Try to get metric value with fallback to alternative names
        prev_val = prev_cell.get(metric)
        curr_val = curr_cell.get(metric)
        
        # Fallback: try alternative metric names (for compatibility)
        if prev_val is None or curr_val is None:
            # Map common metric names to alternatives
            metric_alternatives = {
                "DRB_PdcpSduDelayDl": ["delay_p95_ms", "UE_PDCP_Delay_DL_ms"],
                "delay_p95_ms": ["DRB_PdcpSduDelayDl", "UE_PDCP_Delay_DL_ms"],
                "UE_DRB_PdcpSduDelayDl_UEID": ["UE_PDCP_Delay_DL_ms"],
            }
            
            alternatives = metric_alternatives.get(metric, [])
            for alt in alternatives:
                if prev_val is None:
                    prev_val = prev_cell.get(alt)
                if curr_val is None:
                    curr_val = curr_cell.get(alt)
                if prev_val is not None and curr_val is not None:
                    if should_log(LOG_REWARD):
                        logger.debug(f"[REWARD] Using alternative metric name: {alt} (requested: {metric})")
                    break
        
        # If either value is still missing, log diagnostic info and return neutral reward
        if prev_val is None or curr_val is None:
            if should_log(LOG_REWARD):
                available_metrics = sorted(set(list(prev_cell.keys()) + list(curr_cell.keys())))
                logger.warning(f"[REWARD] Metric '{metric}' not found in CellMetrics. Available metrics: {available_metrics[:10]}...")
                logger.warning(f"[REWARD] prev_kpi CellMetrics keys: {list(prev_cell.keys())[:10]}")
                logger.warning(f"[REWARD] curr_kpi CellMetrics keys: {list(curr_cell.keys())[:10]}")
            return 0.0
        
        # Check for NaN values
        if np.isnan(prev_val) or np.isnan(curr_val):
            if should_log(LOG_REWARD):
                logger.warning(f"[REWARD] Metric '{metric}' has NaN values: prev={prev_val}, curr={curr_val}")
            return 0.0
        
        prev = float(prev_val)
        curr = float(curr_val)
        
        # Ensure values are not normalized (they should be raw values)
        # Normalization only happens in feature extraction, not in reward calculation

        if direction == "lower_better":
            delta_raw = (prev - curr)
            delta_rel = delta_raw / max(abs(prev), 1e-6)
            violation = 1.0 if curr > target else 0.0
        else:  # higher_better
            delta_raw = (curr - prev)
            delta_rel = delta_raw / max(abs(prev), 1e-6)
            violation = 1.0 if curr < target else 0.0

        r = delta_rel
        r -= self.intent.action_cost * float(actions_len)
        r -= violation  # penalty if target not met
        clip = self.intent.reward_clip
        reward = float(np.clip(r, -clip, clip))
        
        if should_log(LOG_REWARD):
            logger.info(f"[REWARD] metric={metric}, prev={prev:.4f}, curr={curr:.4f}, target={target:.4f}, "
                       f"delta_rel={delta_rel:.4f}, violation={violation}, actions={actions_len}, reward={reward:.4f}")
            # Additional diagnostic: show if values are changing
            if abs(prev - curr) < 1e-6:
                logger.warning(f"[REWARD] WARNING: Metric value not changing! prev={prev:.6f}, curr={curr:.6f} (diff={abs(prev-curr):.9f})")
        
        return reward

    def _compute_enhanced_reward(self, prev_kpi: Dict, curr_kpi: Dict,
                               last_playbook: Any) -> float:
        """Compute reward using the active OTM from the LLM, with single-metric fallback."""
        actions_len = len(getattr(last_playbook, "actions", []))
        
        # Extract metrics for reward calculation
        prev_metrics = self._extract_metrics_dict(prev_kpi)
        curr_metrics = self._extract_metrics_dict(curr_kpi)
        
        saved_last_metric_values = self.last_metric_values.copy()
        
        # Merge metrics from both KPIs to handle fragmented KPIs
        # Get all unique metrics from both KPIs
        all_metrics = set(list(prev_metrics.keys()) + list(curr_metrics.keys()))
        
        merged_prev_metrics = {}
        merged_curr_metrics = {}
        
        for metric in all_metrics:
            # For prev: use prev_kpi value if available, otherwise SAVED last known value
            # Use saved_last_metric_values (from before this step) to avoid using curr values
            if metric in prev_metrics and not np.isnan(prev_metrics[metric]):
                merged_prev_metrics[metric] = prev_metrics[metric]
            elif metric in saved_last_metric_values and not np.isnan(saved_last_metric_values[metric]):
                merged_prev_metrics[metric] = saved_last_metric_values[metric]
                if should_log(LOG_REWARD):
                    logger.debug(f"[REWARD] Using saved last known value for prev {metric}: {saved_last_metric_values[metric]:.4f} (not in prev_kpi)")
            
            # For curr: use curr_kpi value if available, otherwise SAVED last known value
            # Use saved_last_metric_values (from before this step) to avoid using prev values
            if metric in curr_metrics and not np.isnan(curr_metrics[metric]):
                merged_curr_metrics[metric] = curr_metrics[metric]
            elif metric in saved_last_metric_values and not np.isnan(saved_last_metric_values[metric]):
                merged_curr_metrics[metric] = saved_last_metric_values[metric]
                if should_log(LOG_REWARD):
                    logger.debug(f"[REWARD] Using saved last known value for curr {metric}: {saved_last_metric_values[metric]:.4f} (not in curr_kpi)")
        
        # update last_metric_values after merging (for next iteration)
        for metric, value in curr_metrics.items():
            if not np.isnan(value):
                self.last_metric_values[metric] = value
        
        # Also update from prev_metrics (in case curr is missing it)
        for metric, value in prev_metrics.items():
            if not np.isnan(value):
                self.last_metric_values[metric] = value
        
        # Use merged metrics for reward calculation
        prev_metrics = merged_prev_metrics
        curr_metrics = merged_curr_metrics
        
        if should_log(LOG_REWARD):
            logger.info(f"[REWARD] Computing reward: has_otm={hasattr(self, 'active_otm') and self.active_otm is not None}, actions_len={actions_len}")
            logger.info(f"[REWARD] Prev Metrics: {{k: round(v, 2) for k, v in prev_metrics.items() if not np.isnan(v)}}")
            logger.info(f"[REWARD] Curr Metrics: {{k: round(v, 2) for k, v in curr_metrics.items() if not np.isnan(v)}}")

        # Primary path: OTM from the LLM (minimal reward: improvement − violation²)
        active_otm = self.active_otm
        if active_otm:
            try:
                from core.common.types import map_otm_metric_name

                # Extract objective from OTM
                obj = active_otm.get("objective", {})
                objective_metric = map_otm_metric_name(obj.get("kpi", ""))
                maximize = obj.get("maximize", True)

                # Extract constraints from OTM (maps 1:1 to the OTM JSON)
                constraints = []
                for c in active_otm.get("constraints", []):
                    metric = map_otm_metric_name(c.get("kpi", ""))
                    threshold = float(c.get("threshold", 0))
                    op = c.get("operator", "le")

                    # Unit conversion for the backend
                    if c.get("unit") == "Mbps" and "thp" in metric.lower():
                        threshold *= 1e6

                    constraints.append({
                        "metric": metric,
                        "operator": op,
                        "threshold": threshold,
                    })

                reward = self.slo_reward_calculator.calculate_reward(
                    prev_metrics=prev_metrics,
                    curr_metrics=curr_metrics,
                    objective_metric=objective_metric,
                    maximize=maximize,
                    constraints=constraints,
                )

                if should_log(LOG_REWARD):
                    logger.info(f"[REWARD] OTM reward={reward:.4f}, obj={objective_metric}, "
                               f"maximize={maximize}, constraints={len(constraints)}")

                return reward

            except Exception as e:
                if should_log(LOG_REWARD):
                    import traceback
                    logger.warning(f"[REWARD] OTM reward failed: {e}, falling back to single-metric")
                    logger.debug(f"[REWARD] Traceback: {traceback.format_exc()}")

        # Fallback is the inherited single metric intent-based reward
        return self._compute_reward(prev_kpi, curr_kpi, actions_len)
    
    # ---------- Public API ----------
    def step(self, last_playbook, kpi_dict: Optional[Dict] = None) -> Optional[np.ndarray]:
        """Enhanced step with contextual intelligence and feature completeness checking."""
        import time
        start_time = time.time()
        
        if kpi_dict is not None:
             kpi_update = kpi_dict
        else:
             kpi_update = self._read_latest_kpi()
        
        if kpi_update is None:
            return None

        # STATEFUL QUERY: Merge updates into persistent state to handle fragmented E2 messages
        if self.known_kpi_state is None:
             from copy import deepcopy
             self.known_kpi_state = deepcopy(kpi_update)
        else:
             # Merge CellMetrics (dictionaries)
             if "CellMetrics" in kpi_update:
                 if "CellMetrics" not in self.known_kpi_state:
                     self.known_kpi_state["CellMetrics"] = {}
                 self.known_kpi_state["CellMetrics"].update(kpi_update["CellMetrics"])
             
             # Merge UEMetrics (list replacement?) or recursive?
             # For now, just replace list-based fields and others
             for k, v in kpi_update.items():
                 if k != "CellMetrics":
                     self.known_kpi_state[k] = v
        
        # Use the accumulated full state for processing
        kpi = self.known_kpi_state

        # Extract features and check completeness
        row, completeness = self._extract_features_row(kpi)
        
        # Check if we have enough features to proceed
        # If we have the target metric (critical for reward), proceed even if completeness is low
        # This matches the logic in ObserverBridge to prevent stalling
        cell_metrics = kpi.get("CellMetrics", {})
        has_critical_metric = False
        
        # Check if intent metric is present (handling potential CSV name mapping)
        if self.intent.metric in cell_metrics and cell_metrics[self.intent.metric] is not None:
             has_critical_metric = True
        else:
             # Check potential alternatives for legacy/CSV compatibility
             alternatives = {
                 "DRB_PdcpSduDelayDl": ["delay_p95_ms", "UE_PDCP_Delay_DL_ms"],
                 "UE_DRB_PdcpSduDelayDl_UEID": ["UE_PDCP_Delay_DL_ms"],
             }.get(self.intent.metric, [])
             
             for alt in alternatives:
                 if alt in cell_metrics and cell_metrics[alt] is not None:
                     has_critical_metric = True
                     break
        
        if completeness < self.min_feature_completeness and not has_critical_metric:
            # Not enough features AND missing critical metric - skip this step
            self.buf.append(row)
            if should_log(LOG_OBSERVER):
                logger.debug(f"[OBSERVER] Skipping step: completeness={completeness:.1%} < min={self.min_feature_completeness:.1%} and critical metric {self.intent.metric} missing")
            return None  # Don't return state until we have enough features
            
        if should_log(LOG_OBSERVER) and completeness < self.min_feature_completeness and has_critical_metric:
             logger.debug(f"[OBSERVER] Proceeding with low completeness ({completeness:.1%}) because critical metric {self.intent.metric} is present")
        
        # Extract context (only if we have enough features)
        if self.enable_contextual_bandit:
            context = self._extract_context(kpi)
            if context:
                self.current_context = context
                self._update_context_history(context)

        s2 = self._update_window(row)  # [W, F] - already cleaned (no NaN values)
        
        # STABILIZATION: Time-based check
        current_time = time.time()
        time_since_action = current_time - self.last_action_time
        stabilization_period = 2.0 # Reduced from 20.0s to accelerate training
        
        if time_since_action < stabilization_period:
            if should_log(LOG_OBSERVER) and self.steps_since_action % 10 == 0:
                 # Log periodically
                 logger.debug(f"[OBSERVER] Stabilizing network: {time_since_action:.1f}/{stabilization_period}s")
            self.steps_since_action += 1 # Keep for logging stats if needed or just remove
            return None # Hold current action

        # If we have a previous KPI, compute reward and push to replay
        if self.last_kpi_raw is not None and self.last_state_win is not None and last_playbook is not None:
            s = self.last_state_win  # previous window [W, F] (from BEFORE stabilization wait)
            
            t0 = time.time()
            # Enhanced reward calculation
            if self.enable_contextual_bandit:
                r = self._compute_enhanced_reward(self.last_kpi_raw, kpi, last_playbook)
            else:
                r = self._compute_reward(self.last_kpi_raw, kpi, len(getattr(last_playbook, "actions", [])))
            
            # Store for external access (e.g. by ObserverBridge to update Cache)
            self.last_reward = r
            
            t_reward = time.time() - t0
            
            t0 = time.time()
            p = self.predictor.encode_playbook_onehot(last_playbook)  # [K, D]
            t_encode = time.time() - t0

            # Push transition to replay and learn
            if should_log(LOG_REWARD):
                logger.debug(f"[REWARD] Pushing experience: reward={r:.4f}, metric={self.intent.metric}, "
                            f"prev_val={self.last_kpi_raw.get('CellMetrics', {}).get(self.intent.metric, 'N/A')}, "
                            f"curr_val={kpi.get('CellMetrics', {}).get(self.intent.metric, 'N/A')}, "
                            f"replay_size={len(self.predictor.replay)}")
            
            t0 = time.time()
            self.predictor.replay.push(s, p, r, s2, False)
            t_push = time.time() - t0
            
            t0 = time.time()
            # Only train every 10 steps (check replay size or counter) -> actually we trigger every action now (every 5s)
            # Since we only run this block every 5s, we can afford to train every time!
            # It takes ~0.2s, which is fine every 5s.
            self.predictor.learn_step()
            t_learn = time.time() - t0
            
            # Log slow operations (adjusted threshold for learn since we expect it to be slower when it runs)
            if t_reward > 0.1 or t_encode > 0.1 or t_push > 0.1 or t_learn > 0.1:
                logger.warning(f"[PERF] Slow step detected: reward={t_reward:.4f}s, encode={t_encode:.4f}s, push={t_push:.4f}s, learn={t_learn:.4f}s")

        # Update previous pointers
        # - Make a deep copy of the KPI dict to avoid reference issues
        # - If we store a reference, modifications to the dict will affect both prev and curr
        from copy import deepcopy
        self.last_kpi_raw = deepcopy(kpi)
        self.last_state_win = s2.copy()  # Store cleaned state window
        self.steps_since_action = 0 # Reset counter after acting
        self.last_action_time = time.time() # Reset stabilization timer
        
        total_time = time.time() - start_time
        if total_time > 0.5:
             logger.warning(f"[PERF] Total step time: {total_time:.4f}s")
             
        return s2  # current window (cleaned, no NaN)

    # Public context access methods
    def get_current_context(self) -> Optional[NetworkContext]:
        """Get current network context."""
        return self.current_context

    def get_context_situation(self) -> str:
        """Get current network situation classification."""
        if self.current_context and self.enable_contextual_bandit:
            return self._classify_network_situation(self.current_context)
        return "unknown"

    def get_context_features(self) -> Optional[np.ndarray]:
        """Get normalized context features for external use."""
        if self.current_context and self.enable_contextual_bandit:
            return self.context_extractor.to_feature_vector(self.current_context)
        return None

    def get_context_tensor(self) -> Optional[np.ndarray]:
        """Get context as [W, F] tensor."""
        if self.current_context and self.enable_contextual_bandit:
            return self.context_extractor.to_state_tensor(self.current_context)
        return None

    def get_context_history_summary(self) -> Dict[str, Any]:
        """Get summary of recent context history."""
        if not self.enable_contextual_bandit or not self.context_history:
            return {}
        
        recent_contexts = self.context_history[-5:]  # Last 5 contexts
        
        return {
            "avg_latency": np.mean([c.latency_ms for c in recent_contexts]),
            "avg_throughput": np.mean([c.throughput_dl_mbps for c in recent_contexts]),
            "avg_bler": np.mean([c.bler for c in recent_contexts]),
            "avg_network_load": np.mean([c.network_load for c in recent_contexts]),
            "avg_quality_index": np.mean([c.quality_index for c in recent_contexts]),
            "situation_trend": [self._classify_network_situation(c) for c in recent_contexts],
            "history_length": len(self.context_history)
        }


if __name__ == "__main__":
    # Simple test
    from dataclasses import dataclass
    
    @dataclass
    class MockPredictor:
        def encode_playbook_onehot(self, playbook):
            return np.array([[1, 0, 0]])
        
        class MockReplay:
            def push(self, s, p, r, s2, done):
                print(f"Replay: reward={r:.3f}")
        
        replay = MockReplay()
        
        def learn_step(self):
            pass
    
    intent = Intent("REDUCE_LATENCY", "delay_p95_ms", 40.0)
    observer = RLObserver(MockPredictor(), intent, enable_contextual_bandit=False)
    print(f"Observer initialized with contextual bandit: {observer.enable_contextual_bandit}")