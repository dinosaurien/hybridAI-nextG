import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class SLORewardCalculator:
    """Enhanced SLO-based reward calculator bridging LLM OTMs and RL logic."""
    
    def __init__(self, action_cost: float = 0.01, reward_clip: float = 100.0):
        self.action_cost = action_cost
        # Increased clip from 20 to 100 so the constraint penalties aren't ignored
        self.reward_clip = reward_clip 
        
    def calculate_reward(self, 
                         prev_state: Dict[str, Any],
                         curr_state: Dict[str, Any], 
                         active_otm: Dict[str, Any],
                         num_actions: int = 1) -> float:
        """Calculate reward using the dynamic LLM OTM and relative improvements."""
        
        if not active_otm:
            return 0.0

        reward = 0.0
        constraint_violated = False

        # Helper to extract flat KPI from nested kpi.raw format
        def _get_kpi(state, kpi_name):
            cell_metrics = state.get("CellMetrics", {})
            ue_metrics = state.get("UEMetrics", [])
            
            if kpi_name in cell_metrics:
                return float(cell_metrics[kpi_name])
            
            ue_vals = [float(ue.get(kpi_name, 0)) for ue in ue_metrics if kpi_name in ue]
            return sum(ue_vals) / len(ue_vals) if ue_vals else None

        # Parses the LLM's dynamically generated rules
        for c in active_otm.get("constraints", []):
            kpi = c.get("kpi")
            op = c.get("operator")
            threshold = float(c.get("threshold", 0.0))
            
            val = _get_kpi(curr_state, kpi)
            if val is not None:
                violation = False
                if op == "le" and val > threshold: violation = True
                elif op == "lt" and val >= threshold: violation = True
                elif op == "ge" and val < threshold: violation = True
                elif op == "gt" and val <= threshold: violation = True
                
                if violation:
                    constraint_violated = True
                    reward -= 50.0  # penalty for violating LLM SLA

        if not constraint_violated:
            obj = active_otm.get("objective", {})
            if obj:
                kpi = obj.get("kpi")
                maximize = obj.get("maximize", True)
                
                prev_val = _get_kpi(prev_state, kpi)
                curr_val = _get_kpi(curr_state, kpi)
                
                if prev_val is not None and curr_val is not None:
                    if maximize:
                        delta_raw = curr_val - prev_val
                    else:
                        delta_raw = prev_val - curr_val
                        
                    delta_rel = delta_raw / max(abs(prev_val), 1e-6)
                    reward += (delta_rel * 10.0) 

        reward -= (self.action_cost * num_actions)

        return float(np.clip(reward, -self.reward_clip, self.reward_clip))