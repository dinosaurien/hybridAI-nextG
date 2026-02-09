from __future__ import annotations
import uuid
import sys
import os
from typing import Any, Dict

from ain.brain.openai_client import fallback_intent_for_deviation

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    llm_dir = os.path.join(current_dir, "..", "RL_demo")
    
    if os.path.exists(llm_dir) and llm_dir not in sys.path:
        sys.path.insert(0, llm_dir)
        
    from demo.ain.RL_demo.ai_logic import generate_biased_gpt2_intent
    HAS_LLM = True
except ImportError:
    print("[GPT-2] Warning: llm_logic.py not found. LLM disabled.")
    HAS_LLM = False

def normalize_deviation(event: Dict[str, Any]) -> Dict[str, Any]:
    dev = {
        "source": event.get("source") or "other",
        "metric": event.get("metric") or event.get("metric_name") or "delay_p95_ms",
        "value": float(event.get("value", 0.0)),
        "baseline": event.get("baseline"),
        "target": event.get("target"),
        "direction": event.get("direction") or ("lower_better" if "delay" in (event.get("metric","")) else "higher_better"),
        "severity": event.get("severity") or "medium",
        "scope": {
            "cell_id": (event.get("scope") or {}).get("cell_id"),
            "slice_id": (event.get("scope") or {}).get("slice_id"),
            "region": (event.get("scope") or {}).get("region") or event.get("region"),
            "service": (event.get("scope") or {}).get("service") or event.get("service") or "demo",
            "tenancy": (event.get("scope") or {}).get("tenancy") or "prod",
        },
        "evidence_ref": event.get("evidence_ref"),
    }
    slo = event.get("slo")
    if slo and dev["target"] is None and "target" in slo:
        dev["target"] = float(slo["target"])
    if dev["target"] is None:
        dev["target"] = dev["value"] * (0.8 if dev["direction"] == "lower_better" else 1.2)
    return dev

def to_rl_intent(net_intent: Dict[str, Any]) -> Dict[str, Any]:
    """
    Maps the LLM's high-level intent to the specific configuration 
    the RL Agent (DQN) requires to select a playbook.
    """
    # Get the raw intent string from the LLM (e.g., "OPTIMIZE_UTILIZATION")
    intent_str = str(net_intent.get("intent", "")).upper()
    
    # Get the metric and target from the deviation data passed through
    slo_data = net_intent.get("slo", {})
    metric = slo_data.get("metric", "DRB_PdcpSduDelayDl")
    target = float(slo_data.get("target", 25.0))

    #Mapping logic
    if "THROUGHPUT" in intent_str:
        return {
            "type": "INCREASE_THROUGHPUT", 
            "metric": "thr_dl_bps", # Use standard internal metric name
            "target": target if target > 1000 else 50e6,
            "direction": "higher_better", 
            "action_cost": 0.01, 
            "reward_clip": 20.0
        }
        
    elif "UTILIZATION" in intent_str:
        # LLM suggested BACKHAUL_LIMIT -> OPTIMIZE_UTILIZATION
        return {
            "type": "OPTIMIZE_UTILIZATION",
            "metric": "RRU_PrbUsedDl", # Target PRB usage
            "target": 80.0, # Target 80% utilization max
            "direction": "lower_better",
            "action_cost": 0.01,
            "reward_clip": 20.0
        }
        
    elif "RELIABILITY" in intent_str:
        # LLM suggested TX_POWER -> INCREASE_RELIABILITY
        return {
            "type": "INCREASE_RELIABILITY",
            "metric": "bler_dl", # Target Block Error Rate
            "target": 0.05, # Target 5% error rate max
            "direction": "lower_better",
            "action_cost": 0.01,
            "reward_clip": 20.0
        }
        
    else:
        # Default / Fallback: REDUCE_LATENCY
        # Handles "BUFFER_SIZE" and generic fallback
        return {
            "type": "REDUCE_LATENCY", 
            "metric": metric, 
            "target": target, 
            "direction": "lower_better", 
            "action_cost": 0.01, 
            "reward_clip": 20.0
        }

def create_network_intent_from_deviation(dev: Dict[str, Any], use_llm: bool = True) -> Dict[str, Any]:
    dev_norm = normalize_deviation(dev)
    if use_llm:
        try:
            ni = reason_from_deviation(dev_norm)
        except OpenAIError:
            ni = fallback_intent_for_deviation(dev_norm)
    else:
        ni = fallback_intent_for_deviation(dev_norm)

    ni.setdefault("intent_id", str(uuid.uuid4()))
    ni.setdefault("category", "performance")
    ni.setdefault("goal", "restore_slo")
    ni.setdefault("scope", {})
    ni["scope"].setdefault("service", dev_norm["scope"]["service"])
    ni["scope"].setdefault("region", dev_norm["scope"]["region"] or "A")
    if dev_norm["scope"].get("cell_id"):
        ni["scope"]["cell_id"] = dev_norm["scope"]["cell_id"]
    if dev_norm["scope"].get("slice_id"):
        ni["scope"]["slice_id"] = dev_norm["scope"]["slice_id"]
    ni.setdefault("constraints", {
        "tenancy": dev_norm["scope"].get("tenancy", "prod"),
        "change_window": "22:00Z/2h",
        "max_risk": "low",
    })
    ni.setdefault("slo", ni.get("slo", {}))
    ni.setdefault("evidence_ref", dev_norm.get("evidence_ref", "telemetry://window/A"))
    return ni