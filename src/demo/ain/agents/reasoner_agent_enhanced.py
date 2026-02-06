"""
Adaptive Reasoner Agent with Static SLOs and Baseline Learning

- SLOs: Static, loaded from operator-defined JSON file
- Baseline: Learned from environment for context understanding
- Reasoner: Uses SLO as target, baseline to understand optimization potential
"""

import asyncio
import logging
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from collections import deque
from datetime import datetime, timezone
from .utils import make_msg
from ain.common.log_config import should_log, LOG_INTENT

logger = logging.getLogger(__name__)


class SLOConfig:
    """Static SLO configuration loaded from JSON file."""
    
    def __init__(self, slo_file: Optional[str] = None):
        """
        Args:
            slo_file: Path to SLO JSON file (e.g., "configs/slos.json")
        """
        self.slos: Dict[str, Dict[str, Any]] = {}  # metric -> SLO config
        self.slo_file = slo_file or "configs/slos.json"
        self._load_slos()
    
    def _load_slos(self):
        """Load SLOs from JSON file."""
        slo_path = Path(self.slo_file)
        if not slo_path.exists():
            logger.warning(f"[SLO] SLO file not found: {self.slo_file}, using defaults")
            return
        
        try:
            with open(slo_path, 'r') as f:
                data = json.load(f)
            
            # Expected format:
            # {
            #   "slos": [
            #     {
            #       "metric": "DRB_PdcpSduDelayDl",
            #       "target": 40.0,
            #       "direction": "lower_better",
            #       "tolerance": 0.1,  # 10% tolerance
            #       "priority": "high"
            #     },
            #     ...
            #   ]
            # }
            
            slos_list = data.get("slos", [])
            for slo in slos_list:
                metric = slo.get("metric")
                if metric:
                    self.slos[metric] = {
                        "target": float(slo.get("target")),
                        "direction": slo.get("direction", "lower_better"),
                        "tolerance": float(slo.get("tolerance", 0.1)),  # 10% default
                        "priority": slo.get("priority", "medium"),
                        "slo_id": slo.get("slo_id", f"slo_{metric}"),
                    }
            
            if should_log(LOG_INTENT):
                logger.info(f"[SLO] Loaded {len(self.slos)} SLOs from {self.slo_file}")
                for metric, config in self.slos.items():
                    logger.info(f"[SLO]   {metric}: target={config['target']}, direction={config['direction']}")
        
        except Exception as e:
            logger.error(f"[SLO] Error loading SLO file: {e}", exc_info=True)
    
    
    def get_slo(self, metric: str) -> Optional[Dict[str, Any]]:
        """Get SLO configuration for a metric (supports regex)."""
        # 1. Exact match
        if metric in self.slos:
            return self.slos[metric]
        
        # 2. Regex match
        for key, config in self.slos.items():
            # If key contains regex special chars, try matching
            if any(c in key for c in "*?+^$[](){}|\\"):
                try:
                    # Anchor match to ensure full string or intended pattern
                    if re.fullmatch(key, metric):
                        return config
                except re.error:
                    continue
        return None
    
    def is_metric_tracked(self, metric: str) -> bool:
        """Check if metric has an SLO defined."""
        return self.get_slo(metric) is not None
    
    def meets_slo(self, metric: str, value: float) -> bool:
        """Check if current value meets SLO target."""
        slo = self.get_slo(metric)
        if not slo:
            return True  # No SLO defined, assume it's met
        
        target = slo["target"]
        tolerance = slo["tolerance"]
        direction = slo["direction"]
        
        if direction == "lower_better":
            # Value should be <= target (with tolerance)
            return value <= target * (1 + tolerance)
        elif direction == "higher_better":
            # Value should be >= target (with tolerance)
            return value >= target * (1 - tolerance)
        else:
            # Moderate: value should be within range
            return abs(value - target) / max(abs(target), 0.001) <= tolerance


class BaselineLearner:
    """Learns baseline distributions from observed KPIs for context."""
    
    def __init__(self, min_samples: int = 50, window_size: int = 200):
        self.min_samples = min_samples
        self.window_size = window_size
        self.metric_baselines: Dict[str, deque] = {}
        self.metric_stats: Dict[str, Dict[str, float]] = {}
        self.is_learned: Dict[str, bool] = {}
    
    def add_observation(self, metric: str, value: float):
        """Add a KPI observation to learn baseline."""
        if metric not in self.metric_baselines:
            self.metric_baselines[metric] = deque(maxlen=self.window_size)
            self.is_learned[metric] = False
        
        self.metric_baselines[metric].append(value)
        
        if len(self.metric_baselines[metric]) >= self.min_samples:
            self._update_stats(metric)
            if not self.is_learned[metric]:
                self.is_learned[metric] = True
                if should_log(LOG_INTENT):
                    stats = self.metric_stats[metric]
                    logger.info(f"[BASELINE] Learned baseline for {metric}: "
                              f"mean={stats['mean']:.2f}, p50={stats['p50']:.2f}, p90={stats['p90']:.2f}")
    
    def _update_stats(self, metric: str):
        """Update statistical measures for a metric."""
        values = np.array(list(self.metric_baselines[metric]))
        
        self.metric_stats[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "p10": float(np.percentile(values, 10)),
            "p25": float(np.percentile(values, 25)),
            "p50": float(np.percentile(values, 50)),
            "p75": float(np.percentile(values, 75)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
        }
    
    def get_baseline(self, metric: str) -> Optional[Dict[str, float]]:
        """Get baseline statistics for a metric."""
        return self.metric_stats.get(metric)
    
    def is_environment_optimal(self, metric: str, value: float, direction: str) -> bool:
        """
        Check if value is already optimal in this environment (based on baseline).
        Used to understand if optimization is even possible.
        """
        baseline = self.get_baseline(metric)
        if not baseline:
            return False  # Don't know yet
        
        if direction == "lower_better":
            # If we're already at p10 or better, environment is optimal
            return value <= baseline.get("p10", baseline["p50"])
        elif direction == "higher_better":
            # If we're already at p90 or better, environment is optimal
            return value >= baseline.get("p90", baseline["p50"])
        return False


from enum import Enum, auto

class IntentState(Enum):
    MONITORING = auto()
    ACTIVATING = auto()
    ASSURANCE = auto()
    WITHDRAWAL = auto()


class EnhancedReasonerAgent:
    """
    Reasoner that uses static SLOs and learned baseline for context-aware intent creation.
    
    Logic:
    1. SLO defines the target (static, operator-defined)
    2. Baseline learns what's normal in this environment
    3. Create intent if: value doesn't meet SLO AND optimization is possible
    """
    
    def __init__(self, bus, knowledge_base=None, use_llm: bool = False, 
                 slo_file: Optional[str] = None):
        """
        Args:
            bus: MemBus instance
            knowledge_base: Knowledge base instance (e.g., CacheLibrary)
            use_llm: Not used, kept for compatibility
            slo_file: Path to SLO JSON file (default: "configs/slos.json")
        """
        self.bus = bus
        self.knowledge_base = knowledge_base
        self.use_llm = use_llm  # Not used, kept for compatibility
        self.re_reason_timer = 0 
        
        self.current_intent: Optional[Dict[str, Any]] = None
        self.active_intent_id: Optional[str] = None
        
        # State Machine
        self.state = IntentState.MONITORING
        
        # Static SLO configuration
        self.slo_config = SLOConfig(slo_file=slo_file)
        
        # Baseline learning for context
        self.baseline_learner = BaselineLearner(min_samples=50, window_size=200)
    
    async def run(self):
        """Subscribe to deviation events and KPIs."""
        q_deviation = await self.bus.sub("deviation.detected")
        q_kpi = await self.bus.sub("kpi.raw")  # Track KPIs for baseline learning
        
        # Learn baseline from KPIs in background
        asyncio.create_task(self._learn_baseline_from_kpis(q_kpi))
        
        while True:
            msg = await q_deviation.get()
            deviation = msg.payload
            metric = deviation.get("metric")
            value = deviation.get("value")
            scope = deviation.get("scope", {})
            
            if should_log(LOG_INTENT) and self.state == IntentState.MONITORING:
                logger.debug(f"[INTENT] Received deviation in MONITORING: {metric}={value:.2f}")
            
            # Check if metric has SLO defined
            slo = self.slo_config.get_slo(metric)
            if not slo:
                continue
            
            # Add to baseline learner for context
            self.baseline_learner.add_observation(metric, value)
            
            # --- State Machine Logic ---

            if self.state == IntentState.ASSURANCE:
                if not self.slo_config.meets_slo(metric, value):
                    self.re_reason_timer += 1
                    if self.re_reason_timer >= 10: # Re-reason every 10 deviations (approx 10 seconds)
                        logger.info(f"[STATE] Still violating SLO. Re-triggering LLM Reasoner...")
                        self.state = IntentState.ACTIVATING 
                        self.re_reason_timer = 0
                else:
                    self.re_reason_timer = 0 # Reset if SLO is met
            
            # 1. MONITORING State
            if self.state == IntentState.MONITORING:
                if not self.slo_config.meets_slo(metric, value):
                    # Check optimization potential
                    baseline = self.baseline_learner.get_baseline(metric)
                    is_optimal = False
                    if baseline:
                        is_optimal = self.baseline_learner.is_environment_optimal(metric, value, slo["direction"])
                    
                    if is_optimal:
                        if should_log(LOG_INTENT):
                            logger.warning(f"[INTENT] SLO violation but environment optimal. Staying in MONITORING.")
                    else:
                        logger.info(f"[STATE] MONITORING -> ACTIVATING (SLO violation: {metric}={value:.2f})")
                        self.state = IntentState.ACTIVATING
            
            # 2. ACTIVATING State (Transient)
            if self.state == IntentState.ACTIVATING:
                await self._create_intent_from_slo(metric, value, slo, scope)
                logger.info(f"[STATE] ACTIVATING -> ASSURANCE")
                self.state = IntentState.ASSURANCE
            
            # 3. ASSURANCE State
            if self.state == IntentState.ASSURANCE:
                # Check if we are satisfying the CURRENT intent
                if self.current_intent and self.current_intent.get("metric") == metric:
                    if self.slo_config.meets_slo(metric, value):
                        logger.info(f"[STATE] ASSURANCE -> WITHDRAWAL (SLO met: {metric}={value:.2f})")
                        self.state = IntentState.WITHDRAWAL
                    else:
                        # Still optimizing
                        if should_log(LOG_INTENT):
                             logger.debug(f"[INTENT] ASSURANCE: Ensuring {metric} (current={value:.2f}, target={slo['target']})")
                else:
                    # Metric mismatch or no intent? specific edge case, stay or reset
                    # If we receive deviation for a DIFFERENT metric, ideally queue it or ignore?
                    # For now, ignore deviations for other metrics while ensuring one.
                    pass

            # 4. WITHDRAWAL State (Transient)
            if self.state == IntentState.WITHDRAWAL:
                if self.current_intent:
                    if should_log(LOG_INTENT):
                        logger.info(f"[INTENT] Clearing satisfied intent for {self.current_intent.get('metric')}")
                    
                    self.current_intent = None
                    self.active_intent_id = None
                    
                    # Publish empty/clear intent
                    await self.bus.pub("intent.current", make_msg(
                        "intent.current", "INTENT", "intent.v1", {}
                    ))
                    # Also publish clear signal to RL observer
                    await self.bus.pub("intent.rl", make_msg(
                        "intent.rl", "RL_INTENT", "rl_intent.v1", {
                            "type": "MONITORING",
                            "metric": "none",
                            "target": 0.0,
                            "direction": "none"
                        }
                    ))
                
                logger.info(f"[STATE] WITHDRAWAL -> MONITORING")
                self.state = IntentState.MONITORING
            
            # --- Visualization ---
            if should_log(LOG_INTENT) and self.state != IntentState.MONITORING:
                active_intents = [self.current_intent] if self.current_intent else []
                # Simple visualization of active intents
                logger.info(f"[INTENT] Active Intents: {json.dumps(active_intents, default=str)}")
    
    async def _learn_baseline_from_kpis(self, q):
        """Continuously learn baseline from KPI stream."""
        while True:
            msg = await q.get()
            kpi = msg.payload.get("kpi", {})
            cell_metrics = kpi.get("CellMetrics", {})
            
            # Track metrics that have SLOs defined
            for metric in self.slo_config.slos.keys():
                value = cell_metrics.get(metric)
                if value is not None:
                    self.baseline_learner.add_observation(metric, float(value))
    
    async def _create_intent_from_slo(self, metric: str, value: float, slo: Dict[str, Any], scope: Dict[str, Any] = None):
        """Create and publish intent to meet SLO target."""
        try:
            # --- START BIASED LLM INJECTION ---
            if self.use_llm:
                print(f"\n[GPT-2] Detected SLO Violation: {metric}={value:.2f}. Triggering Biased GPT-2...")
                try:
                    import sys
                    import os
                    # Ensure the path where llm_logic.py lives is in Python's search list
                    llm_dir = "/home/exposed/Desktop/hybridAI-nextG/src/demo/ain/RL_demo"
                    if llm_dir not in sys.path:
                        sys.path.append(llm_dir)
                    
                    from llm_logic import generate_biased_gpt2_intent
                    
                    # Call your biased GPT-2 logic
                    llm_result = generate_biased_gpt2_intent({"metric": metric, "target": slo["target"], "scope": scope})
                    
                    intent_type = llm_result["intent"]
                    suggestion = llm_result["slo"]["suggestion"]
                    print(f"[GPT-2] LLM Suggestion: {suggestion} -> Intent: {intent_type}\n")
                except Exception as e:
                    logger.error(f"[GPT-2] LLM Failed: {e}. Falling back to default.")
                    intent_type = "REDUCE_LATENCY" if slo["direction"] == "lower_better" else "INCREASE_THROUGHPUT"
                    suggestion = "Default Fallback"
            else:
                # Default non-LLM logic
                if slo["direction"] == "lower_better":
                    intent_type = "REDUCE_LATENCY"
                elif slo["direction"] == "higher_better":
                    intent_type = "INCREASE_THROUGHPUT"
                else:
                    intent_type = "OPTIMIZE_UTILIZATION"
                suggestion = "Rule-based"
            # --- END BIASED LLM INJECTION ---

            target = slo["target"]
            
            # Create RL Intent
            rl_intent = Intent(
                type=intent_type,
                metric=metric,
                target=float(target),
                direction=slo["direction"],
                action_cost=0.01,
                reward_clip=20.0,
            )
            
            # Store current intent
            self.current_intent = {
                "intent_id": f"intent_{metric}_{datetime.now(timezone.utc).timestamp()}",
                "metric": rl_intent.metric,
                "target": rl_intent.target,
                "direction": rl_intent.direction,
                "type": rl_intent.type,
                "slo_id": slo.get("slo_id"),
                "scope": scope or {},
                "llm_suggestion": suggestion # For tracking
            }
            self.active_intent_id = self.current_intent["intent_id"]
            
            # Publish intent to membus
            await self.bus.pub("intent.current", make_msg(
                "intent.current", "INTENT", "intent.v1", self.current_intent
            ))
            
            # Also publish RL intent format for observer
            await self.bus.pub("intent.rl", make_msg(
                "intent.rl", "RL_INTENT", "rl_intent.v1", {
                    "type": rl_intent.type,
                    "metric": rl_intent.metric,
                    "target": rl_intent.target,
                    "direction": rl_intent.direction,
                    "action_cost": rl_intent.action_cost,
                    "reward_clip": rl_intent.reward_clip,
                    "scope": scope or {},
                }
            ))
            
            if should_log(LOG_INTENT):
                logger.info(f"[INTENT] Suggested Action: {intent_type} (LLM: {suggestion})")
            
        except Exception as e:
            logger.error(f"[INTENT] Error creating intent from SLO: {e}", exc_info=True)
