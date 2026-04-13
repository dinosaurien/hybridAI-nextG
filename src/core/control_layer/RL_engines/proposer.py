from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional, Set
import random
import logging
import math
from common.log_config import should_log, LOG_BANDIT
from common.types import ControlAction, Playbook

logger = logging.getLogger(__name__)

# Import contextual bandit components
try:
    from core.control_layer.RL_engines.bandit.context_extractor import NetworkContext
    BANDIT_AVAILABLE = True
except ImportError:
    # Fallback if bandit components not yet created
    BANDIT_AVAILABLE = False
    print("[Warning] Contextual bandit components not found in proposer. Using basic mode.")

# -----------------------------
# Global configuration (PoC defaults)
# -----------------------------

PLAYBOOK_K = 3         # actions per playbook
CANDIDATE_N = 5        # number of candidate playbooks per decision
COOLDOWN_STEPS = 1     # cooldown per (type, scope, entity)

ActionType = str   # {"MCS_CAP","TX_POWER","REPORTING"}
ScopeType = str    # {"CELL"}

# Action parameter ranges - define min, max, and step for continuous/discrete sampling
# These would later be sent by the actual network as a "what can we change now" message

# MCS (Modulation and Coding Scheme) - range: 0-28, typically use 12-28
MCS_DL_MIN = 12
MCS_DL_MAX = 28
MCS_DL_STEP = 2  # Sample every 2 MCS levels

# TX Power - range: 30.0 to 60.0 dBm, step 1.0
TX_POWER_DBM_MIN = 30.0
TX_POWER_DBM_MAX = 60.0
TX_POWER_DBM_STEP = 1.0

def generate_mcs_values(min_val=MCS_DL_MIN, max_val=MCS_DL_MAX, step=MCS_DL_STEP):
    """Generate MCS values in range."""
    return list(range(min_val, max_val + 1, step))

def generate_tx_power_values(min_val=TX_POWER_DBM_MIN, max_val=TX_POWER_DBM_MAX, step=TX_POWER_DBM_STEP):
    """Generate TX power values in range."""
    values = []
    current = min_val
    while current <= max_val:
        values.append(round(current, 1))
        current += step
    return values

@dataclass
class ActionSpace:
    cells: List[str]
    slices: List[str]

    def all_atomic_actions(self) -> List[ControlAction]:
        """Generate all possible atomic actions from defined ranges.
        Only MCS_CAP and TX_POWER are actuated over E2 in ns-3.
        """
        acts: List[ControlAction] = []

        mcs_values = generate_mcs_values()
        tx_power_values = generate_tx_power_values()

        for c in self.cells:
            for m in mcs_values:
                acts.append(ControlAction("MCS_CAP", "CELL", cell_id=c, params={"dl_mcs_max": m}))

            for tx_power in tx_power_values:
                acts.append(ControlAction("TX_POWER", "CELL", cell_id=c, params={"txPowerDbm": tx_power}))

        # NOOP action (kept for safety/fallback)
        acts.append(ControlAction("REPORTING", "CELL", params={"noop": True}))
        return acts

# Conflict and cooldown helpers
def conflict(a: ControlAction, b: ControlAction) -> bool:
    if a.type == "REPORTING" or b.type == "REPORTING":
        return False
    if a.type == b.type and a.scope == b.scope:
        if a.scope == "CELL" and a.cell_id == b.cell_id:
            return True
        if a.scope == "SLICE" and a.slice_id == b.slice_id:
            return True
        if a.scope == "UE" and a.ue_id == b.ue_id:
            return True
    return False

def cooldown_key(a: ControlAction) -> Tuple[str,str,str]:
    if a.scope == "CELL":
        ent = a.cell_id or "GLOBAL"
    elif a.scope == "SLICE":
        ent = a.slice_id or "GLOBAL"
    else:
        ent = a.ue_id or "GLOBAL"
    return (a.type, a.scope, ent)

def violates_cooldown(a: ControlAction, cooldown_clock: Dict[Tuple[str,str,str], int]) -> bool:
    return cooldown_clock.get(cooldown_key(a), 0) > 0

# -----------------------------
# Enhanced Cache of good playbooks with contextual awareness
# -----------------------------

class CacheLibrary:
    def __init__(self, max_per_key=20):
        self.max_per_key = max_per_key
        self.store: Dict[str, List[Tuple[Playbook, float]]] = {}
        # NEW: Context-aware cache
        self.context_store: Dict[str, List[Tuple[Playbook, float, str]]] = {}  # (playbook, score, situation)

    def key(self, intent_meta: Dict[str,Any]) -> str:
        # Debugging key mismatch
        scope = intent_meta.get("scope","GLOBAL")
        # Handle dict scope (convert to string representation or extract ID)
        if isinstance(scope, dict):
             # Simplified scope handling for key matching
             # If scope has 'cell_id', use that. Else 'GLOBAL'
             if 'cell_id' in scope:
                 cid = scope['cell_id']
                 if not str(cid).startswith('CELL_'):
                     scope = f"CELL_{cid}"
                 else:
                     scope = cid
             elif 'slice_id' in scope:
                 sid = scope['slice_id']
                 if not str(sid).startswith('SLICE_'):
                     scope = f"SLICE_{sid}"
                 else:
                     scope = sid
             else:
                 scope = "GLOBAL"
        
        intent = intent_meta.get("intent")
        # If 'intent' key missing, try to map from 'type' (Observer compatibility)
        if not intent:
             i_type = intent_meta.get("type", "UNKNOWN")
             if "LATENCY" in i_type or "delay" in str(intent_meta.get("metric")).lower():
                 intent = "LATENCY_P95"
             elif "THROUGHPUT" in i_type:
                 intent = "THR_DL"
             else:
                 intent = "LATENCY_P95" # Default default

        k = f"{intent}:{scope}"
        if should_log(LOG_BANDIT):
             logger.debug(f"[CACHE_KEY] Generated key '{k}' from meta: {intent_meta}")
        return k

    def context_key(self, intent_meta: Dict[str,Any], situation: str) -> str:
        """Enhanced cache key including network situation."""
        base_key = self.key(intent_meta)
        return f"{base_key}:{situation}"

    def add(self, intent_meta: Dict[str,Any], playbook: Playbook, score: float):
        k = self.key(intent_meta)
        arr = self.store.setdefault(k, [])
        arr.append((playbook, score))
        arr.sort(key=lambda x: x[1], reverse=True)
        if len(arr) > self.max_per_key:
            arr[:] = arr[:self.max_per_key]

    def add_contextual(self, intent_meta: Dict[str,Any], playbook: Playbook, score: float, situation: str):
        if not intent_meta:
            return

        # Add to general store (for backward compatibility/initialization)
        k = self.key(intent_meta)
        self.add(intent_meta, playbook, score)

        if not BANDIT_AVAILABLE:
            return
        
        k = self.context_key(intent_meta, situation)
        arr = self.context_store.setdefault(k, [])
        arr.append((playbook, score, situation))
        arr.sort(key=lambda x: x[1], reverse=True)
        if len(arr) > self.max_per_key:
            arr[:] = arr[:self.max_per_key]

    def sample(self, intent_meta: Dict[str,Any], m=2) -> List[Playbook]:
        k = self.key(intent_meta)
        arr = self.store.get(k, [])
        
        if not arr:
            return []
        take = min(m, len(arr))
        return [pb for (pb, _) in random.sample(arr, take)]

    def sample_contextual(self, intent_meta: Dict[str,Any], situation: str, m=2) -> List[Playbook]:
        """Sample playbooks that worked well in similar situations."""
        if not BANDIT_AVAILABLE:
            return self.sample(intent_meta, m)
        
        # Try exact situation match first
        k = self.context_key(intent_meta, situation)
        arr = self.context_store.get(k, [])
        
        # If no exact match, try related situations
        if not arr:
            related_situations = self._get_related_situations(situation)
            for related_sit in related_situations:
                related_k = self.context_key(intent_meta, related_sit)
                arr = self.context_store.get(related_k, [])
                if arr:
                    break
        
        # Fallback to general cache
        if not arr:
            return self.sample(intent_meta, m)
        
        take = min(m, len(arr))
        return [pb for (pb, _, _) in random.sample(arr, take)]

    def _get_related_situations(self, situation: str) -> List[str]:
        """Get related network situations for fallback."""
        related_map = {
            "high_latency": ["poor_quality", "normal"],
            "low_throughput": ["high_load", "normal"],
            "high_error_rate": ["poor_quality", "normal"],
            "high_load": ["low_throughput", "normal"],
            "poor_quality": ["high_error_rate", "high_latency", "normal"],
            "normal": ["poor_quality", "high_latency", "low_throughput"]
        }
        return related_map.get(situation, ["normal"])

# -----------------------------
# NEW: Contextual Action Weighting
# -----------------------------

class ContextualActionWeights:
    """Calculate action weights based on network context.
    
    NOTE: Heuristics have been disabled to allow for pure specific RL learning.
    This class now returns uniform weights to ensure unbiased exploration.
    """
    
    def __init__(self):
        # Define context-action weight mappings
        # Disabled manual heuristics to allow "propose whatever and learn" behavior
        self.situation_weights = {}

    def get_action_weight(self, action: ControlAction, situation: str) -> float:
        """Get weight for specific action in given situation.
        
        Returns 1.0 for all actions to ensure uniform sampling (unbiased exploration).
        """
        return 1.0

    def get_weighted_actions(self, all_actions: List[ControlAction], situation: str) -> List[ControlAction]:
        """Get weighted action list for contextual sampling.
        
        Since weights are uniform, this simply returns the original list 
        (or a uniformly scaled version, but we just return original for efficiency).
        """
        return all_actions # Return raw actions for uniform sampling

# -----------------------------
# Enhanced Proposer-side sampler with contextual intelligence
# -----------------------------

class ProposerSampler:
    def __init__(self):
        if BANDIT_AVAILABLE:
            self.contextual_weights = ContextualActionWeights()
            print("[Proposer] Contextual bandit mode enabled")
        else:
            print("[Proposer] Basic mode (no contextual bandit)")

    @staticmethod
    def sample_playbooks(action_space: ActionSpace, N=CANDIDATE_N, K=PLAYBOOK_K,
                         cooldown_clock: Optional[Dict[Tuple[str,str,str], int]] = None,
                         cache: Optional[CacheLibrary] = None,
                         intent_meta: Optional[Dict[str,Any]] = None,
                         epsilon: float = 0.1,
                         situation: str = "normal") -> List[Playbook]:
        """Original playbook sampling (fallback/compatibility)."""
        cooldown_clock = cooldown_clock or {}
        # NEW: Sample contextually based on Situation
        seeds = cache.sample_contextual(intent_meta, situation, m=min(2, N)) if cache else []
        playbooks: List[Playbook] = []

        # Extract constraint hints from intent_meta (e.g. {"dl_mcs_max": {"operator": "le", "threshold": 20}})
        constraint_hints = (intent_meta or {}).get("constraint_hints", {})

        # OTM constraint KPI names don't always match action param keys
        _CONSTRAINT_TO_PARAM = {
            "tx_power_dbm": "txPowerDbm",
        }

        def _action_matches_constraints(action: ControlAction) -> bool:
            """Check if an action's params fall within OTM constraint hints."""
            for kpi, hint in constraint_hints.items():
                param_key = _CONSTRAINT_TO_PARAM.get(kpi, kpi)
                val = action.params.get(param_key)
                if val is None:
                    continue
                op = hint.get("operator", "le")
                thr = hint.get("threshold", 0)
                if op in ("le", "lt") and float(val) > float(thr):
                    return False
                if op in ("ge", "gt") and float(val) < float(thr):
                    return False
            return True

        def random_playbook():
            actions = []
            all_acts = action_space.all_atomic_actions()

            # Bias towards actions that satisfy OTM constraints
            if constraint_hints:
                compliant = [a for a in all_acts if _action_matches_constraints(a)]
                pool = compliant if compliant else all_acts
            else:
                pool = all_acts

            tries = 0
            # VARIABLE LENGTH: Randomly choose 1 to K actions (e.g., 1-3)
            # This avoids "weird" consistent 3-command blocks
            target_k = random.randint(1, K)

            while len(actions) < target_k and tries < 50:
                a = random.choice(pool)
                if violates_cooldown(a, cooldown_clock):
                    tries += 1; continue
                if any(conflict(a, b) for b in actions):
                    tries += 1; continue
                actions.append(a)

            # No padding with NOOPs - we want concise playbooks
            return Playbook(actions)

        # Seeds (with mutation probability)
        for i, s in enumerate(seeds):
            if random.random() < epsilon:
                if should_log(LOG_BANDIT):
                    logging.info(f"[BANDIT] Seed {i}: EXPLORING (mutating seed) with eps={epsilon:.2f}")
                pb = ProposerSampler.mutate_playbook(s, action_space, cooldown_clock)
            else:
                if should_log(LOG_BANDIT):
                    logging.info(f"[BANDIT] Seed {i}: EXPLOITING (using seed) with eps={epsilon:.2f}")
                pb = s
            playbooks.append(pb)

        # Fill to N
        fill_count = 0
        while len(playbooks) < N:
            if should_log(LOG_BANDIT):
                # aggregated log to avoid spamming 5 times per step, or just log once
                pass 
            playbooks.append(random_playbook())
            fill_count += 1
            
        if fill_count > 0 and should_log(LOG_BANDIT):
             logging.info(f"[BANDIT] Generated {fill_count} pure random playbooks (exploration/fill)")
        return playbooks

    @staticmethod
    def sample_contextual_playbooks(action_space: ActionSpace, 
                                   context: Optional['NetworkContext'] = None,
                                   situation: str = "normal",
                                   N=CANDIDATE_N, K=PLAYBOOK_K,
                                   cooldown_clock: Optional[Dict[Tuple[str,str,str], int]] = None,
                                   cache: Optional[CacheLibrary] = None,
                                   intent_meta: Optional[Dict[str,Any]] = None,
                                   epsilon: float = 0.1,
                                   contextual_strength: float = 0.7) -> List[Playbook]:
        """Enhanced contextual playbook sampling."""
        
        # Fallback to original if contextual bandit not available
        if not BANDIT_AVAILABLE or context is None:
            return ProposerSampler.sample_playbooks(
                action_space, N, K, cooldown_clock, cache, intent_meta, epsilon
            )
        
        cooldown_clock = cooldown_clock or {}
        contextual_weights = ContextualActionWeights()
        
        # Get contextual seeds from cache
        seeds = []
        if cache:
            contextual_seeds = cache.sample_contextual(intent_meta, situation, m=min(2, N))
            regular_seeds = cache.sample(intent_meta, m=min(1, N-len(contextual_seeds)))
            seeds = contextual_seeds + regular_seeds

        playbooks: List[Playbook] = []

        def contextual_random_playbook():
            """Generate playbook with contextual action weighting."""
            actions = []
            all_acts = action_space.all_atomic_actions()
            
            # Apply contextual weighting
            if random.random() < contextual_strength:  # Use contextual weighting
                weighted_acts = contextual_weights.get_weighted_actions(all_acts, situation)
            else:  # Use uniform sampling
                weighted_acts = all_acts
            
            tries = 0
            # VARIABLE LENGTH: Randomly choose 1 to K actions (e.g., 1-3)
            target_k = random.randint(1, K)
            
            while len(actions) < target_k and tries < 50:
                a = random.choice(weighted_acts)
                if violates_cooldown(a, cooldown_clock):
                    tries += 1; continue
                if any(conflict(a, b) for b in actions):
                    tries += 1; continue
                actions.append(a)
                tries = 0  # Reset tries on successful addition
            
            # No padding with NOOPs
            return Playbook(actions)

        def contextual_mutate_playbook(pb: Playbook) -> Playbook:
            """Mutate playbook with contextual awareness."""
            if not pb.actions:
                return contextual_random_playbook()
            
            idx = random.randrange(len(pb.actions))
            new_actions = pb.actions.copy()
            all_acts = action_space.all_atomic_actions()
            
            # Apply contextual weighting to mutation candidates
            if random.random() < contextual_strength:
                weighted_acts = contextual_weights.get_weighted_actions(all_acts, situation)
            else:
                weighted_acts = all_acts
            
            for _ in range(20):
                cand = random.choice(weighted_acts)
                if violates_cooldown(cand, cooldown_clock):
                    continue
                tmp = new_actions.copy()
                tmp[idx] = cand
                if any(conflict(tmp[i], tmp[j]) for i in range(len(tmp)) for j in range(i+1,len(tmp))):
                    continue
                new_actions = tmp
                break
            
            return Playbook(new_actions)

        # Process seeds with contextual mutation
        for i, s in enumerate(seeds):
            if random.random() < epsilon:
                if should_log(LOG_BANDIT):
                    logging.info(f"[BANDIT] Seed {i}: EXPLORING (mutating seed) with eps={epsilon:.2f}")
                pb = contextual_mutate_playbook(s)
            else:
                if should_log(LOG_BANDIT):
                    logging.info(f"[BANDIT] Seed {i}: EXPLOITING (using seed) with eps={epsilon:.2f}")
                pb = s
            playbooks.append(pb)

        # Fill remaining slots with contextual random playbooks
        fill_count = 0
        while len(playbooks) < N:
            playbooks.append(contextual_random_playbook())
            fill_count += 1
            
        if fill_count > 0 and should_log(LOG_BANDIT):
             logging.info(f"[BANDIT] Generated {fill_count} contextual random playbooks (exploration/fill)")
        
        # Add metadata to track contextual generation
        for pb in playbooks:
            if hasattr(pb, 'metadata'):
                pb.metadata.update({
                    "generation_method": "contextual_bandit",
                    "context_situation": situation,
                    "contextual_strength": contextual_strength,
                    "context_metrics": {
                        "latency_ms": context.latency_ms if context else None,
                        "throughput_dl_mbps": context.throughput_dl_mbps if context else None,
                        "bler": context.bler if context else None,
                        "network_load": context.network_load if context else None
                    }
                })
            elif hasattr(pb, 'actions'):
                # Add simple metadata if playbook doesn't have metadata attribute
                pb.context_situation = situation
                pb.generation_method = "contextual_bandit"
        
        return playbooks

    @staticmethod
    def mutate_playbook(pb: Playbook, action_space: ActionSpace,
                        cooldown_clock: Dict[Tuple[str,str,str], int]) -> Playbook:
        """Original mutation method (kept for compatibility)."""
        if not pb.actions:
            return pb
        
        idx = random.randrange(len(pb.actions))
        new_actions = pb.actions.copy()
        all_acts = action_space.all_atomic_actions()
        
        for _ in range(20):
            cand = random.choice(all_acts)
            if violates_cooldown(cand, cooldown_clock):
                continue
            tmp = new_actions.copy()
            tmp[idx] = cand
            if any(conflict(tmp[i], tmp[j]) for i in range(len(tmp)) for j in range(i+1,len(tmp))):
                continue
            new_actions = tmp
            break
        
        return Playbook(new_actions)

    @staticmethod
    def analyze_playbook_context_fit(playbook: Playbook, context: 'NetworkContext', situation: str) -> Dict[str, Any]:
        """Analyze how well a playbook fits the current network context."""
        if not BANDIT_AVAILABLE or not context:
            return {"fit_score": 0.5, "analysis": "contextual analysis not available"}
        
        contextual_weights = ContextualActionWeights()
        
        total_weight = 0.0
        action_analysis = []
        
        for action in playbook.actions:
            if action.type == "REPORTING" and action.params.get("noop"):
                continue  # Skip NOOP actions
            
            weight = contextual_weights.get_action_weight(action, situation)
            total_weight += weight
            
            action_analysis.append({
                "action_type": action.type,
                "params": action.params,
                "context_weight": weight,
                "fit_assessment": "good" if weight > 1.5 else "neutral" if weight > 0.8 else "poor"
            })
        
        # Calculate overall fit score
        if len(action_analysis) > 0:
            avg_weight = total_weight / len(action_analysis)
            fit_score = min(1.0, avg_weight / 2.0)  # Normalize to [0, 1]
        else:
            fit_score = 0.5  # Neutral for NOOP-only playbooks
        
        return {
            "fit_score": fit_score,
            "avg_action_weight": avg_weight if len(action_analysis) > 0 else 1.0,
            "situation": situation,
            "action_count": len(action_analysis),
            "actions_analysis": action_analysis,
            "context_summary": {
                "latency_ms": context.latency_ms,
                "throughput_dl_mbps": context.throughput_dl_mbps,
                "bler": context.bler,
                "network_load": context.network_load
            }
        }

# Create a global instance for convenience
contextual_proposer = ProposerSampler() if BANDIT_AVAILABLE else None