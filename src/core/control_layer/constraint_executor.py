"""
Objective-Aware Deterministic Constraint Executor.

Deterministic, explainable actuation mechanism. Constraints define the
FEASIBLE SET (e.g., MCS ∈ [0, 20]) and the objective determines WHERE
within that set to operate:
  - "minimize latency" → lower MCS (fewer retransmissions), higher TX power
  - "maximize throughput" → higher MCS (more bits/symbol), higher TX power

Every parameter choice is traceable to a heuristic rule.
"""

import asyncio
import time
import logging
from typing import Any, Dict, Optional, Tuple
from core.bus.messages import make_msg
from core.common.types import ControlAction, Playbook

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────
#  Actuatable parameter definitions
# ────────────────────────────────────────────────────────────────────────

_CONSTRAINT_MAP = {
    "dl_mcs_max": {
        "action_type": "MCS_CAP",
        "param_key": "dl_mcs_max",
        "clamp": (0, 28),
        "cast": int,
    },
    "tx_power_dbm": {
        "action_type": "TX_POWER",
        "param_key": "txPowerDbm",
        "clamp": (30.0, 60.0),
        "cast": float,
    },
}

# ────────────────────────────────────────────────────────────────────────
#  Objective → Parameter Direction Heuristics
#
#  For each known objective KPI, define the preferred direction for each
#  actuatable parameter. This is the deterministic replacement for the
#  RL reward function: instead of learning which direction is better,
#  we encode domain knowledge directly.
#
#  "bias" is a float in [0.0, 1.0] that controls how far from the
#  constraint boundary we operate:
#    0.0 = use the boundary value (no bias)
#    1.0 = push all the way to the opposite clamp limit
#
#  "direction" is "low" or "high":
#    For "le" (upper bound) constraints: "low" moves below threshold
#    For "ge" (lower bound) constraints: "high" moves above threshold
# ────────────────────────────────────────────────────────────────────────

_OBJECTIVE_PROFILES = {
    # Minimize downlink latency:
    # Lower MCS → more robust modulation → fewer retransmissions → lower delay
    # Higher TX power → better SNR → fewer decoding errors → lower delay
    "DRB_PdcpSduDelayDl": {
        "dl_mcs_max":   {"direction": "low",  "bias": 0.3},
        "tx_power_dbm": {"direction": "high", "bias": 0.5},
    },
    # Same profile for per-UE latency variant
    "UE_DRB_PdcpSduDelayDl_UEID": {
        "dl_mcs_max":   {"direction": "low",  "bias": 0.3},
        "tx_power_dbm": {"direction": "high", "bias": 0.5},
    },
    # Maximize downlink throughput:
    # Higher MCS → more bits per symbol → higher throughput
    # Higher TX power → better SNR → supports higher MCS reliably
    "UE_DRB_UEThpDl_UEID": {
        "dl_mcs_max":   {"direction": "high", "bias": 0.0},
        "tx_power_dbm": {"direction": "high", "bias": 0.4},
    },
    # Minimize block error rate:
    # Lower MCS → more conservative modulation → fewer errors
    # Higher TX power → better signal quality → fewer errors
    "UE_DRB_BlerDl_UEID": {
        "dl_mcs_max":   {"direction": "low",  "bias": 0.4},
        "tx_power_dbm": {"direction": "high", "bias": 0.5},
    },
    # PRB utilization management:
    # Lower MCS → each UE uses more PRBs per bit → reduce headroom
    # TX power: moderate, not the primary lever for PRB utilization
    "RRU_PrbUsedDl": {
        "dl_mcs_max":   {"direction": "low",  "bias": 0.2},
        "tx_power_dbm": {"direction": "high", "bias": 0.2},
    },
}

# Neutral profile: no bias, use constraint thresholds as-is
_NEUTRAL_PROFILE = {
    "dl_mcs_max":   {"direction": "neutral", "bias": 0.0},
    "tx_power_dbm": {"direction": "neutral", "bias": 0.0},
}


class ConstraintExecutor:

    MIN_ACTION_INTERVAL = 5.0  # seconds, matches previous RL observer's rate limit

    def __init__(self, bus, default_cell_id: str = "CELL_001",
                 objective_aware: bool = True,
                 allow_adaptive_mcs: bool = True):
        self.bus = bus
        self.default_cell_id = default_cell_id
        self.objective_aware = objective_aware
        # When False, the `ge`/`gt` dl_mcs_max → mcs=-1 escape hatch is
        # disabled and the executor dispatches a concrete integer instead.
        # Used in evaluation mode: ns-3's in-scheduler AMC (mcs=-1) regulates
        # the channel perfectly and the LLM never sees failed episodes, which
        # starves Reflexion of material to reflect on. Forcing specific MCS
        # values makes the LLM's choice evaluable.
        self.allow_adaptive_mcs = allow_adaptive_mcs
        self._last_dispatch_time = 0.0
        # Idempotency: track the last value dispatched per (action_type, cell)
        # so successive OTMs that compute the same operating point don't
        # re-send E2 commands — the extra churn was observed to re-trigger
        # MiniRocket after actuator state settled.
        self._last_dispatched: Dict[Tuple[str, str], Any] = {}

    async def run(self):
        q_intent = await self.bus.sub("intent.current")
        logger.info("[CONSTRAINT_EXEC] Objective-aware executor ready.")

        while True:
            msg = await q_intent.get()
            otm = msg.payload
            constraints = otm.get("constraints", [])

            if not constraints:
                continue

            # Rate limit: don't flood ns-3
            now = time.time()
            elapsed = now - self._last_dispatch_time
            if elapsed < self.MIN_ACTION_INTERVAL:
                await asyncio.sleep(self.MIN_ACTION_INTERVAL - elapsed)

            # Extract objective to drive parameter selection (deploy mode only)
            objective = otm.get("objective", {}) if self.objective_aware else {}

            # Resolve the target cell from OTM metadata stamped by the
            # orchestrator; fall back to default_cell_id only if unset.
            meta = otm.get("metadata") or {}
            target_cell = meta.get("cell_id") or self.default_cell_id

            actions = self._constraints_to_actions(constraints, objective, target_cell)

            if not actions:
                logger.debug("[CONSTRAINT_EXEC] No actuatable constraints in OTM, skipping.")
                continue

            playbook = Playbook(actions)
            self._last_dispatch_time = time.time()

            action_summary = ", ".join(
                f"{a.type}={list(a.params.values())[0]}" for a in actions
            )
            obj_kpi = objective.get("kpi", "none")
            logger.info(f"[CONSTRAINT_EXEC] Executing {len(actions)} action(s) "
                        f"(objective: {obj_kpi}): {action_summary}")

            await self.bus.pub("actor.apply", make_msg(
                "constraint_exec", "APPLY", "v1",
                {"playbook": playbook}
            ))

    def _constraints_to_actions(self, constraints: list,
                                objective: dict,
                                target_cell: Optional[str] = None) -> list:
        """Map OTM constraints to concrete E2 actions, biased by the objective.

        The objective determines the operating point within constraint bounds:
        - Constraints define boundaries (e.g., MCS ≤ 20)
        - The objective profile decides whether to operate near the boundary
          or pull toward the opposite end of the parameter range

        Side-effect: writes 'applied_value' back into each actuatable constraint
        so that episode recording and adaptation prompts know what was actually
        sent to ns-3 (not just the LLM's threshold).
        """
        # Look up the objective profile for this KPI
        obj_kpi = objective.get("kpi", "")
        profile = _OBJECTIVE_PROFILES.get(obj_kpi, _NEUTRAL_PROFILE)

        actions = []
        seen_types = set()

        for c in constraints:
            kpi = c.get("kpi", "")
            mapping = _CONSTRAINT_MAP.get(kpi)

            if not mapping:
                continue  # Not actuatable (observational KPI)

            action_type = mapping["action_type"]

            # Deduplicate — only one action per type per dispatch
            if action_type in seen_types:
                continue
            seen_types.add(action_type)

            raw_threshold = c.get("threshold", 0)
            operator = c.get("operator", "le")
            lo, hi = mapping["clamp"]

            is_llm_adapted = (
                c.get("adapted_by") == "cognitive_llm" or 
                c.get("origin") == "cognitive_llm"
            )
            is_procedure_constraint = (c.get("origin") == "procedure_step")

            if is_procedure_constraint and not is_llm_adapted:
                # Apply heuristic bias ONLY for default procedure steps
                param_profile = profile.get(kpi, {"direction": "neutral", "bias": 0.0})
                direction = param_profile["direction"]
                bias = param_profile["bias"]
            else:
                # It came from the LLM. Apply EXACTLY what it asked for.
                direction = "neutral"
                bias = 0.0

            # Adaptive-MCS escape hatch: ns-3's set-mcs PINS MCS to the
            # supplied value (see eval_scenario.cc ChangeMcs — FixedMcsDl
            # is toggled on). For a `ge` constraint on dl_mcs_max under a
            # throughput-maximising objective, pinning at the threshold
            # would cap the scheduler at the floor, which is the opposite
            # of the expert intent ("allow MCS at least this high"). Dispatch
            # mcs=-1 instead — the scenario's else-branch disables FixedMcs
            # and restores adaptive scheduling.
            #
            # Gated by allow_adaptive_mcs: evaluation mode disables it so the
            # LLM is forced to pick a concrete integer (otherwise ns-3's AMC
            # trivially handles everything and Reflexion has no failures to
            # learn from).
            obj_direction = (_OBJECTIVE_PROFILES.get(obj_kpi, {}).get(kpi, {}).get("direction"))

            use_adaptive_mcs = (
                self.allow_adaptive_mcs
                and kpi == "dl_mcs_max"
                and obj_direction == "high"
                and raw_threshold >= 24  # Catch LLM attempting to max out MCS
            )

            if use_adaptive_mcs:
                value = -1  # ns-3 sentinel: disable FixedMcs, use adaptive
                logger.info(
                    f"[CONSTRAINT_EXEC] {kpi} {operator} {raw_threshold} under "
                    f"objective '{obj_kpi}' → dispatching adaptive MCS (mcs=-1) "
                    f"to prevent hard-pinning packet loss."
                )
            else:
                # Compute the objective-biased operating point
                value = self._compute_biased_value(
                    raw_threshold, operator, lo, hi, direction, bias
                )
                value = mapping["cast"](max(lo, min(hi, value)))

            # Write applied_value back into the constraint for traceability.
            # This is critical: adaptation prompts and episode recording must
            # know what was actually sent to ns-3, not just the threshold.
            c["applied_value"] = value

            # Idempotency: suppress dispatch if this (action_type, cell) pair
            # is already at the computed value. Prevents actuator churn on
            # successive OTMs that settle on the same operating point.
            cell_key = target_cell or self.default_cell_id
            dispatch_key = (action_type, cell_key)
            if self._last_dispatched.get(dispatch_key) == value:
                logger.info(
                    f"[CONSTRAINT_EXEC] {kpi}={value}: unchanged from last "
                    f"dispatch, skipping."
                )
                continue
            self._last_dispatched[dispatch_key] = value

            action = ControlAction(
                type=action_type,
                scope="CELL",
                cell_id=cell_key,
                params={mapping["param_key"]: value},
            )
            actions.append(action)

            # Log the reasoning for auditability
            if direction != "neutral" and bias > 0:
                logger.info(
                    f"[CONSTRAINT_EXEC] {kpi}: threshold={raw_threshold} "
                    f"({operator}), objective bias '{direction}' "
                    f"(bias={bias:.1f}) → applied value={value}"
                )
            elif is_llm_adapted:
                logger.info(
                    f"[CONSTRAINT_EXEC] {kpi}: threshold={raw_threshold} "
                    f"(LLM tuned, bypassing heuristic bias) → applied value={value}"
                )
            elif is_procedure_constraint:
                logger.info(
                    f"[CONSTRAINT_EXEC] {kpi}: threshold={raw_threshold} "
                    f"(procedure step, no bias) → applied value={value}"
                )

        return actions

    @staticmethod
    def _compute_biased_value(threshold: float, operator: str,
                              lo: float, hi: float,
                              direction: str, bias: float) -> float:
        """Compute an objective-biased parameter value within constraint bounds.

        For an upper-bound constraint (operator "le", threshold = 20, range [0, 28]):
          - direction "low", bias 0.3 → 20 - 0.3*(20-0)  = 14.0
          - direction "high", bias 0   → 20 (use boundary)
          - direction "neutral"        → 20 (use boundary)

        For a lower-bound constraint (operator "ge", threshold = 46, range [30, 60]):
          - direction "high", bias 0.5 → 46 + 0.5*(60-46) = 53.0
          - direction "low", bias 0    → 46 (use boundary)
          - direction "neutral"        → 46 (use boundary)

        This is the deterministic equivalent of the RL agent's learned policy:
        instead of exploring the action space, we jump directly to the
        domain-optimal operating point.
        """
        if direction == "neutral" or bias == 0.0:
            return threshold

        if operator in ("le", "lt"):
            # Upper bound: threshold is the max allowed
            if direction == "low":
                # Pull below the threshold toward the lower clamp
                return threshold - bias * (threshold - lo)
            elif direction == "high":
                # Already at the top; can't go higher than threshold
                return threshold

        elif operator in ("ge", "gt"):
            # Lower bound: threshold is the min required
            if direction == "high":
                # Push above the threshold toward the upper clamp
                return threshold + bias * (hi - threshold)
            elif direction == "low":
                # Already at the bottom; can't go lower than threshold
                return threshold

        return threshold
